"""설계 계획 → 코호트 SQL. 실제 카탈로그 스키마 + 합성 데이터(client fixture)로 검증한다."""

import pytest

from req2data.agents.designer import DatasetDesigner
from req2data.agents.sql_builder import CohortSQLBuilder, PlanError
from req2data.schemas import DesignPlan
from tests.fixtures import scenario as sc

REPURCHASE = {
    **sc.DESIGN_V2,
    "features": [
        *sc.DESIGN_V2["features"],
        {"name": "payment_type", "source_column": "order_payments.payment_type", "agg": "sum"},
        {"name": "state", "source_column": "customers.customer_state"},
    ],
}
COMPLAINT = {
    "analysis_unit": "주문",
    "entity_key": "orders.order_id",
    "reference_time": "orders.order_purchase_timestamp",
    "cohort": "every_event",
    "target": {
        "kind": "condition",
        "window_days": 60,
        "condition": "order_reviews.review_score <= 2",
        "description": "불만 리뷰",
    },
    "features": [{"name": "amount", "source_column": "order_payments.payment_value", "agg": "sum"}],
}


def _stats(client, design):
    return client.run_readonly_sql(
        f"SELECT count(*), count(DISTINCT {design.entity_key}), avg(target), max(ref_time),"
        f" count(DISTINCT split) FROM ({design.sql}) t"
    )["rows"][0]


def test_repurchase_cohort(client):
    design = CohortSQLBuilder(client).build(DesignPlan.model_validate(REPURCHASE))
    n, unique, ratio, last_ref, splits = _stats(client, design)
    assert n == unique > 0  # 고객당 1행 (결제 1:N 조인으로 행이 늘지 않음)
    assert 0 < ratio < 1 and splits == 3
    end = client.run_readonly_sql("SELECT max(order_purchase_timestamp) FROM orders")["rows"][0][0]
    assert (end - last_ref).days >= 90  # 타깃 기간을 끝까지 관측할 수 있는 행만
    assert design.reference_time == "orders.order_purchase_timestamp"
    assert (
        "event_id" in client.run_readonly_sql(f"SELECT * FROM ({design.sql}) t LIMIT 0")["columns"]
    )
    types = {f.name: f.dtype for f in design.features}
    assert (
        types["payment_type"] == "categorical" and "first로 집계" in design.rationale
    )  # 문자열 sum → first


def test_condition_target_per_event(client):
    design = CohortSQLBuilder(client).build(DesignPlan.model_validate(COMPLAINT))
    n, unique, ratio, _, _ = _stats(client, design)
    assert n == unique and 0 < ratio < 1
    assert design.target.positive_condition == "order_reviews.review_score <= 2"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"entity_key": "customers.nope"}, "nope 컬럼이 없습니다"),
        ({"reference_time": "orders.order_status"}, r"시점\(TIMESTAMP\) 컬럼"),
        ({"population_filters": ["order_status = 'x'"]}, "table.column 형식"),
        (
            {"target": {"kind": "condition", "window_days": 30, "description": "x"}},
            "condition 조건식이 필요",
        ),
        ({"population_filters": ["orders.order_status = 'never'"]}, "모집단이 비었습니다"),
    ],
)
def test_plan_errors_are_actionable(client, patch, message):
    with pytest.raises(PlanError, match=message):
        CohortSQLBuilder(client).build(DesignPlan.model_validate({**REPURCHASE, **patch}))


def test_designer_fixes_plan_from_builder_error(deps, llm, client):
    from dataclasses import replace

    bad = {**sc.DESIGN_V2, "entity_key": "customers.customer_key"}
    llm.script("DesignPlan", sc.DESIGN_V2)  # 첫 계획(bad)은 직접 넘기고, 수정 요청에 이 응답
    designer = DatasetDesigner(replace(deps, catalog=client))
    design, fixes = designer.build("system", DesignPlan.model_validate(bad), schema="(schema)")
    assert fixes == 1 and design.sql_error is None and design.row_count > 0
    fix_prompt = llm.calls_for("DesignPlan")[0]["messages"][1].content
    assert "customer_key 컬럼이 없습니다" in fix_prompt and "(schema)" in fix_prompt


def test_condition_window_from_data_and_unlabelable_rows_excluded(client):
    """실제 14B 실행에서 본 실패: 모델이 배송 이후에 확정되는 조건에 타깃 기간 1일을 줘 관측 기간 절단이 생김."""
    plan = {**COMPLAINT, "target": {**COMPLAINT["target"], "window_days": 1}}
    design = CohortSQLBuilder(client).build(DesignPlan.model_validate(plan))
    assert design.target.window_days > 1 and "타깃 기간을 1일 →" in design.rationale
    n, with_review = client.run_readonly_sql(
        f"SELECT count(*), count(r.order_id) FROM ({design.sql}) t "
        "LEFT JOIN (SELECT DISTINCT order_id FROM order_reviews) r ON r.order_id = t.order_id"
    )["rows"][0]
    assert n == with_review  # 리뷰가 없어 판정할 수 없는 주문은 제외
