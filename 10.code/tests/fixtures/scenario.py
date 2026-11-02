"""PRD 3.1 재구매 시나리오의 LLM 응답 스크립트."""

DOCUMENT = """마케팅팀 요구사항 정의서
첫 구매 고객 중 재구매 가능성이 낮은 고객에게 쿠폰을 보내고 싶다.
첫 주문 금액과 결제 할부 정보를 활용하면 좋겠다.
쿠폰 발송은 첫 구매 직후에 한다."""

ITEMS = [
    {"id": "R1", "text": "첫 구매 고객 대상", "kind": "population"},
    {"id": "R2", "text": "재구매 여부 예측", "kind": "target"},
    {"id": "R3", "text": "첫 주문 금액 활용", "kind": "feature"},
    {"id": "R4", "text": "결제 할부 정보 활용", "kind": "feature"},
    {"id": "R5", "text": "취소 주문 제외", "kind": "filter"},
    {"id": "R6", "text": "쿠폰 발송", "kind": "constraint"},
]

SPEC_ROUND1 = {
    "business_goal": "재구매 가능성이 낮은 첫 구매 고객에게 쿠폰을 보내 재구매를 늘린다",
    "prediction_target": "첫 구매 고객",
    "analysis_unit": "고객",
    "prediction_time": "첫 구매 직후",
    "target_definition": None,
    "success_metric": "상위 20% 중 미재구매자 비율",
    "requirement_items": ITEMS[:4] + ITEMS[5:],
    "open_questions": [
        {
            "id": "Q1",
            "question": "재구매 판단 기간은 첫 구매 후 며칠로 할까요?",
            "target_field": "target_definition",
            "options": ["30일", "60일", "90일"],
            "default_option": "90일",
        },
        {
            "id": "Q2",
            "question": "취소된 주문도 첫 구매로 볼까요?",
            "target_field": "prediction_target",
            "options": ["포함", "제외"],
            "default_option": "제외",
        },
    ],
    "rationale": "재구매 판단 기간과 취소 주문 처리가 없어 질문으로 남김",
}

SPEC_ROUND2 = {
    **SPEC_ROUND1,
    "prediction_target": "취소되지 않은 첫 구매 고객",
    "target_definition": "첫 구매 후 90일 내 두 번째 주문",
    "requirement_items": ITEMS,
    "open_questions": [],
    "rationale": "사용자 답변으로 타깃 기간(90일)과 취소 제외를 반영",
}

MAPPING = {
    "analysis_key": {"table": "customers", "column": "customer_unique_id"},
    "reference_time": {"table": "orders", "column": "order_purchase_timestamp"},
    "items": [
        {
            "requirement_id": "R1",
            "role": "key",
            "columns": [{"table": "customers", "column": "customer_unique_id"}],
            "evidence": ["customers.customer_unique_id"],
        },
        {
            "requirement_id": "R2",
            "role": "target",
            "columns": [{"table": "orders", "column": "order_purchase_timestamp"}],
            "evidence": ["orders.order_purchase_timestamp"],
        },
        {
            "requirement_id": "R3",
            "role": "feature",
            "columns": [
                {"table": "order_payments", "column": "payment_value"},
                {"table": "orders", "column": "fake_amount"},
            ],
            "evidence": ["order_payments.payment_value"],
        },
        {
            "requirement_id": "R4",
            "role": "feature",
            "columns": [{"table": "order_payments", "column": "payment_installments"}],
            "evidence": ["order_payments.payment_installments"],
        },
        {
            "requirement_id": "R5",
            "role": "filter",
            "columns": [{"table": "orders", "column": "order_status"}],
            "evidence": ["orders.order_status"],
        },
    ],
    "join_paths": [
        {
            "left": {"table": "orders", "column": "customer_id"},
            "right": {"table": "customers", "column": "customer_id"},
        },
        {
            "left": {"table": "order_payments", "column": "order_id"},
            "right": {"table": "orders", "column": "order_id"},
        },
    ],
    "rationale": "customer_id는 주문 수와 고유값이 같아 customer_unique_id를 고객 키로 사용",
}

_BASE_SQL = """
WITH o AS (
  SELECT c.customer_unique_id, o.order_id, o.order_purchase_timestamp,
         row_number() OVER (PARTITION BY c.customer_unique_id ORDER BY o.order_purchase_timestamp) AS rn
  FROM orders o JOIN customers c ON o.customer_id = c.customer_id
  WHERE o.order_status <> 'canceled'
), first AS (SELECT * FROM o WHERE rn = 1)
SELECT f.customer_unique_id, f.order_purchase_timestamp AS ref_time,
       p.payment_value AS first_amount, p.payment_installments AS installments,{extra}
       CASE WHEN EXISTS (SELECT 1 FROM o o2 WHERE o2.customer_unique_id = f.customer_unique_id AND o2.rn = 2
            AND o2.order_purchase_timestamp <= f.order_purchase_timestamp + INTERVAL 90 DAY) THEN 1 ELSE 0 END AS target,
       CASE WHEN f.order_purchase_timestamp < TIMESTAMP '2017-12-01' THEN 'train'
            WHEN f.order_purchase_timestamp < TIMESTAMP '2018-03-01' THEN 'val' ELSE 'test' END AS split
FROM first f
LEFT JOIN order_payments p ON p.order_id = f.order_id
LEFT JOIN order_reviews r ON r.order_id = f.order_id
{where}"""

_FEATURES = [
    {
        "name": "first_amount",
        "description": "첫 주문 결제 금액",
        "source_columns": ["order_payments.payment_value"],
        "dtype": "numeric",
        "requirement_ids": ["R3"],
    },
    {
        "name": "installments",
        "description": "첫 주문 할부 개월",
        "source_columns": ["order_payments.payment_installments"],
        "dtype": "numeric",
        "requirement_ids": ["R4"],
    },
]

_DESIGN = {
    "analysis_unit": "고객",
    "entity_key": "customer_unique_id",
    "reference_time": "orders.order_purchase_timestamp",
    "reference_time_alias": "ref_time",
    "target": {
        "name": "target",
        "description": "첫 구매 후 90일 내 두 번째 주문",
        "window_days": 90,
        "positive_condition": "두 번째 주문이 90일 이내",
    },
    "population_filters": ["order_status <> 'canceled'", "고객별 첫 주문"],
    "split": {
        "column": "split",
        "method": "out_of_time",
        "train_end": "2017-12-01",
        "val_end": "2018-03-01",
    },
}

# v1: 리뷰 점수(배송 후 작성) 누수 + 관측 기간 절단 미처리
DESIGN_V1 = {
    **_DESIGN,
    "features": [
        *_FEATURES,
        {
            "name": "review_score",
            "description": "첫 주문 리뷰 점수",
            "source_columns": ["order_reviews.review_score"],
            "dtype": "numeric",
            "requirement_ids": [],
        },
    ],
    "sql": _BASE_SQL.format(extra="\n       r.review_score,", where=""),
    "rationale": "리뷰 점수가 재구매와 관련이 클 것으로 보고 포함",
}

# v2: 피드백 반영 (누수 피처 제거, 90일 관측 불가 고객 제외)
DESIGN_V2 = {
    **_DESIGN,
    "features": _FEATURES,
    "sql": _BASE_SQL.format(
        extra="",
        where="WHERE f.order_purchase_timestamp <= (SELECT max(order_purchase_timestamp) FROM orders) - INTERVAL 90 DAY",
    ),
    "rationale": "검증 피드백에 따라 리뷰 점수를 제거하고 90일을 관측할 수 없는 고객을 제외",
}
