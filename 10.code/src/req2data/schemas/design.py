from typing import Literal

from pydantic import Field

from req2data.schemas.base import LLMOutput
from req2data.schemas.citation import Citation


class TargetDefinition(LLMOutput):
    name: str = "target"
    description: str
    window_days: int = Field(description="기준 시점 이후 타깃을 관측하는 기간(일)")
    positive_condition: str = Field(description="타깃=1이 되는 조건")


class FeatureSpec(LLMOutput):
    name: str = Field(description="SQL 출력 컬럼 별칭")
    description: str = ""
    source_columns: list[str] = Field(default_factory=list, description="원천 컬럼 (table.column)")
    dtype: Literal["numeric", "categorical", "boolean", "datetime"] = "numeric"
    requirement_ids: list[str] = Field(default_factory=list)


class SplitRule(LLMOutput):
    column: str = "split"
    method: Literal["out_of_time", "random"] = "out_of_time"
    train_end: str | None = Field(default=None, description="이 시각 이전 기준 시점은 train")
    val_end: str | None = Field(default=None, description="이 시각 이전은 val, 이후는 test")
    description: str = ""


class DatasetDesign(LLMOutput):
    analysis_unit: str
    entity_key: str = Field(description="SQL 출력의 분석 단위 키 컬럼명")
    reference_time: str = Field(description="기준 시점 원천 컬럼 (table.column)")
    reference_time_alias: str = Field(default="ref_time", description="SQL 출력의 기준 시점 컬럼명")
    observation_window_days: int | None = None
    target: TargetDefinition
    features: list[FeatureSpec] = Field(default_factory=list)
    population_filters: list[str] = Field(default_factory=list)
    split: SplitRule = Field(default_factory=SplitRule)
    sql: str = Field(description="마트 SQL. 키·기준 시점·피처·타깃·분할 컬럼을 출력")
    rationale: str = Field(
        default="", description="분석 단위→기준 시점→타깃→피처→누수→분할 순 추론 요약"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="판단 근거로 쓴 검색 문서. doc_id는 프롬프트에 표시된 값 그대로",
    )
    # 아래는 Agent가 SQL 실행 후 채운다.
    row_count: int | None = None
    target_ratio: float | None = None
    sql_error: str | None = None
    version: int = 0


# ---- LLM이 채우는 설계 계획. SQL은 코드(agents/sql_builder.py)가 이 계획으로 만든다 ----


class FeaturePlan(LLMOutput):
    name: str = Field(description="출력 컬럼명 (영문 소문자·숫자·밑줄)")
    source_column: str = Field(description="원천 컬럼 table.column")
    agg: Literal["first", "sum", "avg", "min", "max", "count", "count_distinct"] = Field(
        default="first",
        description="기준 이벤트 1건에 원천 행이 여러 개일 때 집계 방법 (예: 결제 금액 합계는 sum)",
    )
    description: str = ""
    requirement_ids: list[str] = Field(default_factory=list)


class TargetPlan(LLMOutput):
    kind: Literal["repeat_event", "condition"] = Field(
        description="repeat_event: 같은 분석 단위의 다음 이벤트가 기간 안에 생기면 1 (재구매 등). "
        "condition: 기준 이벤트에 대한 조건식이 참이면 1 (배송 지연, 낮은 평점 등)"
    )
    window_days: int = Field(description="기준 시점 이후 타깃을 관측하는 기간(일)")
    condition: str | None = Field(
        default=None,
        description="kind=condition일 때 SQL 조건식. 컬럼은 반드시 table.column으로 쓴다 "
        "(예: orders.order_delivered_customer_date > orders.order_estimated_delivery_date)",
    )
    description: str


class DesignPlan(LLMOutput):
    analysis_unit: str = Field(description="한 행의 의미 (예: 고객, 주문)")
    entity_key: str = Field(
        description="분석 단위 키 table.column (예: customers.customer_unique_id)"
    )
    reference_time: str = Field(
        description="기준 시점이 되는 이벤트 시각 table.column (예: orders.order_purchase_timestamp)"
    )
    cohort: Literal["first_event", "every_event"] = Field(
        description="first_event: 분석 단위별 첫 이벤트만 (첫 구매 고객 등). every_event: 이벤트마다 한 행 (주문별 예측 등)"
    )
    population_filters: list[str] = Field(
        default_factory=list,
        description="이벤트 모집단 SQL 조건식 목록, 컬럼은 table.column (예: orders.order_status <> 'canceled')",
    )
    target: TargetPlan
    features: list[FeaturePlan] = Field(default_factory=list)
    rationale: str = Field(
        default="", description="분석 단위→기준 시점→타깃→피처→누수 순 판단 요약"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="판단 근거로 쓴 검색 문서. doc_id는 프롬프트에 표시된 값 그대로",
    )
