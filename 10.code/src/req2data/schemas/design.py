from typing import Literal

from pydantic import BaseModel, Field


class TargetDefinition(BaseModel):
    name: str = "target"
    description: str
    window_days: int = Field(description="기준 시점 이후 타깃을 관측하는 기간(일)")
    positive_condition: str = Field(description="타깃=1이 되는 조건")


class FeatureSpec(BaseModel):
    name: str = Field(description="SQL 출력 컬럼 별칭")
    description: str = ""
    source_columns: list[str] = Field(default_factory=list, description="원천 컬럼 (table.column)")
    dtype: Literal["numeric", "categorical", "boolean", "datetime"] = "numeric"
    requirement_ids: list[str] = Field(default_factory=list)


class SplitRule(BaseModel):
    column: str = "split"
    method: Literal["out_of_time", "random"] = "out_of_time"
    train_end: str | None = Field(default=None, description="이 시각 이전 기준 시점은 train")
    val_end: str | None = Field(default=None, description="이 시각 이전은 val, 이후는 test")
    description: str = ""


class DatasetDesign(BaseModel):
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
    # 아래는 Agent가 SQL 실행 후 채운다.
    row_count: int | None = None
    target_ratio: float | None = None
    sql_error: str | None = None
    version: int = 0
