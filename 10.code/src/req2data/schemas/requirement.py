from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from req2data.schemas.routing import AgentName


class UserType(StrEnum):
    practitioner = "practitioner"  # 실무자: 기술 용어·SQL·근거 상세
    planner = "planner"  # 기획자: 비즈니스 언어 요약, 고객 확인 질문 우선


class ClarificationQuestion(BaseModel):
    id: str = Field(description="질문 ID (예: Q1)")
    question: str
    target_field: str | None = Field(default=None, description="답변이 채우는 명세 필드")
    options: list[str] = Field(default_factory=list)
    default_option: str | None = Field(default=None, description="답이 없을 때 가정할 값")
    reason: str = ""
    asked_by: AgentName = "requirement_analyst"


class RequirementItem(BaseModel):
    id: str = Field(description="요구사항 ID (예: R1)")
    text: str
    kind: Literal["population", "target", "feature", "filter", "metric", "constraint", "other"] = (
        "other"
    )


class RequirementSpec(BaseModel):
    business_goal: str | None = None
    prediction_target: str | None = Field(
        default=None, description="예측 대상 모집단 (예: 첫 구매 고객)"
    )
    analysis_unit: str | None = Field(default=None, description="한 행이 의미하는 단위 (예: 고객)")
    prediction_time: str | None = Field(default=None, description="예측을 수행하는 시점")
    target_definition: str | None = Field(default=None, description="타깃 사건과 기간")
    success_metric: str | None = None
    constraints: list[str] = Field(default_factory=list)
    requirement_items: list[RequirementItem] = Field(default_factory=list)
    open_questions: list[ClarificationQuestion] = Field(default_factory=list)
    rationale: str = ""

    def missing_fields(self, required: list[str]) -> list[str]:
        return [f for f in required if not (getattr(self, f, None) or "").strip()]

    def completeness(self, required: list[str]) -> float:
        if not required:
            return 1.0
        return 1 - len(self.missing_fields(required)) / len(required)
