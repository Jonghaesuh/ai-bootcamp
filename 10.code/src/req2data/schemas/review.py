from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from req2data.schemas.base import LLMOutput
from req2data.schemas.requirement import ClarificationQuestion
from req2data.schemas.routing import AgentName


class CheckStatus(StrEnum):
    passed = "pass"
    warn = "warn"
    fail = "fail"


class CheckResult(BaseModel):
    name: str
    status: CheckStatus
    evidence: str = ""
    affected: list[str] = Field(default_factory=list)
    rework_target: AgentName | None = None


class EDASummary(BaseModel):
    row_count: int = 0
    target_ratio: float | None = None
    split_counts: dict[str, int] = Field(default_factory=dict)
    split_target_ratio: dict[str, float] = Field(default_factory=dict)
    null_ratio: dict[str, float] = Field(default_factory=dict)
    target_correlation: dict[str, float] = Field(default_factory=dict)


class ReviewReport(BaseModel):
    design_version: int
    checks: list[CheckResult] = Field(default_factory=list)
    eda: EDASummary | None = None
    passed: bool = False
    rework_targets: list[AgentName] = Field(default_factory=list)
    feedback: dict[str, list[str]] = Field(
        default_factory=dict, description="Agent별 재작업 피드백"
    )
    feasibility: Literal["ok", "low", "infeasible"] = "ok"
    confirmations: list[ClarificationQuestion] = Field(default_factory=list)
    summary: str = ""


class ReviewNarrative(LLMOutput):
    """검증 Agent의 LLM 부분 출력. 판정은 규칙이 하고 LLM은 설명만 쓴다."""

    summary: str
    additional_feedback: list[str] = Field(default_factory=list)
