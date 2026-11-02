from typing import Literal

from pydantic import BaseModel, Field

AgentName = Literal[
    "requirement_analyst",
    "data_explorer",
    "dataset_designer",
    "reviewer",
    "report",
    "ask_user",
    "end",
]

# 산출물이 만들어지는 순서. 앞 단계가 바뀌면 뒤 단계 산출물은 무효가 된다.
PIPELINE: tuple[AgentName, ...] = (
    "requirement_analyst",
    "data_explorer",
    "dataset_designer",
    "reviewer",
    "report",
)


class RouteDecision(BaseModel):
    """Supervisor의 라우팅 결정."""

    next: AgentName
    reason: str = ""
    instructions: str = Field(default="", description="다음 Agent에게 전달할 재작업 지시")


class ChangeScope(BaseModel):
    """대화형 수정 요청의 영향 범위."""

    changed_fields: list[str] = Field(
        default_factory=list,
        description="바뀌는 State 필드 경로 (예: dataset_design.target.window_days)",
    )
    restart_from: Literal["requirement_analyst", "data_explorer", "dataset_designer"]
    instruction: str = Field(description="재시작 Agent에게 전달할 수정 지시")
    reason: str = ""
