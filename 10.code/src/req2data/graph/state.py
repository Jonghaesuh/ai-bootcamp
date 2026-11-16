import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel

from req2data.schemas.routing import PIPELINE, AgentName

# Agent → 그 Agent가 만드는 State 키
OUTPUT_KEYS: dict[str, str] = {
    "requirement_analyst": "requirement_spec",
    "data_explorer": "data_mapping",
    "dataset_designer": "dataset_design",
    "reviewer": "review_report",
    "report": "artifacts",
}


class Req2DataState(TypedDict, total=False):
    """그래프 공유 State. Pydantic 모델은 `model_dump(mode="json")`한 dict로 저장한다."""

    project_id: str
    user_type: str
    document: str

    # 확인 질문
    pending_questions: list[dict]  # 지금 사용자에게 물을 질문
    deferred_questions: list[dict]  # 검증 통과 후 물을 질문 (예: 기준 시점 변경 대안)
    answered: list[dict]  # ClarificationQuestion + "answer"
    question_rounds: int
    redo_queue: list[AgentName]  # 답변·수정 요청 때문에 다시 실행할 Agent

    # Agent 산출물
    requirement_spec: dict | None
    data_mapping: dict | None
    dataset_design: dict | None
    design_version: int
    review_report: dict | None
    artifacts: dict[str, str] | None

    # 재작업·Reflection
    rework_counts: dict[str, int]
    feedback: dict[str, list[str]]  # Agent → 다음 시도 프롬프트에 넣을 피드백
    change_request: str | None

    citations: Annotated[list[dict], operator.add]  # Agent가 근거로 인용한 검색 문서

    route: dict
    trace: Annotated[list[dict], operator.add]  # UI 진행 표시용 이벤트 로그


def new_project_state(project_id: str, document: str, user_type: str = "practitioner") -> dict:
    return {
        "project_id": project_id,
        "document": document,
        "user_type": user_type,
        "pending_questions": [],
        "deferred_questions": [],
        "answered": [],
        "question_rounds": 0,
        "redo_queue": [],
        "design_version": 0,
        "rework_counts": {},
        "feedback": {},
        "change_request": None,
    }


def load[M: BaseModel](state: Req2DataState, key: str, model: type[M]) -> M | None:
    value = state.get(key)
    return model.model_validate(value) if value else None


def invalidate_from(agent: str) -> dict[str, Any]:
    """`agent`와 그 이후 단계의 산출물을 지우는 State 업데이트."""
    start = PIPELINE.index(agent)
    return {OUTPUT_KEYS[a]: None for a in PIPELINE[start:]}


def trace_event(agent: str, event: str, **detail: Any) -> dict:
    return {"agent": agent, "event": event, **detail}
