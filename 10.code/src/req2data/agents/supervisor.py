from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END
from langgraph.types import Command, interrupt

from req2data.agents.base import BaseAgent, as_json
from req2data.graph.state import Req2DataState, invalidate_from, load, trace_event
from req2data.schemas import (
    PIPELINE,
    AgentName,
    ChangeScope,
    DatasetDesign,
    ReviewReport,
    RouteDecision,
)

NODES: tuple[str, ...] = (*PIPELINE, "ask_user")

# 답변을 반영하는 Agent. 요구사항 질문은 항상 다시 분석하고, 검증 확인 질문은 기본값과 다를 때만 재설계한다.
ANSWER_OWNER: dict[str, AgentName] = {
    "requirement_analyst": "requirement_analyst",
    "reviewer": "dataset_designer",
}

CHANGE_KEYWORDS: list[tuple[AgentName, tuple[str, ...]]] = [
    ("requirement_analyst", ("목표", "대상", "모집단", "지표", "단위")),
    ("data_explorer", ("테이블", "컬럼", "조인", "데이터 소스")),
]


@dataclass
class Candidates:
    options: list[AgentName]
    default: AgentName
    reason: str
    update: dict[str, Any] = field(default_factory=dict)  # 선택과 무관하게 적용할 State 변경
    rework: bool = False  # 검증 미통과로 인한 재작업 (피드백·횟수 반영)


class Supervisor(BaseAgent):
    """규칙이 허용 후보를 정하고, 후보가 여럿일 때만 LLM이 고른다."""

    name = "supervisor"

    def __call__(self, state: Req2DataState) -> Command:
        if state.get("change_request"):
            return self.handle_change(state)

        c = self.candidates(state)
        decision = RouteDecision(next=c.default, reason=c.reason)
        used_llm = False
        if len(c.options) > 1:
            used_llm = True
            result = self.structured(
                self.system(state),
                self.prompt.task(
                    "route",
                    state_summary=summarize(state),
                    candidates=", ".join(c.options),
                    default=c.default,
                ),
                RouteDecision,
                fallback=lambda: decision,
            )
            if result.value.next in c.options:  # 규칙 가드
                decision = result.value

        update = dict(c.update)
        if c.rework:
            update |= self.rework_update(state, decision)

        update["route"] = decision.model_dump(mode="json")
        update["trace"] = [
            trace_event(
                self.name, "route", next=decision.next, reason=decision.reason, llm=used_llm
            )
        ]
        return Command(goto=END if decision.next == "end" else decision.next, update=update)

    def candidates(self, state: Req2DataState) -> Candidates:
        if state.get("pending_questions"):
            return Candidates(["ask_user"], "ask_user", "사용자 확인이 필요한 질문이 있음")

        redo = state.get("redo_queue") or []
        if redo:
            first = min(redo, key=PIPELINE.index)
            return Candidates(
                [first],
                first,
                "사용자 답변 반영",
                update={**invalidate_from(first), "redo_queue": []},
            )

        for agent, key in (
            ("requirement_analyst", "requirement_spec"),
            ("data_explorer", "data_mapping"),
            ("dataset_designer", "dataset_design"),
        ):
            if not state.get(key):
                return Candidates([agent], agent, f"{key} 없음")

        design = load(state, "dataset_design", DatasetDesign)
        review = load(state, "review_report", ReviewReport)
        if review is None or review.design_version != design.version:
            return Candidates(["reviewer"], "reviewer", f"설계 v{design.version} 미검증")

        if not review.passed:
            limit = self.deps.cfg("workflow.max_rework", 2)
            counts = state.get("rework_counts") or {}
            targets = [t for t in review.rework_targets if counts.get(t, 0) < limit]
            if targets:
                return Candidates(targets, targets[0], "검증 미통과: 재작업", rework=True)
            # 한도 초과: 경고를 남긴 채 다음 단계로 진행

        deferred = state.get("deferred_questions") or []
        if deferred:
            return Candidates(
                ["ask_user"],
                "ask_user",
                "검증 후 확인할 대안이 있음",
                update={"pending_questions": deferred, "deferred_questions": []},
            )
        if not state.get("artifacts"):
            return Candidates(["report"], "report", "산출물 생성")
        return Candidates(["end"], "end", "완료")

    def rework_update(self, state: Req2DataState, decision: RouteDecision) -> dict[str, Any]:
        review = load(state, "review_report", ReviewReport)
        target = decision.next
        counts = dict(state.get("rework_counts") or {})
        counts[target] = counts.get(target, 0) + 1
        feedback = dict(state.get("feedback") or {})
        notes = list(review.feedback.get(target, []))
        if decision.instructions:
            notes.append(decision.instructions)
        feedback[target] = notes
        return {**invalidate_from(target), "rework_counts": counts, "feedback": feedback}

    def handle_change(self, state: Req2DataState) -> Command:
        request = state["change_request"]
        result = self.structured(
            self.system(state),
            self.prompt.task(
                "change_scope", change_request=request, state_summary=summarize(state)
            ),
            ChangeScope,
            fallback=lambda: keyword_scope(request),
        )
        scope = result.value
        feedback = dict(state.get("feedback") or {})
        feedback[scope.restart_from] = [f"사용자 수정 요청: {request}", scope.instruction]
        update = {
            **invalidate_from(scope.restart_from),
            "change_request": None,
            "rework_counts": {},
            "feedback": feedback,
            "route": RouteDecision(next=scope.restart_from, reason=scope.reason).model_dump(
                mode="json"
            ),
            "trace": [
                trace_event(
                    self.name,
                    "change_scope",
                    restart_from=scope.restart_from,
                    changed_fields=scope.changed_fields,
                    llm=not result.used_fallback,
                )
            ],
        }
        return Command(goto=scope.restart_from, update=update)


def keyword_scope(request: str) -> ChangeScope:
    for agent, words in CHANGE_KEYWORDS:
        if any(w in request for w in words):
            return ChangeScope(restart_from=agent, instruction=request, reason="키워드 규칙")
    return ChangeScope(
        restart_from="dataset_designer", instruction=request, reason="키워드 규칙(기본)"
    )


def summarize(state: Req2DataState) -> str:
    review = state.get("review_report") or {}
    return as_json(
        {
            "산출물": {
                k: bool(state.get(k))
                for k in (
                    "requirement_spec",
                    "data_mapping",
                    "dataset_design",
                    "review_report",
                    "artifacts",
                )
            },
            "설계_버전": state.get("design_version", 0),
            "검증_통과": review.get("passed"),
            "미통과_검사": [
                {
                    "name": c["name"],
                    "evidence": c["evidence"],
                    "rework_target": c.get("rework_target"),
                }
                for c in review.get("checks", [])
                if c["status"] == "fail"
            ],
            "재작업_횟수": state.get("rework_counts", {}),
        }
    )


def ask_user(state: Req2DataState) -> dict:
    """확인 질문을 보내고 멈춘다. `Command(resume={"answers": {질문 ID: 답변}})`으로 재개한다.

    답변을 "answers" 키로 감싸는 이유: LangGraph는 resume dict의 키를 interrupt ID로 해석할 수 있다.
    빠진 답변은 질문의 default_option으로 채운다.
    """
    questions = state.get("pending_questions") or []
    reply = (interrupt({"type": "clarification", "questions": questions}) or {}).get(
        "answers"
    ) or {}

    new: list[dict] = []
    redo: set[AgentName] = set()
    feedback = dict(state.get("feedback") or {})
    for q in questions:
        answer = reply.get(q["id"]) or q.get("default_option") or "(답변 없음, 판단에 맡김)"
        new.append({**q, "answer": answer})
        owner = ANSWER_OWNER.get(q.get("asked_by", ""), "requirement_analyst")
        if owner == "requirement_analyst" or answer != q.get("default_option"):
            redo.add(owner)
            if owner != "requirement_analyst":
                feedback[owner] = [
                    *feedback.get(owner, []),
                    f"사용자 결정: {q['question']} → {answer}",
                ]
    return {
        "answered": [*(state.get("answered") or []), *new],
        "pending_questions": [],
        "question_rounds": state.get("question_rounds", 0) + 1,
        "redo_queue": sorted(redo, key=PIPELINE.index),
        "feedback": feedback,
        "trace": [trace_event("ask_user", "answered", answers={a["id"]: a["answer"] for a in new})],
    }
