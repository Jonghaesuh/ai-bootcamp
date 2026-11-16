import re

from langchain_core.documents import Document

from req2data.agents.base import BaseAgent
from req2data.graph.state import Req2DataState
from req2data.schemas import ClarificationQuestion, RequirementItem, RequirementSpec

FIELD_QUESTIONS: dict[str, str] = {
    "business_goal": "이 예측 결과로 어떤 업무 결정을 내리려고 하나요?",
    "prediction_target": "예측 대상이 되는 고객·거래의 범위(모집단)는 무엇인가요?",
    "analysis_unit": "한 건의 예측은 어떤 단위(고객, 주문, 상품 등)로 하나요?",
    "prediction_time": "예측은 어느 시점에 실행하나요? (예: 첫 구매 직후, 매월 1일)",
    "target_definition": "어떤 일이 언제까지 일어나면 '해당(양성)'으로 볼까요?",
    "success_metric": "모델이 성공했다고 판단할 지표와 기준은 무엇인가요?",
}


def render_answered(answered: list[dict]) -> str:
    if not answered:
        return "(없음)"
    return "\n".join(f"- Q: {a['question']}\n  A: {a['answer']}" for a in answered)


class RequirementAnalyst(BaseAgent):
    """정의서 → RequirementSpec, 빠진 항목은 확인 질문으로 남긴다."""

    name = "requirement_analyst"

    def __call__(self, state: Req2DataState) -> dict:
        document = state["document"]
        answered = state.get("answered") or []
        required = self.deps.cfg("requirement.required_fields", list(FIELD_QUESTIONS))

        retrieved: list[Document] = []
        extra = "\n\n".join(
            filter(
                None,
                [
                    self.deps.fewshots.render("requirement_spec", document),
                    self.knowledge(
                        document, "glossary", k=4, title="비즈니스 용어집", into=retrieved
                    ),
                    self.knowledge(
                        document, "case", k=2, title="과거 승인 설계 사례", into=retrieved
                    ),
                ],
            )
        )
        result = self.structured(
            self.system(state, extra=extra),
            self.prompt.task("analyze", document=document, answers=render_answered(answered)),
            RequirementSpec,
            fallback=lambda: fallback_spec(document),
        )
        spec = result.value
        citations, rejected = self.cite(spec, retrieved)
        rounds = state.get("question_rounds", 0)
        questions = self._questions(spec, required, answered, rounds)

        assumed: list[str] = []
        if questions and rounds >= self.deps.cfg("requirement.max_question_rounds", 2):
            # 더 묻지 않고 기본값을 가정해 진행한다. 가정은 명세에 남겨 추적한다.
            for q in questions:
                assumed.append(f"{q.question} → {q.default_option or '미정'} (가정)")
            spec.constraints += assumed
            questions = []
        spec.open_questions = questions

        completeness = spec.completeness(required)
        return {
            "requirement_spec": spec.model_dump(mode="json"),
            "pending_questions": [q.model_dump(mode="json") for q in questions],
            "citations": citations,
            "trace": [
                self.result_event(
                    result,
                    completeness=round(completeness, 2),
                    blocking=completeness < self.deps.cfg("requirement.min_completeness", 0.7),
                    questions=len(questions),
                    assumed=assumed,
                    citations=len(citations),
                    rejected_citations=rejected,
                )
            ],
        }

    def _questions(
        self,
        spec: RequirementSpec,
        required: list[str],
        answered: list[dict],
        rounds: int,
    ) -> list[ClarificationQuestion]:
        answered_fields = {a.get("target_field") for a in answered}
        answered_texts = {a["question"] for a in answered}
        questions = [
            q
            for q in spec.open_questions
            if q.question not in answered_texts
            and (q.target_field is None or q.target_field not in answered_fields)
        ]
        covered = {q.target_field for q in questions}
        for f in spec.missing_fields(required):
            if f not in covered and f not in answered_fields and f in FIELD_QUESTIONS:
                questions.append(
                    ClarificationQuestion(
                        id="", question=FIELD_QUESTIONS[f], target_field=f, reason="정의서에 없음"
                    )
                )
        questions = questions[: self.deps.cfg("requirement.max_questions", 5)]
        for i, q in enumerate(questions, 1):
            q.id = f"R{rounds + 1}Q{i}"  # 라운드 간 ID 충돌 방지
            q.asked_by = self.name
        return questions


def fallback_spec(document: str) -> RequirementSpec:
    """LLM 실패 시 규칙 기반 기본값: 문장을 요구사항 항목으로만 쪼개고 나머지는 질문으로 넘긴다."""
    sentences = [s.strip() for s in re.split(r"[.\n!?]+", document) if len(s.strip()) > 3]
    return RequirementSpec(
        requirement_items=[
            RequirementItem(id=f"R{i}", text=s) for i, s in enumerate(sentences[:20], 1)
        ],
        rationale="LLM 구조화 실패로 문장 분리만 수행했습니다. 필수 항목은 확인 질문으로 채웁니다.",
    )
