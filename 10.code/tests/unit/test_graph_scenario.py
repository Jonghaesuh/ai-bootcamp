"""PRD 3.1 재구매 시나리오: 확인 질문 → 매핑 → 설계 → 누수·절단 검출 → 재작업 → 대안 확인 → 산출물."""

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from req2data.graph import build_graph, new_project_state
from tests.fixtures import scenario as sc


def _run(graph, payload, config):
    graph.invoke(payload, config)
    return graph.get_state(config)


def test_repurchase_scenario(deps, llm):
    llm.script("RequirementSpec", sc.SPEC_ROUND1, sc.SPEC_ROUND2)
    llm.script("DataMapping", sc.MAPPING)
    llm.script("DatasetDesign", sc.DESIGN_V1, sc.DESIGN_V2)
    # ReviewNarrative, ReportSummary는 스크립트하지 않아 fallback 경로를 탄다.

    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "p1"}}

    # 1) 요구사항 구조화 후 확인 질문으로 멈춤
    snap = _run(graph, new_project_state("p1", sc.DOCUMENT), config)
    questions = snap.interrupts[0].value["questions"]
    assert [q["question"] for q in questions] == [
        q["question"] for q in sc.SPEC_ROUND1["open_questions"]
    ]
    assert all(q["id"].startswith("R1Q") for q in questions)

    # 2) 답변 → 재분석 → 매핑 → 설계 v1 → 검증 실패 → 재설계 v2 → 통과 → 기준 시점 대안 확인으로 멈춤
    snap = _run(
        graph,
        Command(resume={"answers": {questions[0]["id"]: "90일", questions[1]["id"]: "제외"}}),
        config,
    )
    state = snap.values
    confirm = snap.interrupts[0].value["questions"]
    assert len(confirm) == 1 and confirm[0]["asked_by"] == "reviewer"
    assert "order_reviews.review_creation_date" in confirm[0]["question"]

    assert "orders.fake_amount" in state["data_mapping"]["rejected_refs"]  # 환각 컬럼 제거
    assert state["rework_counts"] == {"dataset_designer": 1}
    feedback = " ".join(state["feedback"]["dataset_designer"])
    assert "[leakage]" in feedback and "[censoring]" in feedback

    # Reflection: 두 번째 설계 프롬프트에 검증 피드백이 들어감
    second_design_system = llm.calls_for("DatasetDesign")[1]["messages"][0].content
    assert "이전 검증 피드백" in second_design_system and "review_score" in second_design_system

    review = state["review_report"]
    assert review["passed"] and review["design_version"] == 2
    assert review["eda"]["split_counts"].keys() == {"train", "val", "test"}
    assert state["dataset_design"]["target_ratio"] > 0

    # 3) 기본값(현재 기준 시점 유지)으로 답하면 재설계 없이 산출물 생성 후 종료
    snap = _run(graph, Command(resume={"answers": {}}), config)
    assert not snap.next
    artifacts = snap.values["artifacts"]
    for name in (
        "summary.md",
        "requirement_spec.md",
        "data_mapping.md",
        "dataset_design.md",
        "mart.sql",
        "traceability.csv",
        "review_report.md",
    ):
        assert Path(artifacts[name]).exists()
    assert "R3" in Path(artifacts["traceability.csv"]).read_text(encoding="utf-8")

    events = [(t["agent"], t["event"]) for t in snap.values["trace"]]
    routed = [t["next"] for t in snap.values["trace"] if t["event"] == "route"]
    assert routed == [
        "requirement_analyst",
        "ask_user",
        "requirement_analyst",
        "data_explorer",
        "dataset_designer",
        "reviewer",
        "dataset_designer",
        "reviewer",
        "ask_user",
        "report",
        "end",
    ]
    assert ("reviewer", "fallback") in events  # ReviewNarrative 미스크립트 → 규칙 기반 요약
    assert not any(
        t.get("llm") for t in snap.values["trace"] if t["event"] == "route"
    )  # 후보가 하나뿐이라 라우팅 LLM 미호출
    assert llm.remaining() == {}


def test_change_request_restarts_designer_only(deps, llm):
    llm.script("RequirementSpec", sc.SPEC_ROUND2)
    llm.script("DataMapping", sc.MAPPING)
    llm.script("DatasetDesign", sc.DESIGN_V2, sc.DESIGN_V2)
    llm.script(
        "ChangeScope",
        {
            "changed_fields": ["dataset_design.target.window_days"],
            "restart_from": "dataset_designer",
            "instruction": "타깃 기간을 60일로",
        },
    )

    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "p2"}}
    snap = _run(graph, new_project_state("p2", sc.DOCUMENT), config)
    assert not snap.next  # 확인 질문·누수 없음 → 한 번에 종료
    mapping_calls = len(llm.calls_for("DataMapping"))

    snap = _run(
        graph, {"change_request": "타깃 기간을 60일로 바꿔줘"}, config
    )  # 같은 스레드에서 새 실행
    assert not snap.next
    assert len(llm.calls_for("DataMapping")) == mapping_calls  # 매핑 재사용
    assert snap.values["design_version"] == 2
    assert "타깃 기간을 60일로" in " ".join(snap.values["feedback"]["dataset_designer"])
