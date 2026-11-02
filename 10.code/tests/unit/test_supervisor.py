from req2data.agents.supervisor import Supervisor, keyword_scope
from req2data.graph.state import new_project_state

DESIGN = {
    "version": 3,
    "analysis_unit": "고객",
    "entity_key": "k",
    "reference_time": "t.c",
    "target": {"description": "d", "window_days": 30, "positive_condition": "p"},
    "sql": "SELECT 1",
}


def _state(**kw):
    s = new_project_state("p", "문서")
    s.update(requirement_spec={"x": 1}, data_mapping={"x": 1}, dataset_design=DESIGN)
    s.update(kw)
    return s


def _failed_review(targets):
    return {
        "design_version": 3,
        "passed": False,
        "rework_targets": targets,
        "feedback": {t: [f"{t} 고칠 것"] for t in targets},
    }


def test_pipeline_order_rules(deps):
    sup = Supervisor(deps)
    assert sup.candidates(new_project_state("p", "문서")).default == "requirement_analyst"
    assert sup.candidates(_state(pending_questions=[{"id": "Q"}])).options == ["ask_user"]
    assert sup.candidates(_state()).default == "reviewer"  # 설계 v3 미검증
    stale = _state(review_report={"design_version": 2, "passed": True})
    assert sup.candidates(stale).default == "reviewer"


def test_redo_queue_invalidates_downstream(deps):
    c = Supervisor(deps).candidates(_state(redo_queue=["dataset_designer", "requirement_analyst"]))
    assert c.default == "requirement_analyst"
    assert c.update["data_mapping"] is None and c.update["redo_queue"] == []


def test_single_rework_target_skips_llm_and_applies_feedback(deps, llm):
    cmd = Supervisor(deps)(_state(review_report=_failed_review(["dataset_designer"])))
    assert cmd.goto == "dataset_designer"
    assert cmd.update["rework_counts"] == {"dataset_designer": 1}
    assert cmd.update["feedback"]["dataset_designer"] == ["dataset_designer 고칠 것"]
    assert cmd.update["dataset_design"] is None and "data_mapping" not in cmd.update  # 매핑은 유지
    assert llm.calls == []


def test_guard_rejects_llm_choice_outside_candidates(deps, llm):
    llm.script("RouteDecision", {"next": "report", "reason": "빨리 끝내자"})
    cmd = Supervisor(deps)(
        _state(review_report=_failed_review(["data_explorer", "dataset_designer"]))
    )
    assert cmd.goto == "data_explorer"  # 후보 밖 선택 → 규칙 기본값
    assert len(llm.calls_for("RouteDecision")) == 1


def test_llm_choice_within_candidates_is_used(deps, llm):
    llm.script(
        "RouteDecision",
        {"next": "dataset_designer", "reason": "매핑은 충분", "instructions": "SQL만 수정"},
    )
    cmd = Supervisor(deps)(
        _state(review_report=_failed_review(["data_explorer", "dataset_designer"]))
    )
    assert cmd.goto == "dataset_designer"
    assert cmd.update["feedback"]["dataset_designer"][-1] == "SQL만 수정"


def test_rework_limit_moves_on_to_report(deps):
    s = _state(
        review_report=_failed_review(["dataset_designer"]), rework_counts={"dataset_designer": 2}
    )
    assert Supervisor(deps).candidates(s).default == "report"


def test_keyword_scope_fallback():
    assert keyword_scope("분석 단위를 주문으로 바꿔줘").restart_from == "requirement_analyst"
    assert keyword_scope("결제 테이블도 써줘").restart_from == "data_explorer"
    assert keyword_scope("타깃 기간을 60일로").restart_from == "dataset_designer"
