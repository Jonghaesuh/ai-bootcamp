"""코드 리뷰에서 찾은 결함의 회귀 테스트."""

import threading
from types import SimpleNamespace

import duckdb
import pytest

from req2data.agents.report import current_citations
from req2data.agents.sql_builder import _ident
from req2data.rag.index import corpus_fingerprint
from req2data.schemas import Citation
from req2data.service.runner import GraphRunner

# ---- SQL 별칭 ----


@pytest.mark.parametrize(
    ("name", "alias"),
    [
        ("30일 주문수", "feature_30"),
        ("7d_orders", "feature_7d_orders"),
        ("order_count", "order_count"),
    ],
)
def test_ident_never_starts_with_digit(name, alias):
    assert _ident(name) == alias
    duckdb.sql(f"SELECT 1 AS {alias}")  # 따옴표 없이 파싱되어야 한다


# ---- 프로젝트 잠금 ----


def _bare_runner() -> GraphRunner:
    runner = object.__new__(GraphRunner)
    runner._locks, runner._guard = {}, threading.Lock()
    runner.projects = SimpleNamespace(update=lambda *a, **k: None)
    return runner


def test_lock_released_when_pre_stream_step_fails():
    runner = _bare_runner()
    events = runner._stream("p1", None)

    def boom():
        raise ValueError("demo_inputs.json 손상")

    with pytest.raises(ValueError):
        runner._or_close(events, boom)
    assert runner._lock_for("p1").acquire(blocking=False)


def test_lock_released_when_stream_is_never_iterated():
    runner = _bare_runner()
    events = runner._stream("p1", None)
    events.close()  # SSE 본문을 읽지 않고 버린 경우 (GC도 close를 부른다)
    assert runner._lock_for("p1").acquire(blocking=False)


# ---- 인용 근거 ----


def _cited(*ids: str) -> SimpleNamespace:
    return SimpleNamespace(citations=[Citation(doc_id=i) for i in ids])


def test_report_keeps_only_citations_of_final_outputs():
    records = [
        {"agent": "dataset_designer", "doc_id": "guide:A"},  # 반려된 v1 설계의 인용
        {"agent": "dataset_designer", "doc_id": "guide:B"},
        {"agent": "data_explorer", "doc_id": "column:orders.order_id"},
    ]
    kept = current_citations(
        records, _cited(), _cited("column:orders.order_id"), _cited("guide:B"), None
    )
    assert [r["doc_id"] for r in kept] == ["guide:B", "column:orders.order_id"]


# ---- 지식 베이스 ----


def test_add_case_again_drops_stale_chunks_and_stays_current(kb):
    long = "# 사례\n\n" + "\n\n".join(f"## 절 {i}\n" + "내용 " * 200 for i in range(5))
    first = kb.add_case("shrink_case", long)
    second = kb.add_case("shrink_case", "# 사례\n\n## 절\n짧은 내용\n")
    assert len(first) > len(second)
    ids = {d.id for d in kb.index.documents() if d.id.startswith("case:shrink_case#")}
    assert ids == {d.id for d in second}
    # 지문이 갱신되어 다음 시작 때 전체 재구축을 하지 않는다
    assert kb.index.is_current(corpus_fingerprint(kb.documents()))


def test_bm25_is_shared_across_scoped_retrievers(kb):
    kb.search("재구매", source="glossary")
    built = kb.index.bm25()
    kb.search("주문", source="catalog")
    assert kb.index.bm25() is built


# ---- RAG 리뷰 결함 (BM25 대체, 사례 롤백, 소스 재적재, 동시성, ReAct 근거) ----


def _kb_paths(tmp_path, glossary=None):
    from req2data.rag import KnowledgePaths
    from tests.conftest import ROOT

    data = KnowledgePaths.under(ROOT / "data")
    return KnowledgePaths(
        data.catalog,
        tmp_path / "none.json",
        glossary or data.glossary,
        data.guides,
        tmp_path / "cases",
    )


def test_bm25_works_when_index_was_never_built(tmp_path):
    from req2data.rag import KnowledgeBase
    from req2data.rag.embeddings import UnavailableEmbeddings

    kb = KnowledgeBase(_kb_paths(tmp_path), UnavailableEmbeddings(), persist_dir=tmp_path / "c")
    assert kb.ensure_current() is False and kb.index.count() == 0
    hits = kb.search("재구매", source="glossary")
    assert hits and all(h["source"] == "glossary" for h in hits)


def test_failed_add_case_rolls_back_file(tmp_path):
    from req2data.rag import KnowledgeBase
    from req2data.rag.embeddings import UnavailableEmbeddings
    from tests.fixtures.rag import HashEmbeddings

    kb = KnowledgeBase(_kb_paths(tmp_path), HashEmbeddings(), persist_dir=tmp_path / "c")
    kb.rebuild()
    kb.add_case("kept", "# 사례\n\n## 절\n원래 내용\n")
    kb.index.embeddings = UnavailableEmbeddings()
    with pytest.raises(RuntimeError):
        kb.add_case("orphan", "# 고아 사례\n\n## 절\n내용\n")
    with pytest.raises(RuntimeError):
        kb.add_case("kept", "# 사례\n\n## 절\n바뀐 내용\n")
    assert not (kb.paths.cases / "orphan.md").exists()
    assert "원래 내용" in (kb.paths.cases / "kept.md").read_text(encoding="utf-8")


def test_add_case_fingerprint_does_not_hide_unindexed_files(tmp_path):
    from req2data.rag import KnowledgeBase
    from tests.fixtures.rag import HashEmbeddings

    kb = KnowledgeBase(_kb_paths(tmp_path), HashEmbeddings(), persist_dir=tmp_path / "c")
    kb.rebuild()
    kb.paths.cases.mkdir(exist_ok=True)
    (kb.paths.cases / "manual.md").write_text(
        "# 수동 사례\n\n## 절\n색인 안 됨\n", encoding="utf-8"
    )
    kb.add_case("approved", "# 승인 사례\n\n## 절\n내용\n")
    # 색인되지 않은 파일이 있으므로 다음 시작 때 재구축 대상이어야 한다
    assert not kb.index.is_current(corpus_fingerprint(kb.documents()))
    assert kb.ensure_current() and kb.search("수동 사례", source="case")


def test_rebuild_reloads_edited_glossary(tmp_path):
    import shutil

    import yaml

    from req2data.rag import KnowledgeBase
    from tests.conftest import ROOT
    from tests.fixtures.rag import HashEmbeddings

    glossary = tmp_path / "glossary.yaml"
    shutil.copy(ROOT / "data" / "glossary" / "glossary.yaml", glossary)
    kb = KnowledgeBase(_kb_paths(tmp_path, glossary), HashEmbeddings(), persist_dir=tmp_path / "c")
    kb.rebuild()
    terms = yaml.safe_load(glossary.read_text(encoding="utf-8"))
    terms.append({"term": "휴면고객", "synonyms": ["잠자는 고객"], "definition": "180일 미구매"})
    glossary.write_text(yaml.safe_dump(terms, allow_unicode=True), encoding="utf-8")
    kb.rebuild()
    assert any(t.term == "휴면고객" for t in kb.retriever.glossary)
    assert kb.search("휴면고객", source="glossary")[0]["doc_id"] == "glossary:휴면고객"


def test_search_keeps_working_during_rebuild(tmp_path):
    from req2data.rag import KnowledgeBase
    from tests.fixtures.rag import HashEmbeddings

    kb = KnowledgeBase(_kb_paths(tmp_path), HashEmbeddings(), persist_dir=tmp_path / "c")
    kb.rebuild()
    errors: list[Exception] = []

    def rebuild_many():
        try:
            for _ in range(3):
                kb.rebuild()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=rebuild_many)
    t.start()
    while t.is_alive():
        try:
            assert kb.search("주문 배송", source="catalog")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            break
    t.join()
    assert errors == []


def test_react_notes_carry_full_search_documents():
    import json

    from langchain_core.messages import ToolMessage

    from req2data.agents.base import _summarize_react, _unique_docs
    from req2data.agents.tool_loop import retrieved_from_tools

    hits = [
        {"doc_id": f"guide:g#{i}", "source": "guide", "title": f"절 {i}", "text": f"본문{i} " * 150}
        for i in range(4)
    ]
    msgs = [
        ToolMessage(
            content=json.dumps(hits, ensure_ascii=False), name="search_knowledge", tool_call_id="c1"
        )
    ]
    docs = _unique_docs(retrieved_from_tools(msgs) * 2)
    notes = _summarize_react(msgs, docs)
    assert [d.id for d in docs] == [h["doc_id"] for h in hits]
    for h in hits:  # 마지막 문서까지 잘리지 않고 [doc_id]와 함께 실린다
        assert f"[{h['doc_id']}]" in notes and h["text"].strip() in notes


# ---- 에이전트 로직 리뷰 결함 (피드백, 확인 질문, 완성도, 실익, SQL 가드) ----

_DESIGN = {
    "version": 3,
    "analysis_unit": "고객",
    "entity_key": "k",
    "reference_time": "t.c",
    "target": {"description": "d", "window_days": 30, "positive_condition": "p"},
    "sql": "SELECT 1",
}
_FULL_SPEC = {
    f: "값"
    for f in (
        "business_goal",
        "prediction_target",
        "analysis_unit",
        "prediction_time",
        "target_definition",
        "success_metric",
    )
}


def _sup_state(**kw):
    from req2data.graph.state import new_project_state

    s = new_project_state("p", "문서")
    s.update(requirement_spec=_FULL_SPEC, data_mapping={"x": 1}, dataset_design=_DESIGN)
    s.update(kw)
    return s


def test_rework_keeps_user_decision_in_feedback(deps):
    from req2data.agents.supervisor import Supervisor

    review = {
        "design_version": 3,
        "passed": False,
        "rework_targets": ["dataset_designer"],
        "feedback": {"dataset_designer": ["[censoring] 새 피드백"]},
    }
    state = _sup_state(
        review_report=review,
        feedback={"dataset_designer": ["사용자 결정: 기준 시점? → 변경", "[leakage] 옛 피드백"]},
    )
    cmd = Supervisor(deps)(state)
    assert cmd["feedback"]["dataset_designer"] == [
        "사용자 결정: 기준 시점? → 변경",
        "[censoring] 새 피드백",
    ]


def test_change_request_drops_stale_feedback_and_confirmations(deps, llm):
    from req2data.agents.supervisor import Supervisor

    llm.script(
        "ChangeScope",
        {"restart_from": "data_explorer", "instruction": "배송 테이블 추가"},
    )
    state = _sup_state(
        change_request="배송 테이블도 써줘",
        feedback={
            "dataset_designer": ["[leakage] 옛 피처", "사용자 결정: Q → A"],
            "data_explorer": ["[coverage] 옛 피드백"],
        },
        deferred_questions=[{"id": "V1Q1", "question": "옛 질문", "asked_by": "reviewer"}],
    )
    cmd = Supervisor(deps)(state)
    assert cmd["feedback"]["dataset_designer"] == ["사용자 결정: Q → A"]
    assert all("[coverage]" not in n for n in cmd["feedback"]["data_explorer"])
    assert "배송 테이블도 써줘" in " ".join(cmd["feedback"]["data_explorer"])
    assert cmd["deferred_questions"] == []


def test_reviewer_confirmations_do_not_use_analyst_question_budget(monkeypatch):
    from req2data.agents import supervisor

    monkeypatch.setattr(supervisor, "interrupt", lambda value: {"answers": {}})
    state = {
        "pending_questions": [
            {
                "id": "V1Q1",
                "question": "기준 시점?",
                "asked_by": "reviewer",
                "default_option": "유지",
            }
        ],
        "question_rounds": 1,
        "analyst_rounds": 1,
    }
    out = supervisor.ask_user(state)
    assert out["question_rounds"] == 2 and out["analyst_rounds"] == 1


def test_newer_confirmation_replaces_older_version():
    from req2data.agents.reviewer import merge_confirmations
    from req2data.schemas import ClarificationQuestion

    def q(version, features):
        return ClarificationQuestion(
            id=f"V{version}Q1",
            question=f"{features} 피처를 쓰려면 기준 시점을 바꿀까요?",
            target_field="dataset_design.reference_time",
            asked_by="reviewer",
        )

    deferred = merge_confirmations([], [q(1, "A")], [])
    deferred = merge_confirmations(deferred, [q(2, "B")], [])
    assert [d["id"] for d in deferred] == ["V2Q1"]
    # 새 질문이 없으면 이전 대안 질문은 남는다
    assert merge_confirmations(deferred, [], []) == deferred


def test_infeasible_design_fails_review(deps):
    from req2data.agents.reviewer import Reviewer
    from req2data.schemas import DatasetDesign, EDASummary

    design = DatasetDesign.model_validate(_DESIGN)
    report = Reviewer(deps).compile(design, [], EDASummary(row_count=500, target_ratio=0.01), [])
    assert report.feasibility == "infeasible" and not report.passed
    assert report.rework_targets == ["dataset_designer"]


def test_sql_guard_table_names_are_case_insensitive():
    from req2data.tools.sql_guard import check_readonly_sql

    check_readonly_sql("WITH A AS (SELECT * FROM Orders) SELECT * FROM a", {"orders"})
    check_readonly_sql("SELECT * FROM range(3)", {"orders"})


@pytest.mark.parametrize(
    "sql", ["SELECT * FROM duckdb_settings()", "SELECT * FROM duckdb_secrets()"]
)
def test_sql_guard_blocks_system_table_functions(sql):
    from req2data.tools.catalog import SQLExecutionError
    from req2data.tools.sql_guard import check_readonly_sql

    with pytest.raises(SQLExecutionError):
        check_readonly_sql(sql, {"orders"})
