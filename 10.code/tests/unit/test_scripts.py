"""인덱스 자동 재구축과 실행 확인 스크립트(scripts/rag_flow.py, scripts/check_env.py)."""

import importlib.util
import shutil

from req2data.rag import KnowledgeBase, KnowledgePaths
from req2data.rag.embeddings import UnavailableEmbeddings
from tests.conftest import ROOT
from tests.fixtures import scenario as sc
from tests.fixtures.rag import HashEmbeddings

DATA = KnowledgePaths.under(ROOT / "data")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths(tmp_path) -> KnowledgePaths:
    return KnowledgePaths(
        DATA.catalog, tmp_path / "none.json", DATA.glossary, DATA.guides, tmp_path / "cases"
    )


def test_ensure_current_rebuilds_for_new_embedding(tmp_path):
    paths, store = _paths(tmp_path), tmp_path / "chroma"
    KnowledgeBase(paths, HashEmbeddings(), persist_dir=store, embedding_name="a").rebuild()
    kb = KnowledgeBase(paths, HashEmbeddings(), persist_dir=store, embedding_name="b")
    assert not kb.index.is_current()
    assert kb.ensure_current() and kb.index.built_with() == "b"


def test_ensure_current_keeps_index_when_embeddings_down(tmp_path):
    paths, store = _paths(tmp_path), tmp_path / "chroma"
    counts = KnowledgeBase(paths, HashEmbeddings(), persist_dir=store, embedding_name="a").rebuild()
    kb = KnowledgeBase(paths, UnavailableEmbeddings(), persist_dir=store, embedding_name="b")
    assert kb.ensure_current() is False and kb.index.count() == sum(counts.values())


def test_ensure_current_rebuilds_when_sources_change(tmp_path):
    paths, store = _paths(tmp_path), tmp_path / "chroma"
    kb = KnowledgeBase(paths, HashEmbeddings(), persist_dir=store, embedding_name="a")
    kb.rebuild()
    assert kb.ensure_current() and kb.search("통신 해지", source="case") == []
    paths.cases.mkdir()
    (paths.cases / "telecom.md").write_text("# 통신 해지 예측\n\n## 설계\n회선 단위 해지 예측\n")
    reopened = KnowledgeBase(paths, HashEmbeddings(), persist_dir=store, embedding_name="a")
    assert reopened.ensure_current()
    assert reopened.search("통신 해지", source="case")[0]["doc_id"] == "case:telecom#1"


def test_rag_flow_runs_without_llm(kb):
    out = load_script("rag_flow").run(kb, "재구매 가능성이 낮은 고객", llm=None)
    assert set(out["documents"]) == {"catalog", "glossary", "guide"}
    assert out["hits"] and out["answer"] is None and "content_hash" in out["sample"]


def test_rag_flow_answers_with_citations(kb, llm):
    llm.script(
        "GroundedAnswer",
        {"answer": "customer_unique_id로 판별", "citations": [{"doc_id": "glossary:재구매"}]},
    )
    out = load_script("rag_flow").run(kb, "재구매는 어떻게 정의하나", llm=llm)
    assert out["answer"].citations[0].doc_id == "glossary:재구매"


def test_check_env_full_run_answers_questions_with_defaults(deps, llm):
    llm.script("RequirementSpec", sc.SPEC_ROUND1, sc.SPEC_ROUND2)
    llm.script("DataMapping", sc.MAPPING)
    llm.script("DesignPlan", sc.DESIGN_V1, sc.DESIGN_V2)
    out = load_script("check_env").run_full(deps, sc.DOCUMENT)
    assert out["finished"] and out["row_count"] > 0 and out["passed"] is True
    assert "summary.md" in out["artifacts"]
    shutil.rmtree(deps.artifact_dir, ignore_errors=True)
