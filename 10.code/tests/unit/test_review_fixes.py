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
