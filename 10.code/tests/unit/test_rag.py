import pytest

from req2data.rag import KnowledgeBase, KnowledgePaths
from req2data.rag.documents import catalog_documents, markdown_chunks
from req2data.rag.tokenize import tokenize
from req2data.tools.catalog import SQLExecutionError
from req2data.tools.local_catalog import LocalCatalogClient
from req2data.tools.sql_guard import check_readonly_sql
from tests.conftest import ROOT
from tests.fixtures.rag import HashEmbeddings, build_local_db

PATHS = KnowledgePaths.under(ROOT / "data")


@pytest.fixture(scope="module")
def kb(tmp_path_factory) -> KnowledgeBase:
    root = tmp_path_factory.mktemp("kb")
    paths = KnowledgePaths(
        PATHS.catalog, root / "none.json", PATHS.glossary, PATHS.guides, root / "cases"
    )
    kb = KnowledgeBase(paths, HashEmbeddings(), persist_dir=root / "chroma")
    kb.rebuild()
    return kb


@pytest.fixture(scope="module")
def client(kb, tmp_path_factory) -> LocalCatalogClient:
    db = build_local_db(tmp_path_factory.mktemp("db") / "olist.duckdb", kb.spec)
    return LocalCatalogClient(db, kb, max_rows=50, timeout_seconds=5)


# ---- 지식 소스 ----


def test_catalog_spec_is_consistent(kb):
    spec = kb.spec
    assert len(spec.tables) == 9
    for left, right in spec.foreign_keys():
        assert spec.column(right) is not None, f"{left} → {right}"
    for t in spec.tables.values():
        for c in t.columns:
            if c.available_at:
                target = spec.column(c.available_at)
                assert target is not None and target.is_time, f"{t.name}.{c.name}"
    ranks = [c.time_rank for c in spec.tables["orders"].columns if c.is_time]
    assert ranks == sorted(ranks)


def test_catalog_documents_table_and_column_level(kb):
    docs = catalog_documents(kb.spec, {"orders.order_status": {"null_ratio": 0.0, "distinct": 8}})
    n_cols = sum(len(t.columns) for t in kb.spec.tables.values())
    assert len(docs) == len(kb.spec.tables) + n_cols
    assert len({d.doc_id for d in docs}) == len(docs)
    status = next(d for d in docs if d.doc_id == "column:orders.order_status")
    assert "프로파일: 결측률 0.0" in status.text and "주의:" in status.text
    review = next(d for d in docs if d.doc_id == "column:order_reviews.review_score")
    assert review.metadata["available_at"] == "order_reviews.review_creation_date"
    assert all(v is not None for d in docs for v in d.chroma_metadata().values())


def test_guide_chunked_by_heading():
    chunks = markdown_chunks(PATHS.guides / "dataset_design_guide.md", "guide")
    sections = (PATHS.guides / "dataset_design_guide.md").read_text(encoding="utf-8").count("\n## ")
    assert len(chunks) == sections
    assert all(c.text.startswith("[데이터셋 설계 가이드]") for c in chunks)


def test_tokenize_mixes_korean_bigrams_and_column_parts():
    toks = tokenize("재구매 customer_unique_id")
    assert {"재구", "구매", "customer_unique_id", "unique"} <= set(toks)


# ---- 검색 ----


def test_query_expansion_uses_glossary(kb):
    q = kb.retriever.expand_query("재구매 가능성이 낮은 고객")
    assert "재구매" in q.terms and "customers.customer_unique_id" in q.text
    assert kb.retriever.expand_query("배송이 늦어진 주문").terms == ["배송 지연"]
    assert kb.retriever.expand_query("주문 상태").terms == []  # 한 글자 동의어로 오탐하지 않음


def test_source_filter_and_rrf(kb):
    hits = kb.search("customer_unique_id 재구매", k=5, source="catalog")
    assert hits and all(h["source"] == "catalog" for h in hits)
    assert hits[0]["doc_id"] == "column:customers.customer_unique_id"
    assert set(hits[0]["ranks"]) == {"dense", "bm25"}
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    guide = kb.search("리뷰 점수 누수", k=3, source="guide")
    assert all(h["source"] == "guide" for h in guide)
    assert any("누수" in h["text"] for h in guide)


def test_add_case_is_searchable_immediately(kb):
    assert kb.search("통신 해지 예측", source="case") == []
    kb.add_case(
        "telecom_case", "# 통신 해지 예측 설계\n\n## 설계\n회선 단위, 매월 1일 기준 해지 예측\n"
    )
    hits = kb.search("통신 해지 예측", source="case")
    assert hits and hits[0]["doc_id"].startswith("case:telecom_case")


def test_fk_graph_prefers_pk_joins_over_zip_codes(kb):
    path = kb.fk_graph.join_path(["customers", "order_payments"])
    assert {j["left"] for j in path} == {"orders.customer_id", "order_payments.order_id"}
    via = kb.fk_graph.via_tables(["customers", "sellers"])
    assert via == ["order_items", "orders"]  # geolocation(우편번호 N:M) 경유 안 함


# ---- 로컬 카탈로그 클라이언트·SQL 가드 ----


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "SELECT 1; DROP TABLE orders",
        "SELECT * FROM read_csv('/etc/hosts')",
        "SELECT * FROM '/etc/hosts'",
        "COPY orders TO 'x.csv'",
        "ATTACH 'x.db'",
        "SELECT * FROM not_in_catalog",
    ],
)
def test_sql_guard_blocks(sql):
    with pytest.raises(SQLExecutionError):
        check_readonly_sql(sql, {"orders", "customers"})


def test_sql_guard_allows_cte_and_union():
    check_readonly_sql(
        "WITH a AS (SELECT * FROM orders) SELECT * FROM a UNION ALL SELECT * FROM a", {"orders"}
    )


def test_local_client_describe_search_and_sql(client):
    desc = client.describe_table("order_reviews")
    score = next(c for c in desc["columns"] if c["name"] == "review_score")
    assert score["available_at"] == "order_reviews.review_creation_date"
    assert desc["row_count"] > 0

    assert (
        client.search_catalog("재구매 고객", k=3, source="glossary")[0]["doc_id"]
        == "glossary:재구매"
    )
    assert client.find_join_path(["customers", "orders"]) == [
        {"left": "orders.customer_id", "right": "customers.customer_id"}
    ]

    res = client.run_readonly_sql("SELECT order_id FROM orders")
    assert res["truncated"] and res["row_count"] == 50
    prof = client.profile_column("customers", "customer_id")
    assert prof["distinct"] > 0 and prof["null_ratio"] == 0


def test_local_client_rejects_file_access_even_without_guard(client):
    with pytest.raises(SQLExecutionError):
        client.run_readonly_sql("SELECT * FROM read_csv('/etc/hosts')")
    con = client._connection()
    with pytest.raises(Exception, match="(?i)permission|disabled"):
        con.execute("SELECT * FROM read_csv('/etc/hosts')")  # DuckDB enable_external_access=false


def test_local_client_missing_db(kb, tmp_path):
    c = LocalCatalogClient(tmp_path / "none.duckdb", kb)
    assert c.describe_table("orders")["row_count"] is None  # 카탈로그 정의는 DB 없이도 동작
    with pytest.raises(SQLExecutionError, match="make data"):
        c.run_readonly_sql("SELECT 1 FROM orders")


# ---- 에이전트 연결 ----


def test_agents_receive_rag_context(deps, llm, client):
    from dataclasses import replace

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from req2data.graph import build_graph, new_project_state
    from tests.fixtures import scenario as sc

    llm.script("RequirementSpec", sc.SPEC_ROUND2)
    llm.script("DataMapping", sc.MAPPING)
    llm.script("DatasetDesign", sc.DESIGN_V1, sc.DESIGN_V2)
    graph = build_graph(replace(deps, catalog=client), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "rag"}}
    graph.invoke(new_project_state("rag", sc.DOCUMENT), config)
    graph.invoke(Command(resume={"answers": {}}), config)  # 기준 시점 대안 → 기본값 유지
    assert not graph.get_state(config).next

    req_system = llm.calls_for("RequirementSpec")[0]["messages"][0].content
    assert "비즈니스 용어집" in req_system and "glossary:재구매" in req_system
    map_task = llm.calls_for("DataMapping")[0]["messages"][1].content
    assert "glossary:재구매" in map_task and "customers.customer_unique_id" in map_task
    design_system = llm.calls_for("DatasetDesign")[0]["messages"][0].content
    assert "데이터셋 설계 가이드" in design_system and "guide:dataset_design_guide" in design_system
    narrative_system = llm.calls_for("ReviewNarrative")[0]["messages"][0].content
    assert "guide:dataset_design_guide" in narrative_system  # 누수·절단 실패 근거 가이드
