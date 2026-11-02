"""실제 bge-m3(Ollama)로 검색 품질을 확인한다. `pytest -m llm`으로 실행."""

import httpx
import pytest

from req2data.config import get_settings
from req2data.rag import KnowledgeBase, KnowledgePaths
from req2data.rag.embeddings import get_embeddings
from tests.conftest import ROOT

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    try:
        httpx.get(f"{get_settings().ollama_base_url}/api/tags", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("Ollama에 연결할 수 없음")
    kb = KnowledgeBase(
        KnowledgePaths.under(ROOT / "data"),
        get_embeddings(),
        persist_dir=tmp_path_factory.mktemp("c"),
    )
    kb.rebuild()
    return kb


@pytest.mark.parametrize(
    ("query", "source", "expected", "top"),
    [
        ("재구매 가능성이 낮은 첫 구매 고객", "catalog", "column:customers.customer_unique_id", 3),
        ("고객 단위 키", "catalog", "column:customers.customer_unique_id", 3),
        ("배송이 늦어진 주문", "catalog", "column:orders.order_delivered_customer_date", 5),
        ("결제 할부", "catalog", "column:order_payments.payment_installments", 3),
        ("재구매", "glossary", "glossary:재구매", 1),
        ("리뷰 점수를 피처로 써도 되나", "guide", "guide:dataset_design_guide#3", 2),
        ("마지막 달 데이터는 타깃을 다 볼 수 없다", "guide", "guide:dataset_design_guide#4", 2),
    ],
)
def test_retrieval_quality(kb, query, source, expected, top):
    ids = [h["doc_id"] for h in kb.search(query, k=top, source=source)]
    assert expected in ids, ids
