import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from req2data.rag.embeddings import UnavailableEmbeddings
from req2data.rag.index import KnowledgeIndex
from tests.fixtures.rag import HashEmbeddings

DOCS = [
    Document(
        id="glossary:재구매",
        page_content="[용어] 재구매 두 번째 주문",
        metadata={"source": "glossary", "kind": "term"},
    ),
    Document(
        id="guide:g#1",
        page_content="[가이드] 누수 방지 기준 시점",
        metadata={"source": "guide", "kind": "section"},
    ),
    Document(
        id="table:orders",
        page_content="[테이블] orders 주문",
        metadata={"source": "catalog", "kind": "table"},
    ),
]


def test_add_and_similarity_search_roundtrip(tmp_path):
    index = KnowledgeIndex(HashEmbeddings(), tmp_path, embedding_name="hash")
    assert isinstance(index.store, Chroma) and index.rebuild(DOCS) == 3
    doc, _score = index.similarity_search("재구매 주문", k=1)[0]
    assert doc.id == "glossary:재구매" and doc.metadata["kind"] == "term"


def test_persisted_index_reopens(tmp_path):
    KnowledgeIndex(HashEmbeddings(), tmp_path, embedding_name="hash").rebuild(DOCS)
    again = KnowledgeIndex(HashEmbeddings(), tmp_path, embedding_name="hash")
    assert again.count() == 3 and again.is_current()
    assert not KnowledgeIndex(HashEmbeddings(), tmp_path, embedding_name="other").is_current()


def test_source_filter(tmp_path):
    index = KnowledgeIndex(HashEmbeddings(), tmp_path)
    index.rebuild(DOCS)
    assert index.dense("주문", k=3, sources=["catalog"]) == ["table:orders"]


def test_rebuild_keeps_old_index_when_embeddings_down(tmp_path):
    KnowledgeIndex(HashEmbeddings(), tmp_path, embedding_name="hash").rebuild(DOCS)
    broken = KnowledgeIndex(UnavailableEmbeddings(), tmp_path, embedding_name="hash")
    with pytest.raises(RuntimeError):
        broken.rebuild(DOCS[:1])
    assert broken.count() == 3


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="중복 문서 ID"):
        KnowledgeIndex(HashEmbeddings(), tmp_path).rebuild([DOCS[0], DOCS[0]])
