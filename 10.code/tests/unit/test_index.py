import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from req2data.rag.embeddings import UnavailableEmbeddings
from req2data.rag.index import KnowledgeIndex, corpus_fingerprint
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


class FlakyEmbeddings(HashEmbeddings):
    """N번째 문서 임베딩 호출에서 실패한다 (분당 요청 제한 등 일시 오류)."""

    def __init__(self, fail_on: int):
        self.calls = 0
        self.fail_on = fail_on

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("429 Too Many Requests")
        return super().embed_documents(texts)


@pytest.mark.parametrize("fail_on", [1, 2])
def test_rebuild_failure_midway_keeps_previous_index(tmp_path, fail_on):
    """재구축 도중 임베딩이 실패해도 기존 인덱스가 비거나 일부만 남지 않는다."""
    docs = [
        Document(id=f"guide:d#{i}", page_content=f"문서 {i} 재구매", metadata={"source": "guide"})
        for i in range(70)  # 배치(64)가 두 번 나뉜다
    ]
    KnowledgeIndex(HashEmbeddings(), tmp_path, embedding_name="hash").rebuild(docs)
    broken = KnowledgeIndex(FlakyEmbeddings(fail_on), tmp_path, embedding_name="hash")
    changed = [
        *docs,
        Document(id="guide:new", page_content="새 문서", metadata={"source": "guide"}),
    ]
    with pytest.raises(RuntimeError):
        broken.rebuild(changed)
    assert broken.count() == 70 and len(broken.documents()) == 70
    assert broken.is_current(corpus_fingerprint(docs))
    assert not broken.is_current(corpus_fingerprint(changed))
