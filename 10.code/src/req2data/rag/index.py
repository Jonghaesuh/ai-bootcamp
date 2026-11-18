"""Chroma 벡터 인덱스 (langchain_chroma). BM25 코퍼스도 여기 저장된 문서에서 다시 만들어 저장소를 하나로 유지한다."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from req2data.rag.tokenize import tokenize

log = logging.getLogger(__name__)

COLLECTION = "req2data_kb"
BATCH = 64


def corpus_fingerprint(docs: list[Document]) -> str:
    """문서 ID와 내용으로 만든 지문. 지식 소스나 분할 방식이 바뀌면 달라진다."""
    lines = sorted(f"{d.id}:{d.metadata.get('content_hash', '')}" for d in docs)
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:16]


class KnowledgeIndex:
    def __init__(
        self,
        embeddings: Embeddings,
        persist_dir: Path | None = None,
        collection: str = COLLECTION,
        embedding_name: str = "",
    ):
        self.embeddings = embeddings
        self.embedding_name = embedding_name
        self.persist_dir = Path(persist_dir) if persist_dir else None
        settings = ChromaSettings(anonymized_telemetry=False)
        self.client = (
            chromadb.PersistentClient(path=str(self.persist_dir), settings=settings)
            if self.persist_dir
            else chromadb.EphemeralClient(settings=settings)
        )
        self.collection_name = collection
        self.store = self._open()
        self._docs: list[Document] | None = None
        self._bm25: tuple[BM25Okapi, list[Document]] | None = None

    def _open(self, corpus: str = "", name: str | None = None) -> Chroma:
        """컬렉션 메타데이터에 만든 임베딩 이름과 문서 지문을 남겨, 다시 만들어야 하는지 판정한다."""
        return Chroma(
            client=self.client,
            collection_name=name or self.collection_name,
            embedding_function=self.embeddings,
            collection_metadata={
                "hnsw:space": "cosine",
                "embedding": self.embedding_name or "unknown",
                "corpus": corpus,
            },
        )

    def rebuild(self, docs: list[Document]) -> int:
        """인덱스를 새로 만든다. 문서 ID는 고유해야 한다.
        임시 컬렉션에 모두 저장한 뒤에야 기존 컬렉션과 바꾸므로, 도중에 임베딩이 실패해도
        기존 인덱스가 비거나 일부만 남지 않는다."""
        ids = [d.id for d in docs]
        if len(ids) != len(set(ids)):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"중복 문서 ID: {dup}")
        staging_name = f"{self.collection_name}__staging"
        self._drop(staging_name)
        staging = self._open(corpus_fingerprint(docs), name=staging_name)
        try:
            self._add(staging, docs)
        except Exception:
            self._drop(staging_name)
            raise
        self._drop(self.collection_name)
        self.client.get_collection(staging_name).modify(name=self.collection_name)
        self.store = self._open(corpus_fingerprint(docs))
        self._docs = None
        log.info("인덱스 %s: %d개 문서로 재구축", self.collection_name, len(docs))
        return len(docs)

    def _drop(self, name: str) -> None:
        try:
            self.client.delete_collection(name)
        except Exception:  # 없는 컬렉션
            pass

    @staticmethod
    def _add(store: Chroma, docs: list[Document]) -> None:
        for i in range(0, len(docs), BATCH):
            batch = docs[i : i + BATCH]
            store.add_documents(batch, ids=[d.id for d in batch])

    def upsert(self, docs: list[Document], stale_prefix: str | None = None) -> int:
        """문서를 넣거나 갱신한다. `stale_prefix`로 시작하는 ID 가운데 이번에 넣지 않은 문서는
        지운다 (같은 사례를 다시 저장해 청크 수가 줄어든 경우)."""
        self._add(self.store, docs)
        if stale_prefix:
            fresh = {d.id for d in docs}
            stale = [
                d.id
                for d in self.documents()
                if d.id and d.id.startswith(stale_prefix) and d.id not in fresh
            ]
            if stale:
                self.store.delete(ids=stale)
        self._docs = None
        log.info("인덱스 %s: %d개 문서 반영", self.collection_name, len(docs))
        return len(docs)

    def set_corpus(self, corpus: str) -> None:
        """증분 반영 뒤 문서 지문을 갱신해, 다음 시작 때 불필요한 전체 재구축을 막는다.
        거리 함수(hnsw:*)는 생성 후 바꿀 수 없어 메타데이터에서 빼고 넘긴다 (설정은 유지된다)."""
        col = self.client.get_collection(self.collection_name)
        meta = {k: v for k, v in (col.metadata or {}).items() if not k.startswith("hnsw:")}
        col.modify(metadata={**meta, "corpus": corpus})

    def bm25(self) -> tuple[BM25Okapi, list[Document]] | None:
        """저장된 문서로 만든 BM25 색인. 문서 목록이 바뀔 때만 다시 만든다 (범위만 바꾼 검색기끼리 공유)."""
        docs = self.documents()
        if not docs:
            return None
        if self._bm25 is None or self._bm25[1] is not docs:
            self._bm25 = (BM25Okapi([tokenize(d.page_content) for d in docs]), docs)
        return self._bm25

    def count(self) -> int:
        return self.client.get_collection(self.collection_name).count()

    def built_with(self) -> str:
        """인덱스를 만든 임베딩 이름 (예: azure:text-embedding-3-small)."""
        meta = self.client.get_collection(self.collection_name).metadata or {}
        return meta.get("embedding", "unknown")

    def is_current(self, corpus: str | None = None) -> bool:
        """문서가 있고 지금 설정한 임베딩으로 만든 인덱스인지. 아니면 의미 검색 차원이 맞지 않는다.
        `corpus`(문서 지문)를 주면 지식 소스가 바뀌었는지도 함께 본다."""
        if self.count() == 0 or self.built_with() != (self.embedding_name or "unknown"):
            return False
        meta = self.client.get_collection(self.collection_name).metadata or {}
        return corpus is None or meta.get("corpus") == corpus

    def documents(self) -> list[Document]:
        if self._docs is None:
            got = self.store.get(include=["documents", "metadatas"])
            self._docs = [
                Document(id=i, page_content=t, metadata=m or {})
                for i, t, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
            ]
        return self._docs

    def similarity_search(
        self, query: str, k: int, sources: list[str] | None = None
    ) -> list[tuple[Document, float]]:
        """코사인 거리가 가까운 순서의 (문서, 거리). `sources`로 지식 소스를 거른다."""
        n = self.count()
        if n == 0:
            return []
        where: dict[str, Any] | None = {"source": {"$in": sources}} if sources else None
        return self.store.similarity_search_with_score(query, k=min(k, n), filter=where)

    def dense(self, query: str, k: int, sources: list[str] | None = None) -> list[str]:
        """의미 검색 상위 문서 ID."""
        return [doc.id for doc, _ in self.similarity_search(query, k, sources)]
