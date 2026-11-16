"""Chroma 벡터 인덱스 (langchain_chroma). BM25 코퍼스도 여기 저장된 문서에서 다시 만들어 저장소를 하나로 유지한다."""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)

COLLECTION = "req2data_kb"
BATCH = 64


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
        self.store = Chroma(
            client=self.client,
            collection_name=collection,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine", "embedding": embedding_name or "unknown"},
        )
        self._docs: list[Document] | None = None

    def rebuild(self, docs: list[Document]) -> int:
        """컬렉션을 지우고 다시 만든다. 문서 ID는 고유해야 한다."""
        ids = [d.id for d in docs]
        if len(ids) != len(set(ids)):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"중복 문서 ID: {dup}")
        # 임베딩 서버를 먼저 확인해, 오류 시 기존 인덱스가 지워지지 않게 한다
        self.embeddings.embed_query("ping")
        self.store.reset_collection()
        return self.upsert(docs)

    def upsert(self, docs: list[Document]) -> int:
        for i in range(0, len(docs), BATCH):
            batch = docs[i : i + BATCH]
            self.store.add_documents(batch, ids=[d.id for d in batch])
        self._docs = None
        log.info("인덱스 %s: %d개 문서 반영", self.collection_name, len(docs))
        return len(docs)

    def count(self) -> int:
        return self.client.get_collection(self.collection_name).count()

    def built_with(self) -> str:
        """인덱스를 만든 임베딩 이름 (예: azure:text-embedding-3-small)."""
        meta = self.client.get_collection(self.collection_name).metadata or {}
        return meta.get("embedding", "unknown")

    def is_current(self) -> bool:
        """문서가 있고 지금 설정한 임베딩으로 만든 인덱스인지. 아니면 의미 검색 차원이 맞지 않는다."""
        return self.count() > 0 and self.built_with() == (self.embedding_name or "unknown")

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
