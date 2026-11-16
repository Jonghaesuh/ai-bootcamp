"""Chroma 벡터 인덱스. BM25 코퍼스도 Chroma에 저장된 문서에서 다시 만들어 저장소를 하나로 유지한다."""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
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
        settings = ChromaSettings(anonymized_telemetry=False)
        self.client = (
            chromadb.PersistentClient(path=str(persist_dir), settings=settings)
            if persist_dir
            else chromadb.EphemeralClient(settings=settings)
        )
        self.collection_name = collection
        self._collection = self._open()
        self._docs: list[Document] | None = None

    def _open(self):
        return self.client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding": self.embedding_name or "unknown"},
            embedding_function=None,
        )

    def rebuild(self, docs: list[Document]) -> int:
        """컬렉션을 지우고 다시 만든다. 문서 ID는 고유해야 한다."""
        ids = [d.id for d in docs]
        if len(ids) != len(set(ids)):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"중복 문서 ID: {dup}")
        # 임베딩을 먼저 계산해, 임베딩 서버 오류 시 기존 인덱스가 지워지지 않게 한다
        vectors = self._embed(docs)
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._open()
        return self._write(docs, vectors)

    def upsert(self, docs: list[Document]) -> int:
        return self._write(docs, self._embed(docs))

    def _embed(self, docs: list[Document]) -> list[list[float]]:
        return [
            v
            for i in range(0, len(docs), BATCH)
            for v in self.embeddings.embed_documents([d.page_content for d in docs[i : i + BATCH]])
        ]

    def _write(self, docs: list[Document], vectors: list[list[float]]) -> int:
        for i in range(0, len(docs), BATCH):
            batch = docs[i : i + BATCH]
            self._collection.upsert(
                ids=[d.id for d in batch],
                documents=[d.page_content for d in batch],
                metadatas=[d.metadata for d in batch],
                embeddings=vectors[i : i + BATCH],
            )
        self._docs = None
        log.info("인덱스 %s: %d개 문서 반영", self.collection_name, len(docs))
        return len(docs)

    def count(self) -> int:
        return self._collection.count()

    def built_with(self) -> str:
        """인덱스를 만든 임베딩 이름 (예: ollama:bge-m3)."""
        return (self._collection.metadata or {}).get("embedding", "unknown")

    def is_current(self) -> bool:
        """문서가 있고 지금 설정한 임베딩으로 만든 인덱스인지. 아니면 의미 검색 차원이 맞지 않는다."""
        return self.count() > 0 and self.built_with() == (self.embedding_name or "unknown")

    def documents(self) -> list[Document]:
        if self._docs is None:
            got = self._collection.get(include=["documents", "metadatas"])
            self._docs = [
                Document(id=i, page_content=t, metadata=m)
                for i, t, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
            ]
        return self._docs

    def dense(self, query: str, k: int, sources: list[str] | None = None) -> list[str]:
        """코사인 유사도 상위 문서 ID."""
        n = self.count()
        if n == 0:
            return []
        where: dict[str, Any] | None = {"source": {"$in": sources}} if sources else None
        res = self._collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=min(k, n),
            where=where,
            include=[],
        )
        return res["ids"][0]
