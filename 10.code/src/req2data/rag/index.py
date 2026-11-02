"""Chroma 벡터 인덱스. BM25 코퍼스도 Chroma에 저장된 문서에서 다시 만들어 저장소를 하나로 유지한다."""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings

from req2data.rag.documents import RagDocument

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
        self._docs: list[RagDocument] | None = None

    def _open(self):
        return self.client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding": self.embedding_name or "unknown"},
            embedding_function=None,
        )

    def rebuild(self, docs: list[RagDocument]) -> int:
        """컬렉션을 지우고 다시 만든다. 문서 ID는 고유해야 한다."""
        ids = [d.doc_id for d in docs]
        if len(ids) != len(set(ids)):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"중복 문서 ID: {dup}")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._open()
        return self.upsert(docs)

    def upsert(self, docs: list[RagDocument]) -> int:
        for i in range(0, len(docs), BATCH):
            batch = docs[i : i + BATCH]
            self._collection.upsert(
                ids=[d.doc_id for d in batch],
                documents=[d.text for d in batch],
                metadatas=[d.chroma_metadata() for d in batch],
                embeddings=self.embeddings.embed_documents([d.text for d in batch]),
            )
        self._docs = None
        log.info("인덱스 %s: %d개 문서 반영", self.collection_name, len(docs))
        return len(docs)

    def count(self) -> int:
        return self._collection.count()

    def documents(self) -> list[RagDocument]:
        if self._docs is None:
            got = self._collection.get(include=["documents", "metadatas"])
            self._docs = [
                RagDocument(doc_id=i, text=t, source=m.pop("source", ""), metadata=m)
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
