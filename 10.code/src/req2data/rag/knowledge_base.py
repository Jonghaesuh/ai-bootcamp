"""RAG 진입점: 지식 소스 경로 → 문서 → 인덱스 → 검색기. 승인된 설계를 사례로 추가하는 장기 기억 쓰기도 담당한다."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from req2data.config import Settings, get_settings
from req2data.rag.documents import (
    RagDocument,
    catalog_documents,
    directory_chunks,
    glossary_documents,
    markdown_chunks,
)
from req2data.rag.embeddings import get_embeddings
from req2data.rag.fk_graph import FKGraph
from req2data.rag.index import KnowledgeIndex
from req2data.rag.profiling import load_profiles
from req2data.rag.recording import RECORD_FILE as RETRIEVAL_FILE
from req2data.rag.recording import DenseRecorder
from req2data.rag.retriever import HybridRetriever
from req2data.rag.sources import CatalogSpec, load_catalog_spec, load_glossary


@dataclass(frozen=True)
class KnowledgePaths:
    catalog: Path
    profiles: Path
    glossary: Path
    guides: Path
    cases: Path

    @classmethod
    def under(cls, data_dir: Path) -> "KnowledgePaths":
        d = Path(data_dir)
        return cls(
            catalog=d / "catalog" / "olist_catalog.yaml",
            profiles=d / "catalog" / "profiles.json",
            glossary=d / "glossary" / "glossary.yaml",
            guides=d / "guides",
            cases=d / "cases",
        )


class KnowledgeBase:
    def __init__(
        self,
        paths: KnowledgePaths,
        embeddings: Embeddings,
        persist_dir: Path | None = None,
        rag_policy: dict[str, Any] | None = None,
        embedding_name: str = "",
    ):
        policy = rag_policy or {}
        self.paths = paths
        self.spec: CatalogSpec = load_catalog_spec(paths.catalog)
        self.glossary = load_glossary(paths.glossary)
        self.profiles = load_profiles(paths.profiles)
        self.fk_graph = FKGraph(self.spec)
        self.index = KnowledgeIndex(embeddings, persist_dir, embedding_name=embedding_name)
        self.retriever = HybridRetriever(
            self.index,
            self.glossary,
            top_k=policy.get("top_k", 8),
            rrf_k=policy.get("rrf_k", 60),
            weights=policy.get("hybrid_weights"),
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "KnowledgeBase":
        s = settings or get_settings()
        kb = cls(
            KnowledgePaths.under(s.data_dir),
            get_embeddings(s),
            persist_dir=s.chroma_dir,
            rag_policy=s.policy.get("rag", {}),
            embedding_name=s.embedding_label,
        )
        if s.demo_mode or s.llm_record:
            kb.retriever.dense_recorder = DenseRecorder(
                Path(s.demo_recordings_dir) / RETRIEVAL_FILE,
                "replay" if s.demo_mode else "record",
            )
        return kb

    def documents(self) -> list[RagDocument]:
        return [
            *catalog_documents(self.spec, self.profiles),
            *glossary_documents(self.glossary),
            *directory_chunks(self.paths.guides, "guide"),
            *directory_chunks(self.paths.cases, "case"),
        ]

    def rebuild(self) -> dict[str, int]:
        docs = self.documents()
        self.index.rebuild(docs)
        counts: dict[str, int] = {}
        for d in docs:
            counts[d.source] = counts.get(d.source, 0) + 1
        return counts

    def search(
        self, query: str, k: int | None = None, source: str | list[str] | None = None
    ) -> list[dict[str, Any]]:
        sources = [source] if isinstance(source, str) else source
        return [h.as_dict() for h in self.retriever.search(query, k, sources)]

    def add_case(self, case_id: str, markdown: str) -> list[RagDocument]:
        """승인된 설계를 사례 문서로 저장하고 바로 검색 대상에 넣는다 (장기 기억)."""
        self.paths.cases.mkdir(parents=True, exist_ok=True)
        path = self.paths.cases / f"{case_id}.md"
        path.write_text(markdown, encoding="utf-8")
        docs = markdown_chunks(path, "case")
        self.index.upsert(docs)
        return docs
