"""RAG 진입점: 지식 소스 경로 → 문서 → 인덱스 → 검색기. 승인된 설계를 사례로 추가하는 장기 기억 쓰기도 담당한다."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from req2data.config import Settings, get_settings
from req2data.rag.documents import build_documents, markdown_chunks
from req2data.rag.embeddings import get_embeddings
from req2data.rag.fk_graph import FKGraph
from req2data.rag.index import KnowledgeIndex, corpus_fingerprint
from req2data.rag.profiling import load_profiles
from req2data.rag.retriever import HybridRetriever, hit_dict
from req2data.rag.sources import CatalogSpec, load_catalog_spec, load_glossary

log = logging.getLogger(__name__)


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
        self.chunking = {
            "chunk_size": policy.get("chunk_size", 600),
            "chunk_overlap": policy.get("chunk_overlap", 80),
        }
        self.spec: CatalogSpec = load_catalog_spec(paths.catalog)
        self.glossary = load_glossary(paths.glossary)
        self.profiles = load_profiles(paths.profiles)
        self.fk_graph = FKGraph(self.spec)
        self.index = KnowledgeIndex(embeddings, persist_dir, embedding_name=embedding_name)
        self.retriever = HybridRetriever(
            index=self.index,
            glossary=self.glossary,
            top_k=policy.get("top_k", 8),
            rrf_k=policy.get("rrf_k", 60),
            weights=policy.get("hybrid_weights") or {"dense": 0.5, "bm25": 0.5},
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "KnowledgeBase":
        s = settings or get_settings()
        return cls(
            KnowledgePaths.under(s.data_dir),
            get_embeddings(s),
            persist_dir=s.chroma_dir,
            rag_policy=s.policy.get("rag", {}),
            embedding_name=s.embedding_label,
        )

    def documents(self) -> list[Document]:
        return build_documents(
            self.spec,
            self.glossary,
            self.profiles,
            self.paths.guides,
            self.paths.cases,
            **self.chunking,
        )

    def rebuild(self) -> dict[str, int]:
        docs = self.documents()
        self.index.rebuild(docs)
        counts: dict[str, int] = {}
        for d in docs:
            source = d.metadata["source"]
            counts[source] = counts.get(source, 0) + 1
        return counts

    def ensure_current(self) -> bool:
        """인덱스가 지금 임베딩·지식 소스와 맞지 않으면 다시 만든다 (서비스 시작 시 호출).
        임베딩 서버 오류 등으로 실패하면 기존 인덱스를 두고 False를 돌려준다 (검색은 BM25로 계속)."""
        if self.index.is_current(corpus_fingerprint(self.documents())):
            return True
        try:
            counts = self.rebuild()
        except Exception as exc:
            log.warning("RAG 인덱스 재구축 실패, 기존 인덱스로 계속: %s", exc)
            return False
        log.info("RAG 인덱스 재구축: %s (%s)", counts, self.index.built_with())
        return True

    def search(
        self, query: str, k: int | None = None, source: str | list[str] | None = None
    ) -> list[dict[str, Any]]:
        sources = [source] if isinstance(source, str) else source
        return [hit_dict(d) for d in self.retriever.with_scope(k, sources).invoke(query)]

    def add_case(self, case_id: str, markdown: str) -> list[Document]:
        """승인된 설계를 사례 문서로 저장하고 바로 검색 대상에 넣는다 (장기 기억)."""
        self.paths.cases.mkdir(parents=True, exist_ok=True)
        path = self.paths.cases / f"{case_id}.md"
        path.write_text(markdown, encoding="utf-8")
        docs = markdown_chunks(path, "case", **self.chunking)
        self.index.upsert(docs)
        return docs
