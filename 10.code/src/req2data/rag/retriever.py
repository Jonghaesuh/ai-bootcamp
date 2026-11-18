"""하이브리드 검색기 (LangChain BaseRetriever): 용어집 쿼리 확장 → 의미 검색(Chroma) + 키워드 검색(BM25) → RRF 결합 → 소스 필터."""

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from req2data.rag.index import KnowledgeIndex
from req2data.rag.sources import GlossaryTerm
from req2data.rag.tokenize import tokenize

log = logging.getLogger(__name__)


def _squash(text: str) -> str:
    """공백 차이("배송이 늦" / "배송이늦")를 무시하고 비교한다."""
    return "".join(text.lower().split())


def hit_dict(doc: Document) -> dict[str, Any]:
    """검색 결과 문서 → CatalogClient·API·UI가 쓰는 dict."""
    m = doc.metadata
    return {
        "doc_id": doc.id,
        "source": m.get("source", ""),
        "table": m.get("table", ""),
        "column": m.get("column", ""),
        "title": m.get("title", ""),
        "text": doc.page_content,
        "score": m.get("score", 0.0),
        "ranks": m.get("ranks", {}),
    }


@dataclass
class ExpandedQuery:
    text: str
    terms: list[str]


class HybridRetriever(BaseRetriever):
    index: KnowledgeIndex
    glossary: list[GlossaryTerm] = Field(default_factory=list)
    top_k: int = 8
    rrf_k: int = 60
    weights: dict[str, float] = Field(default_factory=lambda: {"dense": 0.5, "bm25": 0.5})
    candidate_k: int = 30
    sources: list[str] | None = None  # 지식 소스 필터 (catalog | glossary | guide | case)

    def with_scope(
        self, k: int | None = None, sources: list[str] | None = None
    ) -> "HybridRetriever":
        """결과 수와 소스 필터만 바꾼 검색기. 인덱스와 BM25 코퍼스는 공유한다."""
        return self.model_copy(update={"top_k": k or self.top_k, "sources": sources})

    def expand_query(self, query: str) -> ExpandedQuery:
        """용어집 표면형이 쿼리에 있으면 동의어와 관련 컬럼을 덧붙인다 (예: 재구매 → 두 번째 주문, repeat purchase)."""
        q = _squash(query)
        matched = [
            t
            for t in self.glossary
            if any(len(f) >= 2 and f in q for f in map(_squash, t.surface_forms))
        ]
        extra = [w for t in matched for w in (*t.synonyms, *t.related_columns)]
        return ExpandedQuery(" ".join([query, *extra]), [t.term for t in matched])

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        expanded = self.expand_query(query)
        rankings = {
            "dense": self._dense_rank(expanded.text),
            "bm25": self._bm25_rank(expanded.text),
        }
        docs = {d.id: d for d in self.index.documents()}
        scores: dict[str, float] = {}
        ranks: dict[str, dict[str, int]] = {}
        for name, ids in rankings.items():
            w = self.weights.get(name, 0)
            for rank, doc_id in enumerate(ids, 1):
                if doc_id not in docs:
                    continue
                scores[doc_id] = scores.get(doc_id, 0.0) + w / (self.rrf_k + rank)
                ranks.setdefault(doc_id, {})[name] = rank  # 검색기별 순위 (1부터)
        top = sorted(scores, key=lambda i: -scores[i])[: self.top_k]
        return [
            Document(
                id=i,
                page_content=docs[i].page_content,
                metadata={**docs[i].metadata, "score": round(scores[i], 5), "ranks": ranks[i]},
            )
            for i in top
        ]

    def _dense_rank(self, query: str) -> list[str]:
        """임베딩 서버가 없거나 인덱스의 임베딩 차원이 다르면 키워드 검색만으로 계속한다."""
        try:
            return self.index.dense(query, self.candidate_k, self.sources)
        except Exception as exc:
            log.warning("의미 검색 실패, BM25만 사용: %s", exc)
            return []

    def _bm25_rank(self, query: str) -> list[str]:
        built = self.index.bm25()  # with_scope 사본마다 다시 만들지 않도록 인덱스에 캐시한다
        if built is None:
            return []
        bm25, corpus = built
        scores = bm25.get_scores(tokenize(query))
        order = sorted(range(len(corpus)), key=lambda i: -scores[i])
        return [
            corpus[i].id
            for i in order
            if scores[i] > 0
            and (not self.sources or corpus[i].metadata.get("source") in self.sources)
        ][: self.candidate_k]
