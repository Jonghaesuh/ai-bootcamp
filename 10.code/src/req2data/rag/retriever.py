"""하이브리드 검색: 용어집 쿼리 확장 → 의미 검색(Chroma) + 키워드 검색(BM25) → RRF 결합 → 소스 필터."""

import logging
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from req2data.rag.documents import RagDocument
from req2data.rag.index import KnowledgeIndex
from req2data.rag.sources import GlossaryTerm
from req2data.rag.tokenize import tokenize

log = logging.getLogger(__name__)


def _squash(text: str) -> str:
    """공백 차이("배송이 늦" / "배송이늦")를 무시하고 비교한다."""
    return "".join(text.lower().split())


@dataclass
class SearchHit:
    doc: RagDocument
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # 검색기별 순위 (1부터)

    def as_dict(self) -> dict[str, Any]:
        m = self.doc.metadata
        return {
            "doc_id": self.doc.doc_id,
            "source": self.doc.source,
            "table": m.get("table", ""),
            "column": m.get("column", ""),
            "title": m.get("title", ""),
            "text": self.doc.text,
            "score": round(self.score, 5),
            "ranks": self.ranks,
        }


@dataclass
class ExpandedQuery:
    text: str
    terms: list[str]


class HybridRetriever:
    def __init__(
        self,
        index: KnowledgeIndex,
        glossary: list[GlossaryTerm] | None = None,
        *,
        top_k: int = 8,
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
        candidate_k: int = 30,
    ):
        self.index = index
        self.glossary = glossary or []
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.weights = weights or {"dense": 0.5, "bm25": 0.5}
        self.candidate_k = candidate_k
        self._bm25: tuple[int, BM25Okapi, list[RagDocument]] | None = None

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

    def search(
        self, query: str, k: int | None = None, sources: list[str] | None = None
    ) -> list[SearchHit]:
        k = k or self.top_k
        expanded = self.expand_query(query)
        rankings = {
            "dense": self._dense_rank(expanded.text, sources),
            "bm25": self._bm25_rank(expanded.text, sources),
        }
        docs = {d.doc_id: d for d in self.index.documents()}
        fused: dict[str, SearchHit] = {}
        for name, ids in rankings.items():
            w = self.weights.get(name, 0)
            for rank, doc_id in enumerate(ids, 1):
                if doc_id not in docs:
                    continue
                hit = fused.setdefault(doc_id, SearchHit(docs[doc_id], 0.0))
                hit.score += w / (self.rrf_k + rank)
                hit.ranks[name] = rank
        return sorted(fused.values(), key=lambda h: -h.score)[:k]

    def _dense_rank(self, query: str, sources: list[str] | None) -> list[str]:
        """임베딩 서버가 없으면(데모 모드, Ollama 중단) 키워드 검색만으로 계속한다."""
        try:
            return self.index.dense(query, self.candidate_k, sources)
        except Exception as exc:
            log.warning("의미 검색 실패, BM25만 사용: %s", exc)
            return []

    def _bm25_rank(self, query: str, sources: list[str] | None) -> list[str]:
        docs = self.index.documents()
        if not docs:
            return []
        if self._bm25 is None or self._bm25[0] != len(docs) or self._bm25[2] is not docs:
            self._bm25 = (len(docs), BM25Okapi([tokenize(d.text) for d in docs]), docs)
        _, bm25, corpus = self._bm25
        scores = bm25.get_scores(tokenize(query))
        order = sorted(range(len(corpus)), key=lambda i: -scores[i])
        return [
            corpus[i].doc_id
            for i in order
            if scores[i] > 0 and (not sources or corpus[i].source in sources)
        ][: self.candidate_k]
