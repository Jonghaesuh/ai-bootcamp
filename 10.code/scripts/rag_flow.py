"""RAG 흐름을 단계별로 실행해 보여준다: 전처리 → 인덱싱 → 검색 → 근거 포함 응답.

서비스가 쓰는 구성요소(rag/preprocess.py, rag/index.py, rag/retriever.py, llm/chains.py)를 그대로 호출한다.
LLM 키가 없으면 검색까지만 실행한다.

    python scripts/rag_flow.py [--query "..."] [--rebuild]
"""

import argparse
import logging
import sys
from typing import Any

from langchain_core.language_models import BaseChatModel

from req2data.agents.citations import validate_citations
from req2data.config import get_settings
from req2data.llm import get_chat_model
from req2data.llm.chains import rag_chain
from req2data.rag.knowledge_base import KnowledgeBase
from req2data.schemas import GroundedAnswer

DEFAULT_QUERY = "재구매 가능성이 낮은 고객을 예측하려면 어떤 데이터가 필요한가?"
INSTRUCTION = (
    "당신은 데이터 카탈로그 안내자입니다. 검색된 문맥만 근거로 한국어로 답하세요. "
    "문맥에 없는 내용은 모른다고 답하고, 근거로 쓴 문서의 doc_id를 citations에 적으세요."
)


def run(kb: KnowledgeBase, query: str, llm: BaseChatModel | None) -> dict[str, Any]:
    # 1. 전처리: 정제 → 분할 → 메타데이터 부여 (rag/documents.py build_documents)
    docs = kb.documents()
    counts: dict[str, int] = {}
    for d in docs:
        counts[d.metadata["source"]] = counts.get(d.metadata["source"], 0) + 1
    sample = next((d for d in docs if d.metadata["kind"] == "section"), docs[0])

    # 3. 검색: 용어집 쿼리 확장 → 벡터 검색 + BM25 → RRF (rag/retriever.py HybridRetriever)
    retrieved = kb.retriever.invoke(query)
    hits = kb.search(query)

    # 4. 근거 포함 응답: 검색 → 프롬프트 → LLM → 파서 (llm/chains.py rag_chain)
    answer = None
    if llm is not None:
        answer = rag_chain(kb.retriever, llm, GroundedAnswer, INSTRUCTION).invoke(query)
        answer.citations, _ = validate_citations(answer.citations, retrieved)
    return {
        "documents": counts,
        "sample": {"id": sample.id, **sample.metadata},
        "hits": hits,
        "answer": answer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--rebuild", action="store_true", help="인덱스를 무조건 다시 만든다")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    s = get_settings()
    kb = KnowledgeBase.from_settings(s)

    if args.rebuild:
        kb.rebuild()
        index_note = ""
    else:
        index_note = (
            ""
            if kb.ensure_current()
            else "   임베딩 서버에 연결하지 못해 기존 인덱스를 씁니다 (의미 검색 없이 키워드 검색만).\n"
        )

    llm = get_chat_model(s)
    configured = type(llm).__name__ != "UnconfiguredChatModel"
    llm_error = "" if configured else "LLM 키가 없어 검색까지만 실행했습니다."
    try:
        out = run(kb, args.query, llm if configured else None)
    except Exception as exc:  # LLM 서버에 닿지 못해도 검색 단계까지는 보여준다
        llm_error = f"LLM 호출 실패로 검색까지만 실행했습니다: {str(exc)[:200]}"
        out = run(kb, args.query, None)

    print("1. 전처리 (정제 → 분할 → 메타데이터)")
    print(f"   소스별 문서 수: {out['documents']}")
    print(f"   청크 예시 메타데이터: {out['sample']}")

    print("\n2. 인덱싱 (Chroma)")
    print(
        f"{index_note}   {s.chroma_dir}: {kb.index.count()}개 문서, 임베딩 {kb.index.built_with()}"
    )

    print(f"\n3. 검색: {args.query}")
    for h in out["hits"]:
        print(f"   {h['score']:.4f} {h['ranks']} {h['doc_id']}")

    print("\n4. 근거 포함 응답")
    if out["answer"] is None:
        print(f"   {llm_error}")
        return 0
    print(f"   {out['answer'].answer}")
    for c in out["answer"].citations:
        print(f"   - 근거 [{c.doc_id}] {c.used_for}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
