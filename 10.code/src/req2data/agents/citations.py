"""인용 근거 검증: LLM이 적은 citations 가운데 실제로 검색해 프롬프트에 넣은 문서만 남긴다."""

from langchain_core.documents import Document

from req2data.schemas import Citation


def normalize_doc_id(doc_id: str) -> str:
    """프롬프트가 문서를 [doc_id]로 보여 주므로 모델이 붙여 쓴 대괄호와 공백을 걷어낸다."""
    return doc_id.strip().strip("[]").strip()


def validate_citations(
    citations: list[Citation], retrieved: list[Document]
) -> tuple[list[Citation], list[str]]:
    """(검색된 문서를 가리키는 인용, 버린 doc_id). 같은 문서를 여러 번 인용하면 첫 번째만 남긴다."""
    known = {d.id for d in retrieved}
    kept: list[Citation] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for c in citations:
        c = c.model_copy(update={"doc_id": normalize_doc_id(c.doc_id)})
        if c.doc_id in seen:
            continue
        seen.add(c.doc_id)
        if c.doc_id in known:
            kept.append(c)
        else:
            rejected.append(c.doc_id)
    return kept, rejected


def citation_records(agent: str, kept: list[Citation], retrieved: list[Document]) -> list[dict]:
    """State·리포트에 싣는 인용 기록. 출처 종류와 제목은 검색 문서에서 가져온다."""
    docs = {d.id: d for d in retrieved}
    return [
        {
            "agent": agent,
            "doc_id": c.doc_id,
            "source": docs[c.doc_id].metadata.get("source", ""),
            "title": docs[c.doc_id].metadata.get("title", ""),
            "used_for": c.used_for,
        }
        for c in kept
    ]
