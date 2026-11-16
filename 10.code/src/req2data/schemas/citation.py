from pydantic import Field

from req2data.schemas.base import LLMOutput


class Citation(LLMOutput):
    """판단 근거로 쓴 검색 문서 하나."""

    doc_id: str = Field(description="프롬프트에 [doc_id]로 표시된 문서 ID 그대로")
    used_for: str = Field(default="", description="이 문서를 어떤 판단에 썼는지 한 구절")


class GroundedAnswer(LLMOutput):
    """검색 문맥만 근거로 한 응답 (scripts/rag_flow.py)."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
