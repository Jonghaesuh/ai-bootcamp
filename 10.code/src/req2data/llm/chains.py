"""LCEL 체인: 프롬프트 템플릿 → LLM → 출력 파서, 그리고 검색 결과를 문맥으로 넣는 RAG 체인.

- generation_chain | parsing_chain: 모든 Agent의 구조화 출력 호출 (llm/structured.py가 재시도를 감싼다)
- retrieval_chain: 검색 → 문서와 프롬프트용 문맥 블록 (Agent의 지식 주입)
- rag_chain: 검색 → 프롬프트 → LLM → 파서를 한 체인으로 (scripts/rag_flow.py)
"""

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pydantic import BaseModel

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

STRUCTURED_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "{system}\n\n{format_instructions}"), MessagesPlaceholder("messages")]
)
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{instruction}\n\n{format_instructions}"),
        ("human", "## 검색된 문맥\n{context}\n\n## 질문\n{question}"),
    ]
)


def format_instructions(schema: type[BaseModel]) -> str:
    """출력 형식 안내. '출력 스키마: 이름' 줄은 로그·테스트에서 호출을 구분하는 표지로도 쓴다."""
    parser = PydanticOutputParser(pydantic_object=schema)
    return (
        f"## 출력 형식\n출력 스키마: {schema.__name__}\n"
        "아래 안내를 따르는 JSON 객체 하나만 출력하세요. 설명 문장은 쓰지 마세요.\n"
        f"{parser.get_format_instructions()}"
    )


def extract_json(text: str) -> str:
    """모델 출력에서 JSON 객체 부분만 꺼낸다 (코드 블록·앞뒤 설명 허용)."""
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("JSON 객체를 찾지 못했습니다.")
    return text[start : end + 1]


def native_schema_kwargs(llm: BaseChatModel, schema: type[BaseModel]) -> dict[str, Any]:
    """Ollama는 JSON Schema로 디코딩을 제약할 수 있다 (작은 모델이 임의 구조를 만드는 것을 막음).
    기록 모드 래퍼는 내부 모델 기준으로 판단한다."""
    target = getattr(llm, "inner", None) or llm
    if type(target).__name__ == "ChatOllama":
        return {"format": schema.model_json_schema()}
    return {}


def _bound(llm: BaseChatModel, schema: type[BaseModel]) -> Runnable:
    kwargs = native_schema_kwargs(llm, schema)
    return llm.bind(**kwargs) if kwargs else llm


def generation_chain(llm: BaseChatModel, schema: type[BaseModel]) -> Runnable[dict, str]:
    """{"system", "messages"} → 모델 원문. 시스템 프롬프트 끝에 출력 형식 안내를 붙인다."""
    prompt = STRUCTURED_PROMPT.partial(format_instructions=format_instructions(schema))
    return prompt | _bound(llm, schema) | StrOutputParser()


def parsing_chain[T: BaseModel](schema: type[T]) -> Runnable[str, T]:
    """모델 원문 → 스키마 객체. 검증에 실패하면 예외를 던진다."""
    return RunnableLambda(extract_json) | PydanticOutputParser(pydantic_object=schema)


def structured_chain[T: BaseModel](llm: BaseChatModel, schema: type[T]) -> Runnable[dict, T]:
    return generation_chain(llm, schema) | parsing_chain(schema)


def format_docs(docs: list[Document]) -> str:
    """검색 문서 → 프롬프트용 문맥. 모델이 인용할 수 있게 문서마다 [doc_id]를 붙인다."""
    return "\n\n".join(
        f"[{d.id}] {d.metadata.get('title', '')}".rstrip() + f"\n{d.page_content}" for d in docs
    )


def retrieval_chain(retriever: BaseRetriever) -> Runnable[str, dict]:
    """질의 → {"docs": 검색 문서, "context": 프롬프트용 문맥}."""
    return retriever | RunnableParallel(
        docs=RunnablePassthrough(), context=RunnableLambda(format_docs)
    )


def rag_chain[T: BaseModel](
    retriever: BaseRetriever, llm: BaseChatModel, schema: type[T], instruction: str
) -> Runnable[str, T]:
    """질문 → 검색 → 문맥을 넣은 프롬프트 → LLM → 스키마 객체."""
    prompt = RAG_PROMPT.partial(
        instruction=instruction, format_instructions=format_instructions(schema)
    )
    return (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | _bound(llm, schema)
        | StrOutputParser()
        | parsing_chain(schema)
    )


__all__ = [
    "extract_json",
    "format_docs",
    "format_instructions",
    "generation_chain",
    "native_schema_kwargs",
    "parsing_chain",
    "rag_chain",
    "retrieval_chain",
    "structured_chain",
]
