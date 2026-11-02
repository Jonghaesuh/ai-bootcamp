import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class StructuredResult[T: BaseModel]:
    value: T
    attempts: int
    used_fallback: bool = False
    error: str | None = None


def schema_instruction(schema: type[BaseModel]) -> str:
    return (
        f"## 출력 형식\n출력 스키마: {schema.__name__}\n"
        "아래 JSON Schema를 따르는 JSON 객체 하나만 출력하세요. 설명 문장이나 코드 블록 표시는 쓰지 마세요.\n"
        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
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


def _content(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in message.content
    )


def invoke_structured[T: BaseModel](
    llm: BaseChatModel,
    messages: list[BaseMessage],
    schema: type[T],
    *,
    retries: int = 1,
    fallback: Callable[[], T],
) -> StructuredResult[T]:
    """스키마 검증 실패 시 오류를 넣어 재요청하고, 그래도 실패하면 규칙 기반 기본값으로 대체한다."""
    convo = list(messages)
    if convo and isinstance(convo[0], SystemMessage):
        convo[0] = SystemMessage(content=f"{convo[0].content}\n\n{schema_instruction(schema)}")
    else:
        convo.insert(0, SystemMessage(content=schema_instruction(schema)))

    error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            raw = _content(llm.invoke(convo))
        except Exception as exc:  # 네트워크·모델 오류는 재요청해도 같은 경우가 많아 바로 대체
            log.warning("LLM 호출 실패(%s): %s", schema.__name__, exc)
            return StructuredResult(fallback(), attempt, used_fallback=True, error=str(exc))
        try:
            return StructuredResult(schema.model_validate_json(extract_json(raw)), attempt)
        except (ValueError, ValidationError) as exc:
            error = str(exc)
            log.info("스키마 검증 실패(%s, %d회차): %s", schema.__name__, attempt, error)
            convo += [
                AIMessage(content=raw),
                HumanMessage(
                    content=f"출력이 {schema.__name__} 스키마 검증에 실패했습니다.\n오류: {error}\n"
                    "오류를 고쳐 JSON 객체 하나만 다시 출력하세요."
                ),
            ]
    return StructuredResult(fallback(), retries + 1, used_fallback=True, error=error)
