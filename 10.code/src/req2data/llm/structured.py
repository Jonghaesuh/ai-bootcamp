import logging
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from req2data.llm.chains import (
    extract_json,
    generation_chain,
    native_schema_kwargs,
    parsing_chain,
)
from req2data.llm.recording import DemoReplayMiss

log = logging.getLogger(__name__)

__all__ = ["StructuredResult", "extract_json", "invoke_structured", "native_schema_kwargs"]


@dataclass
class StructuredResult[T: BaseModel]:
    value: T
    attempts: int
    used_fallback: bool = False
    error: str | None = None


def invoke_structured[T: BaseModel](
    llm: BaseChatModel,
    messages: list[BaseMessage],
    schema: type[T],
    *,
    retries: int = 1,
    fallback: Callable[[], T],
) -> StructuredResult[T]:
    """LCEL 체인(프롬프트 → LLM → 파서)으로 구조화 출력을 받는다.
    스키마 검증 실패 시 오류를 넣어 재요청하고, 그래도 실패하면 규칙 기반 기본값으로 대체한다."""
    convo = list(messages)
    system = ""
    if convo and isinstance(convo[0], SystemMessage):
        system = str(convo.pop(0).content)
    generate = generation_chain(llm, schema)
    parse = parsing_chain(schema)

    error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            raw = generate.invoke({"system": system, "messages": convo})
        except DemoReplayMiss as exc:  # 데모 녹화에 없는 입력: 다시 보내도 같으므로 바로 기본값
            return StructuredResult(fallback(), attempt, used_fallback=True, error=str(exc))
        except Exception as exc:
            # 타임아웃·연결 끊김은 같은 요청을 다시 보내고, 재시도를 다 쓰면 대체
            error = f"LLM 호출 실패: {exc}"
            log.warning("LLM 호출 실패(%s, %d회차): %s", schema.__name__, attempt, exc)
            continue
        try:
            return StructuredResult(parse.invoke(raw), attempt)
        except (ValueError, ValidationError, OutputParserException) as exc:
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
