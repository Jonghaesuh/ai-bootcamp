import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from req2data.llm.recording import DemoReplayMiss

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


def native_schema_kwargs(llm: BaseChatModel, schema: type[BaseModel]) -> dict[str, Any]:
    """Ollama는 JSON Schema로 디코딩을 제약할 수 있다 (작은 모델이 임의 구조를 만드는 것을 막음).
    기록 모드 래퍼는 내부 모델 기준으로 판단한다."""
    target = getattr(llm, "inner", None) or llm
    if type(target).__name__ == "ChatOllama":
        return {"format": schema.model_json_schema()}
    return {}


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
            raw = _content(llm.invoke(convo, **native_schema_kwargs(llm, schema)))
        except DemoReplayMiss as exc:  # 데모 녹화에 없는 입력: 다시 보내도 같으므로 바로 기본값
            return StructuredResult(fallback(), attempt, used_fallback=True, error=str(exc))
        except Exception as exc:
            # 타임아웃·연결 끊김은 같은 요청을 다시 보내고, 재시도를 다 쓰면 대체
            error = f"LLM 호출 실패: {exc}"
            log.warning("LLM 호출 실패(%s, %d회차): %s", schema.__name__, attempt, exc)
            continue
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
