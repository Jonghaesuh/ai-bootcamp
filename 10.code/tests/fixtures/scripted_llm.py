"""테스트용 LLM: 프롬프트의 '출력 스키마: X'를 보고 X에 등록된 응답을 순서대로 돌려준다."""

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

_SCHEMA = re.compile(r"출력 스키마: (\w+)")


class ScriptedChatModel(BaseChatModel):
    _responses: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def script(self, schema: str, *responses: Any) -> "ScriptedChatModel":
        self._responses.setdefault(schema, []).extend(
            r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in responses
        )
        return self

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._calls

    def calls_for(self, schema: str) -> list[dict[str, Any]]:
        return [c for c in self._calls if c["schema"] == schema]

    def remaining(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._responses.items() if v}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        match = _SCHEMA.search(str(system))
        schema = match.group(1) if match else "react"
        self._calls.append({"schema": schema, "messages": messages})
        if schema == "react":
            content = "조사 완료: 추가로 확인할 사항 없음"
        else:
            queue = self._responses.get(schema) or []
            content = (
                queue.pop(0) if queue else "{}"
            )  # 응답이 없으면 스키마 검증 실패 → fallback 경로
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
