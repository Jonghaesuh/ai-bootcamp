"""데모 모드: 실제 LLM 응답을 기록하고(LLM_RECORD=true), LLM 없이 재생한다(DEMO_MODE=true).

입력 메시지·바인딩된 도구 이름의 해시를 키로 쓴다. temperature 0이라 같은 입력이면 같은 흐름이 재현된다.
기록에 없는 호출은 "{}"를 돌려줘 각 Agent의 규칙 기반 기본값 경로로 넘어간다.
"""

import hashlib
import json
import logging
import threading
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, message_to_dict, messages_from_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

log = logging.getLogger(__name__)

RECORD_FILE = "llm_recordings.jsonl"
_write_lock = threading.Lock()


def recording_key(label: str, messages: list[BaseMessage], tool_names: list[str]) -> str:
    payload = json.dumps(
        [
            label,
            tool_names,
            [[m.type, m.content, getattr(m, "tool_calls", None)] for m in messages],
        ],
        ensure_ascii=False,
        default=str,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@cache
def _store(path: Path) -> dict[str, dict]:
    store: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                store[rec["key"]] = rec["message"]
    return store


class RecordReplayChatModel(BaseChatModel):
    label: str
    path: Path
    mode: Literal["record", "replay"]
    inner: BaseChatModel | None = None
    tool_names: list[str] = []
    _bound: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return f"{self.mode}-{self.label}"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "RecordReplayChatModel":
        clone = self.model_copy(update={"tool_names": [getattr(t, "name", str(t)) for t in tools]})
        if self.mode == "record" and self.inner is not None:
            clone._bound = self.inner.bind_tools(tools, **kwargs)
        return clone

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        key = recording_key(self.label, messages, self.tool_names)
        store = _store(self.path)
        if self.mode == "replay":
            data = store.get(key)
            if data is None:
                log.warning("데모 기록에 없는 LLM 호출(%s) → 기본값 경로", self.label)
                message: BaseMessage = AIMessage(content="{}")
            else:
                message = messages_from_dict([data])[0]
        else:
            message = (self._bound or self.inner).invoke(messages, **kwargs)
            data = message_to_dict(message)
            store[key] = data
            with _write_lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {"key": key, "label": self.label, "message": data}, ensure_ascii=False
                        )
                        + "\n"
                    )
        return ChatResult(generations=[ChatGeneration(message=message)])


def wrap_for_demo(deps, settings):
    """설정에 따라 AgentDeps의 LLM을 기록·재생 모델로 바꾼다."""
    path = Path(settings.demo_recordings_dir) / RECORD_FILE
    if settings.demo_mode:
        return replace(
            deps,
            llm=RecordReplayChatModel(label="chat", path=path, mode="replay"),
            json_llm=RecordReplayChatModel(label="json", path=path, mode="replay"),
        )
    if settings.llm_record:
        return replace(
            deps,
            llm=RecordReplayChatModel(label="chat", path=path, mode="record", inner=deps.llm),
            json_llm=RecordReplayChatModel(
                label="json", path=path, mode="record", inner=deps.json_llm
            ),
        )
    return deps
