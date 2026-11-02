import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from req2data.llm import extract_json, invoke_structured
from req2data.schemas import RouteDecision
from tests.fixtures.scripted_llm import ScriptedChatModel

MESSAGES = [SystemMessage(content="역할"), HumanMessage(content="작업")]


def test_extract_json_handles_fence_and_prose():
    assert extract_json('설명입니다\n```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('결과: {"a": {"b": 2}} 끝') == '{"a": {"b": 2}}'


def test_retry_with_error_then_success():
    llm = ScriptedChatModel().script("RouteDecision", '{"next": "nowhere"}', {"next": "reviewer"})
    result = invoke_structured(
        llm, MESSAGES, RouteDecision, retries=1, fallback=lambda: RouteDecision(next="end")
    )
    assert result.value.next == "reviewer" and result.attempts == 2 and not result.used_fallback
    retry_prompt = llm.calls[1]["messages"][-1].content
    assert "스키마 검증에 실패" in retry_prompt and "next" in retry_prompt


def test_fallback_after_retries_exhausted():
    llm = ScriptedChatModel().script("RouteDecision", "모르겠습니다", "{}")
    result = invoke_structured(
        llm, MESSAGES, RouteDecision, retries=1, fallback=lambda: RouteDecision(next="end")
    )
    assert result.used_fallback and result.value.next == "end" and result.attempts == 2


def test_schema_instruction_is_appended_to_system_prompt():
    llm = ScriptedChatModel().script("RouteDecision", {"next": "end"})
    invoke_structured(llm, MESSAGES, RouteDecision, fallback=lambda: RouteDecision(next="end"))
    system = llm.calls[0]["messages"][0].content
    assert system.startswith("역할") and "출력 스키마: RouteDecision" in system


def test_wrapped_structure_is_rejected_not_silently_empty():
    """실제 7B 실행에서 본 실패: {"data_mapping": {...}}로 감싼 출력이 빈 DataMapping으로 통과하면 안 된다."""
    from req2data.schemas import DataMapping

    wrapped = {"data_mapping": {"source_tables": [{"table_name": "orders"}]}}
    good = {"analysis_key": {"table": "customers", "column": "customer_unique_id"}}
    llm = ScriptedChatModel().script("DataMapping", wrapped, good)
    result = invoke_structured(llm, MESSAGES, DataMapping, fallback=DataMapping)
    assert result.attempts == 2 and result.value.analysis_key.column == "customer_unique_id"
    assert "data_mapping" in llm.calls[1]["messages"][-1].content  # 오류 내용이 재요청에 들어감


def test_ollama_gets_json_schema_format():
    ChatOllama = pytest.importorskip("langchain_ollama").ChatOllama

    from req2data.llm.recording import RecordReplayChatModel
    from req2data.llm.structured import native_schema_kwargs

    ollama = ChatOllama(model="x")
    kwargs = native_schema_kwargs(ollama, RouteDecision)
    assert kwargs["format"]["additionalProperties"] is False
    wrapped = RecordReplayChatModel(label="json", path="x.jsonl", mode="record", inner=ollama)
    assert native_schema_kwargs(wrapped, RouteDecision) == kwargs
    assert native_schema_kwargs(ScriptedChatModel(), RouteDecision) == {}


def test_transient_call_failure_is_retried():
    """연결 끊김·타임아웃은 재시도 횟수 안에서 같은 요청을 다시 보낸다."""

    class Flaky(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if not self.calls_for("RouteDecision"):
                self.calls.append({"schema": "RouteDecision", "messages": messages})
                raise TimeoutError("read timeout")
            return super()._generate(messages, stop, run_manager, **kwargs)

    llm = Flaky().script("RouteDecision", {"next": "reviewer"})
    result = invoke_structured(
        llm, MESSAGES, RouteDecision, retries=1, fallback=lambda: RouteDecision(next="end")
    )
    assert result.value.next == "reviewer" and result.attempts == 2 and not result.used_fallback
