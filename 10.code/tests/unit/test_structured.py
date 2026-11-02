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
