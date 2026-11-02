"""LLM 제공자 설정 (네트워크 호출 없이 생성만 확인)."""

import pytest

from req2data.config import Settings
from req2data.llm.factory import get_chat_model
from req2data.llm.structured import native_schema_kwargs
from req2data.schemas import RouteDecision


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_ollama_gets_context_and_output_caps():
    m = get_chat_model(_settings(llm_provider="ollama", llm_num_ctx=16384, llm_num_predict=2048))
    assert m.num_ctx == 16384 and m.num_predict == 2048


def test_gemini_json_mode_and_no_temperature():
    s = _settings(llm_provider="gemini", llm_model="gemini-3.6-flash", google_api_key="test-key")
    chat, structured = get_chat_model(s), get_chat_model(s, json_mode=True)
    assert type(chat).__name__ == "ChatGoogleGenerativeAI" and chat.model == "gemini-3.6-flash"
    assert structured.response_mime_type == "application/json" and chat.response_mime_type is None
    assert chat.temperature is None  # Gemini 3.x는 샘플링 파라미터 폐기
    assert chat.max_output_tokens == s.llm_num_predict
    assert (
        native_schema_kwargs(structured, RouteDecision) == {}
    )  # JSON 모드 + 프롬프트 스키마 + 검증·재요청


def test_gemini_requires_api_key():
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        get_chat_model(_settings(llm_provider="gemini", google_api_key=None))
