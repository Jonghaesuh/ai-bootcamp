"""LLM 제공자 설정 (네트워크 호출 없이 생성만 확인)."""

import pytest

from req2data.config import Settings
from req2data.llm.factory import get_chat_model
from req2data.llm.structured import native_schema_kwargs
from req2data.schemas import RouteDecision


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_azure_uses_aoai_env_and_json_mode(monkeypatch):
    monkeypatch.setenv("AOAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AOAI_API_KEY", "test-key")
    monkeypatch.setenv("AOAI_DEPLOY_GPT41_MINI", "my-gpt-4.1-mini")
    s = _settings(llm_provider="azure")
    chat, structured = get_chat_model(s), get_chat_model(s, json_mode=True)
    assert type(chat).__name__ == "AzureChatOpenAI"
    assert chat.deployment_name == "my-gpt-4.1-mini" and s.llm_label == "azure:my-gpt-4.1-mini"
    assert structured.model_kwargs["response_format"] == {"type": "json_object"}
    assert native_schema_kwargs(structured, RouteDecision) == {}


def test_azure_without_keys_starts_but_calls_fail(monkeypatch):
    for k in ("AOAI_ENDPOINT", "AOAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    chat = get_chat_model(_settings(llm_provider="azure"))
    with pytest.raises(ValueError, match="AOAI_ENDPOINT"):
        chat.invoke("hi")


def test_ollama_gets_context_and_output_caps():
    pytest.importorskip("langchain_ollama")
    m = get_chat_model(_settings(llm_provider="ollama", llm_num_ctx=16384, llm_num_predict=2048))
    assert m.num_ctx == 16384 and m.num_predict == 2048


def test_gemini_json_mode_and_no_temperature():
    pytest.importorskip("langchain_google_genai")
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
    pytest.importorskip("langchain_google_genai")
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        get_chat_model(_settings(llm_provider="gemini", google_api_key=None))
