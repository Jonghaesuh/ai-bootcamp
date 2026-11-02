from langchain_core.language_models import BaseChatModel

from req2data.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None, *, json_mode: bool = False) -> BaseChatModel:
    """LLM 제공자 추상화. `json_mode`는 Structured Output 전용 호출에 쓴다(도구 호출과 함께 쓰지 않음)."""
    s = settings or get_settings()
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=s.llm_model,
            base_url=s.ollama_base_url,
            temperature=s.llm_temperature,
            format="json" if json_mode else None,
        )
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        return ChatOpenAI(
            model=s.llm_model,
            api_key=s.openai_api_key,
            base_url=s.openai_base_url or None,
            temperature=s.llm_temperature,
            model_kwargs=kwargs,
        )
    if s.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=s.llm_model, api_key=s.anthropic_api_key, temperature=s.llm_temperature
        )
    raise ValueError(f"지원하지 않는 LLM 제공자: {s.llm_provider}")
