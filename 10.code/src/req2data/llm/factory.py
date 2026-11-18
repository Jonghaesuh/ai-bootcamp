import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult

from req2data.config import Settings, get_settings

log = logging.getLogger(__name__)

AZURE_KEYS_MISSING = (
    "Azure OpenAI 설정이 없습니다. AOAI_ENDPOINT·AOAI_API_KEY"
    "(또는 AZURE_OPENAI_ENDPOINT·AZURE_OPENAI_API_KEY) 환경변수를 지정하세요."
)


class UnconfiguredChatModel(BaseChatModel):
    """키가 없을 때 서비스는 뜨게 하고(데모 모드 등) 실제 호출만 실패시킨다 → 규칙 기반 기본값으로 대체."""

    reason: str

    @property
    def _llm_type(self) -> str:
        return "unconfigured"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise ValueError(self.reason)


def get_chat_model(settings: Settings | None = None, *, json_mode: bool = False) -> BaseChatModel:
    """LLM 제공자 추상화. `json_mode`는 Structured Output 전용 호출에 쓴다(도구 호출과 함께 쓰지 않음)."""
    s = settings or get_settings()
    if s.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI

        if not (s.azure_openai_endpoint and s.azure_openai_api_key):
            if not s.demo_mode:
                log.warning(AZURE_KEYS_MISSING)
            return UnconfiguredChatModel(reason=AZURE_KEYS_MISSING)
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        return AzureChatOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            azure_deployment=s.azure_chat_deployment,
            api_key=s.azure_openai_api_key,
            api_version=s.azure_openai_api_version,
            temperature=s.llm_temperature,
            max_tokens=s.llm_num_predict,
            timeout=s.llm_timeout_seconds,
            max_retries=6,  # 분당 요청 제한(429) 시 지수 백오프 재시도
            model_kwargs=kwargs,
        )
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=s.llm_model,
            base_url=s.ollama_base_url,
            temperature=s.llm_temperature,
            num_ctx=s.llm_num_ctx,
            num_predict=s.llm_num_predict,
            client_kwargs={"timeout": s.llm_timeout_seconds},
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
            max_tokens=s.llm_num_predict,
            timeout=s.llm_timeout_seconds,
            model_kwargs=kwargs,
        )
    if s.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=s.llm_model,
            api_key=s.anthropic_api_key,
            temperature=s.llm_temperature,
            max_tokens=s.llm_num_predict,
            timeout=s.llm_timeout_seconds,
        )
    if s.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not s.google_api_key:
            raise ValueError("LLM_PROVIDER=gemini에는 GOOGLE_API_KEY가 필요합니다.")
        # Gemini 3.x는 temperature 등 샘플링 파라미터가 폐기되어 넘기지 않는다 (통합 라이브러리가 모델별로 처리).
        return ChatGoogleGenerativeAI(
            model=s.llm_model,
            google_api_key=s.google_api_key,
            max_output_tokens=s.llm_num_predict,
            max_retries=6,  # 무료 등급 분당 요청 제한(429) 시 지수 백오프 재시도
            timeout=s.llm_timeout_seconds,
            response_mime_type="application/json" if json_mode else None,
        )
    raise ValueError(f"지원하지 않는 LLM 제공자: {s.llm_provider}")
