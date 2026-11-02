from langchain_core.embeddings import Embeddings

from req2data.config import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """한글 요구사항과 영문 컬럼명을 함께 다루므로 다국어 모델(bge-m3)을 기본으로 쓴다."""
    s = settings or get_settings()
    if s.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=s.embedding_model, base_url=s.ollama_base_url)
    if s.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=s.embedding_model, api_key=s.openai_api_key, base_url=s.openai_base_url or None
        )
    raise ValueError(f"지원하지 않는 임베딩 제공자: {s.embedding_provider}")
