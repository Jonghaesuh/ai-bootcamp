from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from req2data.config import Settings, get_settings


class UnavailableEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("임베딩 설정 없음 (AOAI_ENDPOINT·AOAI_API_KEY)")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("임베딩 설정 없음 (AOAI_ENDPOINT·AOAI_API_KEY)")


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """한글 요구사항과 영문 컬럼명을 함께 다루므로 다국어 임베딩을 쓴다 (기본 Azure text-embedding-3-small)."""
    s = settings or get_settings()
    if s.embedding_provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        if not (s.azure_openai_endpoint and s.azure_openai_api_key):
            return (
                UnavailableEmbeddings()
            )  # 의미 검색은 실패하고 BM25로 계속 (retriever._dense_rank)

        return AzureOpenAIEmbeddings(
            azure_endpoint=s.azure_openai_endpoint,
            azure_deployment=s.azure_embedding_deployment,
            api_key=SecretStr(s.azure_openai_api_key),
            api_version=s.azure_openai_api_version,
        )
    if s.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=s.embedding_model, base_url=s.ollama_base_url)
    if s.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=s.embedding_model,
            api_key=SecretStr(s.openai_api_key) if s.openai_api_key else None,
            base_url=s.openai_base_url or None,
        )
    raise ValueError(f"지원하지 않는 임베딩 제공자: {s.embedding_provider}")
