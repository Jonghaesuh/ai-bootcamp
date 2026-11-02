from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """.env 기반 환경 설정. 정책값은 config.yaml(`policy`)에서 읽는다."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["ollama", "openai", "anthropic", "gemini"] = "ollama"
    llm_model: str = "qwen2.5:14b-instruct"
    llm_temperature: float = 0.0
    # Ollama 컨텍스트 길이. 기본 4096이면 긴 프롬프트가 잘려 지시·스키마가 사라진다
    llm_num_ctx: int = 16384
    # 응답 토큰 상한. 없으면 반복 생성이 컨텍스트 이동으로 끝없이 이어질 수 있다
    llm_num_predict: int = 2048
    # LLM 요청 하나의 최대 대기 시간(초). 연결이 조용히 끊겨도 무한 대기하지 않게
    llm_timeout_seconds: float = 600
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None  # LLM_PROVIDER=gemini

    embedding_provider: Literal["ollama", "openai"] = "ollama"
    embedding_model: str = "bge-m3"

    data_dir: Path = Path("./data")
    raw_data_dir: Path = Path("./data/raw/olist")
    db_path: Path = Path("./data/db/olist.duckdb")
    chroma_dir: Path = Path("./storage/chroma")
    checkpoint_db: Path = Path("./storage/checkpoints/checkpoints.sqlite")
    store_db: Path = Path("./storage/store/store.sqlite")
    project_db: Path = Path("./storage/store/projects.sqlite")
    artifact_dir: Path = Path("./storage/artifacts")
    prompt_dir: Path = Path("./prompts")
    config_path: Path = Path("./config.yaml")

    mcp_transport: Literal["stdio", "streamable-http"] = "streamable-http"
    mcp_catalog_url: str = "http://localhost:8765/mcp"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    ui_port: int = 8501

    demo_mode: bool = False  # 기록된 LLM 응답을 재생 (LLM 없이 실행)
    llm_record: bool = False  # 실제 LLM 응답을 demo_recordings_dir에 기록
    demo_recordings_dir: Path = Path("./demo/recordings")

    a2a_modeling_agent_url: str = "http://localhost:9999"

    log_level: str = "INFO"

    @property
    def policy(self) -> dict[str, Any]:
        return load_policy(self.config_path)


@lru_cache
def load_policy(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_settings() -> Settings:
    return Settings()
