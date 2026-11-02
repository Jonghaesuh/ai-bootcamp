# 두 가지 타깃
#   app    : docker-compose용. 데이터·인덱스·프로젝트 상태는 볼륨(data/, storage/)으로 붙인다.
#   submit : 제출용 단일 이미지. 데이터(Olist DuckDB·프로파일)·RAG 인덱스·데모 녹화·테스트를 모두 포함해
#            `docker run` 하나로 API(8000)+UI(8501)가 뜨고, LLM 없이 데모 모드로 바로 실행된다.
FROM python:3.12-slim AS app

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 의존성 레이어를 먼저 만들어 소스 변경 시 재사용 (UV_NO_CACHE로 다운로드 캐시를 이미지에 남기지 않음)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY ui ./ui
COPY scripts ./scripts
COPY prompts ./prompts
COPY config.yaml ./
RUN uv sync --frozen --no-dev

EXPOSE 8000 8501
CMD ["uvicorn", "req2data.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM app AS submit

# 테스트 실행용 (uv.lock과 같은 버전)
RUN uv pip install --python /opt/venv/bin/python pytest==9.2.0 pytest-asyncio==1.4.1

# 지식 소스·샘플·문서
COPY data/catalog ./data/catalog
COPY data/glossary ./data/glossary
COPY data/guides ./data/guides
COPY data/samples ./data/samples
COPY ["Req2Data Agent – (서종해).md", "./docs/PRD.md"]
COPY docs ./docs
COPY tests ./tests
# 적재된 Olist DB와 bge-m3로 만든 RAG 인덱스 (CSV 원본은 넣지 않음)
COPY data/db/olist.duckdb ./data/db/olist.duckdb
COPY storage/chroma ./storage/chroma
# 데모 녹화 (Qwen2.5 14B로 실행한 샘플 01)
COPY demo/recordings ./demo/recordings
COPY docker/entrypoint.sh /usr/local/bin/req2data

# 기본은 LLM 없이 실행되는 데모 모드. 실제 LLM은 docs/SUBMISSION.md 참고 (DEMO_MODE=false + Ollama/Gemini)
ENV DEMO_MODE=true \
    API_BASE_URL=http://localhost:8000 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["req2data"]
CMD ["serve"]
