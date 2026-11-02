# 두 가지 타깃
#   app    : docker-compose용. 데이터·인덱스·프로젝트 상태는 볼륨(data/, storage/)으로 붙인다.
#   submit : 제출용 단일 이미지. 데이터(Olist DuckDB·프로파일)·RAG 인덱스·데모 녹화·테스트를 모두 포함해
#            `docker run` 하나로 API(8000)+UI(8501)가 뜬다. LLM은 Azure OpenAI(GPT, AOAI_* 환경변수)이고,
#            키가 없으면 녹화 재생(데모 모드)으로 실행된다.
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
# 기본은 Azure OpenAI(GPT)만 설치. Ollama 등 로컬 제공자가 필요하면 --build-arg UV_EXTRAS="--extra local"
ARG UV_EXTRAS=""
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project $UV_EXTRAS

COPY src ./src
COPY ui ./ui
COPY scripts ./scripts
COPY prompts ./prompts
COPY config.yaml ./
RUN uv sync --frozen --no-dev $UV_EXTRAS

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
# 적재된 Olist DB와 RAG 인덱스 (CSV 원본은 넣지 않음). 인덱스는 GPT 임베딩 사용 시 시작할 때 다시 만든다
COPY data/db/olist.duckdb ./data/db/olist.duckdb
COPY storage/chroma ./storage/chroma
# 데모 녹화 (Qwen2.5 14B로 실행한 샘플 01)
COPY demo/recordings ./demo/recordings
COPY docker/entrypoint.sh /usr/local/bin/req2data

# LLM·임베딩은 Azure OpenAI. DEMO_MODE는 entrypoint가 AOAI 키 유무로 정한다 (docs/SUBMISSION.md)
ENV LLM_PROVIDER=azure \
    EMBEDDING_PROVIDER=azure \
    API_BASE_URL=http://localhost:8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["req2data"]
CMD ["serve"]
