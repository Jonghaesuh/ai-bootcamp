# api·ui 공용 이미지. 데이터·인덱스·프로젝트 상태는 볼륨(data/, storage/)으로 붙인다.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 의존성 레이어를 먼저 만들어 소스 변경 시 재사용
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
