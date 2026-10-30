.PHONY: setup env models data mcp api ui run test lint

setup: env
	uv sync --all-groups --extra a2a

env:
	@test -f .env || cp .env.example .env

models:
	ollama pull qwen2.5:7b-instruct
	ollama pull bge-m3

# 데이터 적재·카탈로그 프로파일·인덱스 구축 (스크립트는 구현 단계에서 작성)
data:
	uv run python scripts/load_olist.py
	uv run python scripts/build_index.py

mcp:
	uv run python -m mcp_catalog.server

api:
	uv run uvicorn req2data.api.main:app --reload --port 8000

ui:
	uv run streamlit run ui/app.py --server.port 8501

run:
	$(MAKE) -j3 mcp api ui

test:
	uv run pytest -m "not llm"

lint:
	uv run ruff check src tests
