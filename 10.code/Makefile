.PHONY: setup env models data index mcp api ui run demo record up up-demo down test test-llm lint

setup: env
	uv sync --all-groups --extra a2a

env:
	@test -f .env || cp .env.example .env

models:
	ollama pull qwen2.5:14b-instruct
	ollama pull bge-m3

# 데이터 적재·카탈로그 프로파일·인덱스 구축
data:
	uv run python scripts/load_olist.py
	$(MAKE) index

# RAG 인덱스만 다시 구축 (용어집·가이드 수정 후). CSV 없이도 카탈로그 정의만으로 동작
index:
	uv run python scripts/build_index.py

# MCP 카탈로그 서버 (PRD 2.5 B에서 구현 예정)
mcp:
	uv run python -m mcp_catalog.server

api:
	uv run uvicorn req2data.api.main:app --reload --reload-dir src --port 8000

ui:
	uv run streamlit run ui/app.py --server.port 8501

# 로컬 실행: API(8000) + UI(8501). Ollama는 호스트에서 실행 중이어야 한다.
run:
	$(MAKE) -j2 api ui

# LLM 없이 기록된 응답으로 실행 (demo/recordings)
demo:
	DEMO_MODE=true $(MAKE) run

# 실제 LLM 응답을 demo/recordings에 기록하며 실행
record:
	LLM_RECORD=true $(MAKE) run

# Docker Compose (api + ui, Ollama는 호스트)
up:
	docker compose up --build -d

down:
	docker compose down

# Docker로 LLM 없이 데모 재생
up-demo:
	DEMO_MODE=true docker compose up --build -d

test:
	uv run pytest -m "not llm"

# Ollama가 필요한 테스트 (실제 bge-m3 검색 품질 등)
test-llm:
	uv run pytest -m llm

lint:
	uv run ruff check src tests ui scripts
