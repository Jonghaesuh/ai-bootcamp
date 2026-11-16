.PHONY: setup env models data index rag-flow check mcp api ui run demo record up up-demo down submit source-zip test test-llm lint

setup: env
	uv sync --all-groups --all-extras

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

# RAG 흐름을 단계별로 실행: 전처리 → 인덱싱 → 검색 → 근거 포함 응답
rag-flow:
	uv run python scripts/rag_flow.py

# 실행 환경 점검: LLM·임베딩 호출, 인덱스, 샘플 1건 전체 실행
check:
	uv run python scripts/check_env.py --full

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

# LLM 없이 기록된 응답으로 실행 (demo/recordings, 녹화는 make record로 직접 만든다)
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

# 제출용 단일 이미지 (amd64·arm64) → dist/*.tar.gz. 실행 안내: docs/SUBMISSION.md
submit:
	mkdir -p dist
	for arch in amd64 arm64; do \
		docker buildx build -f docker/Dockerfile --target submit --platform linux/$$arch -t req2data-agent:submit --load . && \
		docker save req2data-agent:submit | gzip -1 > dist/req2data-agent-submit-$$arch.tar.gz; \
	done
	cp docs/SUBMISSION.md dist/
	cd dist && shasum -a 256 *.tar.gz > SHA256SUMS

# 제출용 소스 zip → dist/req2data-agent-source.zip (가상환경·Docker 이미지 제외).
# storage/와 data/db/는 gitignore 대상이라 git에 등록된 파일 목록에 직접 더한다.
# 한글 파일 이름이 이스케이프되지 않도록 quotePath를 끄고, 작업용 설계·계획 문서(docs/superpowers)는 뺀다.
source-zip:
	mkdir -p dist && rm -f dist/req2data-agent-source.zip
	(git -c core.quotePath=false ls-files | grep -v '^docs/superpowers/'; ls data/db/olist.duckdb data/catalog/profiles.json; find storage/chroma -type f) | zip -q dist/req2data-agent-source.zip -@

# Ollama가 필요한 테스트 (실제 bge-m3 검색 품질 등)
test-llm:
	uv run pytest -m llm

lint:
	uv run ruff check src tests ui scripts
