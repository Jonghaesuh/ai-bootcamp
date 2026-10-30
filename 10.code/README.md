# Req2Data Agent

요구사항 정의서 → ML 데이터셋 설계(구조화 → 데이터 매핑 → 타깃·피처 → 마트 SQL → 검증) LangGraph Multi-Agent.
상세 기획: `Req2Data Agent – (서종해).md`

## 환경 설정

```bash
make setup     # .env 생성 + uv sync (Python 3.12, .venv)
make models    # Ollama 모델: qwen2.5:7b-instruct, bge-m3
```

Olist 데이터: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce 의 CSV 9개를 `data/raw/olist/`에 둡니다.

## 디렉토리

```
src/req2data/
  agents/      Supervisor + 요구사항·탐색·설계·검증·리포트 Agent
  graph/       LangGraph State·그래프·라우팅
  schemas/     Pydantic 계약 (RequirementSpec, DataMapping, DatasetDesign, ReviewReport, RouteDecision)
  llm/         LLM·임베딩 제공자 추상화 (ollama/openai/anthropic), 데모 모드
  rag/         카탈로그 인덱싱, Chroma+BM25 하이브리드, FK 그래프
  tools/       Agent 도구 (MCP 클라이언트 포함)
  memory/      체크포인터(단기)·Store(장기)
  reporting/   명세서·추적표·리포트 생성
  api/         FastAPI (SSE)
  config.py    pydantic-settings
src/mcp_catalog/  FastMCP 카탈로그·읽기 전용 SQL 서버
ui/               Streamlit
prompts/          Agent별 프롬프트 YAML, fewshot/
data/             raw/olist, db, catalog, glossary, guides, cases, samples
storage/          chroma, checkpoints, store, artifacts
scripts/ tests/ demo/recordings docker/ docs/
```
