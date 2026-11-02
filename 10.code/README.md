# Req2Data Agent

요구사항 정의서 → ML 데이터셋 설계(구조화 → 데이터 매핑 → 타깃·피처 → 마트 SQL → 검증) LangGraph Multi-Agent.
상세 기획: `Req2Data Agent – (서종해).md`

## 환경 설정

```bash
make setup     # .env 생성 + uv sync (Python 3.12, .venv)
make models    # Ollama 모델: qwen2.5:14b-instruct, bge-m3
```

Olist 데이터: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce 의 CSV 9개를 `data/raw/olist/`에 둡니다.

```bash
make data      # CSV → DuckDB 적재 → 컬럼 프로파일 → RAG 인덱스 (상세: docs/rag.md)
make index     # 인덱스만 다시 구축 (CSV 없이도 카탈로그 정의·용어집·가이드로 동작)
```

## 실행

```bash
make run       # API http://localhost:8000/docs + UI http://localhost:8501 (Ollama는 호스트에서 실행)
make demo      # LLM 없이 기록된 응답으로 실행
make up        # Docker Compose
```

API·UI·데모 모드·Docker 상세: [docs/service.md](docs/service.md)

## Agent 구조

설계 문서: [docs/agent-architecture.md](docs/agent-architecture.md)

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from req2data.agents import AgentDeps
from req2data.graph import build_graph, new_project_state
from req2data.tools.local_catalog import LocalCatalogClient

graph = build_graph(AgentDeps.from_settings(catalog=LocalCatalogClient.from_settings()), checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "p1"}}
graph.invoke(new_project_state("p1", 정의서_텍스트, "practitioner"), config)   # 확인 질문에서 멈춤
graph.invoke(Command(resume={"answers": {"R1Q1": "90일"}}), config)          # 답변으로 재개
graph.invoke({"change_request": "타깃 기간을 60일로 바꿔줘"}, config)          # 대화형 수정
```

`LocalCatalogClient`는 DuckDB(읽기 전용)와 RAG 지식 베이스를 묶은 `CatalogClient` 구현체입니다. MCP 카탈로그 서버는 이를 감싸 노출할 예정이고, 테스트는 `tests/fixtures/`의 가짜 구현을 씁니다.

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
