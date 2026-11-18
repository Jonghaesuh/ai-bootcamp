# Req2Data Agent

요구사항 정의서 → ML 데이터셋 설계(구조화 → 데이터 매핑 → 타깃·피처 → 마트 SQL → 검증) LangGraph Multi-Agent.
상세 기획: `Req2Data Agent – (서종해).md`

## 평가 항목별 구현 위치

| 평가 항목 | 구현 위치 (`src/req2data/`) | 실행 |
| --- | --- | --- |
| 임베딩·Vector DB 인덱싱, 저장·조회 | `rag/index.py` `KnowledgeIndex` (`langchain_chroma.Chroma`의 `add_documents`·`similarity_search_with_score`), `scripts/build_index.py` | `make index` |
| 문서 분할·메타데이터·정제 | `rag/preprocess.py` (`clean_text`, `split_markdown`, `enrich_metadata`, `dedupe`), `rag/documents.py` `build_documents` | `make rag-flow` |
| Retrieval | `rag/retriever.py` `HybridRetriever(BaseRetriever)`: 용어집 쿼리 확장 → 벡터 검색 + BM25 → RRF, `tools/catalog.py` `CatalogRetriever` | `make rag-flow` |
| 검색 지식을 근거와 함께 반영 | `llm/chains.py` `retrieval_chain`·`rag_chain`, `agents/base.py` `knowledge`·`cite`, `agents/citations.py` `validate_citations`, `reporting/render.py` `citations_md` | 산출물 `summary.md`의 "참고 근거" |
| LangChain 체인·LLM 호출 | `llm/chains.py` (`ChatPromptTemplate \| LLM \| StrOutputParser \| PydanticOutputParser`), `llm/structured.py` `invoke_structured` | `make test` |
| LangGraph 노드·엣지·상태 | `graph/builder.py` (`StateGraph`, `add_node`, `add_edge`, `add_conditional_edges`), `graph/state.py` `Req2DataState`, `agents/supervisor.py` (`interrupt`) | `make run` |
| Tool binding·Tool 선택 호출 | `agents/tool_loop.py` (`bind_tools`, `ToolNode`, `tools_condition`), `tools/catalog.py` `build_catalog_tools` (도구 6개) | `make test` |
| 도메인 맞춤 검증 | `agents/reviewer.py` 검사 7종: `check_existence`, `check_sql`, `check_leakage`, `check_censoring`, `check_quality`, `check_coverage`, `check_grounding` | `make run` |

실행 순서:

```bash
make setup                                    # uv sync (Python 3.12)
export AOAI_ENDPOINT=... AOAI_API_KEY=...     # Azure OpenAI. 배포 이름은 AOAI_DEPLOY_GPT41_MINI, AOAI_DEPLOY_EMBED_3_SMALL
uv run python scripts/check_env.py --full     # LLM·임베딩·Tool 호출 점검 → 인덱스 구축 → 샘플 1건 전체 실행
```

## 환경 설정

```bash
make setup     # .env 생성 + uv sync (Python 3.12, .venv)
# LLM·임베딩 기본값은 Azure OpenAI(GPT): .env 또는 환경변수에 AOAI_ENDPOINT·AOAI_API_KEY
make models    # (선택) 로컬 Ollama 모델: qwen2.5:14b-instruct, bge-m3 → LLM_PROVIDER=ollama
```

Olist 데이터: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce 의 CSV 9개를 `data/raw/olist/`에 둡니다.

```bash
make data      # CSV → DuckDB 적재 → 컬럼 프로파일 → RAG 인덱스 (상세: docs/rag.md)
make index     # 인덱스만 다시 구축 (CSV 없이도 카탈로그 정의·용어집·가이드로 동작)
```

## 실행

```bash
make run       # API http://localhost:8000/docs + UI http://localhost:8501
make rag-flow  # RAG 흐름만 단계별로 실행: 전처리 → 인덱싱 → 검색 → 근거 포함 응답
make check     # 실행 환경 점검 + 샘플 1건 전체 실행
make up        # Docker Compose
```

API·UI·Docker 상세: [docs/service.md](docs/service.md)

제출용 소스 zip: `make source-zip` → `dist/req2data-agent-source.zip`, 실행 안내 [docs/SUBMISSION.md](docs/SUBMISSION.md)

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
  llm/         LLM 제공자 추상화 (azure/ollama/openai/gemini), LCEL 체인, 구조화 출력
  rag/         전처리(정제·분할·메타데이터), Chroma 인덱스, 하이브리드 검색기, FK 그래프
  tools/       Agent 도구 6개, CatalogClient 경계, 읽기 전용 SQL
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
