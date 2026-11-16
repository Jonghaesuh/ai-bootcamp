# 재제출 보강 설계: 평가 코멘트 8개 항목을 표준 LangChain/LangGraph 코드로 충족

작성일: 2026-11-16 (2차 수정) · 분량: 1일 · 상태: 승인 대기

## 1. 의도

과제 제출(zip)이 불합격했고, 코멘트 8개가 모두 "실제 코드로 구현해야 한다"였다. 8개 항목은 이미 동작하지만 대부분 직접 만든 방식이라 표준 LangChain/LangGraph 구성요소로 보이지 않는다. 제출 zip(1.2GB)에는 `.venv` 파일 3만여 개가 들어 있어 소스 88개가 묻혀 있었다.

채점은 자동으로 이뤄진 것으로 추정되며 기준표는 없다. 이번 작업의 기준은 하나다: **최종 제출 코드가 코멘트 8개 항목을 명확히 충족한다.** 데모 녹화 재생은 고려하지 않는다 (사용자 결정).

## 2. 평가 코멘트와 대응

| # | 코멘트 | 대응 (절) |
| --- | --- | --- |
| 1 | 임베딩 모델로 Vector DB에 인덱싱, 실행 가능한 저장·조회 흐름 | 5.2, 5.9 |
| 2 | 문서 분할, 메타데이터 부여, 정제를 포함한 전처리·구조화 | 5.1 |
| 3 | 사용자 쿼리로 Vector DB에서 검색하는 retrieval | 5.3 |
| 4 | 검색된 지식을 근거와 함께 Agent 응답에 반영 | 5.5 |
| 5 | LangChain 전처리, 체인 구성, LLM 호출, RAG 파이프라인 | 5.1~5.5 |
| 6 | LangGraph 노드, 엣지, 상태 관리 | 5.6 |
| 7 | 상황에 따라 Tool을 선택·호출하는 Tool binding | 5.7 |
| 8 | 도메인 맞춤 검증 로직, 업무 특화 Agent 동작 | 5.8 |

## 3. 목표와 비목표

**목표**
- 8개 항목 각각이 표준 API 이름으로 소스에 드러나고, 실제 서비스 경로에서 호출된다.
- 테스트가 통과하고, 실제 LLM으로 샘플 1건이 끝까지 실행된다.
- 제출 zip은 소스만 담고, README 첫 화면에서 항목별 구현 위치를 찾을 수 있다.

**비목표**
- 데모 녹화 유지·재생성. 기존 녹화 파일은 프롬프트가 바뀌어 맞지 않으므로 삭제한다.
- 새 Agent, 새 API 엔드포인트, UI 개편.
- Docker 이미지 tar 재빌드.

**실패로 보는 경우**
- 표준 예제를 별도로 두고 실제 파이프라인은 그대로인 상태 (연결되지 않은 보여주기용 코드).
- 표준 이름만 넣고 호출되지 않는 코드.

## 4. 검토한 접근

| 접근 | 내용 | 장단점 |
| --- | --- | --- |
| A | zip 정리와 README 대응표만 | 반나절. 채점기가 표준 API 사용을 본다면 6개 항목이 그대로 걸림 |
| **B (선택)** | 핵심 경로를 표준 구성요소로 교체 + A | 파일을 못 찾은 경우와 형태를 못 알아본 경우를 모두 덮음 |
| C | 기존 코드는 두고 교과서형 예제를 별도 추가 | 하루. 같은 기능이 두 벌이 되고 실제 경로와 연결되지 않음 |

사용자가 B를 선택했다. 조건: 1일 분량, 데모 녹화는 고려하지 않음.

1차 설계는 기존 녹화가 재생되도록 프롬프트·청크·출력 스키마를 한 글자도 바꾸지 않는 것이었다. 이 제약을 없앴으므로 분할, 정제, 인용 근거를 실제로 동작이 달라지는 수준으로 구현한다.

## 5. 설계

### 5.1 전처리·문서 구조화 (`rag/preprocess.py` 신규, `rag/documents.py`)

처리 순서는 `load → clean → split → enrich`이고, 함수 `build_documents()` 하나에서 이 순서가 보이게 한다.

- **load**: 카탈로그 YAML, 용어집 YAML, 가이드·사례 Markdown을 LangChain `Document`로 읽는다. 기존 `RagDocument`는 `Document`(`page_content`, `metadata`, `id`)로 대체한다.
- **clean** (`clean_text`): 유니코드 NFC 정규화, 제어 문자 제거, 줄 끝 공백 제거, 빈 줄 3개 이상 축약. 정제 후 빈 문서와 내용이 같은 중복 문서를 버린다.
- **split**
  - Markdown은 `MarkdownHeaderTextSplitter`(`#`, `##`, `###`)로 구간을 나눈 뒤 `RecursiveCharacterTextSplitter`로 크기를 맞춘다.
  - 카탈로그는 테이블 문서 1개와 컬럼 문서(컬럼당 1개), 용어집은 용어당 1개로 유지한다. 이미 의미 단위라 더 쪼개지 않는다.
  - `chunk_size`, `chunk_overlap`은 `config.yaml`의 `rag` 설정에서 읽는다 (기본 600자, 80자).
- **enrich** (`enrich_metadata`): 모든 청크에 `source`, `kind`, `title`, `path`, `chunk_index`, `table`, `column`, `content_hash`를 붙인다. 컬럼 문서의 `is_pk`, `fk`, `is_time`, `time_rank`, `available_at`은 유지한다.

청크 ID 규칙이 바뀌므로 인덱스를 다시 만든다.

### 5.2 인덱싱 (`rag/index.py`)

- `chromadb` 직접 호출을 `langchain_chroma.Chroma`로 바꾼다. 저장 경로와 컬렉션 이름은 그대로다.
- 저장은 `add_documents(documents, ids=...)`, 조회는 `similarity_search_with_score(query, k, filter=...)`를 쓴다.
- 유지하는 동작: 임베딩을 먼저 계산한 뒤 컬렉션을 교체(임베딩 실패 시 기존 인덱스 보존), 중복 ID 검사, 임베딩 이름이 다르면 오래된 인덱스로 판정(`is_current`).
- 의존성 추가: `langchain-chroma`, `langchain-text-splitters`.

### 5.3 검색 (`rag/retriever.py`)

- `HybridRetriever`를 LangChain `BaseRetriever`의 하위 클래스로 만든다. `_get_relevant_documents(query)`가 용어집 쿼리 확장 → 벡터 검색(Chroma) + 키워드 검색(BM25) → RRF 결합 → 소스 필터를 수행한다.
- 반환은 `Document` 목록이고, 점수와 검색기별 순위를 메타데이터(`score`, `ranks`)에 넣는다.
- 기존 `search()`는 `invoke()`를 감싼 얇은 함수로 남겨 `LocalCatalogClient`와 API가 그대로 쓴다.
- 임베딩 서버가 없을 때 BM25만으로 계속하는 동작은 유지한다. 검색 순위 녹화(`DenseRecorder`)는 제거한다.

### 5.4 체인과 LLM 호출 (`llm/chains.py` 신규, `llm/structured.py`)

- 구조화 출력 체인: `ChatPromptTemplate | llm | PydanticOutputParser(schema)`.
  - 출력 형식 안내는 직접 만든 `schema_instruction` 대신 파서의 `get_format_instructions()`를 쓴다.
  - 스키마 검증 실패 시 오류를 넣어 재요청하고, 그래도 실패하면 규칙 기반 기본값으로 대체하는 동작은 유지한다.
- RAG 체인: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | parser`.
  - `format_docs`는 문서마다 `[doc_id] 제목\n본문` 블록을 만든다.
- 각 Agent는 직접 메시지를 조립하지 않고 이 두 체인을 호출한다 (`BaseAgent.structured`, `BaseAgent.knowledge`가 체인을 감싼다).

### 5.5 근거 반영 (`schemas/*`, `agents/*`, `reporting/render.py`, 프롬프트 YAML)

- LLM 출력 스키마에 `citations: list[Citation]`를 추가한다. `Citation`은 `{doc_id, used_for}`이다. 대상은 `RequirementSpec`, `DataMapping`, `DatasetDesign`(실제로는 LLM이 내는 `DesignPlan`), 검증 요약이다. 기본값이 빈 목록이라 기존 테스트 응답도 그대로 통과한다.
- 각 Agent 프롬프트에 "판단에 쓴 검색 문서의 `doc_id`를 `citations`에 적을 것"을 추가한다.
- **인용 검증** (`validate_citations`): LLM이 적은 `doc_id` 중 실제로 검색해 프롬프트에 넣은 문서만 남긴다. 없는 ID는 버리고 로그와 `trace`에 남긴다.
- State에 `citations: Annotated[list[dict], operator.add]`를 추가해 Agent별 인용을 누적한다. 항목은 `{agent, doc_id, source, title, used_for}`.
- 최종 리포트 Markdown에 "참고 근거" 절을 추가한다: Agent, 문서 제목, 출처 종류, 어디에 썼는지를 표로 출력한다. UI 결과 탭의 리포트에도 그대로 보인다.

### 5.6 그래프 (`graph/builder.py`, `agents/supervisor.py`)

- Supervisor는 `Command(goto=...)` 대신 State 업데이트(`route` 포함)만 반환한다.
- `builder.py`에 라우팅 함수 `route_next(state)`를 두고 `add_conditional_edges("supervisor", route_next, {...노드, "end": END})`로 연결한다. 노드, 엣지, 조건 분기가 `builder.py` 한 파일에서 모두 보인다.
- State 정의(`graph/state.py`)에는 `citations`가 추가된다. 재작업 루프, `interrupt` 기반 확인 질문, SQLite 체크포인터는 그대로다.
- `tests/unit/test_supervisor.py`에서 `goto`를 확인하는 3곳을 `route.next` 확인으로 바꾼다.

### 5.7 Tool 호출 (`agents/tool_loop.py` 신규, `tools/catalog.py`, `agents/base.py`)

- `BaseAgent.react()`가 쓰던 `create_agent`를 명시적 LangGraph 하위 그래프로 바꾼다.
  - `agent` 노드: `llm.bind_tools(tools)`로 호출한다.
  - `tools` 노드: `ToolNode(tools)`.
  - `tools_condition`으로 "도구 호출이 있으면 tools, 없으면 종료"를 연결한다.
- 도구 목록에 `search_knowledge(query, source)`를 추가한다. 용어집·설계 가이드·과거 사례를 검색하는 도구로, Agent가 필요할 때 스스로 검색을 호출한다. 기존 도구 5개(`search_catalog`, `describe_table`, `profile_column`, `find_join_path`, `run_readonly_sql`)는 그대로다.
- 도구가 돌려준 검색 문서도 5.5의 인용 후보에 포함한다.
- 단계 한도(`max_react_steps`)와 실패 시 메모로 대체하는 동작은 유지한다.
- 데이터 탐색 Agent와 데이터셋 설계 Agent가 이 루프를 쓴다.

### 5.8 도메인 검증 (`agents/reviewer.py`)

기존 검사(컬럼 존재, SQL 실행, 시점 누수, 타깃 절단, 품질, 요구사항 반영률)는 그대로 둔다. 추가는 하나다.

- **근거 검사** (`check_grounding`): 설계가 인용한 문서가 실제 검색 결과에 있는지, 그리고 타깃·기준 시점 정의에 인용 근거가 하나 이상 있는지 확인한다. 없으면 경고로 표시한다.

README 대응표에 검사 7종을 함수 이름과 함께 나열한다.

### 5.9 실행 스크립트와 제출물

- `scripts/rag_flow.py`(신규): 전처리 → 인덱싱 → 검색 → 근거 포함 응답을 순서대로 실행해 단계별 결과를 출력한다. 5.1~5.4의 구성요소를 그대로 호출한다. LLM 키가 없으면 검색 단계까지만 실행한다.
- README 맨 위에 "평가 항목 → 구현 위치" 표를 둔다. 8개 항목마다 파일·함수와 실행 명령 한 줄.
- `docs/rag.md`, `docs/agent-architecture.md`, `docs/SUBMISSION.md`를 바뀐 구조에 맞게 고친다.
- `make source-zip`(신규): git에 등록된 파일 목록에 DuckDB와 인덱스를 더해 소스만 담은 zip을 만든다 (`storage/`와 `data/db/`는 gitignore 대상이라 `git archive`로는 빠진다). `.venv`, `dist`, 캐시, `__MACOSX`가 들어가지 않는다. DuckDB와 다시 만든 인덱스는 포함한다.

### 5.10 데모 모드 정리

- `demo/recordings/`의 녹화 파일 3개를 삭제한다.
- LLM 기록·재생 코드(`llm/recording.py`)와 그 테스트는 남긴다. 테스트가 임시 폴더에 녹화를 만들어 쓰므로 영향이 없다.
- README와 `SUBMISSION.md`에서 "LLM 없이 버튼 두 번 데모" 안내를 빼고, 실행에 LLM 키가 필요하다고 적는다.
- `Dockerfile`은 녹화 폴더를 복사하는 줄만 고쳐 빌드가 깨지지 않게 한다.

## 6. 작업 순서

1. 전처리 → 인덱싱 → 검색 (5.1~5.3), 인덱스 재구축
2. 체인과 근거 반영 (5.4, 5.5)
3. 그래프와 Tool 호출 (5.6, 5.7)
4. 근거 검사, 데모 정리 (5.8, 5.10)
5. 스크립트, README, 문서, zip (5.9)
6. 실제 LLM으로 샘플 01 실행

각 단계 끝에서 전체 테스트를 돌린다. 시간이 부족하면 `docs/` 수정과 `check_grounding`을 먼저 줄인다.

## 7. 완료 기준

- `pytest -m "not llm"` 통과. 새 테스트: 전처리(정제·분할·메타데이터), Chroma 저장·조회, 인용 검증, Tool 루프, 조건부 엣지 라우팅.
- `ruff check` 통과.
- 실제 LLM으로 샘플 01을 끝까지 실행해 검증 통과 결과와 "참고 근거" 절이 나온다.
- `scripts/rag_flow.py`가 단계별 결과를 출력한다.
- 소스의 실제 호출 경로에 다음 이름이 있다: `Document`, `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`, `Chroma`, `BaseRetriever`, `ChatPromptTemplate`, `PydanticOutputParser`, `RunnablePassthrough`, `StateGraph`, `add_conditional_edges`, `bind_tools`, `ToolNode`.
- 새 zip에 `.venv`와 `dist`가 없다.

## 8. 구현자에게 맡기는 것

- `langchain-chroma`, `langchain-text-splitters`, `langgraph.prebuilt`의 정확한 시그니처는 설치된 버전 문서로 확인한다.
- `clean_text` 규칙의 세부, 청크 ID 형식, "참고 근거" 표의 열 구성.
- `Citation.used_for`를 자유 문장으로 둘지 정해진 값으로 둘지.
- Supervisor의 `handle_change` 경로를 조건부 엣지로 옮기는 세부.

## 9. 위험과 확인되지 않은 것

- **실습 환경의 Azure OpenAI는 로컬 Mac에서 호출할 수 없다** (2026-11-16 확인). 코드는 제공된 `AOAI_*` 변수 이름을 그대로 읽지만, 엔드포인트가 사설망 전용이라 외부에서는 403 "Public access is disabled"로 거절된다. 실습 환경(`atl-dev`) 안에서만 쓸 수 있다.
- 따라서 검증을 둘로 나눈다.
  - 로컬: 실제 LLM 검증(6단계)은 Gemini로, 인덱스 재구축은 Ollama `bge-m3`로 한다. Ollama 설치는 되어 있으나 서버가 꺼져 있어 `ollama serve`가 필요하다.
  - 실습 환경: 사용자가 zip을 올려 `scripts/check_env.py`(신규)를 실행한다. 채팅, JSON 출력, Tool 호출, 임베딩을 한 번씩 호출해 성공 여부를 출력하고, 이어서 인덱스 재구축과 샘플 01 실행까지 확인한다.
- 제출 환경에서는 인덱스가 다른 임베딩(`bge-m3`)으로 만들어져 있으므로 Azure 임베딩으로 다시 만들어야 한다. Docker 진입점에만 있는 자동 재구축(`build_index.py --if-stale`)을 API 서버 시작 시에도 수행하게 한다.
- `langchain-chroma`가 설치된 `chromadb 1.5.9`와 맞지 않을 수 있다. 맞지 않으면 `chromadb` 버전을 조정한다.
- 인용은 LLM이 적는다. 모델이 `citations`를 비워 두면 "참고 근거" 절이 비고 `check_grounding`이 경고를 낸다. 프롬프트로 유도하지만 보장되지는 않는다.
- 채점기가 실제로 무엇을 보는지 모른다. 이 설계가 합격을 보장하지는 않는다.
