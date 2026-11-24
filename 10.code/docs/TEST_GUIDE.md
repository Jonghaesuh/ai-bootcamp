# Req2Data Agent 테스트 가이드

제출 환경에서 Req2Data Agent가 동작하는지 순서대로 확인하는 안내입니다. 실행과 구성 설명은 [SUBMISSION.md](SUBMISSION.md)에 있습니다.

- 1~3단계는 Azure OpenAI 키 없이도 돌아갑니다. 키가 없는 환경에서도 RAG 검색과 단위 테스트는 확인할 수 있습니다.
- 4단계부터는 Azure OpenAI 키가 필요합니다.

| 단계 | 내용 | 키 필요 | 소요 시간 |
| --- | --- | --- | --- |
| 1 | 압축 해제·체크섬·설치 | 아니오 | 1~2분 |
| 2 | 단위 테스트 | 아니오 | 10초 |
| 3 | RAG 흐름 (키워드 검색만) | 아니오 | 수 초 |
| 4 | 실행 환경 점검 | 예 | 수 초 |
| 5 | 샘플 정의서 끝까지 실행 | 예 | 수 분 |
| 6 | 서비스(UI·API) 시나리오 | 예 | 5~10분 |

## 1. 준비와 설치

필요한 것:
- Python 3.12와 [uv](https://docs.astral.sh/uv/)
- (4단계부터) Azure OpenAI 엔드포인트·키, 채팅 배포 `gpt-4.1-mini`, 임베딩 배포 `text-embedding-3-small`

```bash
shasum -a 256 -c SHA256SUMS                   # 제출 파일이 손상되지 않았는지 확인
unzip req2data-agent-source.zip -d req2data-agent
cd req2data-agent
uv sync
```

키는 환경변수로 넘깁니다. `.env.example`을 `.env`로 복사해 채워도 됩니다.

```bash
export AOAI_ENDPOINT="https://<리소스>.openai.azure.com/"
export AOAI_API_KEY="<키>"
# 배포 이름이 기본값과 다를 때만
export AOAI_DEPLOY_GPT41_MINI="<채팅 배포>"
export AOAI_DEPLOY_EMBED_3_SMALL="<임베딩 배포>"
```

zip에는 적재된 DuckDB(`data/db/olist.duckdb`), 컬럼 프로파일(`data/catalog/profiles.json`), RAG 인덱스(`storage/chroma`)가 들어 있습니다. 그래서 원본 CSV를 받지 않아도 됩니다.

## 2. 단위 테스트 (키 불필요)

```bash
uv run pytest -m "not llm" -q
```

기대 결과: `136 passed, 4 skipped, 7 deselected`

- `deselected`는 로컬 Ollama가 필요한 테스트(`-m llm`)라서 제외된 것입니다.
- `skipped`는 실제 Azure 키가 필요한 통합 테스트입니다.

평가 항목별로 나눠 돌리려면 다음 명령을 씁니다.

| 평가 항목 | 명령 | 확인하는 내용 |
| --- | --- | --- |
| 1. 임베딩·Vector DB 인덱싱 | `uv run pytest tests/unit/test_index.py -q` | Chroma 저장·조회, 재구축 때 기존 인덱스 보존, 문서 지문 |
| 2. 전처리·문서 구조화 | `uv run pytest tests/unit/test_preprocess.py -q` | 정제, Markdown 제목 분할, 메타데이터, 중복 제거 |
| 3. Retrieval | `uv run pytest tests/unit/test_rag.py -q` | 하이브리드 검색(벡터+BM25, RRF), 용어집 쿼리 확장, 소스 필터 |
| 4. 근거 기반 응답 | `uv run pytest tests/unit/test_citations.py -q` | 검색하지 않은 문서 인용 제거, 참고 근거 기록 |
| 5. LangChain 체인 | `uv run pytest tests/unit/test_chains.py tests/unit/test_structured.py -q` | 프롬프트 → LLM → 파서, 스키마 실패 재요청·기본값 대체 |
| 6. LangGraph | `uv run pytest tests/unit/test_graph_scenario.py tests/unit/test_supervisor.py -q` | 노드·조건부 엣지, 확인 질문 중단·재개, 재작업 루프, 수정 요청 |
| 7. Tool Calling | `uv run pytest tests/unit/test_tool_loop.py -q` | `bind_tools`로 도구 선택, `ToolNode` 실행, 도구 오류 처리, 단계 한도 |
| 8. 맞춤형 검증 루프 | `uv run pytest tests/unit/test_review_fixes.py tests/unit/test_supervisor.py -q` | 누수·절단·실익 검사, 재작업 판단, 확인 질문 관리, SQL 가드 |

`tests/unit/test_graph_scenario.py::test_repurchase_scenario`는 LLM 응답만 스크립트로 바꾼 전체 흐름입니다. 다음 순서를 한 번에 확인합니다: 요구사항 분석 → 확인 질문 → 매핑 → 설계 v1 누수로 검증 실패 → 피드백 반영 재설계 v2 → 통과 → 기준 시점 확인 → 산출물.

## 3. RAG 흐름 (키 불필요)

```bash
uv run python scripts/rag_flow.py
```

전처리 → 인덱싱 → 검색 → 근거 포함 응답을 단계별로 출력합니다. 키가 없으면 아래처럼 나옵니다.

```text
1. 전처리 (정제 → 분할 → 메타데이터)
   소스별 문서 수: {'catalog': 61, 'glossary': 13, 'guide': 10}
2. 인덱싱 (Chroma)
   임베딩 서버에 연결하지 못해 기존 인덱스를 씁니다 (의미 검색 없이 키워드 검색만).
   storage/chroma: 84개 문서, ...
3. 검색: 재구매 가능성이 낮은 고객을 예측하려면 어떤 데이터가 필요한가?
   0.0082 {'bm25': 1} glossary:재구매
   ...
4. 근거 포함 응답
   LLM 키가 없어 검색까지만 실행했습니다.
```

키가 있으면 다르게 나옵니다.
- 2단계에서 Azure 임베딩으로 인덱스를 한 번 다시 만듭니다 (수 초).
- 3단계 순위에 `{'dense': n, 'bm25': m}`이 함께 나옵니다.
- 4단계에 답변과 `근거 [doc_id]` 목록이 나옵니다.

다른 질문으로 검색하려면 다음 명령을 씁니다.

```bash
uv run python scripts/rag_flow.py --query "배송 지연이 리뷰 점수에 미치는 영향"
uv run python scripts/build_index.py --query "주문 시점" --source catalog
```

## 4. 실행 환경 점검 (키 필요)

```bash
uv run python scripts/check_env.py
```

모든 줄이 `[OK]`여야 합니다: LLM 준비, 지식 베이스 준비, 채팅 호출, JSON 출력, Tool 호출, 임베딩 호출, RAG 인덱스. 마지막 줄은 `결과: 모두 정상`입니다.

키가 없으면 채팅·JSON·Tool·임베딩·인덱스가 `[FAIL]`로 나옵니다. 이는 정상입니다.

## 5. 샘플 정의서 끝까지 실행 (키 필요)

```bash
uv run python scripts/check_env.py --full
```

4단계 점검 뒤에 `data/samples/01_repurchase.md`(첫 구매 고객 재구매 예측)를 Supervisor 그래프로 끝까지 실행합니다. 확인 질문에는 기본값으로 답합니다. 마지막 줄에서 다음을 확인합니다.

```text
[OK]   전체 실행 (01_repurchase.md): 행 수 ..., 타깃 비율 ...%, 검증 통과, 인용 N건, 기본값 대체 없음, 산출물 storage/artifacts/<프로젝트 ID>
결과: 모두 정상
```

- 행 수와 타깃 비율이 0보다 커야 합니다.
- 인용이 1건 이상이면 검색한 지식을 근거로 썼다는 뜻입니다.
- `기본값 대체`에 Agent 이름이 나오면 그 Agent의 LLM 출력이 스키마 검증에 실패해 규칙 기반 기본값을 쓴 것입니다.
- 검증이 `미통과 [...]`이면 재작업 한도를 넘은 것입니다. 실패한 검사 이름이 괄호 안에 나옵니다.

산출물 폴더에는 `summary.md`(마지막에 "참고 근거" 절), `requirement_spec.md`, `data_mapping.md`, `dataset_design.md`, `mart.sql`, `traceability.csv`, `review_report.md`가 생깁니다.

## 6. 서비스 시나리오 (키 필요)

```bash
make run          # 또는 터미널 두 개에서: make api / make ui
```

- UI: http://localhost:8501
- API 문서(Swagger): http://localhost:8000/docs

### 6.1 UI로 확인

1. **새 프로젝트**에서 정의서 입력을 **샘플 선택**으로 두고 `01_repurchase`를 고른 뒤 **프로젝트 만들고 실행**을 누릅니다.
   - 진행 로그에 `supervisor → requirement_analyst` 같은 라우팅이 실시간으로 나옵니다 (항목 6).
2. 확인 질문 카드가 나오면 답하거나 비워 두고 **답변하고 계속**을 누릅니다. 비워 두면 기본값을 씁니다.
   - 답변이 반영되어 요구사항 분석이 다시 실행됩니다.
3. 데이터 탐색과 설계 단계에서 진행 로그에 도구 호출이 보입니다: `search_catalog`, `describe_table`, `profile_column`, `run_readonly_sql` 등 (항목 7).
4. 검증 결과를 확인합니다 (항목 8).
   - **검증 리포트** 탭: 검사 항목별 pass/warn/fail과 근거
   - 미통과 항목이 있으면 진행 로그에 `검증 미통과: 재작업`과 재실행된 Agent가 보입니다.
   - 기준 시점 이후 정보를 쓰는 피처가 있으면 "기준 시점을 바꿀까요?" 확인 질문이 나옵니다.
5. 결과 탭에서 요약과 근거를 확인합니다 (항목 4).
   - **요약**의 "참고 근거" 절에 인용한 문서(`glossary:재구매`, `guide:dataset_design_guide#3` 등)와 용도가 나옵니다.
6. 수정 요청을 해 봅니다.
   - **대화로 수정**에 "타깃 기간을 60일로 바꿔줘"를 입력합니다.
   - 설계 단계부터 다시 실행되고, 설계 버전이 올라갑니다.
7. 검증을 통과했다면 **설계 승인**을 누릅니다.
   - **데이터 카탈로그** 화면에서 소스를 **과거 사례**로 두고 검색하면 방금 승인한 설계가 나옵니다 (장기 기억, 항목 1·3).

### 6.2 API로 확인

```bash
# 카탈로그 검색 (RAG retrieval). 결과마다 doc_id, score, ranks(검색기별 순위)
curl -s "http://localhost:8000/catalog/search?q=재구매&source=glossary&k=3"

# 프로젝트 생성
PID=$(python3 -c 'import json; print(json.dumps({"document": open("data/samples/01_repurchase.md", encoding="utf-8").read()}))' \
  | curl -s -X POST http://localhost:8000/projects -H "Content-Type: application/json" -d @- \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])')

# 실행. Agent 진행 이벤트가 SSE로 흐른다 (trace / node / interrupt / done)
curl -N -X POST "http://localhost:8000/projects/$PID/run"

# 확인 질문(interrupt)에서 멈췄으면 기본값으로 답하고 이어서 실행
curl -N -X POST "http://localhost:8000/projects/$PID/resume" \
  -H "Content-Type: application/json" -d '{"answers": {}}'

# 상태와 산출물
curl -s "http://localhost:8000/projects/$PID" | python3 -m json.tool | head -40
curl -s "http://localhost:8000/projects/$PID/artifacts"
```

## 7. 확인할 만한 동작

| 상황 | 기대 동작 |
| --- | --- |
| 임베딩 키가 없음 | 의미 검색 없이 BM25 키워드 검색으로 결과가 나옴 (3단계) |
| 정의서에 필수 항목이 빠짐 | 빠진 항목을 확인 질문으로 최대 2라운드 묻습니다. 그 뒤에도 답이 없으면 기본값을 가정하고 진행하며, 가정은 요구사항 명세의 제약 조건에 남습니다. |
| 양성 건수가 30건 미만인 설계 | `feasibility` 검사 실패 → 설계 재작업. 통과하지 못하면 승인할 수 없음 |
| 검증 실패가 반복됨 | Agent별 재작업은 최대 2회. 한도를 넘으면 경고를 남기고 산출물을 만들지만 승인은 막힘 |
| 모델이 카탈로그에 없는 테이블·쓰기 SQL을 시도 | 읽기 전용 SQL 가드가 거부하고, 오류를 모델에 돌려줌 |

## 8. 문제 해결

- **`[FAIL] 채팅 호출: ... 403 Public access is disabled`**
  - Azure OpenAI 엔드포인트가 사설망 전용입니다. 허용된 네트워크(과제 실행 환경) 안에서 실행하세요.
- **`[FAIL] 채팅 호출: ... DeploymentNotFound`**
  - 배포 이름이 다릅니다. `AOAI_DEPLOY_GPT41_MINI` 또는 `AZURE_CHAT_DEPLOYMENT`를 지정하세요.
- **`[FAIL] RAG 인덱스`**
  - 임베딩 배포 이름(`AOAI_DEPLOY_EMBED_3_SMALL`)을 확인하세요.
  - 실패해도 서비스는 키워드 검색으로 동작합니다.
- **`[FAIL] Tool 호출: NotImplementedError`**
  - 키가 없어 LLM이 설정되지 않은 상태입니다. 4단계의 환경변수를 확인하세요.
- **포트가 이미 사용 중**
  - `Makefile`의 `api`·`ui` 타깃에서 포트(8000, 8501)를 바꿉니다. 바꾼 API 주소는 UI에 `API_BASE_URL`로 알려 줍니다.
- **처음부터 다시 실행하려면**
  - `storage/checkpoints`, `storage/store`, `storage/artifacts`를 지웁니다. 프로젝트와 실행 기록이 지워집니다.
  - `storage/chroma`(RAG 인덱스)는 남겨 두세요.
