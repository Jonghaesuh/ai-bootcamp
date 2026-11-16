# Req2Data Agent 제출물 실행 안내

요구사항 정의서 → ML 데이터셋 설계(요구사항 구조화 → 데이터 매핑 → 타깃·피처 설계 → 마트 SQL → 검증)를 만드는 LangGraph Multi-Agent 서비스입니다. 제출물은 소스 zip 하나입니다.

평가 항목별 구현 위치는 [README.md](../README.md) 맨 위의 표에 있습니다.

## 1. 제출 파일

`req2data-agent-source.zip`: 소스, 테스트, 프롬프트, 지식 소스(카탈로그·용어집·가이드), 적재된 DuckDB(`data/db/olist.duckdb`), RAG 인덱스(`storage/chroma`)가 들어 있습니다. 가상환경과 Docker 이미지는 들어 있지 않습니다.

필요한 것: Python 3.12, [uv](https://docs.astral.sh/uv/), Azure OpenAI 키.

## 2. 설치와 환경변수

```bash
unzip req2data-agent-source.zip -d req2data-agent && cd req2data-agent
uv sync
```

LLM과 임베딩은 Azure OpenAI를 씁니다. 과제 실행 환경의 `AOAI_*` 환경변수를 그대로 읽습니다.

| 환경변수 | 용도 | 기본값 |
| --- | --- | --- |
| `AOAI_ENDPOINT`, `AOAI_API_KEY` | Azure OpenAI 엔드포인트·키 (`AZURE_OPENAI_ENDPOINT`·`AZURE_OPENAI_API_KEY`도 가능) | 없음 (필수) |
| `AOAI_DEPLOY_GPT41_MINI` | 채팅 배포 이름 | `gpt-4.1-mini` |
| `AOAI_DEPLOY_EMBED_3_SMALL` | 임베딩 배포 이름 | `text-embedding-3-small` |
| `AOAI_API_VERSION` | API 버전 | `2024-10-21` |
| `AZURE_CHAT_DEPLOYMENT` | 다른 채팅 배포를 쓸 때 | |

키가 없으면 LLM 호출이 실패하고 각 Agent가 규칙 기반 기본값으로 진행합니다. 의미 검색은 키워드 검색(BM25)으로 대체됩니다.

## 3. 동작 확인

```bash
uv run pytest -m "not llm" -q                 # 단위 테스트 (LLM 불필요)
uv run python scripts/check_env.py            # 채팅·JSON 출력·Tool 호출·임베딩·인덱스 점검 (수 초)
uv run python scripts/check_env.py --full     # 위 점검 + 샘플 01 정의서를 끝까지 실행 (수 분)
uv run python scripts/rag_flow.py             # RAG 흐름: 전처리 → 인덱싱 → 검색 → 근거 포함 응답
```

`check_env.py`는 단계마다 `[OK]` 또는 `[FAIL]`을 출력합니다. `--full`의 마지막 줄에 행 수, 타깃 비율, 검증 통과 여부, 인용 건수, 산출물 폴더가 나옵니다.

들어 있는 RAG 인덱스는 로컬 임베딩으로 만든 것입니다. 임베딩 설정이 다르면 처음 실행할 때 Azure 임베딩으로 한 번 다시 만듭니다 (문서 84개, 수 초).

## 4. 서비스 실행 (API + UI)

```bash
make run
```

- UI: http://localhost:8501
- API 문서(Swagger): http://localhost:8000/docs

1. **새 프로젝트** 화면에서 샘플 정의서를 고르거나 직접 붙여 넣고 **프로젝트 만들고 실행**을 누릅니다.
2. Agent 진행이 실시간으로 표시되고, 정의서에 빠진 내용이 있으면 확인 질문이 나옵니다. 답하고 **답변하고 계속**을 누릅니다.
3. 결과 탭에서 요약(참고 근거 포함), 요구사항 명세, 데이터 매핑, 설계·SQL, 검증 리포트, 추적표, 진행 로그를 봅니다.
4. **데이터 카탈로그** 화면에서 카탈로그·용어집·설계 가이드 검색과 테이블 프로파일을 볼 수 있습니다.

Docker로 실행하려면 `make up`(Docker Compose) 또는 `make submit`(단일 이미지 빌드)을 씁니다. 자세한 내용은 [service.md](service.md)에 있습니다.

## 5. 구성

| 경로 | 내용 |
| --- | --- |
| `src/req2data/agents` | Supervisor, 요구사항 분석, 데이터 탐색, 데이터셋 설계, 검증, 리포트, 도구 호출 루프, 인용 검증 |
| `src/req2data/graph` | LangGraph 그래프(`StateGraph`, 조건부 엣지)와 State |
| `src/req2data/rag` | 전처리, Chroma 인덱스, 하이브리드 검색기, FK 그래프 |
| `src/req2data/llm` | LLM 제공자, LCEL 체인, 구조화 출력 |
| `src/req2data/tools` | Agent 도구 6개, 읽기 전용 SQL |
| `src/req2data/api`, `ui` | FastAPI(SSE), Streamlit |
| `prompts` | Agent별 역할 프롬프트 YAML, Few-shot 예시 |
| `data` | Olist 카탈로그(한글 설명·PK/FK·시점 순서), 용어집, 설계 가이드, 샘플 정의서 3종, DuckDB |
| `storage/chroma` | RAG 벡터 인덱스 |
| `scripts` | 인덱스 구축, RAG 흐름 실행, 환경 점검, 데이터 적재 |
| `tests` | 단위 테스트 |
| `docs` | Agent 구조, RAG, 서비스 설계 문서 |

## 6. 문제 해결

- **`[FAIL] 채팅 호출: ... 403 Public access is disabled`**: Azure OpenAI 엔드포인트가 사설망 전용입니다. 허용된 네트워크(과제 실행 환경) 안에서 실행하세요.
- **`[FAIL] RAG 인덱스`**: 임베딩 호출이 실패했습니다. 임베딩 배포 이름(`AOAI_DEPLOY_EMBED_3_SMALL`)을 확인하세요. 실패해도 서비스는 키워드 검색으로 동작합니다.
- **포트가 이미 사용 중**: `Makefile`의 `api`·`ui` 타깃에서 포트(8000, 8501)를 바꿉니다.

## 7. 데이터 출처

Olist Brazilian E-Commerce Public Dataset (Kaggle, CC BY-NC-SA 4.0), https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce. 원본 CSV 대신 적재·프로파일된 DuckDB 파일을 넣었습니다.
