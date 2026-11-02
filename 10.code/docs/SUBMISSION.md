# Req2Data Agent 제출용 Docker 이미지 실행 안내

요구사항 정의서 → ML 데이터셋 설계(요구사항 구조화 → 데이터 매핑 → 타깃·피처 설계 → 마트 SQL → 검증)를 자동으로 만드는 LangGraph Multi-Agent 서비스입니다. 이미지 하나에 API·UI·데이터·RAG 인덱스·데모 녹화·테스트가 모두 들어 있어, **인터넷·LLM·GPU 없이** 바로 실행됩니다.

## 1. 제출 파일

| 파일 | 대상 환경 |
| --- | --- |
| `req2data-agent-submit-amd64.tar.gz` | x86_64: 일반 Linux 서버, Windows, Intel Mac |
| `req2data-agent-submit-arm64.tar.gz` | ARM: Apple Silicon Mac, ARM 서버 |

두 파일 모두 이미지 이름은 `req2data-agent:submit`이라 아래 명령은 동일합니다. CPU 확인: `uname -m` (`x86_64` → amd64, `arm64`/`aarch64` → arm64).

필요한 것: Docker 20 이상, 디스크 약 3GB, 포트 8000·8501.

## 2. 이미지 불러오기

```bash
docker load -i req2data-agent-submit-amd64.tar.gz
```

`Loaded image: req2data-agent:submit`가 나오면 됩니다.

## 3. 동작 확인 (브라우저 없이, 1분 안에)

```bash
docker run --rm req2data-agent:submit test       # 테스트 72개 (LLM 불필요)
docker run --rm req2data-agent:submit scenario   # 샘플 01 정의서를 API로 끝까지 실행하고 결과 요약
```

`scenario` 정상 결과(데모 모드, 수 초):

```
행 수: 86,924, 타깃 비율: 1.96%
검증: 통과 (모델링 실익 low)
  - leakage: pass  기준 시점 이후 정보를 쓰는 피처 없음
  - censoring: pass  모든 행이 90일 타깃 기간을 관측 가능 ...
```

## 4. 서비스 실행 (UI)

```bash
docker run --rm -p 8501:8501 -p 8000:8000 req2data-agent:submit
```

- UI: http://localhost:8501
- API 문서(Swagger): http://localhost:8000/docs

**데모 시연 (버튼 두 번)**
1. **새 프로젝트** 화면: 데모 안내와 함께 샘플 `01_repurchase`, 사용자 유형 "실무자"가 미리 선택되어 있습니다. **프로젝트 만들고 실행**을 누릅니다.
2. Agent 진행이 실시간으로 표시되고 확인 질문 2개가 나옵니다. 녹화 당시 답변이 채워져 있으니 **답변하고 계속**을 누릅니다.
3. 결과 탭에서 요약, 요구사항 명세, 데이터 매핑, 설계·SQL, 검증 리포트, 추적표, 진행 로그를 보고 산출물을 내려받습니다.
4. **데이터 카탈로그** 화면에서 카탈로그·용어집·설계 가이드 검색과 테이블 프로파일을 볼 수 있습니다.

데모 모드는 실제 LLM(Qwen2.5 14B)으로 실행한 응답을 녹화해 재생합니다. 재생은 녹화와 같은 입력일 때만 맞습니다. 다른 정의서나 답변을 넣으면 "🎬 데모 녹화와 다른 입력" 표시와 함께 규칙 기반 기본값으로 진행합니다.

## 5. 실제 LLM으로 실행 (선택)

데모가 아닌 실제 실행에는 LLM이 필요합니다. 컨테이너를 실행하는 머신에서 Ollama를 띄우는 방법이 기본입니다.

```bash
ollama pull qwen2.5:14b-instruct && ollama pull bge-m3      # 약 10GB, 메모리 16GB 이상 권장
docker run --rm -p 8501:8501 -p 8000:8000 \
  --add-host host.docker.internal:host-gateway \
  -e DEMO_MODE=false \
  req2data-agent:submit
```

샘플 1건에 약 7~15분 걸립니다 (Apple M4 Pro 기준).

Google Gemini를 쓰려면 다음과 같이 실행합니다. 의미 검색 임베딩(bge-m3)은 Ollama가 없으면 키워드 검색으로 대체됩니다.

```bash
docker run --rm -p 8501:8501 -p 8000:8000 \
  -e DEMO_MODE=false -e LLM_PROVIDER=gemini -e LLM_MODEL=gemini-3.8-flash -e GOOGLE_API_KEY=<키> \
  req2data-agent:submit
```

## 6. 이미지 구성

| 경로 | 내용 |
| --- | --- |
| `/app/src/req2data` | Agent(Supervisor, 요구사항 분석, 데이터 탐색, 데이터셋 설계, 검증, 리포트), LangGraph 그래프, RAG, SQL 빌더, FastAPI 서비스 |
| `/app/ui` | Streamlit UI |
| `/app/prompts` | Agent별 역할 프롬프트 YAML, Few-shot 예시 |
| `/app/data` | Olist 카탈로그(한글 설명·PK/FK·시점 순서)·용어집·설계 가이드·샘플 정의서 3종, 적재된 DuckDB(`data/db/olist.duckdb`) |
| `/app/storage/chroma` | RAG 벡터 인덱스 (bge-m3) |
| `/app/demo/recordings` | 데모 녹화 (LLM 응답·검색 순위·사용자 입력) |
| `/app/tests` | 테스트 (`test` 명령) |
| `/app/docs` | 기획서(`PRD.md`), Agent 구조, RAG, 서비스 설계 문서 |

컨테이너 안을 직접 보려면 `docker run --rm -it req2data-agent:submit bash`를 실행합니다.

## 7. 문제 해결

- **포트가 이미 사용 중**: `-p 18501:8501 -p 18000:8000`처럼 바꾸고 http://localhost:18501 로 접속합니다.
- **Apple Silicon에서 amd64 파일을 불러옴**: 에뮬레이션으로 느리게 동작합니다. arm64 파일을 쓰세요.
- **UI에 "API 서버에 연결할 수 없습니다"**: 컨테이너 시작 직후 몇 초간 나올 수 있습니다. 새로고침하세요.

## 8. 데이터 출처

Olist Brazilian E-Commerce Public Dataset (Kaggle, CC BY-NC-SA 4.0), https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce. 이미지에는 원본 CSV 대신 적재·프로파일된 DuckDB 파일을 넣었습니다.
