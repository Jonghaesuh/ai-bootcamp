# Req2Data Agent GitHub 저장소 정보

제출한 zip과 같은 코드가 GitHub에 있습니다.

| 항목 | 내용 |
| --- | --- |
| 저장소 | https://github.com/Jonghaesuh/ai-bootcamp |
| 제출 버전 | 태그 [`resubmission-2026-11`](https://github.com/Jonghaesuh/ai-bootcamp/tree/resubmission-2026-11/10.code) |
| 프로젝트 위치 | 저장소 안의 `10.code/` 폴더 |
| 작성자 | 서종해 |

## 받기와 실행

```bash
git clone --branch resubmission-2026-11 https://github.com/Jonghaesuh/ai-bootcamp.git
cd ai-bootcamp/10.code
uv sync
export AOAI_ENDPOINT=... AOAI_API_KEY=...     # Azure OpenAI
```

GitHub에는 적재된 DuckDB(`data/db/olist.duckdb`)와 RAG 인덱스(`storage/chroma`)가 없습니다. 용량 때문에 저장소에서 뺐고, 제출 zip에는 들어 있습니다. GitHub에서 받았다면 데이터를 먼저 준비합니다.

1. [Olist 데이터셋](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)의 CSV 9개를 `data/raw/olist/`에 둡니다.
2. `make data`로 DuckDB 적재, 컬럼 프로파일, RAG 인덱스 구축을 한 번에 합니다.

이후 동작 확인은 제출 zip과 같습니다.

```bash
uv run python scripts/check_env.py --full     # LLM·임베딩·Tool 호출 점검 → 샘플 1건 전체 실행
```

자세한 실행 방법은 [docs/SUBMISSION.md](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/docs/SUBMISSION.md)에 있습니다.

## 평가 항목별 구현 위치

| 평가 항목 | 구현 위치 |
| --- | --- |
| 임베딩·Vector DB 인덱싱, 저장·조회 | [`rag/index.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/rag/index.py) `KnowledgeIndex` (`langchain_chroma.Chroma`) |
| 문서 분할·메타데이터·정제 | [`rag/preprocess.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/rag/preprocess.py), [`rag/documents.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/rag/documents.py) `build_documents` |
| Retrieval | [`rag/retriever.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/rag/retriever.py) `HybridRetriever(BaseRetriever)` |
| 검색 지식을 근거와 함께 반영 | [`llm/chains.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/llm/chains.py) `retrieval_chain`, [`agents/citations.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/agents/citations.py), 리포트 "참고 근거" 절 |
| LangChain 체인·LLM 호출 | [`llm/chains.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/llm/chains.py) (`ChatPromptTemplate \| LLM \| PydanticOutputParser`) |
| LangGraph 노드·엣지·상태 | [`graph/builder.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/graph/builder.py) (`StateGraph`, `add_conditional_edges`), [`graph/state.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/graph/state.py) |
| Tool binding·Tool 선택 호출 | [`agents/tool_loop.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/agents/tool_loop.py) (`bind_tools`, `ToolNode`), [`tools/catalog.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/tools/catalog.py) 도구 6개 |
| 도메인 맞춤 검증 | [`agents/reviewer.py`](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/src/req2data/agents/reviewer.py) 검사 7종 (누수, 절단, 반영률, 근거 등) |

경로는 모두 `10.code/src/req2data/` 기준입니다. 전체 표는 [README.md](https://github.com/Jonghaesuh/ai-bootcamp/blob/resubmission-2026-11/10.code/README.md) 맨 위에 있습니다.
