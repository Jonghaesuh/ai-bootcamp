# 표준 LangChain/LangGraph 형태로 재제출 보강 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 평가 코멘트 8개 항목이 표준 LangChain/LangGraph API로 실제 서비스 경로에서 호출되도록 RAG, 체인, 그래프, Tool 호출, 근거 반영을 교체한다.

**Architecture:** RAG 계층(`rag/`)은 LangChain `Document` → text splitter → `langchain_chroma.Chroma` → `BaseRetriever`로 바꾼다. LLM 호출은 LCEL 체인(`llm/chains.py`)으로 통일하고, Agent 출력에 인용(`citations`)을 추가해 코드가 검증한 뒤 State와 리포트에 싣는다. 그래프는 `add_conditional_edges`로 라우팅을 드러내고, 도구 호출은 `bind_tools` + `ToolNode` 하위 그래프로 바꾼다.

**Tech Stack:** Python 3.12, uv, LangChain 1.4 (`langchain-core` 1.7), LangGraph 1.2, `langchain-chroma` 1.1.0, `langchain-text-splitters` 1.1.3, ChromaDB 1.5.9, DuckDB, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-16-standard-langchain-rag-design.md`

## Global Constraints

- 작업 브랜치: `resubmit-langchain-standard` (`main`에서 분기). 커밋은 로컬에만 하고 푸시하지 않는다.
- 커밋 메시지 끝: `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`
- 테스트 명령: `uv run pytest -m "not llm" -q -p no:cacheprovider`. 각 Task 끝에서 전체가 통과해야 한다.
- 린트: `uv run ruff check src tests ui scripts`와 `uv run ruff format --check src tests ui scripts`.
- 새 의존성은 `langchain-chroma`, `langchain-text-splitters` 두 개만. `chromadb` 버전은 바꾸지 않는다 (호환 확인됨).
- 설치된 라이브러리의 정확한 시그니처는 구현 전에 확인한다: `uv run python -c "import inspect, langchain_chroma; print(inspect.signature(langchain_chroma.Chroma.__init__))"` 방식.
- 검색 결과 dict 형식은 유지한다 (`CatalogClient.search_catalog`, API, UI가 사용): `{"doc_id", "source", "table", "column", "title", "text", "score", "ranks"}`.
- 시스템 프롬프트에 `출력 스키마: <스키마 이름>` 문자열이 반드시 들어가야 한다 (테스트용 `ScriptedChatModel`이 이 문자열로 응답을 고른다).
- API 키는 저장소에 넣지 않는다. `.env`는 gitignore 대상이다.
- 데모 녹화 재생은 고려하지 않는다. 프롬프트·스키마·청크를 바꿔도 된다.
- 주석과 문서는 한국어, 기존 코드의 주석 밀도와 명명 방식을 따른다.

## Review Focus

설계가 암시하지만 기본 테스트로는 드러나지 않는 입력들. 각 항목의 테스트는 해당 Task에 들어 있다.

1. **제목(`#`)이 없는 Markdown이나 빈 문서를 사례로 추가** → 오류 없이 1개 청크 또는 0개 청크가 된다 (Task 1).
2. **임베딩 서버가 꺼져 있거나 인덱스가 다른 임베딩으로 만들어져 차원이 다름** → 검색은 BM25로 계속되고 서비스가 뜬다 (Task 3, Task 9).
3. **LLM이 검색되지 않은 `doc_id`를 인용하거나 같은 문서를 중복 인용** → 버리고 `trace`에 남기며 파이프라인은 계속된다 (Task 5).
4. **LLM이 없는 도구 이름을 호출하거나 도구가 예외를 냄** → 오류가 `ToolMessage`로 모델에 돌아가고 단계 한도 안에서 끝난다 (Task 7).
5. **LLM이 JSON 앞뒤에 설명을 붙이거나 코드 블록으로 감쌈** → LCEL 체인으로 바꾼 뒤에도 파싱된다 (Task 4).

## File Structure

| 파일 | 책임 | 변경 |
| --- | --- | --- |
| `src/req2data/rag/preprocess.py` | 정제 → 분할 → 메타데이터 부여 | 신규 |
| `src/req2data/rag/documents.py` | 소스별 `Document` 생성 | 수정 |
| `src/req2data/rag/index.py` | `langchain_chroma.Chroma` 저장·조회 | 수정 |
| `src/req2data/rag/retriever.py` | `BaseRetriever` 하이브리드 검색 | 수정 |
| `src/req2data/rag/recording.py` | 검색 순위 녹화 | 삭제 |
| `src/req2data/rag/knowledge_base.py` | 조립, `ensure_current()` | 수정 |
| `src/req2data/llm/chains.py` | LCEL 체인 | 신규 |
| `src/req2data/llm/structured.py` | 재시도·기본값 대체 (체인 호출) | 수정 |
| `src/req2data/schemas/citation.py` | `Citation`, `GroundedAnswer` | 신규 |
| `src/req2data/agents/citations.py` | 인용 검증·기록 | 신규 |
| `src/req2data/agents/tool_loop.py` | `bind_tools` + `ToolNode` 하위 그래프 | 신규 |
| `src/req2data/tools/catalog.py` | `CatalogRetriever`, `search_knowledge` 도구 | 수정 |
| `src/req2data/graph/builder.py` | `add_conditional_edges` | 수정 |
| `scripts/rag_flow.py`, `scripts/check_env.py` | 실행 확인 스크립트 | 신규 |

---

### Task 1: 전처리 파이프라인과 LangChain `Document`

**Files:**
- Create: `src/req2data/rag/preprocess.py`, `tests/unit/test_preprocess.py`
- Modify: `pyproject.toml` (의존성 2개), `config.yaml` (`rag.chunk_size: 600`, `rag.chunk_overlap: 80`)
- Modify: `src/req2data/rag/documents.py`, `src/req2data/rag/knowledge_base.py`, `src/req2data/rag/__init__.py`
- Modify (필드 이름만): `src/req2data/rag/index.py`, `src/req2data/rag/retriever.py`
- Modify: `tests/unit/test_rag.py:37,47` (`.text` → `.page_content`)

**Interfaces:**
- Produces (`rag/preprocess.py`):
  - `clean_text(text: str) -> str`
  - `split_markdown(text: str, *, chunk_size: int = 600, chunk_overlap: int = 80) -> list[Document]` — 메타데이터에 `h1`, `h2`, `h3`
  - `flatten_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]` — `None` 제거, 리스트는 `", "`로 연결
  - `enrich_metadata(doc: Document, *, doc_id: str, source: str, kind: str, chunk_index: int = 0, **extra: Any) -> Document` — `Document.id = doc_id`, `content_hash` 추가
  - `dedupe(docs: list[Document]) -> list[Document]`
  - `build_documents(spec: CatalogSpec, glossary: list[GlossaryTerm], profiles: dict, guides_dir: Path, cases_dir: Path, *, chunk_size: int = 600, chunk_overlap: int = 80) -> list[Document]`
- Produces (`rag/documents.py`): `catalog_documents`, `glossary_documents`, `markdown_chunks(path, source, *, chunk_size=600, chunk_overlap=80)`, `directory_chunks`가 모두 `list[Document]`를 반환. `RagDocument`는 삭제.
- 문서 ID 규칙: `table:{t}`, `column:{t}.{c}`, `glossary:{term}`, `{source}:{파일 stem}#{1부터 순번}`.
- 모든 `Document.metadata`의 키: `source`, `kind`, `title`, `path`, `chunk_index`, `table`, `column`, `content_hash` (컬럼 문서는 `type`, `is_pk`, `fk`, `is_time`, `time_rank`, `available_at` 추가).

- [ ] **Step 1: 브랜치와 의존성**

```bash
git checkout -b resubmit-langchain-standard
uv add langchain-chroma langchain-text-splitters
```

Expected: `langchain-chroma==1.1.0`, `langchain-text-splitters==1.1.3` 설치, `chromadb` 버전 변동 없음.

- [ ] **Step 2: 실패하는 테스트 작성 (`tests/unit/test_preprocess.py`)**

```python
def test_clean_text_normalizes_and_strips():
    assert clean_text("가  \nb\x00\n\n\n\nc") == "가\nb\n\nc"

def test_split_markdown_by_heading_then_size():
    body = "문장입니다. " * 250  # 약 1,750자
    docs = split_markdown(f"# 제목\n## 가\n{body}\n## 나\n짧은 본문", chunk_size=600, chunk_overlap=80)
    assert len(docs) >= 4 and all(len(d.page_content) <= 600 for d in docs)
    assert {d.metadata["h1"] for d in docs} == {"제목"}
    assert {d.metadata["h2"] for d in docs} == {"가", "나"}

def test_markdown_without_headings_or_empty(tmp_path):   # Review Focus 1
    plain = tmp_path / "plain.md"; plain.write_text("제목 없는 메모", encoding="utf-8")
    empty = tmp_path / "empty.md"; empty.write_text("  \n\n", encoding="utf-8")
    assert [d.id for d in markdown_chunks(plain, "case")] == ["case:plain#1"]
    assert markdown_chunks(empty, "case") == []

def test_enrich_metadata_is_chroma_safe_and_unique(kb):
    docs = kb.documents()
    assert len({d.id for d in docs}) == len(docs)
    for d in docs:
        assert {"source", "kind", "title", "chunk_index", "content_hash"} <= set(d.metadata)
        assert all(isinstance(v, str | int | float | bool) for v in d.metadata.values())

def test_dedupe_drops_empty_and_duplicate_content():
    a = Document(id="a", page_content="같음"); b = Document(id="b", page_content="같음")
    assert [d.id for d in dedupe([a, b, Document(id="c", page_content=" ")])] == ["a"]

def test_build_documents_covers_all_sources(kb):
    assert {d.metadata["source"] for d in kb.documents()} == {"catalog", "glossary", "guide"}
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/unit/test_preprocess.py -q -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError: req2data.rag.preprocess`)

- [ ] **Step 4: `rag/preprocess.py` 구현**

- `clean_text`: NFC 정규화 → `\n`·`\t` 외 제어 문자 제거 → 줄 끝 공백 제거 → 연속 줄바꿈 3개 이상을 2개로 → 양끝 공백 제거.
- `split_markdown`: `MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])` 다음 `RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=...).split_documents(...)`.
- `content_hash`: `sha1(page_content)`의 앞 12자. `dedupe`는 이 값이 같은 문서 중 첫 번째만 남긴다.
- `build_documents`는 `load → clean → split → enrich → dedupe` 순서가 함수 본문에서 그대로 읽히게 쓴다.

- [ ] **Step 5: `rag/documents.py`를 `Document` 반환으로 변경**

Markdown 청크 본문 형식: `[{h1 또는 파일 stem}] {h2}{" / " + h3 (있을 때)}\n{본문}`. 제목이 없으면 `[{stem}]\n{본문}`. `metadata["title"]`은 `"{h1} / {h2}"`.

- [ ] **Step 6: `KnowledgeBase.documents()`가 `build_documents`를 호출하게 하고, `index.py`·`retriever.py`의 필드 접근을 바꾼다**

`d.doc_id` → `d.id`, `d.text` → `d.page_content`, `d.source` → `d.metadata["source"]`, `d.chroma_metadata()` → `d.metadata`. `chunk_size`·`chunk_overlap`은 `rag_policy`에서 읽는다. 동작 변경은 Task 2, 3에서 한다.

- [ ] **Step 7: 전체 테스트 통과 확인 후 커밋**

Run: `uv run pytest -m "not llm" -q -p no:cacheprovider` → PASS

```bash
git add -A && git commit -m "Preprocess knowledge sources into LangChain Documents with cleaning, splitting and metadata"
```

---

### Task 2: `langchain_chroma.Chroma` 인덱싱

**Files:**
- Modify: `src/req2data/rag/index.py`
- Test: `tests/unit/test_index.py` (신규)

**Interfaces:**
- Consumes: Task 1의 `Document` (`id`, `page_content`, Chroma에 안전한 `metadata`)
- Produces (`KnowledgeIndex`, 생성자 시그니처는 그대로):
  - `store: Chroma`, `persist_dir: Path | None` 속성
  - `rebuild(docs: list[Document]) -> int`, `upsert(docs: list[Document]) -> int`
  - `similarity_search(query: str, k: int, sources: list[str] | None = None) -> list[tuple[Document, float]]`
  - `dense(query: str, k: int, sources: list[str] | None = None) -> list[str]` — 문서 ID 목록
  - `documents() -> list[Document]`, `count() -> int`, `built_with() -> str`, `is_current() -> bool`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
DOCS = [
    Document(id="glossary:재구매", page_content="[용어] 재구매 두 번째 주문", metadata={"source": "glossary", "kind": "term"}),
    Document(id="guide:g#1", page_content="[가이드] 누수 방지 기준 시점", metadata={"source": "guide", "kind": "section"}),
    Document(id="table:orders", page_content="[테이블] orders 주문", metadata={"source": "catalog", "kind": "table"}),
]

def test_add_and_similarity_search_roundtrip(tmp_path):
    index = KnowledgeIndex(HashEmbeddings(), tmp_path, embedding_name="hash")
    assert isinstance(index.store, Chroma) and index.rebuild(DOCS) == 3
    doc, _score = index.similarity_search("재구매 주문", k=1)[0]
    assert doc.id == "glossary:재구매" and doc.metadata["kind"] == "term"

def test_persisted_index_reopens(tmp_path):
    KnowledgeIndex(HashEmbeddings(), tmp_path, embedding_name="hash").rebuild(DOCS)
    again = KnowledgeIndex(HashEmbeddings(), tmp_path, embedding_name="hash")
    assert again.count() == 3 and again.is_current()
    assert not KnowledgeIndex(HashEmbeddings(), tmp_path, embedding_name="other").is_current()

def test_source_filter(tmp_path):
    index = KnowledgeIndex(HashEmbeddings(), tmp_path); index.rebuild(DOCS)
    assert index.dense("주문", k=3, sources=["catalog"]) == ["table:orders"]

def test_rebuild_keeps_old_index_when_embeddings_down(tmp_path):
    KnowledgeIndex(HashEmbeddings(), tmp_path, embedding_name="hash").rebuild(DOCS)
    broken = KnowledgeIndex(UnavailableEmbeddings(), tmp_path, embedding_name="hash")
    with pytest.raises(RuntimeError):
        broken.rebuild(DOCS[:1])
    assert broken.count() == 3

def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="중복 문서 ID"):
        KnowledgeIndex(HashEmbeddings(), tmp_path).rebuild([DOCS[0], DOCS[0]])
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/unit/test_index.py -q -p no:cacheprovider` → FAIL (`store` 속성 없음)

- [ ] **Step 3: `KnowledgeIndex`를 `Chroma`로 구현**

- 생성: `Chroma(collection_name=..., embedding_function=embeddings, persist_directory=..., collection_metadata={"hnsw:space": "cosine", "embedding": name}, client_settings=ChromaSettings(anonymized_telemetry=False))`. `persist_dir`이 없으면 메모리 컬렉션.
- 저장: `store.add_documents(docs, ids=[d.id for d in docs])`를 64개씩.
- `rebuild`: 중복 ID 검사 → `embeddings.embed_query("ping")`로 임베딩 서버 확인 → 컬렉션 초기화(`reset_collection()`) → 저장. 확인 호출이 실패하면 기존 컬렉션을 건드리지 않는다.
- 조회: `store.similarity_search_with_score(query, k=min(k, n), filter={"source": {"$in": sources}} if sources else None)`.
- `documents()`: `store.get(include=["documents", "metadatas"])`를 `Document`로 변환하고 캐시한다 (쓰기 후 캐시 무효화).

- [ ] **Step 4: 통과 확인 후 커밋**

Run: 전체 테스트 → PASS

```bash
git add -A && git commit -m "Store and query the knowledge index through langchain_chroma.Chroma"
```

---

### Task 3: `BaseRetriever` 하이브리드 검색

**Files:**
- Modify: `src/req2data/rag/retriever.py`, `src/req2data/rag/knowledge_base.py`, `src/req2data/rag/__init__.py`, `src/req2data/tools/catalog.py`
- Delete: `src/req2data/rag/recording.py`
- Modify: `tests/unit/test_rag.py`, `tests/unit/test_demo.py` (`test_dense_ranking_replays_without_embedding_server` 삭제)

**Interfaces:**
- Consumes: Task 2의 `KnowledgeIndex.dense`, `KnowledgeIndex.documents`, `KnowledgeIndex.persist_dir`
- Produces (`rag/retriever.py`):
  - `class HybridRetriever(BaseRetriever)` — 필드 `index: KnowledgeIndex`, `glossary: list[GlossaryTerm] = []`, `top_k: int = 8`, `rrf_k: int = 60`, `weights: dict[str, float] = {"dense": 0.5, "bm25": 0.5}`, `candidate_k: int = 30`, `sources: list[str] | None = None`. 키워드 인자로만 생성한다.
  - `_get_relevant_documents(query, *, run_manager) -> list[Document]` — 반환 문서의 `metadata`에 `score: float`, `ranks: dict[str, int]` 추가 (원본 문서는 변경하지 않고 복사본을 만든다)
  - `with_scope(k: int | None = None, sources: list[str] | None = None) -> HybridRetriever`
  - `expand_query(query: str) -> ExpandedQuery` (그대로)
  - `hit_dict(doc: Document) -> dict[str, Any]` — Global Constraints의 검색 결과 dict
- Produces (`tools/catalog.py`):
  - `hit_to_document(hit: dict[str, Any]) -> Document` — `id=hit["doc_id"]`, `page_content=hit["text"]`, 메타데이터 `source`, `table`, `column`, `title`, `score`
  - `class CatalogRetriever(BaseRetriever)` — 필드 `client: Any`, `source: str = "catalog"`, `k: int = 8`. `client.search_catalog(query, k=self.k, source=self.source)` 결과를 `hit_to_document`로 변환
- `SearchHit`, `DenseRecorder`는 삭제. `KnowledgeBase.search()`는 `[hit_dict(d) for d in self.retriever.with_scope(k, sources).invoke(query)]`.

- [ ] **Step 1: 실패하는 테스트 작성 (`tests/unit/test_rag.py`에 추가)**

```python
def test_hybrid_retriever_is_langchain_retriever(kb):
    assert isinstance(kb.retriever, BaseRetriever)
    docs = kb.retriever.invoke("재구매 가능성이 낮은 고객")
    assert docs and all(isinstance(d, Document) for d in docs)
    assert docs[0].metadata["score"] > 0 and set(docs[0].metadata["ranks"]) <= {"dense", "bm25"}

def test_retriever_composes_in_lcel(kb):
    chain = kb.retriever.with_scope(k=2, sources=["glossary"]) | (lambda docs: [d.id for d in docs])
    ids = chain.invoke("재구매")
    assert 1 <= len(ids) <= 2 and all(i.startswith("glossary:") for i in ids)

def test_dense_failure_falls_back_to_bm25(kb, monkeypatch):   # Review Focus 2
    def down(*a, **k): raise RuntimeError("임베딩 서버 없음")
    monkeypatch.setattr(kb.index, "dense", down)
    docs = kb.retriever.invoke("재구매 고객")
    assert docs and all(set(d.metadata["ranks"]) == {"bm25"} for d in docs)

def test_dimension_mismatch_falls_back_to_bm25(kb):   # Review Focus 2: 다른 임베딩으로 만든 인덱스
    class Tiny(HashEmbeddings):
        def _embed(self, text): return [1.0] * 8
    stale = KnowledgeIndex(Tiny(), kb.index.persist_dir, embedding_name="tiny")
    docs = HybridRetriever(index=stale, glossary=kb.glossary).invoke("재구매 고객")
    assert docs and all("dense" not in d.metadata["ranks"] for d in docs)

def test_catalog_retriever_wraps_client(client):
    docs = CatalogRetriever(client=client, source="glossary", k=2).invoke("재구매")
    assert docs and all(d.id.startswith("glossary:") and d.page_content for d in docs)
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/unit/test_rag.py -q -p no:cacheprovider` → 새 테스트 FAIL

- [ ] **Step 3: 구현** — 검색 흐름(쿼리 확장 → dense + BM25 → RRF → 소스 필터)은 지금 로직을 `_get_relevant_documents`로 옮긴다. BM25 캐시는 `PrivateAttr`로 둔다. dense 예외는 지금처럼 잡아 경고를 남기고 빈 순위로 처리한다.

- [ ] **Step 4: 기존 테스트 정리** — `test_source_filter_and_rrf` 등 `SearchHit`를 쓰는 곳을 새 반환 형식에 맞춘다. `rag/recording.py`와 `knowledge_base.from_settings`의 녹화 연결, `test_demo.py`의 해당 테스트를 지운다.

- [ ] **Step 5: 통과 확인 후 커밋**

```bash
git add -A && git commit -m "Expose hybrid search as a LangChain BaseRetriever"
```

---

### Task 4: LCEL 체인으로 LLM 호출 통일

**Files:**
- Create: `src/req2data/llm/chains.py`, `src/req2data/schemas/citation.py`, `tests/unit/test_chains.py`
- Modify: `src/req2data/llm/structured.py`, `src/req2data/llm/__init__.py`, `src/req2data/schemas/__init__.py`

**Interfaces:**
- Consumes: Task 3의 `CatalogRetriever`
- Produces (`schemas/citation.py`):
  - `class Citation(LLMOutput)` — `doc_id: str`, `used_for: str = ""`
  - `class GroundedAnswer(LLMOutput)` — `answer: str`, `citations: list[Citation] = []`
- Produces (`llm/chains.py`):
  - `format_instructions(schema: type[BaseModel]) -> str` — `"## 출력 형식\n출력 스키마: {이름}\n" + PydanticOutputParser(pydantic_object=schema).get_format_instructions()`
  - `STRUCTURED_PROMPT: ChatPromptTemplate` — `[("system", "{system}\n\n{format_instructions}"), MessagesPlaceholder("messages")]`
  - `generation_chain(llm: BaseChatModel, schema: type[T]) -> Runnable[dict, str]` — `STRUCTURED_PROMPT.partial(...) | llm.bind(**native_schema_kwargs(llm, schema)) | StrOutputParser()`
  - `parsing_chain(schema: type[T]) -> Runnable[str, T]` — `RunnableLambda(extract_json) | PydanticOutputParser(pydantic_object=schema)`
  - `structured_chain(llm, schema) -> Runnable[dict, T]` — `generation_chain | parsing_chain`
  - `format_docs(docs: list[Document]) -> str` — 문서마다 `[{d.id}] {title}\n{page_content}`, 빈 줄로 구분
  - `retrieval_chain(retriever: BaseRetriever) -> Runnable[str, dict]` — `retriever | RunnableParallel(docs=RunnablePassthrough(), context=RunnableLambda(format_docs))`
  - `rag_chain(retriever: BaseRetriever, llm: BaseChatModel, schema: type[T], instruction: str) -> Runnable[str, T]` — `{"context": retriever | format_docs, "question": RunnablePassthrough()} | 프롬프트 | llm | StrOutputParser() | parsing_chain(schema)`
- `invoke_structured(llm, messages, schema, *, retries=1, fallback)`의 시그니처와 반환(`StructuredResult`)은 그대로. `extract_json`, `native_schema_kwargs`는 `chains.py`로 옮기고 `structured.py`와 `llm/__init__.py`에서 다시 내보낸다. `schema_instruction`은 `format_instructions`로 대체한다.
- 서비스 경로에서는 Agent가 `retrieval_chain`과 `generation_chain`·`parsing_chain`을 쓴다. `rag_chain`은 이 둘을 한 체인으로 묶은 것으로 `scripts/rag_flow.py`(Task 9)가 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_structured_chain_is_lcel_and_parses(llm):
    llm.script("RouteDecision", {"next": "end", "reason": "완료"})
    chain = structured_chain(llm, RouteDecision)
    assert isinstance(chain, RunnableSequence)
    out = chain.invoke({"system": "역할", "messages": [HumanMessage(content="과제")]})
    assert out == RouteDecision(next="end", reason="완료")
    system, human = llm.calls[0]["messages"]
    assert system.content.startswith("역할\n\n## 출력 형식\n출력 스키마: RouteDecision")
    assert human.content == "과제"

def test_prose_and_fenced_json_still_parse(llm):   # Review Focus 5
    llm.script("RouteDecision", '설명입니다.\n```json\n{"next": "end", "reason": "r"}\n```\n끝')
    assert structured_chain(llm, RouteDecision).invoke({"system": "", "messages": []}).next == "end"

def test_retrieval_chain_returns_docs_and_context(client):
    out = retrieval_chain(CatalogRetriever(client=client, source="glossary", k=2)).invoke("재구매")
    assert out["docs"] and f"[{out['docs'][0].id}]" in out["context"]

def test_rag_chain_grounds_answer_in_retrieved_context(llm, client):
    llm.script("GroundedAnswer", {"answer": "customer_unique_id 기준", "citations": []})
    chain = rag_chain(CatalogRetriever(client=client, source="glossary", k=2), llm, GroundedAnswer, "문맥만 근거로 답하세요.")
    assert chain.invoke("재구매는 어떻게 정의하나").answer == "customer_unique_id 기준"
    assert "[glossary:" in llm.calls[0]["messages"][-1].content
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/unit/test_chains.py -q -p no:cacheprovider` → FAIL

- [ ] **Step 3: `schemas/citation.py`, `chains.py` 구현, `invoke_structured`가 체인을 호출하도록 변경**

재시도 루프는 `generation_chain`과 `parsing_chain`을 따로 호출한다 (검증 실패 시 원문을 `AIMessage`로 대화에 넣어야 하므로). 입력은 `{"system": 첫 SystemMessage의 내용 또는 "", "messages": 나머지 메시지 + 재시도 대화}`. 잡는 예외에 `OutputParserException`을 추가한다. `DemoReplayMiss`, 호출 실패 재시도, 기본값 대체는 그대로다.

- [ ] **Step 4: 통과 확인** — `tests/unit/test_structured.py`는 수정 없이 통과해야 한다 (`schema_instruction`을 직접 import하는 테스트가 있으면 `format_instructions`로 바꾼다).

- [ ] **Step 5: 커밋**

```bash
git add -A && git commit -m "Route LLM calls through LCEL chains with ChatPromptTemplate and PydanticOutputParser"
```

---

### Task 5: 인용 근거 수집·검증·리포트

**Files:**
- Create: `src/req2data/agents/citations.py`, `tests/unit/test_citations.py`
- Modify: `src/req2data/schemas/{requirement,mapping,design,review}.py`
- Modify: `src/req2data/graph/state.py`, `src/req2data/agents/{base,requirement,explorer,designer,reviewer,report}.py`
- Modify: `src/req2data/reporting/render.py`, `prompts/{requirement_analyst,data_explorer,dataset_designer,reviewer}.yaml`
- Modify: `tests/unit/test_graph_scenario.py`

**Interfaces:**
- Consumes: Task 3의 `CatalogRetriever`, `hit_to_document`; Task 4의 `retrieval_chain`, `Citation`
- Produces:
  - 스키마 필드 `citations: list[Citation] = Field(default_factory=list, description="판단 근거로 쓴 검색 문서. doc_id는 프롬프트에 [doc_id]로 표시된 값 그대로")` — `RequirementSpec`, `DataMapping`, `DesignPlan`, `DatasetDesign`, `ReviewNarrative`, `ReviewReport`
  - `agents/citations.py`:
    - `validate_citations(citations: list[Citation], retrieved: list[Document]) -> tuple[list[Citation], list[str]]` — (검색된 문서만 남긴 목록, 버린 `doc_id` 목록). 같은 `doc_id`는 첫 번째만 남긴다.
    - `citation_records(agent: str, kept: list[Citation], retrieved: list[Document]) -> list[dict]` — `{"agent", "doc_id", "source", "title", "used_for"}`
  - `BaseAgent.knowledge(query: str, source: str, k: int = 3, title: str = "", into: list[Document] | None = None) -> str` — 내부에서 `retrieval_chain(CatalogRetriever(...))` 호출, 검색 문서를 `into`에 추가
  - `BaseAgent.cite(result_value, retrieved: list[Document]) -> tuple[list[dict], list[str]]` — 검증 후 `result_value.citations`를 교체하고 (State용 기록, 버린 ID)를 반환
  - `Req2DataState.citations: Annotated[list[dict], operator.add]`
  - `render.citations_md(citations: list[dict]) -> str` — `## 참고 근거` 절

- [ ] **Step 1: 실패하는 테스트 작성**

```python
RETRIEVED = [Document(id="glossary:재구매", page_content="...", metadata={"source": "glossary", "title": "재구매"})]

def test_validate_citations_drops_unknown_and_duplicate():   # Review Focus 3
    cites = [Citation(doc_id="glossary:재구매", used_for="타깃 정의"),
             Citation(doc_id="glossary:재구매"), Citation(doc_id="glossary:없는용어")]
    kept, rejected = validate_citations(cites, RETRIEVED)
    assert [c.doc_id for c in kept] == ["glossary:재구매"] and rejected == ["glossary:없는용어"]

def test_citation_records_carry_source_and_title():
    kept = [Citation(doc_id="glossary:재구매", used_for="타깃 정의")]
    assert citation_records("requirement_analyst", kept, RETRIEVED) == [
        {"agent": "requirement_analyst", "doc_id": "glossary:재구매", "source": "glossary",
         "title": "재구매", "used_for": "타깃 정의"}]

def test_requirement_analyst_records_validated_citations(deps, llm, client):
    deps = replace(deps, catalog=client)
    llm.script("RequirementSpec", {"business_goal": "재구매 예측", "citations": [
        {"doc_id": "glossary:재구매", "used_for": "타깃 정의"}, {"doc_id": "glossary:없는용어"}]})
    out = RequirementAnalyst(deps)(new_project_state("p1", "첫 구매 고객의 재구매 가능성을 예측한다."))
    assert [c["doc_id"] for c in out["citations"]] == ["glossary:재구매"]
    assert out["trace"][0]["rejected_citations"] == ["glossary:없는용어"]
    assert out["requirement_spec"]["citations"] == [{"doc_id": "glossary:재구매", "used_for": "타깃 정의"}]

def test_citations_md_table_and_empty():
    md = citations_md([{"agent": "requirement_analyst", "doc_id": "glossary:재구매",
                        "source": "glossary", "title": "재구매", "used_for": "타깃 정의"}])
    assert "## 참고 근거" in md and "| requirement_analyst | 재구매 | glossary | 타깃 정의 |" in md
    assert "인용된 문서가 없습니다" in citations_md([])
```

`tests/unit/test_graph_scenario.py`의 전체 실행 테스트에 추가: `assert "## 참고 근거" in Path(state["artifacts"]["summary.md"]).read_text(encoding="utf-8")`.

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/unit/test_citations.py -q -p no:cacheprovider` → FAIL

- [ ] **Step 3: 스키마 필드, `agents/citations.py`, State 구현**

- [ ] **Step 4: Agent 4개에 적용**

각 Agent의 `__call__`에서 `retrieved: list[Document] = []`를 만들고 `knowledge(..., into=retrieved)`로 채운 뒤, 구조화 출력 결과에 `self.cite(result.value, retrieved)`를 적용한다. 반환 dict에 `"citations": 기록`을 넣고 `trace` 이벤트에 `citations=<건수>`, `rejected_citations=<버린 ID 목록>`을 넣는다.

- 데이터 탐색 Agent는 `prefetch`의 `search_hits`와 `glossary` 항목을 `hit_to_document`로 바꿔 `retrieved`에 넣는다.
- 데이터셋 설계 Agent는 `DesignPlan.citations`를 `DatasetDesign.citations`로 옮긴다 (`failed_design` 포함).
- 검증 Agent는 `ReviewNarrative.citations`를 `ReviewReport.citations`로 옮긴다.

- [ ] **Step 5: 프롬프트와 리포트**

- 프롬프트 YAML 4개의 `responsibilities`에 추가: `판단에 쓴 검색 문서의 doc_id를 citations에 적는다. 프롬프트에 [doc_id]로 표시된 값만 쓴다.`
- `citations_md`: 표 열은 `Agent | 문서 | 출처 | 사용처`. 같은 `(agent, doc_id)`는 마지막 것만 남긴다 (재작업으로 누적된 중복 제거).
- `ReportNode`: `summary.md` 끝에 `citations_md(state.get("citations") or [])`를 붙인다.

- [ ] **Step 6: 통과 확인 후 커밋** — `tests/unit/test_prompts.py`의 Few-shot 스키마 검증이 통과하는지 확인한다.

```bash
git add -A && git commit -m "Carry validated citations from retrieval through agent outputs into the report"
```

---

### Task 6: 조건부 엣지로 그래프 라우팅 명시

**Files:**
- Modify: `src/req2data/graph/builder.py`, `src/req2data/agents/supervisor.py`
- Modify: `tests/unit/test_supervisor.py:47,59,71`
- Test: `tests/unit/test_graph_scenario.py`

**Interfaces:**
- Produces:
  - `Supervisor.__call__(state) -> dict`, `Supervisor.handle_change(state) -> dict` — 반환 dict의 `route["next"]`가 다음 노드 이름 또는 `"end"`
  - `graph/builder.py`: `route_next(state: Req2DataState) -> str` — `state["route"]["next"]`
  - `build_graph(deps, checkpointer=None) -> CompiledStateGraph` (시그니처 그대로)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_graph_declares_nodes_and_conditional_edges(deps):
    g = build_graph(deps).get_graph()
    workers = {"requirement_analyst", "data_explorer", "dataset_designer", "reviewer", "report", "ask_user"}
    assert workers | {"supervisor"} <= set(g.nodes)
    edges = {(e.source, e.target, e.conditional) for e in g.edges}
    assert all(("supervisor", w, True) in edges and (w, "supervisor", False) in edges for w in workers)
    assert ("supervisor", "__end__", True) in edges and ("__start__", "supervisor", False) in edges

def test_route_next_reads_route_from_state():
    assert route_next({"route": {"next": "reviewer", "reason": ""}}) == "reviewer"
```

`test_supervisor.py`의 `cmd.goto == X` 3곳은 `out["route"]["next"] == X`로 바꾼다.

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/unit/test_graph_scenario.py tests/unit/test_supervisor.py -q -p no:cacheprovider` → FAIL

- [ ] **Step 3: 구현**

`builder.py`:

```python
g.add_node("supervisor", Supervisor(deps))
g.add_conditional_edges("supervisor", route_next, {**{name: name for name in workers}, "end": END})
```

Supervisor의 두 반환 지점은 `Command` 대신 State 업데이트 dict를 반환한다. `handle_change`는 `route`를 `RouteDecision(next=scope.restart_from, reason=...)`으로 채운다. `ask_user`의 `interrupt`와 재개(`Command(resume=...)`)는 그대로다.

- [ ] **Step 4: 통과 확인 후 커밋**

```bash
git add -A && git commit -m "Declare supervisor routing with add_conditional_edges"
```

---

### Task 7: `bind_tools` + `ToolNode` 도구 호출 루프

**Files:**
- Create: `src/req2data/agents/tool_loop.py`, `tests/unit/test_tool_loop.py`
- Modify: `src/req2data/agents/base.py`, `src/req2data/tools/catalog.py`, `src/req2data/agents/{explorer,designer}.py`
- Modify: `tests/fixtures/scripted_llm.py`

**Interfaces:**
- Consumes: Task 3의 `hit_to_document`; Task 5의 `into` 수집 방식
- Produces:
  - `agents/tool_loop.py`:
    - `build_tool_loop(llm: BaseChatModel, tools: list[BaseTool], system: str) -> CompiledStateGraph` — `StateGraph(MessagesState)`, 노드 `agent`·`tools`
    - `run_tool_loop(llm, tools, system: str, task: str, *, max_steps: int) -> list[BaseMessage]`
    - `retrieved_from_tools(messages: list[BaseMessage]) -> list[Document]` — `search_catalog`·`search_knowledge`의 `ToolMessage` 내용(JSON 목록)을 `hit_to_document`로 변환. JSON이 아니거나 오류 응답이면 건너뛴다.
  - `tools/catalog.py`: 도구 `search_knowledge(query: str, source: Literal["glossary", "guide", "case"]) -> str`. `build_catalog_tools`는 6개를 이 순서로 반환: `search_catalog`, `search_knowledge`, `describe_table`, `profile_column`, `find_join_path`, `run_readonly_sql`
  - `BaseAgent.react(system: str, task: str, tools: list[BaseTool], into: list[Document] | None = None) -> str`
  - `ScriptedChatModel.script_react(*messages: AIMessage)` — 스키마가 없는 호출에서 순서대로 반환, 다 쓰면 기존 기본 응답. `bound_tools: list[str]` 속성에 `bind_tools`로 받은 도구 이름을 기록.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def call(name, args, i): return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])

def test_agent_selects_and_calls_tool(llm, catalog):
    tools = build_catalog_tools(catalog)
    llm.script_react(call("describe_table", {"table": "orders"}, 1))
    msgs = run_tool_loop(llm, tools, "시스템", "orders 구조 확인", max_steps=4)
    assert llm.bound_tools == [t.name for t in tools]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_msgs] == ["describe_table"] and "order_id" in tool_msgs[0].content
    assert isinstance(msgs[-1], AIMessage) and not msgs[-1].tool_calls

def test_tool_loop_graph_has_agent_and_tool_nodes(llm, catalog):
    g = build_tool_loop(llm, build_catalog_tools(catalog), "시스템").get_graph()
    assert {"agent", "tools"} <= set(g.nodes)

def test_unknown_tool_and_tool_error_return_to_model(llm, catalog):   # Review Focus 4
    llm.script_react(call("없는_도구", {}, 1), call("describe_table", {"table": "없는_테이블"}, 2))
    msgs = run_tool_loop(llm, build_catalog_tools(catalog), "시스템", "과제", max_steps=4)
    assert len([m for m in msgs if isinstance(m, ToolMessage)]) == 2
    assert isinstance(msgs[-1], AIMessage) and not msgs[-1].tool_calls

def test_loop_stops_at_step_limit(llm, catalog):
    llm.script_react(*[call("describe_table", {"table": "orders"}, i) for i in range(10)])
    msgs = run_tool_loop(llm, build_catalog_tools(catalog), "시스템", "과제", max_steps=2)
    assert len([m for m in msgs if isinstance(m, ToolMessage)]) <= 2

def test_search_knowledge_tool_and_retrieved_docs(llm, client):
    tools = build_catalog_tools(client)
    assert [t.name for t in tools] == ["search_catalog", "search_knowledge", "describe_table",
                                       "profile_column", "find_join_path", "run_readonly_sql"]
    llm.script_react(call("search_knowledge", {"query": "재구매", "source": "glossary"}, 1))
    docs = retrieved_from_tools(run_tool_loop(llm, tools, "시스템", "과제", max_steps=4))
    assert docs and all(d.id.startswith("glossary:") for d in docs)
```

`FakeCatalog`의 `orders` 테이블에 `order_id` 컬럼이 없으면 그 테이블의 실제 컬럼 이름으로 단언을 바꾼다.

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/unit/test_tool_loop.py -q -p no:cacheprovider` → FAIL

- [ ] **Step 3: 구현**

```python
def build_tool_loop(llm, tools, system):
    model = llm.bind_tools(tools)
    def agent(state: MessagesState) -> dict:
        return {"messages": [model.invoke([SystemMessage(content=system), *state["messages"]])]}
    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()
```

`run_tool_loop`는 `stream_mode="values"`, `recursion_limit=max_steps * 2 + 1`로 실행하고 `GraphRecursionError`가 나면 그때까지의 메시지를 반환한다. `BaseAgent.react`는 `create_agent` 대신 이것을 호출하고, `into`가 있으면 `retrieved_from_tools` 결과를 추가한다. `react_enabled`, 실패 시 메모 대체, `_summarize_react`는 그대로다.

- [ ] **Step 4: 탐색·설계 Agent가 `react(..., into=retrieved)`로 호출하게 변경**

- [ ] **Step 5: 통과 확인 후 커밋**

```bash
git add -A && git commit -m "Run agent tool calls through an explicit bind_tools and ToolNode loop"
```

---

### Task 8: 근거 검사 (도메인 검증 추가)

**Files:**
- Modify: `src/req2data/agents/reviewer.py`
- Test: `tests/unit/test_citations.py`

**Interfaces:**
- Consumes: Task 5의 `DataMapping.citations`, `DatasetDesign.citations`
- Produces: `Reviewer.check_grounding(mapping: DataMapping, design: DatasetDesign) -> CheckResult` — `name="grounding"`, 상태는 `pass` 또는 `warn`만 (재작업을 유발하지 않음, `rework_target=None`)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_grounding_warns_when_design_has_no_citation(deps, mapping, design):
    r = Reviewer(deps).check_grounding(mapping, design.model_copy(update={"citations": []}))
    assert r.name == "grounding" and r.status == CheckStatus.warn and "dataset_designer" in r.evidence

def test_grounding_passes_when_both_cite(deps, mapping, design):
    cite = [Citation(doc_id="guide:dataset_design_guide#1", used_for="기준 시점")]
    r = Reviewer(deps).check_grounding(mapping.model_copy(update={"citations": cite}),
                                       design.model_copy(update={"citations": cite}))
    assert r.status == CheckStatus.passed and "2건" in r.evidence
```

`mapping`, `design` 픽스처는 `tests/fixtures/scenario.py`의 기존 예시로 만든다. 맞는 예시가 없으면 테스트 파일 안에서 최소 객체를 만든다.

- [ ] **Step 2: 실패 확인 → Step 3: 구현** — 인용이 없는 쪽의 Agent 이름을 `evidence`에 나열한다 (`인용 근거 없음: data_explorer, dataset_designer`). 통과 시 `evidence`는 `매핑·설계가 검색 문서를 근거로 인용 (N건)`. `__call__`에서 `check_coverage` 다음에 추가한다.

- [ ] **Step 4: 통과 확인 후 커밋** — 검사 개수를 단언하는 기존 테스트가 있으면 맞춘다.

```bash
git add -A && git commit -m "Add a grounding check to the domain review"
```

---

### Task 9: 데모 정리, 인덱스 자동 재구축, 실행 확인 스크립트

**Files:**
- Delete: `demo/recordings/llm_recordings.jsonl`, `demo/recordings/retrieval_recordings.json`, `demo/recordings/demo_inputs.json` (폴더는 `.gitkeep`으로 유지)
- Modify: `src/req2data/rag/knowledge_base.py`, `src/req2data/service/services.py`, `scripts/build_index.py`
- Modify: `docker/Dockerfile` (`ENV DEMO_MODE=true` 제거), `docker/entrypoint.sh` (키가 없어도 `DEMO_MODE=false`, 경고만 출력)
- Create: `scripts/rag_flow.py`, `scripts/check_env.py`, `tests/unit/test_scripts.py`
- Modify: `Makefile` (`rag-flow`, `check` 타깃)

**Interfaces:**
- Consumes: Task 1~5의 `build_documents`, `KnowledgeIndex`, `HybridRetriever`, `rag_chain`, `GroundedAnswer`
- Produces:
  - `KnowledgeBase.ensure_current() -> bool` — 인덱스가 최신이면 `True`. 아니면 재구축을 시도해 성공하면 `True`, 실패하면 경고를 남기고 `False` (기존 인덱스 유지)
  - `scripts/rag_flow.py`: `run(kb: KnowledgeBase, query: str, llm: BaseChatModel | None) -> dict` — 키 `documents`(소스별 개수), `sample`(청크 1개의 메타데이터), `hits`(검색 결과 dict 목록), `answer`(`GroundedAnswer` 또는 `None`)
  - `scripts/check_env.py`: `main() -> int` — 단계별 `[OK]`/`[FAIL]` 출력, 하나라도 실패하면 1

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_ensure_current_rebuilds_for_new_embedding(tmp_path):
    paths = KnowledgePaths.under(ROOT / "data")
    KnowledgeBase(paths, HashEmbeddings(), persist_dir=tmp_path, embedding_name="a").rebuild()
    kb = KnowledgeBase(paths, HashEmbeddings(), persist_dir=tmp_path, embedding_name="b")
    assert not kb.index.is_current() and kb.ensure_current() and kb.index.built_with() == "b"

def test_ensure_current_keeps_index_when_embeddings_down(tmp_path):   # Review Focus 2
    paths = KnowledgePaths.under(ROOT / "data")
    n = KnowledgeBase(paths, HashEmbeddings(), persist_dir=tmp_path, embedding_name="a").rebuild()
    kb = KnowledgeBase(paths, UnavailableEmbeddings(), persist_dir=tmp_path, embedding_name="b")
    assert kb.ensure_current() is False and kb.index.count() == sum(n.values())

def test_rag_flow_runs_without_llm(kb):
    out = load_script("rag_flow").run(kb, "재구매 가능성이 낮은 고객", llm=None)
    assert set(out["documents"]) == {"catalog", "glossary", "guide"}
    assert out["hits"] and out["answer"] is None and "content_hash" in out["sample"]

def test_rag_flow_answers_with_citations(kb, llm):
    llm.script("GroundedAnswer", {"answer": "customer_unique_id로 판별", "citations": [{"doc_id": "glossary:재구매"}]})
    out = load_script("rag_flow").run(kb, "재구매는 어떻게 정의하나", llm=llm)
    assert out["answer"].citations[0].doc_id == "glossary:재구매"
```

`load_script(name)`은 테스트 파일 안의 도우미로, `importlib.util.spec_from_file_location`으로 `scripts/{name}.py`를 읽는다.

- [ ] **Step 2: 실패 확인 → Step 3: `ensure_current` 구현**

`build_services`는 `demo_mode`가 아닐 때 `kb.ensure_current()`를 호출한다. `build_index.py --if-stale`도 이 메서드를 쓴다.

- [ ] **Step 4: `scripts/rag_flow.py`**

단계마다 제목을 출력한다: `1. 전처리`, `2. 인덱싱`, `3. 검색`, `4. 근거 포함 응답`. 인자: `--query`(기본 `재구매 가능성이 낮은 고객을 예측하려면 어떤 데이터가 필요한가?`), `--rebuild`. LLM이 설정되지 않았으면 4단계에서 "LLM 키가 없어 검색까지만 실행"을 출력한다. 4단계는 `rag_chain(kb.retriever, llm, GroundedAnswer, ...)`을 쓴다.

- [ ] **Step 5: `scripts/check_env.py`**

순서대로 한 번씩 호출하고 결과를 출력한다. 키 값은 출력하지 않는다.

1. 설정 요약 (제공자, 채팅 배포 이름, 임베딩 배포 이름, API 버전)
2. 채팅 호출
3. JSON 출력 호출 (`get_chat_model(s, json_mode=True)`)
4. Tool 호출 (`bind_tools`로 도구 1개를 묶어 `tool_calls`가 오는지)
5. 임베딩 호출 (차원 출력)
6. `KnowledgeBase.ensure_current()`
7. `--full`을 주면 샘플 `data/samples/01_repurchase.md`를 그래프로 끝까지 실행한다. `build_graph(AgentDeps.from_settings(catalog=...), InMemorySaver())`로 만들고, 확인 질문이 나오면 각 질문의 `default_option`으로 답해 재개한다. 행 수, 타깃 비율, 검증 통과 여부, 인용 건수를 출력한다.

- [ ] **Step 6: 데모 정리** — 녹화 파일 3개 삭제, `Dockerfile`·`entrypoint.sh` 수정. `llm/recording.py`와 `tests/unit/test_recording.py`, `test_demo.py`의 나머지 테스트는 남긴다.

- [ ] **Step 7: 통과 확인 후 커밋**

```bash
git add -A && git commit -m "Rebuild a stale index on startup and add runnable RAG flow and environment checks"
```

---

### Task 10: 문서, 제출 zip, 최종 검증

**Files:**
- Modify: `README.md`, `docs/rag.md`, `docs/agent-architecture.md`, `docs/service.md`, `docs/SUBMISSION.md`, `Makefile`, `.env.example`

**Interfaces:**
- Consumes: 앞선 모든 Task

- [ ] **Step 1: README 맨 위에 "평가 항목 → 구현 위치" 표**

| 평가 항목 | 구현 위치 | 실행 |
| --- | --- | --- |
| 임베딩·Vector DB 인덱싱, 저장·조회 | `rag/index.py` `KnowledgeIndex` (`langchain_chroma.Chroma`), `scripts/build_index.py` | `make index` |
| 문서 분할·메타데이터·정제 | `rag/preprocess.py` `build_documents` (`clean_text`, `split_markdown`, `enrich_metadata`) | `make rag-flow` |
| Retrieval | `rag/retriever.py` `HybridRetriever(BaseRetriever)`, `tools/catalog.py` `CatalogRetriever` | `make rag-flow` |
| RAG 근거 반영 | `llm/chains.py` `retrieval_chain`·`rag_chain`, `agents/citations.py`, `reporting/render.py` `citations_md` | 결과의 `summary.md` "참고 근거" |
| LangChain 체인·LLM 호출 | `llm/chains.py` (`ChatPromptTemplate \| LLM \| PydanticOutputParser`), `llm/structured.py` | `make test` |
| LangGraph 노드·엣지·상태 | `graph/builder.py` (`StateGraph`, `add_conditional_edges`), `graph/state.py` | `make run` |
| Tool binding | `agents/tool_loop.py` (`bind_tools`, `ToolNode`), `tools/catalog.py` 도구 6개 | `make test` |
| 도메인 검증 | `agents/reviewer.py` 검사 7종: `check_existence`, `check_sql`, `check_leakage`, `check_censoring`, `check_quality`, `check_coverage`, `check_grounding` | `make run` |

표 아래에 실행 순서 3줄: `make setup` → `AOAI_*` 환경변수 설정 → `uv run python scripts/check_env.py --full`.

- [ ] **Step 2: 문서 수정**

- `docs/rag.md`, `docs/agent-architecture.md`: 바뀐 클래스·함수 이름과 처리 순서를 반영한다.
- `docs/SUBMISSION.md`, `docs/service.md`, README: "LLM 없이 버튼 두 번 데모" 안내를 지우고, 실행에 LLM 키가 필요하다고 적는다. Docker tar 안내는 "이미지는 `make submit`으로 직접 빌드"로 바꾼다.

- [ ] **Step 3: `make source-zip`**

```make
source-zip:
	mkdir -p dist && rm -f dist/req2data-agent-source.zip
	(git ls-files; ls data/db/olist.duckdb data/catalog/profiles.json; find storage/chroma -type f) | zip -q dist/req2data-agent-source.zip -@
```

`storage/`와 `data/db/`는 gitignore 대상이라 `git archive`로는 빠지므로 목록을 직접 넘긴다.

- [ ] **Step 4: 린트와 전체 테스트**

Run: `uv run ruff check src tests ui scripts && uv run ruff format --check src tests ui scripts && uv run pytest -m "not llm" -q -p no:cacheprovider`
Expected: 오류 0건, 전체 PASS

- [ ] **Step 5: 표준 API가 호출 경로에 있는지 확인**

Run:
```bash
for n in MarkdownHeaderTextSplitter RecursiveCharacterTextSplitter "langchain_chroma" BaseRetriever ChatPromptTemplate PydanticOutputParser RunnablePassthrough StateGraph add_conditional_edges bind_tools ToolNode; do printf "%-32s %s\n" "$n" "$(grep -rl "$n" src | tr '\n' ' ')"; done
```
Expected: 모든 이름에 `src/` 파일이 1개 이상 나온다.

- [ ] **Step 6: 인덱스 재구축 (로컬)**

사용자가 `ollama serve`를 실행한 상태에서: `make index`
Expected: `인덱스 구축 완료: ... {'catalog': N, 'glossary': N, 'guide': N}`. Ollama가 꺼져 있으면 여기서 멈추고 사용자에게 알린다.

- [ ] **Step 7: 실제 LLM으로 샘플 1건 실행 (로컬)**

Run: `LLM_PROVIDER=gemini uv run python scripts/check_env.py --full` (모델 이름은 `.env.example`의 Gemini 항목). Gemini가 안 되면 `.env`의 Ollama `qwen2.5:14b-instruct`로 실행한다.
Expected: 모든 단계 `[OK]`, 검증 통과, 인용 1건 이상. 인용이 0건이면 프롬프트 문구(Task 5 Step 5)를 보강하고 다시 실행한다.

- [ ] **Step 8: 커밋, zip 생성, 내용 확인**

```bash
git add -A && git commit -m "Map review criteria to code in README and add a source-only submission zip"
make source-zip
unzip -l dist/req2data-agent-source.zip | grep -cE "\.venv/|__MACOSX| dist/"
```
Expected: 마지막 명령의 출력이 `0`. `unzip -l dist/req2data-agent-source.zip | grep -c "storage/chroma"`는 1 이상.

- [ ] **Step 9: 실습 환경 검증 안내 (사용자 수행)**

zip을 실습 환경(`atl-dev`)에 풀고 실행할 명령을 사용자에게 전달한다: `uv sync && uv run python scripts/check_env.py --full`. 이 단계의 결과는 사용자가 알려줄 때까지 "미확인"으로 보고한다.
