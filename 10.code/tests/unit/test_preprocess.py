from langchain_core.documents import Document

from req2data.rag.documents import markdown_chunks
from req2data.rag.preprocess import clean_text, dedupe, split_markdown


def test_clean_text_normalizes_and_strips():
    assert clean_text("가  \nb\x00\n\n\n\nc") == "가\nb\n\nc"


def test_split_markdown_by_heading_then_size():
    body = "문장입니다. " * 250  # 약 1,750자
    docs = split_markdown(
        f"# 제목\n## 가\n{body}\n## 나\n짧은 본문", chunk_size=600, chunk_overlap=80
    )
    assert len(docs) >= 4 and all(len(d.page_content) <= 600 for d in docs)
    assert {d.metadata["h1"] for d in docs} == {"제목"}
    assert {d.metadata["h2"] for d in docs} == {"가", "나"}


def test_markdown_without_headings_or_empty(tmp_path):
    plain = tmp_path / "plain.md"
    plain.write_text("제목 없는 메모", encoding="utf-8")
    empty = tmp_path / "empty.md"
    empty.write_text("  \n\n", encoding="utf-8")
    assert [d.id for d in markdown_chunks(plain, "case")] == ["case:plain#1"]
    assert markdown_chunks(empty, "case") == []


def test_enrich_metadata_is_chroma_safe_and_unique(kb):
    docs = kb.documents()
    assert len({d.id for d in docs}) == len(docs)
    for d in docs:
        assert {"source", "kind", "title", "chunk_index", "content_hash"} <= set(d.metadata)
        assert all(isinstance(v, str | int | float | bool) for v in d.metadata.values())


def test_dedupe_drops_empty_and_duplicate_content():
    a = Document(id="a", page_content="같음")
    b = Document(id="b", page_content="같음")
    assert [d.id for d in dedupe([a, b, Document(id="c", page_content=" ")])] == ["a"]


def test_build_documents_covers_all_sources(kb):
    assert {d.metadata["source"] for d in kb.documents()} == {"catalog", "glossary", "guide"}
