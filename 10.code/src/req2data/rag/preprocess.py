"""지식 소스 전처리: 정제(clean) → 분할(split) → 메타데이터 부여(enrich) → 중복 제거(dedupe).

소스가 무엇이든 같은 단계를 거친다. 소스별 문서 조립은 rag/documents.py가 맡는다.
"""

import hashlib
import re
import unicodedata
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # 탭·줄바꿈은 남긴다
_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """유니코드 NFC 정규화, 제어 문자·줄 끝 공백 제거, 빈 줄 3개 이상 축약."""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    text = _TRAILING.sub("", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def split_markdown(
    text: str, *, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    """제목(#, ##, ###) 단위로 나눈 뒤 긴 구간은 크기에 맞춰 다시 나눈다. 제목은 메타데이터 h1~h3에 남는다."""
    sections = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS).split_text(text)
    sizer = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return sizer.split_documents(sections)


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def flatten_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma는 None·리스트 메타데이터를 받지 않는다."""
    return {
        k: (", ".join(map(str, v)) if isinstance(v, list | tuple) else v)
        for k, v in meta.items()
        if v is not None
    }


def enrich_metadata(
    doc: Document, *, doc_id: str, source: str, kind: str, chunk_index: int = 0, **extra: Any
) -> Document:
    """검색 필터·인용 표시에 쓰는 공통 메타데이터를 붙이고 문서 ID를 정한다."""
    meta = {
        "source": source,
        "kind": kind,
        "title": "",
        "path": "",
        "chunk_index": chunk_index,
        "table": "",
        "column": "",
        **extra,
        "content_hash": content_hash(doc.page_content),
    }
    return Document(id=doc_id, page_content=doc.page_content, metadata=flatten_metadata(meta))


def dedupe(docs: list[Document]) -> list[Document]:
    """빈 문서와 내용이 같은 문서를 버린다 (먼저 나온 것을 남긴다)."""
    seen: set[str] = set()
    kept = []
    for d in docs:
        key = d.metadata.get("content_hash") or content_hash(d.page_content)
        if not d.page_content.strip() or key in seen:
            continue
        seen.add(key)
        kept.append(d)
    return kept
