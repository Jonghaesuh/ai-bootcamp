"""지식 소스 → 검색 문서(LangChain Document).

- 카탈로그: 테이블 문서 1개 + 컬럼 문서(컬럼별 1개). 정의서 설명과 자동 프로파일을 합친다.
- 용어집: 용어당 1개.
- 가이드·사례: Markdown 제목 단위로 나누고, 긴 구간은 크기에 맞춰 다시 나눈다.

모든 문서는 rag/preprocess.py의 정제 → 분할 → 메타데이터 부여 → 중복 제거를 거친다.
"""

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from req2data.rag.preprocess import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    clean_text,
    dedupe,
    enrich_metadata,
    split_markdown,
)
from req2data.rag.sources import CatalogSpec, ColumnSpec, GlossaryTerm, TableSpec

SOURCES = ("catalog", "glossary", "guide", "case")


def build_documents(
    spec: CatalogSpec,
    glossary: list[GlossaryTerm],
    profiles: dict[str, dict[str, Any]],
    guides_dir: Path,
    cases_dir: Path,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """전체 지식 소스를 색인용 문서로 만든다: load → clean → split → enrich → dedupe."""
    size = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    return dedupe(
        [
            *catalog_documents(spec, profiles),
            *glossary_documents(glossary),
            *directory_chunks(guides_dir, "guide", **size),
            *directory_chunks(cases_dir, "case", **size),
        ]
    )


def _document(lines: list[str], **meta: Any) -> Document:
    return enrich_metadata(Document(page_content=clean_text("\n".join(lines))), **meta)


def catalog_documents(
    spec: CatalogSpec, profiles: dict[str, dict[str, Any]] | None = None
) -> list[Document]:
    profiles = profiles or {}
    docs = []
    for t in spec.tables.values():
        docs.append(_table_doc(t, profiles.get(t.name, {})))
        docs += [_column_doc(t, c, profiles.get(f"{t.name}.{c.name}")) for c in t.columns]
    return docs


def _table_doc(t: TableSpec, profile: dict[str, Any]) -> Document:
    keys = [c.name for c in t.columns if c.pk]
    fks = [f"{c.name} → {c.fk}" for c in t.columns if c.fk]
    times = [f"{c.name}(순서 {c.time_rank})" for c in t.columns if c.is_time]
    lines = [
        f"[테이블] {t.name}",
        f"설명: {t.description}",
        f"행 단위: {t.grain}",
        f"컬럼: {', '.join(c.name for c in t.columns)}",
        f"기본키: {', '.join(keys) or '없음'}",
        f"외래키: {'; '.join(fks) or '없음'}",
        f"시점 컬럼: {', '.join(times) or '없음'}",
    ]
    if profile.get("row_count") is not None:
        lines.append(f"행 수: {profile['row_count']}")
    return _document(
        lines, doc_id=f"table:{t.name}", source="catalog", kind="table", title=t.name, table=t.name
    )


def _column_doc(t: TableSpec, c: ColumnSpec, profile: dict[str, Any] | None) -> Document:
    lines = [
        f"[컬럼] {t.name}.{c.name} ({c.type})",
        f"설명: {c.description}",
        f"영문 설명: {c.description_en}",
        f"테이블: {t.name} — {t.description}",
    ]
    if c.pk:
        lines.append("기본키")
    if c.fk:
        lines.append(f"외래키: {c.fk}")
    if c.is_time:
        lines.append(f"시점 컬럼 (업무 흐름 순서 {c.time_rank})")
    if c.available_at:
        lines.append(f"값 확정 시점: {c.available_at}")
    if c.note:
        lines.append(f"주의: {c.note}")
    if profile:
        lines.append(
            f"프로파일: 결측률 {profile.get('null_ratio')}, 고유값 {profile.get('distinct')}, "
            f"범위 {profile.get('min')} ~ {profile.get('max')}, 예시 {profile.get('examples')}"
        )
    return _document(
        lines,
        doc_id=f"column:{t.name}.{c.name}",
        source="catalog",
        kind="column",
        title=f"{t.name}.{c.name}",
        table=t.name,
        column=c.name,
        type=c.type,
        is_pk=c.pk,
        fk=c.fk,
        is_time=c.is_time,
        time_rank=c.time_rank,
        available_at=c.available_at,
    )


def glossary_documents(terms: list[GlossaryTerm]) -> list[Document]:
    return [
        _document(
            [
                f"[용어] {t.term}",
                f"동의어: {', '.join(t.synonyms)}",
                f"정의: {t.definition}",
                f"계산: {t.calculation}",
                f"관련 컬럼: {', '.join(t.related_columns)}",
            ],
            doc_id=f"glossary:{t.term}",
            source="glossary",
            kind="term",
            title=t.term,
            related_columns=t.related_columns,
        )
        for t in terms
    ]


def markdown_chunks(
    path: Path, source: str, *, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    """Markdown 파일 하나를 청크로 만든다. 각 청크 앞에 문서·구간 제목을 붙여 맥락을 남긴다."""
    path = Path(path)
    text = clean_text(path.read_text(encoding="utf-8"))
    docs = []
    for i, chunk in enumerate(
        split_markdown(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap), 1
    ):
        title = chunk.metadata.get("h1") or path.stem
        heading = " / ".join(filter(None, [chunk.metadata.get("h2"), chunk.metadata.get("h3")]))
        prefix = f"[{title}] {heading}".rstrip()
        docs.append(
            enrich_metadata(
                Document(page_content=f"{prefix}\n{chunk.page_content}"),
                doc_id=f"{source}:{path.stem}#{i}",
                source=source,
                kind="section",
                chunk_index=i,
                title=" / ".join(filter(None, [title, heading])),
                path=str(path),
            )
        )
    return dedupe(docs)


def directory_chunks(
    directory: Path,
    source: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return [
        doc
        for p in sorted(directory.glob("*.md"))
        for doc in markdown_chunks(p, source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ]
