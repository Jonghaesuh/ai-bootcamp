"""지식 소스 → 검색 문서(청크).

- 카탈로그: 테이블 문서 1개 + 컬럼 문서(컬럼별 1개). 정의서 설명과 자동 프로파일을 합친다.
- 용어집: 용어당 1개.
- 가이드·사례: Markdown 제목(##) 단위.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from req2data.rag.sources import CatalogSpec, ColumnSpec, GlossaryTerm, TableSpec

SOURCES = ("catalog", "glossary", "guide", "case")


@dataclass
class RagDocument:
    doc_id: str
    text: str
    source: str  # catalog | glossary | guide | case
    metadata: dict[str, Any] = field(default_factory=dict)

    def chroma_metadata(self) -> dict[str, str | int | float | bool]:
        """Chroma는 None·리스트 메타데이터를 받지 않는다."""
        meta = {"source": self.source, **self.metadata}
        return {
            k: (", ".join(map(str, v)) if isinstance(v, list) else v)
            for k, v in meta.items()
            if v is not None
        }


def catalog_documents(
    spec: CatalogSpec, profiles: dict[str, dict[str, Any]] | None = None
) -> list[RagDocument]:
    profiles = profiles or {}
    docs = []
    for t in spec.tables.values():
        docs.append(_table_doc(t, profiles.get(t.name, {})))
        docs += [_column_doc(t, c, profiles.get(f"{t.name}.{c.name}")) for c in t.columns]
    return docs


def _table_doc(t: TableSpec, profile: dict[str, Any]) -> RagDocument:
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
    return RagDocument(
        doc_id=f"table:{t.name}",
        text="\n".join(lines),
        source="catalog",
        metadata={"kind": "table", "table": t.name, "column": ""},
    )


def _column_doc(t: TableSpec, c: ColumnSpec, profile: dict[str, Any] | None) -> RagDocument:
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
            f"프로파일: 결측률 {profile.get('null_ratio')}, 고유값 약 {profile.get('distinct')}, "
            f"범위 {profile.get('min')} ~ {profile.get('max')}, 예시 {profile.get('examples')}"
        )
    return RagDocument(
        doc_id=f"column:{t.name}.{c.name}",
        text="\n".join(lines),
        source="catalog",
        metadata={
            "kind": "column",
            "table": t.name,
            "column": c.name,
            "type": c.type,
            "is_pk": c.pk,
            "fk": c.fk,
            "is_time": c.is_time,
            "time_rank": c.time_rank,
            "available_at": c.available_at,
        },
    )


def glossary_documents(terms: list[GlossaryTerm]) -> list[RagDocument]:
    return [
        RagDocument(
            doc_id=f"glossary:{t.term}",
            text="\n".join(
                [
                    f"[용어] {t.term}",
                    f"동의어: {', '.join(t.synonyms)}",
                    f"정의: {t.definition}",
                    f"계산: {t.calculation}",
                    f"관련 컬럼: {', '.join(t.related_columns)}",
                ]
            ),
            source="glossary",
            metadata={"kind": "term", "title": t.term, "related_columns": t.related_columns},
        )
        for t in terms
    ]


_HEADING = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def markdown_chunks(path: Path, source: str) -> list[RagDocument]:
    """## 제목 단위로 자른다. 각 청크에는 문서 제목을 앞에 붙여 맥락을 남긴다."""
    text = Path(path).read_text(encoding="utf-8")
    headings = list(_HEADING.finditer(text))
    title = next((m.group(2).strip() for m in headings if len(m.group(1)) == 1), Path(path).stem)
    sections = [m for m in headings if len(m.group(1)) == 2]
    if not sections:
        return [RagDocument(f"{source}:{Path(path).stem}", text.strip(), source, {"title": title})]
    docs = []
    for i, m in enumerate(sections):
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        heading = m.group(2).strip()
        body = text[m.end() : end].strip()
        docs.append(
            RagDocument(
                doc_id=f"{source}:{Path(path).stem}#{i + 1}",
                text=f"[{title}] {heading}\n{body}",
                source=source,
                metadata={"kind": "section", "title": f"{title} / {heading}", "path": str(path)},
            )
        )
    return docs


def directory_chunks(directory: Path, source: str) -> list[RagDocument]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return [doc for p in sorted(directory.glob("*.md")) for doc in markdown_chunks(p, source)]
