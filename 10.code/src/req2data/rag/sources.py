"""RAG 지식 소스 로더: 데이터 카탈로그, 용어집. (가이드·사례는 Markdown 파일 그대로 청크한다.)"""

from functools import cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ColumnSpec(BaseModel):
    name: str
    type: str
    description: str = ""
    description_en: str = ""
    pk: bool = False
    fk: str | None = None
    time_rank: int | None = None
    available_at: str | None = None
    note: str = ""

    @property
    def is_time(self) -> bool:
        return self.time_rank is not None


class TableSpec(BaseModel):
    name: str
    source_file: str
    description: str = ""
    grain: str = ""
    columns: list[ColumnSpec]

    def column(self, name: str) -> ColumnSpec | None:
        return next((c for c in self.columns if c.name == name), None)


class CatalogSpec(BaseModel):
    dataset: str
    source: str = ""
    period: str = ""
    tables: dict[str, TableSpec]

    def column(self, ref: str) -> ColumnSpec | None:
        table, _, column = ref.partition(".")
        spec = self.tables.get(table)
        return spec.column(column) if spec else None

    def foreign_keys(self) -> list[tuple[str, str]]:
        return [
            (f"{t.name}.{c.name}", c.fk) for t in self.tables.values() for c in t.columns if c.fk
        ]


class GlossaryTerm(BaseModel):
    term: str
    synonyms: list[str] = Field(default_factory=list)
    definition: str = ""
    calculation: str = ""
    related_columns: list[str] = Field(default_factory=list)

    @property
    def surface_forms(self) -> list[str]:
        return [self.term, *self.synonyms]


def load_catalog_spec(path: Path) -> CatalogSpec:
    data = dict(_read_yaml(Path(path)))  # 캐시된 원본을 바꾸지 않도록 복사
    tables = {name: TableSpec(name=name, **t) for name, t in data.pop("tables").items()}
    return CatalogSpec(tables=tables, **data)


def load_glossary(path: Path) -> list[GlossaryTerm]:
    path = Path(path)
    return [GlossaryTerm(**t) for t in _read_yaml(path)] if path.exists() else []


@cache
def _read_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
