from typing import Literal

from pydantic import Field

from req2data.schemas.base import LLMOutput


class ColumnRef(LLMOutput):
    table: str
    column: str

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.column}"

    @classmethod
    def parse(cls, ref: str) -> "ColumnRef":
        table, _, column = ref.partition(".")
        return cls(table=table, column=column)


class MappingItem(LLMOutput):
    requirement_id: str
    role: Literal["key", "target", "feature", "filter", "time"] = "feature"
    columns: list[ColumnRef] = Field(default_factory=list)
    evidence: list[str] = Field(
        default_factory=list, description="근거 카탈로그 문서 (table.column)"
    )
    note: str = ""


class JoinStep(LLMOutput):
    left: ColumnRef
    right: ColumnRef
    how: Literal["inner", "left"] = "left"


class DataMapping(LLMOutput):
    analysis_key: ColumnRef | None = Field(default=None, description="분석 단위 키")
    reference_time: ColumnRef | None = Field(default=None, description="기준 시점 컬럼")
    items: list[MappingItem] = Field(default_factory=list)
    join_paths: list[JoinStep] = Field(default_factory=list)
    unmapped_requirements: list[str] = Field(default_factory=list)
    proxy_suggestions: list[str] = Field(
        default_factory=list, description="대체 데이터·추가 수집 제안"
    )
    rejected_refs: list[str] = Field(default_factory=list, description="스키마에 없어 제거한 참조")
    rationale: str = ""

    def column_refs(self) -> set[str]:
        refs = {c.ref for item in self.items for c in item.columns}
        for c in (self.analysis_key, self.reference_time):
            if c:
                refs.add(c.ref)
        return refs
