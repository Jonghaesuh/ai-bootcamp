import re
from typing import Any

from req2data.agents.base import BaseAgent, as_json
from req2data.agents.requirement import render_answered
from req2data.graph.state import Req2DataState, load
from req2data.schemas import DataMapping, DatasetDesign, RequirementSpec, TargetDefinition
from req2data.tools.catalog import SQLExecutionError, build_catalog_tools, schema_index


def strip_sql(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def render_schema(index: dict[str, dict[str, Any]]) -> str:
    lines = []
    for ref, col in index.items():
        meta = [col.get("type", "")]
        if col.get("is_pk"):
            meta.append("PK")
        if col.get("fk"):
            meta.append(f"FK→{col['fk']}")
        if col.get("is_time"):
            meta.append(f"time_rank={col.get('time_rank')}")
        if col.get("available_at"):
            meta.append(f"available_at={col['available_at']}")
        lines.append(f"- {ref} ({', '.join(m for m in meta if m)}): {col.get('description', '')}")
    return "\n".join(lines)


class DatasetDesigner(BaseAgent):
    """Plan-and-Solve 설계 → SQL 실행으로 확인 → 실패 시 SQL 수정 요청."""

    name = "dataset_designer"

    def __call__(self, state: Req2DataState) -> dict:
        spec = load(state, "requirement_spec", RequirementSpec)
        mapping = load(state, "data_mapping", DataMapping)
        assert spec and mapping, "requirement_spec과 data_mapping이 필요합니다."

        tables = sorted({ref.split(".")[0] for ref in mapping.column_refs()})
        schema = render_schema(schema_index(self.catalog, tables))
        requirements = as_json(spec.model_copy(update={"open_questions": []}))
        mapping_text = as_json(mapping)
        query = " ".join(filter(None, [spec.prediction_target, spec.target_definition]))
        system = self.system(state, extra=self.deps.fewshots.render("dataset_design", query))

        notes = self.react(
            system,
            self.prompt.task(
                "investigate", requirements=requirements, mapping=mapping_text, schema=schema
            ),
            build_catalog_tools(self.catalog),
        )
        result = self.structured(
            system,
            self.prompt.task(
                "design",
                requirements=requirements,
                answers=render_answered(state.get("answered") or []),
                mapping=mapping_text,
                schema=schema,
                notes=notes,
            ),
            DatasetDesign,
            fallback=lambda: fallback_design(spec, mapping),
        )
        design = result.value.model_copy(update={"version": state.get("design_version", 0) + 1})
        design, fixes = self.verify_sql(system, design)
        return {
            "dataset_design": design.model_dump(mode="json"),
            "design_version": design.version,
            "trace": [
                self.result_event(
                    result,
                    version=design.version,
                    row_count=design.row_count,
                    target_ratio=design.target_ratio,
                    sql_error=design.sql_error,
                    sql_fixes=fixes,
                )
            ],
        }

    def verify_sql(self, system: str, design: DatasetDesign) -> tuple[DatasetDesign, int]:
        """SQL을 실제로 실행해 행 수·타깃 비율을 채운다. 실패하면 오류를 넣어 수정을 요청한다."""
        attempts = self.deps.cfg("workflow.sql_fix_attempts", 1)
        for fix in range(attempts + 1):
            try:
                return self._measure(design), fix
            except SQLExecutionError as exc:
                design = design.model_copy(update={"sql_error": str(exc)})
                if fix == attempts:
                    break
                fixed = self.structured(
                    system,
                    self.prompt.task("fix_sql", sql=design.sql, error=str(exc)),
                    DatasetDesign,
                    fallback=lambda current=design: current,
                ).value
                design = fixed.model_copy(update={"version": design.version, "sql_error": None})
        return design, attempts

    def _measure(self, design: DatasetDesign) -> DatasetDesign:
        if not design.sql.strip():
            raise SQLExecutionError("SQL이 비어 있습니다.")
        sql = strip_sql(design.sql)
        res = self.catalog.run_readonly_sql(
            f"SELECT count(*) AS n, avg(CAST({design.target.name} AS DOUBLE)) AS r FROM ({sql}) t"
        )
        n, ratio = res["rows"][0]
        return design.model_copy(
            update={
                "sql": sql,
                "row_count": int(n),
                "target_ratio": None if ratio is None else float(ratio),
                "sql_error": None,
            }
        )


def fallback_design(spec: RequirementSpec, mapping: DataMapping) -> DatasetDesign:
    days = re.search(r"(\d+)\s*일", spec.target_definition or "")
    return DatasetDesign(
        analysis_unit=spec.analysis_unit or "미정",
        entity_key=mapping.analysis_key.column if mapping.analysis_key else "id",
        reference_time=mapping.reference_time.ref if mapping.reference_time else "",
        target=TargetDefinition(
            description=spec.target_definition or "미정",
            window_days=int(days.group(1)) if days else 30,
            positive_condition="미정",
        ),
        sql="",
        rationale="LLM 설계 실패로 규칙 기반 골격만 만들었습니다. 검증 후 재작업합니다.",
    )
