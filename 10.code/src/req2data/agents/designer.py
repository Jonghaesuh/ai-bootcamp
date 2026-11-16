import re
from typing import Any

from langchain_core.documents import Document

from req2data.agents.base import BaseAgent, as_json
from req2data.agents.requirement import render_answered
from req2data.agents.sql_builder import CohortSQLBuilder, PlanError
from req2data.graph.state import Req2DataState, load
from req2data.schemas import (
    DataMapping,
    DatasetDesign,
    DesignPlan,
    FeaturePlan,
    RequirementSpec,
    TargetDefinition,
    TargetPlan,
)
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
    """Plan-and-Solve로 설계 계획(DesignPlan)을 세우고, SQL은 CohortSQLBuilder가 만든다.
    계획이 카탈로그와 맞지 않거나 실행에 실패하면 오류를 넣어 계획 수정을 요청한다."""

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
        retrieved: list[Document] = []
        extra = "\n\n".join(
            filter(
                None,
                [
                    self.deps.fewshots.render("dataset_design", query),
                    self.knowledge(
                        f"{query} 분석 단위 기준 시점 누수 관측 기간 분할",
                        "guide",
                        k=4,
                        title="데이터셋 설계 가이드",
                        into=retrieved,
                    ),
                    self.knowledge(query, "case", k=2, title="과거 승인 설계 사례", into=retrieved),
                ],
            )
        )
        system = self.system(state, extra=extra)

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
            DesignPlan,
            fallback=lambda: fallback_plan(spec, mapping),
        )
        citations, rejected = self.cite(result.value, retrieved)
        design, fixes = self.build(system, result.value, schema)
        design = design.model_copy(
            update={
                "version": state.get("design_version", 0) + 1,
                "citations": result.value.citations,
            }
        )
        return {
            "dataset_design": design.model_dump(mode="json"),
            "design_version": design.version,
            "citations": citations,
            "trace": [
                self.result_event(
                    result,
                    version=design.version,
                    row_count=design.row_count,
                    target_ratio=design.target_ratio,
                    sql_error=design.sql_error,
                    sql_fixes=fixes,
                    citations=len(citations),
                    rejected_citations=rejected,
                )
            ],
        }

    def build(self, system: str, plan: DesignPlan, schema: str) -> tuple[DatasetDesign, int]:
        """계획 → SQL(코드) → 실행 확인. 계획 오류는 오류 내용을 넣어 계획 수정을 요청한다."""
        builder = CohortSQLBuilder(self.catalog)
        attempts = self.deps.cfg("workflow.sql_fix_attempts", 1)
        for fix in range(attempts + 1):
            try:
                return self._measure(builder.build(plan)), fix
            except (PlanError, SQLExecutionError) as exc:
                error = str(exc)
                if fix == attempts:
                    break
                plan = self.structured(
                    system,
                    self.prompt.task("fix_plan", plan=as_json(plan), error=error, schema=schema),
                    DesignPlan,
                    fallback=lambda current=plan: current,
                ).value
        return failed_design(plan, error), attempts

    def _measure(self, design: DatasetDesign) -> DatasetDesign:
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


def fallback_plan(spec: RequirementSpec, mapping: DataMapping) -> DesignPlan:
    """LLM 출력 실패 시 매핑으로 만든 규칙 기반 계획. 분석 단위와 기준 이벤트 테이블이 다르면
    (고객 단위 + 주문 시점) 첫 이벤트 코호트의 재발생 타깃으로 본다. 키·시점이 없으면 빌더가 오류를 낸다."""
    key = mapping.analysis_key.ref if mapping.analysis_key else ""
    ref = mapping.reference_time.ref if mapping.reference_time else ""
    days = re.search(r"(\d+)\s*일", spec.target_definition or "")
    features = [
        FeaturePlan(name=c.column, source_column=c.ref, requirement_ids=[item.requirement_id])
        for item in mapping.items
        if item.role == "feature"
        for c in item.columns
    ]
    return DesignPlan(
        analysis_unit=spec.analysis_unit or "미정",
        entity_key=key,
        reference_time=ref,
        cohort="first_event" if key.split(".")[0] != ref.split(".")[0] else "every_event",
        target=TargetPlan(
            kind="repeat_event",
            window_days=int(days.group(1)) if days else 30,
            description=spec.target_definition or "미정",
        ),
        features=features,
        rationale="LLM 설계 실패로 데이터 매핑에서 규칙 기반 계획을 만들었습니다.",
    )


def failed_design(plan: DesignPlan, error: str) -> DatasetDesign:
    """SQL을 만들지 못한 계획. 검증 Agent가 sql_execution 미통과로 재작업을 지시한다."""
    return DatasetDesign(
        analysis_unit=plan.analysis_unit,
        entity_key=plan.entity_key.split(".")[-1] or "id",
        reference_time=plan.reference_time,
        target=TargetDefinition(
            description=plan.target.description,
            window_days=max(int(plan.target.window_days), 1),
            positive_condition=plan.target.condition or plan.target.kind,
        ),
        sql="",
        sql_error=error,
        rationale=plan.rationale,
    )
