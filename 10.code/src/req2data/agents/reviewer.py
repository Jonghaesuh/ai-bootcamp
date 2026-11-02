import logging
from typing import Any

from sqlglot import exp
from sqlglot.lineage import lineage

from req2data.agents.base import BaseAgent, as_json
from req2data.graph.state import Req2DataState, load
from req2data.schemas import (
    PIPELINE,
    CheckResult,
    CheckStatus,
    ClarificationQuestion,
    DataMapping,
    DatasetDesign,
    EDASummary,
    RequirementSpec,
    ReviewNarrative,
    ReviewReport,
)
from req2data.tools.catalog import SQLExecutionError, schema_index

log = logging.getLogger(__name__)

DATA_KINDS = {"population", "target", "feature", "filter"}  # 데이터로 반영돼야 하는 요구사항 종류
MIN_POSITIVES = 30  # 양성 수가 이보다 적으면 모델링 불가로 판정


def feature_sources(sql: str, column: str, schema: dict[str, dict[str, str]]) -> set[str]:
    """SQL 출력 컬럼의 원천 table.column을 sqlglot lineage로 추출한다."""
    try:
        node = lineage(column, sql, schema=schema or None, dialect="duckdb")
    except Exception as exc:
        log.debug("lineage 실패(%s): %s", column, exc)
        return set()
    return {
        f"{n.source.name}.{n.name.split('.')[-1]}"
        for n in node.walk()
        if not n.downstream and isinstance(n.source, exp.Table)
    }


class Reviewer(BaseAgent):
    """판정은 규칙 검사로, 설명은 LLM으로. 미통과 항목은 원인 Agent와 피드백을 지정한다."""

    name = "reviewer"

    def __call__(self, state: Req2DataState) -> dict:
        spec = load(state, "requirement_spec", RequirementSpec)
        mapping = load(state, "data_mapping", DataMapping)
        design = load(state, "dataset_design", DatasetDesign)
        assert spec and mapping and design, "명세·매핑·설계가 모두 필요합니다."

        tables = {r.split(".")[0] for r in mapping.column_refs()}
        tables |= {r.split(".")[0] for f in design.features for r in f.source_columns}
        if design.reference_time:
            tables.add(design.reference_time.split(".")[0])
        index = schema_index(self.catalog, sorted(t for t in tables if t))

        checks: list[CheckResult] = []
        confirmations: list[ClarificationQuestion] = []
        eda: EDASummary | None = None

        checks.append(self.check_existence(mapping, design, index))
        sql_ok = self.check_sql(design, checks)
        if sql_ok:
            leak, confirmation = self.check_leakage(design, index)
            checks.append(leak)
            if confirmation:
                confirmations.append(confirmation)
            checks.append(self.check_censoring(design, index))
            eda = self.eda(design)
            checks += self.check_quality(design, eda)
        checks.append(self.check_coverage(spec, mapping, design))

        report = self.compile(design, checks, eda, confirmations)
        narrative = self.structured(
            self.system(state),
            self.prompt.task(
                "narrative",
                design=as_json(
                    design.model_dump(
                        include={
                            "analysis_unit",
                            "reference_time",
                            "target",
                            "features",
                            "population_filters",
                            "split",
                            "rationale",
                        }
                    )
                ),
                checks=as_json([c.model_dump(mode="json") for c in checks]),
                eda=as_json(eda) if eda else "(SQL 실행 실패로 없음)",
            ),
            ReviewNarrative,
            fallback=lambda: ReviewNarrative(summary=default_summary(report)),
        )
        report.summary = narrative.value.summary
        if report.rework_targets and narrative.value.additional_feedback:
            report.feedback.setdefault(report.rework_targets[0], []).extend(
                narrative.value.additional_feedback
            )

        deferred = state.get("deferred_questions") or []
        known = {q["question"] for q in deferred} | {
            a["question"] for a in state.get("answered") or []
        }
        deferred += [q.model_dump(mode="json") for q in confirmations if q.question not in known]
        return {
            "review_report": report.model_dump(mode="json"),
            "deferred_questions": deferred,
            "trace": [
                self.result_event(
                    narrative,
                    passed=report.passed,
                    failed=[c.name for c in checks if c.status == CheckStatus.fail],
                    feasibility=report.feasibility,
                )
            ],
        }

    # ---- 개별 검사 ----

    def check_existence(
        self, mapping: DataMapping, design: DatasetDesign, index: dict[str, Any]
    ) -> CheckResult:
        bad_design = [r for f in design.features for r in f.source_columns if r not in index] + (
            [design.reference_time] if design.reference_time not in index else []
        )
        bad_mapping = [r for r in mapping.column_refs() if r not in index]
        if not bad_design and not bad_mapping:
            return CheckResult(
                name="schema_existence",
                status=CheckStatus.passed,
                evidence="모든 참조 컬럼이 스키마에 존재",
            )
        return CheckResult(
            name="schema_existence",
            status=CheckStatus.fail,
            evidence=f"스키마에 없는 컬럼: {', '.join(bad_design + bad_mapping)}",
            affected=bad_design + bad_mapping,
            rework_target="data_explorer" if bad_mapping else "dataset_designer",
        )

    def check_sql(self, design: DatasetDesign, checks: list[CheckResult]) -> bool:
        if design.sql_error or not design.sql.strip():
            checks.append(
                CheckResult(
                    name="sql_execution",
                    status=CheckStatus.fail,
                    evidence=design.sql_error or "SQL 없음",
                    rework_target="dataset_designer",
                )
            )
            return False
        try:
            cols = self.catalog.run_readonly_sql(f"SELECT * FROM ({design.sql}) t LIMIT 0")[
                "columns"
            ]
        except SQLExecutionError as exc:
            checks.append(
                CheckResult(
                    name="sql_execution",
                    status=CheckStatus.fail,
                    evidence=str(exc),
                    rework_target="dataset_designer",
                )
            )
            return False
        expected = [
            design.entity_key,
            design.reference_time_alias,
            design.target.name,
            design.split.column,
        ]
        expected += [f.name for f in design.features]
        missing = [c for c in expected if c not in cols]
        checks.append(
            CheckResult(
                name="sql_execution",
                status=CheckStatus.fail if missing else CheckStatus.passed,
                evidence=f"SQL 출력에 없는 컬럼: {missing}"
                if missing
                else f"{design.row_count}행 생성",
                affected=missing,
                rework_target="dataset_designer" if missing else None,
            )
        )
        return not missing

    def check_leakage(
        self, design: DatasetDesign, index: dict[str, Any]
    ) -> tuple[CheckResult, ClarificationQuestion | None]:
        ref_rank = (index.get(design.reference_time) or {}).get("time_rank")
        if ref_rank is None:
            return CheckResult(
                name="leakage",
                status=CheckStatus.warn,
                evidence=f"기준 시점 {design.reference_time}의 time_rank 메타데이터가 없어 누수를 판정하지 못함",
            ), None

        schema: dict[str, dict[str, str]] = {}
        for ref, col in index.items():
            t, c = ref.split(".", 1)
            schema.setdefault(t, {})[c] = col.get("type") or "text"

        leaks: dict[str, str] = {}  # 피처 → 근거
        latest: tuple[int, str] | None = None
        for f in design.features:
            for src in set(f.source_columns) | feature_sources(design.sql, f.name, schema):
                col = index.get(src) or {}
                avail = col.get("available_at") or (src if col.get("is_time") else None)
                rank = (index.get(avail) or {}).get("time_rank") if avail else None
                if rank is not None and rank > ref_rank:
                    leaks[f.name] = (
                        f"{src}는 {avail} 시점에 확정 (기준 시점 {design.reference_time}보다 늦음)"
                    )
                    if latest is None or rank > latest[0]:
                        latest = (rank, avail)
        if not leaks:
            return CheckResult(
                name="leakage",
                status=CheckStatus.passed,
                evidence="기준 시점 이후 정보를 쓰는 피처 없음",
            ), None

        confirmation = ClarificationQuestion(
            id=f"V{design.version}Q1",
            question=(
                f"기준 시점을 {latest[1]}(으)로 옮기면 {', '.join(leaks)} 피처를 쓸 수 있지만 "
                "예측·조치 시점도 그만큼 늦어집니다. 기준 시점을 바꿀까요?"
            ),
            target_field="dataset_design.reference_time",
            options=["현재 기준 시점 유지", f"기준 시점을 {latest[1]}(으)로 변경"],
            default_option="현재 기준 시점 유지",
            reason="기준 시점은 비즈니스 결정이라 자동으로 바꾸지 않음",
            asked_by="reviewer",
        )
        return (
            CheckResult(
                name="leakage",
                status=CheckStatus.fail,
                evidence="; ".join(f"{k}: {v}" for k, v in leaks.items()),
                affected=list(leaks),
                rework_target="dataset_designer",
            ),
            confirmation,
        )

    def check_censoring(self, design: DatasetDesign, index: dict[str, Any]) -> CheckResult:
        table, _, column = design.reference_time.partition(".")
        try:
            end = self.catalog.run_readonly_sql(f"SELECT max({column}) FROM {table}")["rows"][0][0]
            n = self.catalog.run_readonly_sql(
                f"SELECT count(*) FROM ({design.sql}) t WHERE CAST({design.reference_time_alias} AS TIMESTAMP)"
                f" > CAST('{end}' AS TIMESTAMP) - INTERVAL {int(design.target.window_days)} DAY"
            )["rows"][0][0]
        except SQLExecutionError as exc:
            return CheckResult(
                name="censoring", status=CheckStatus.warn, evidence=f"절단 검사 실행 실패: {exc}"
            )
        if n:
            return CheckResult(
                name="censoring",
                status=CheckStatus.fail,
                evidence=f"데이터 마지막 시각 {end} 기준 타깃 기간 {design.target.window_days}일을 관측할 수 없는 행 {n}개 포함",
                rework_target="dataset_designer",
            )
        return CheckResult(
            name="censoring",
            status=CheckStatus.passed,
            evidence=f"모든 행이 {design.target.window_days}일 타깃 기간을 관측 가능 (데이터 끝 {end})",
        )

    def eda(self, design: DatasetDesign) -> EDASummary:
        sql, target = design.sql, design.target.name
        eda = EDASummary(row_count=design.row_count or 0, target_ratio=design.target_ratio)
        for split, n, r in self.catalog.run_readonly_sql(
            f"SELECT CAST({design.split.column} AS VARCHAR), count(*), avg(CAST({target} AS DOUBLE)) FROM ({sql}) t GROUP BY 1 ORDER BY 1"
        )["rows"]:
            eda.split_counts[str(split)] = int(n)
            eda.split_target_ratio[str(split)] = round(float(r or 0), 4)
        names = [f.name for f in design.features]
        if names:
            nulls = ", ".join(f"avg(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0 END)" for c in names)
            row = self.catalog.run_readonly_sql(f"SELECT {nulls} FROM ({sql}) t")["rows"][0]
            eda.null_ratio = {c: round(float(v or 0), 4) for c, v in zip(names, row, strict=True)}
        for f in design.features:
            if f.dtype in ("numeric", "boolean"):
                try:
                    v = self.catalog.run_readonly_sql(
                        f"SELECT corr(CAST({f.name} AS DOUBLE), CAST({target} AS DOUBLE)) FROM ({sql}) t"
                    )["rows"][0][0]
                except SQLExecutionError:
                    continue
                if v is not None:
                    eda.target_correlation[f.name] = round(float(v), 4)
        return eda

    def check_quality(self, design: DatasetDesign, eda: EDASummary) -> list[CheckResult]:
        cfg = self.deps.cfg
        checks = []
        high_null = {
            k: v for k, v in eda.null_ratio.items() if v > cfg("review.max_null_ratio", 0.3)
        }
        checks.append(
            CheckResult(
                name="quality",
                status=CheckStatus.warn if high_null else CheckStatus.passed,
                evidence=f"결측률 기준 초과: {high_null}" if high_null else "피처 결측률 기준 이내",
                affected=list(high_null),
            )
        )
        min_rows = cfg("review.min_rows", 1000)
        checks.append(
            CheckResult(
                name="row_count",
                status=CheckStatus.warn if eda.row_count < min_rows else CheckStatus.passed,
                evidence=f"{eda.row_count}행 (기준 {min_rows})",
            )
        )
        min_ratio = cfg("review.min_target_ratio", 0.02)
        ratio = eda.target_ratio or 0
        checks.append(
            CheckResult(
                name="target_ratio",
                status=CheckStatus.warn if ratio < min_ratio else CheckStatus.passed,
                evidence=f"타깃 비율 {ratio:.4f} (기준 {min_ratio}), 양성 {round(ratio * eda.row_count)}건",
            )
        )
        missing = {"train", "val", "test"} - set(eda.split_counts)
        checks.append(
            CheckResult(
                name="split",
                status=CheckStatus.fail if missing else CheckStatus.passed,
                evidence=f"분할 값 누락: {sorted(missing)}"
                if missing
                else f"분할별 행 수 {eda.split_counts}",
                rework_target="dataset_designer" if missing else None,
            )
        )
        return checks

    def check_coverage(
        self, spec: RequirementSpec, mapping: DataMapping, design: DatasetDesign
    ) -> CheckResult:
        items = [i for i in spec.requirement_items if i.kind in DATA_KINDS]
        if not items:
            return CheckResult(
                name="coverage",
                status=CheckStatus.warn,
                evidence="데이터로 반영할 요구사항 항목이 없음",
            )
        mapped = {m.requirement_id for m in mapping.items}
        in_design = {rid for f in design.features for rid in f.requirement_ids}
        covered = []
        for i in items:
            if (
                i.id in mapped
                or i.id in in_design
                or (i.kind == "target" and design.target.positive_condition)
                or (i.kind == "population" and design.population_filters)
            ):
                covered.append(i.id)
        ratio = len(covered) / len(items)
        uncovered = [i.id for i in items if i.id not in covered]
        threshold = self.deps.cfg("review.min_requirement_coverage", 0.9)
        if ratio >= threshold:
            return CheckResult(
                name="coverage",
                status=CheckStatus.passed,
                evidence=f"반영률 {ratio:.0%}",
                affected=uncovered,
            )
        return CheckResult(
            name="coverage",
            status=CheckStatus.fail,
            evidence=f"반영률 {ratio:.0%} < {threshold:.0%}, 미반영: {uncovered}",
            affected=uncovered,
            rework_target="data_explorer" if set(uncovered) - mapped else "dataset_designer",
        )

    # ---- 종합 ----

    def compile(
        self,
        design: DatasetDesign,
        checks: list[CheckResult],
        eda: EDASummary | None,
        confirmations: list[ClarificationQuestion],
    ) -> ReviewReport:
        failed = [c for c in checks if c.status == CheckStatus.fail]
        targets = sorted({c.rework_target for c in failed if c.rework_target}, key=PIPELINE.index)
        feedback: dict[str, list[str]] = {}
        for c in failed:
            if c.rework_target:
                feedback.setdefault(c.rework_target, []).append(f"[{c.name}] {c.evidence}")

        feasibility = "ok"
        if eda is not None:
            positives = (eda.target_ratio or 0) * eda.row_count
            if eda.row_count == 0 or positives < MIN_POSITIVES:
                feasibility = "infeasible"
            elif any(
                c.name in ("row_count", "target_ratio") and c.status == CheckStatus.warn
                for c in checks
            ):
                feasibility = "low"
        return ReviewReport(
            design_version=design.version,
            checks=checks,
            eda=eda,
            passed=not failed,
            rework_targets=targets,
            feedback=feedback,
            feasibility=feasibility,
            confirmations=confirmations,
        )


def default_summary(report: ReviewReport) -> str:
    lines = [f"검증 {'통과' if report.passed else '미통과'} (모델링 실익: {report.feasibility})"]
    lines += [
        f"- {c.name}: {c.status.value} — {c.evidence}"
        for c in report.checks
        if c.status != CheckStatus.passed
    ]
    return "\n".join(lines)
