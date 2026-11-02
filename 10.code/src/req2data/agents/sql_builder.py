"""설계 계획(DesignPlan) → 코호트 마트 SQL.

로컬 7B·14B 모델이 가장 자주 틀린 부분(코호트·윈도우·CTE 범위·1:N 조인 중복·관측 기간 절단)을 코드가 만든다.
LLM은 무엇을(분석 단위·기준 시점·타깃 종류·피처) 정하고, 코드는 어떻게(SQL)를 책임진다.

생성되는 SQL 구조:
    events  기준 이벤트 테이블의 모집단 (분석 키, 이벤트 ID, 기준 시점)
    cohort  first_event면 분석 단위별 첫 이벤트, every_event면 이벤트 전체
    fN      피처마다 이벤트 ID 단위로 미리 집계한 서브쿼리 (1:N 테이블도 행이 늘지 않음)
    tgt     repeat_event: 기간 안 다음 이벤트 여부 / condition: 조건식 여부
    최종    타깃 기간을 끝까지 관측할 수 있는 행만 남기고, 기준 시점 분위수로 train/val/test 분할
"""

import math
import re
from typing import Any

from req2data.schemas import DatasetDesign, DesignPlan, FeatureSpec, SplitRule, TargetDefinition
from req2data.tools.catalog import CatalogClient, SQLExecutionError

_REF = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
_QUOTED = re.compile(r"'[^']*'")
_IDENT = re.compile(r"[^a-z0-9_]+")
NUMERIC_TYPES = ("INTEGER", "BIGINT", "SMALLINT", "DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC")
AGG_SQL = {
    "first": "any_value({c})",
    "sum": "sum({c})",
    "avg": "avg({c})",
    "min": "min({c})",
    "max": "max({c})",
    "count": "count({c})",
    "count_distinct": "count(DISTINCT {c})",
}
RESERVED = {"entity_key", "event_id", "ref_time", "target", "split", "rn", "value"}
SPLIT_QUANTILES = (0.7, 0.85)


class PlanError(ValueError):
    """계획이 카탈로그·규칙과 맞지 않음. 메시지를 그대로 LLM 수정 요청에 넣는다."""


class CohortSQLBuilder:
    def __init__(self, catalog: CatalogClient):
        self.catalog = catalog
        self._tables: dict[str, dict[str, Any]] = {}

    # ---- 카탈로그 확인 ----

    def _table(self, table: str) -> dict[str, Any]:
        if table not in self._tables:
            try:
                self._tables[table] = self.catalog.describe_table(table)
            except Exception as exc:
                raise PlanError(f"카탈로그에 없는 테이블: {table}") from exc
        return self._tables[table]

    def column(self, ref: str) -> dict[str, Any]:
        table, _, name = ref.partition(".")
        if not table or not name:
            raise PlanError(f"컬럼은 table.column 형식이어야 합니다: {ref!r}")
        col = next((c for c in self._table(table)["columns"] if c["name"] == name), None)
        if col is None:
            names = ", ".join(c["name"] for c in self._table(table)["columns"])
            raise PlanError(f"{table}에 {name} 컬럼이 없습니다. 사용 가능한 컬럼: {names}")
        return col

    def primary_key(self, table: str) -> str:
        pk = next((c["name"] for c in self._table(table)["columns"] if c.get("is_pk")), None)
        if pk is None:
            raise PlanError(f"{table}에 기본키가 없어 기준 이벤트 테이블로 쓸 수 없습니다.")
        return pk

    def expression_refs(self, expr: str) -> list[str]:
        """조건식의 table.column 참조를 검증해 돌려준다."""
        refs = list(dict.fromkeys(f"{t}.{c}" for t, c in _REF.findall(_QUOTED.sub("''", expr))))
        if not refs:
            raise PlanError(f"조건식에 table.column 형식의 컬럼이 없습니다: {expr!r}")
        for ref in refs:
            self.column(ref)
        return refs

    def expression_tables(self, expr: str) -> set[str]:
        return {ref.split(".")[0] for ref in self.expression_refs(expr)}

    def time_rank(self, ref: str) -> int | None:
        """컬럼 값이 확정되는 시점의 업무 순서 (시점 컬럼은 자기 자신, 아니면 available_at 기준)."""
        col = self.column(ref)
        if col.get("is_time"):
            return col.get("time_rank")
        avail = col.get("available_at")
        return self.column(avail).get("time_rank") if avail else None

    def observation_days(self, reference_time: str, refs: list[str]) -> int:
        """타깃 조건이 기준 시점보다 늦게 확정되는 컬럼에 의존하면, 그 지연의 99분위(일)를 돌려준다.
        예: 결제 승인 → 배송 완료는 수십 일 걸리므로, 그만큼 지나야 타깃을 판정할 수 있다."""
        ref_rank = self.time_rank(reference_time) or 0
        days = 0
        for ref in refs:
            rank = self.time_rank(ref)
            if rank is None or rank <= ref_rank:
                continue
            col = self.column(ref)
            when = ref if col.get("is_time") else col["available_at"]
            event_table = reference_time.split(".")[0]
            sql = (
                f"SELECT quantile_cont(date_diff('day', {reference_time}, {when}), 0.99) "
                f"FROM {self.join_from(event_table, {when.split('.')[0]})} "
                f"WHERE {when} IS NOT NULL AND {when} >= {reference_time}"
            )
            try:
                value = self.catalog.run_readonly_sql(sql)["rows"][0][0]
            except SQLExecutionError as exc:
                raise PlanError(f"타깃 관측 기간 계산 실패: {exc}") from exc
            if value is not None:
                days = max(days, math.ceil(float(value)))
        return days

    def join_from(self, base: str, tables: set[str]) -> str:
        """base 테이블에서 FK 경로로 tables를 잇는 FROM 절."""
        needed = set(tables) - {base}
        if not needed:
            return base
        edges = self.catalog.find_join_path([base, *sorted(needed)])
        joined, clauses = {base}, [base]
        pending = list(edges)
        while pending:
            progressed = False
            for edge in list(pending):
                lt, rt = edge["left"].split(".")[0], edge["right"].split(".")[0]
                new = rt if lt in joined else lt if rt in joined else None
                if new is None:
                    continue
                pending.remove(edge)
                progressed = True
                if new not in joined:
                    clauses.append(f"JOIN {new} ON {edge['left']} = {edge['right']}")
                    joined.add(new)
            if not progressed:
                break
        missing = needed - joined
        if missing:
            raise PlanError(
                f"{base}에서 {', '.join(sorted(missing))}(으)로 가는 조인 경로가 없습니다."
            )
        return "\n    ".join(clauses)

    # ---- SQL 생성 ----

    def build(self, plan: DesignPlan) -> DatasetDesign:
        ref_col = self.column(plan.reference_time)
        if not ref_col.get("is_time"):
            raise PlanError(f"기준 시점 {plan.reference_time}은 시점(TIMESTAMP) 컬럼이어야 합니다.")
        self.column(plan.entity_key)
        event_table = plan.reference_time.split(".")[0]
        key_table = plan.entity_key.split(".")[0]
        event_id = f"{event_table}.{self.primary_key(event_table)}"
        window = int(plan.target.window_days)
        if window <= 0:
            raise PlanError("타깃 기간(window_days)은 1일 이상이어야 합니다.")
        notes: list[str] = []

        filter_tables: set[str] = set()
        for f in plan.population_filters:
            filter_tables |= self.expression_tables(f)
        where = " AND ".join(f"({f})" for f in plan.population_filters) or "TRUE"
        ctes = [
            (
                "events",
                f"SELECT DISTINCT {plan.entity_key} AS entity_key, {event_id} AS event_id, "
                f"{plan.reference_time} AS ref_time\n  FROM {self.join_from(event_table, {key_table} | filter_tables)}\n"
                f"  WHERE {where} AND {plan.reference_time} IS NOT NULL",
            ),
            (
                "cohort",
                "SELECT entity_key, event_id, ref_time FROM (\n"
                "    SELECT *, row_number() OVER (PARTITION BY entity_key ORDER BY ref_time, event_id) AS rn\n"
                "    FROM events) WHERE rn = 1"
                if plan.cohort == "first_event"
                else "SELECT entity_key, event_id, ref_time FROM events",
            ),
        ]

        key_alias = _ident(plan.entity_key.split(".")[1])
        used = {key_alias} | RESERVED
        features: list[FeatureSpec] = []
        select, joins = [], []
        for i, f in enumerate(plan.features):
            col = self.column(f.source_column)
            agg = f.agg
            if agg in ("sum", "avg") and not str(col.get("type", "")).upper().startswith(
                NUMERIC_TYPES
            ):
                notes.append(f"{f.name}: 숫자가 아닌 컬럼이라 {agg} 대신 first로 집계")
                agg = "first"
            name = _unique(_ident(f.name) or f"feature_{i}", used)
            table = f.source_column.split(".")[0]
            ctes.append(
                (
                    f"f{i}",
                    f"SELECT {event_id} AS event_id, {AGG_SQL[agg].format(c=f.source_column)} AS value\n"
                    f"  FROM {self.join_from(event_table, {table})}\n  GROUP BY 1",
                )
            )
            select.append(f"f{i}.value AS {name}")
            joins.append(f"LEFT JOIN f{i} ON f{i}.event_id = c.event_id")
            features.append(
                FeatureSpec(
                    name=name,
                    description=f.description or f"{agg}({f.source_column})",
                    source_columns=[f.source_column],
                    dtype=_dtype(col, agg),
                    requirement_ids=f.requirement_ids,
                )
            )

        if plan.target.kind == "repeat_event":
            ctes.append(
                (
                    "tgt",
                    "SELECT c.event_id, max(CASE WHEN e.ref_time > c.ref_time "
                    f"AND e.ref_time <= c.ref_time + INTERVAL {window} DAY THEN 1 ELSE 0 END) AS target\n"
                    "  FROM cohort c LEFT JOIN events e ON e.entity_key = c.entity_key AND e.event_id <> c.event_id\n"
                    "  GROUP BY 1",
                )
            )
            condition = (
                f"기준 시점 이후 {window}일 안에 같은 {plan.analysis_unit}의 다음 이벤트 발생"
            )
        else:
            if not plan.target.condition:
                raise PlanError("target.kind=condition이면 target.condition 조건식이 필요합니다.")
            cond_refs = self.expression_refs(plan.target.condition)
            needed = self.observation_days(plan.reference_time, cond_refs)
            if needed > window:
                notes.append(
                    f"타깃 조건 컬럼이 기준 시점 뒤 최대 {needed}일(99분위)에 확정되어 타깃 기간을 {window}일 → {needed}일로 늘림"
                )
                window = needed
            observable = " AND ".join(f"{r} IS NOT NULL" for r in cond_refs)
            ctes.append(
                (
                    "tgt",
                    f"SELECT {event_id} AS event_id, max(CASE WHEN {plan.target.condition} THEN 1 ELSE 0 END) AS target,\n"
                    f"    max(CASE WHEN {observable} THEN 1 ELSE 0 END) AS observable\n"
                    f"  FROM {self.join_from(event_table, {r.split('.')[0] for r in cond_refs})}\n  GROUP BY 1",
                )
            )
            condition = plan.target.condition
            notes.append(f"타깃 조건 컬럼({', '.join(cond_refs)})이 비어 판정할 수 없는 행은 제외")

        data_end = f"(SELECT max({plan.reference_time}) FROM {event_table})"
        # 분석 키가 이벤트 ID와 다르면(예: 환자 단위 키 + 입원마다 한 행) 행을 식별할 이벤트 ID도 낸다
        event_col = [] if plan.entity_key == event_id else ["c.event_id AS event_id"]
        body_select = ", ".join(
            [
                f"c.entity_key AS {key_alias}",
                *event_col,
                "c.ref_time",
                *select,
                "coalesce(tgt.target, 0) AS target",
            ]
        )
        tail = (
            "FROM cohort c\nLEFT JOIN tgt ON tgt.event_id = c.event_id\n"
            + "".join(f"{j}\n" for j in joins)
            + f"WHERE c.ref_time <= {data_end} - INTERVAL {window} DAY"
            + (" AND tgt.observable = 1" if plan.target.kind == "condition" else "")
        )
        with_clause = "WITH " + ",\n".join(f"{n} AS (\n  {sql}\n)" for n, sql in ctes)

        train_end, val_end = self._split_bounds(f"{with_clause}\nSELECT {body_select}\n{tail}")
        split = (
            f"CASE WHEN c.ref_time < TIMESTAMP '{train_end}' THEN 'train' "
            f"WHEN c.ref_time < TIMESTAMP '{val_end}' THEN 'val' ELSE 'test' END AS split"
        )
        sql = f"{with_clause}\nSELECT {body_select},\n  {split}\n{tail}"

        filters = [*plan.population_filters]
        if plan.cohort == "first_event":
            filters.append(f"{plan.analysis_unit}별 첫 이벤트")
        filters.append(
            f"데이터 마지막 시각에서 {window}일 이전까지의 기준 시점만 (관측 기간 절단 방지)"
        )
        rationale = plan.rationale + ("\n" + "\n".join(notes) if notes else "")
        return DatasetDesign(
            analysis_unit=plan.analysis_unit,
            entity_key=key_alias,
            reference_time=plan.reference_time,
            reference_time_alias="ref_time",
            target=TargetDefinition(
                description=plan.target.description,
                window_days=window,
                positive_condition=condition,
            ),
            features=features,
            population_filters=filters,
            split=SplitRule(
                method="out_of_time",
                train_end=train_end,
                val_end=val_end,
                description="기준 시점 70%·85% 분위 기준 시간 순 분할",
            ),
            sql=sql,
            rationale=rationale.strip(),
        )

    def _split_bounds(self, body_sql: str) -> tuple[str, str]:
        q1, q2 = SPLIT_QUANTILES
        try:
            row = self.catalog.run_readonly_sql(
                f"SELECT quantile_disc(ref_time, {q1}), quantile_disc(ref_time, {q2}), count(*) FROM ({body_sql}) t"
            )["rows"][0]
        except SQLExecutionError as exc:
            raise PlanError(f"생성한 SQL 실행 실패: {exc}") from exc
        if not row[2]:
            raise PlanError("모집단이 비었습니다. 모집단 조건과 타깃 기간을 확인하세요.")
        return str(row[0]), str(row[1])


def _ident(text: str) -> str:
    return _IDENT.sub("_", text.lower()).strip("_")


def _unique(name: str, used: set[str]) -> str:
    base, i = name, 2
    while name in used:
        name, i = f"{base}_{i}", i + 1
    used.add(name)
    return name


def _dtype(col: dict[str, Any], agg: str) -> str:
    if agg in ("count", "count_distinct"):
        return "numeric"
    typ = str(col.get("type", "")).upper()
    if typ.startswith(NUMERIC_TYPES):
        return "numeric"
    if typ.startswith("TIMESTAMP") or typ == "DATE":
        return "datetime"
    if typ.startswith("BOOL"):
        return "boolean"
    return "categorical"
