"""산출물 Markdown/CSV 템플릿."""

import csv
import io

from req2data.schemas import (
    CheckStatus,
    DataMapping,
    DatasetDesign,
    RequirementSpec,
    ReviewReport,
    TraceabilityRow,
)

FIELD_LABELS = {
    "business_goal": "비즈니스 목표",
    "prediction_target": "예측 대상",
    "analysis_unit": "분석 단위",
    "prediction_time": "예측 시점",
    "target_definition": "타깃 정의",
    "success_metric": "성공 지표",
}


def traceability(
    spec: RequirementSpec, mapping: DataMapping, design: DatasetDesign, review: ReviewReport | None
) -> list[TraceabilityRow]:
    uncovered: set[str] = set()
    if review:
        for c in review.checks:
            if c.name == "coverage":
                uncovered = set(c.affected)
    rows = []
    for item in spec.requirement_items:
        columns = sorted(
            {c.ref for m in mapping.items if m.requirement_id == item.id for c in m.columns}
        )
        features = [f.name for f in design.features if item.id in f.requirement_ids]
        sql_ref = ", ".join(features)
        if item.kind == "target":
            sql_ref = design.target.name
        rows.append(
            TraceabilityRow(
                requirement_id=item.id,
                requirement=item.text,
                columns=columns,
                features=features,
                sql_ref=sql_ref,
                check_status="미반영" if item.id in uncovered else "반영",
            )
        )
    return rows


def spec_md(spec: RequirementSpec) -> str:
    lines = ["# 요구사항 명세", "", "| 항목 | 내용 |", "| --- | --- |"]
    lines += [f"| {label} | {getattr(spec, f) or '(미정)'} |" for f, label in FIELD_LABELS.items()]
    lines += ["", "## 제약", *[f"- {c}" for c in spec.constraints or ["(없음)"]]]
    lines += ["", "## 요구사항 항목", "| ID | 종류 | 내용 |", "| --- | --- | --- |"]
    lines += [f"| {i.id} | {i.kind} | {i.text} |" for i in spec.requirement_items]
    return "\n".join(lines) + "\n"


def mapping_md(mapping: DataMapping) -> str:
    lines = ["# 데이터 매핑", ""]
    if mapping.analysis_key:
        lines.append(f"- 분석 키: `{mapping.analysis_key.ref}`")
    if mapping.reference_time:
        lines.append(f"- 기준 시점 후보: `{mapping.reference_time.ref}`")
    lines += ["", "| 요구사항 | 역할 | 컬럼 | 근거 | 메모 |", "| --- | --- | --- | --- | --- |"]
    for m in mapping.items:
        cols = ", ".join(f"`{c.ref}`" for c in m.columns)
        lines.append(
            f"| {m.requirement_id} | {m.role} | {cols} | {', '.join(m.evidence)} | {m.note} |"
        )
    if mapping.join_paths:
        lines += [
            "",
            "## 조인 경로",
            *[f"- `{j.left.ref}` = `{j.right.ref}` ({j.how})" for j in mapping.join_paths],
        ]
    if mapping.unmapped_requirements:
        lines += ["", f"## 미매핑 요구사항\n{', '.join(mapping.unmapped_requirements)}"]
    if mapping.proxy_suggestions:
        lines += ["", "## 대체 데이터 제안", *[f"- {p}" for p in mapping.proxy_suggestions]]
    return "\n".join(lines) + "\n"


def design_md(design: DatasetDesign) -> str:
    t = design.target
    lines = [
        f"# 데이터셋 설계 (v{design.version})",
        "",
        f"- 분석 단위: {design.analysis_unit} (`{design.entity_key}`)",
        f"- 기준 시점: `{design.reference_time}` → `{design.reference_time_alias}`",
        f"- 타깃: {t.description} ({t.window_days}일, 조건: {t.positive_condition})",
        f"- 행 수: {design.row_count}, 타깃 비율: {design.target_ratio}",
        f"- 분할: {design.split.method} (train < {design.split.train_end} ≤ val < {design.split.val_end} ≤ test)",
        "",
        "## 모집단 필터",
        *[f"- {p}" for p in design.population_filters or ["(없음)"]],
        "",
        "## 피처",
        "| 이름 | 타입 | 원천 | 설명 |",
        "| --- | --- | --- | --- |",
        *[
            f"| {f.name} | {f.dtype} | {', '.join(f.source_columns)} | {f.description} |"
            for f in design.features
        ],
        "",
        "## 설계 근거",
        design.rationale,
        "",
        "## SQL",
        "```sql",
        design.sql,
        "```",
    ]
    return "\n".join(lines) + "\n"


def review_md(review: ReviewReport) -> str:
    icon = {CheckStatus.passed: "통과", CheckStatus.warn: "경고", CheckStatus.fail: "미통과"}
    lines = [
        f"# 검증 리포트 (설계 v{review.design_version})",
        "",
        f"- 결과: {'통과' if review.passed else '미통과'} / 모델링 실익: {review.feasibility}",
        "",
        "| 검사 | 판정 | 근거 |",
        "| --- | --- | --- |",
        *[f"| {c.name} | {icon[c.status]} | {c.evidence} |" for c in review.checks],
        "",
        "## 요약",
        review.summary,
    ]
    if review.eda:
        e = review.eda
        lines += [
            "",
            "## EDA",
            f"- 행 수 {e.row_count}, 타깃 비율 {e.target_ratio}",
            f"- 분할별 행 수 {e.split_counts}, 분할별 타깃 비율 {e.split_target_ratio}",
            f"- 결측률 {e.null_ratio}",
            f"- 타깃 상관 {e.target_correlation}",
        ]
    return "\n".join(lines) + "\n"


def traceability_csv(rows: list[TraceabilityRow]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["requirement_id", "requirement", "columns", "features", "sql_ref", "status"])
    for r in rows:
        w.writerow(
            [
                r.requirement_id,
                r.requirement,
                "; ".join(r.columns),
                "; ".join(r.features),
                r.sql_ref,
                r.check_status,
            ]
        )
    return buf.getvalue()
