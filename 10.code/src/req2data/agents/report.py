from pathlib import Path

from req2data.agents.base import BaseAgent, as_json
from req2data.graph.state import Req2DataState, load
from req2data.reporting import render
from req2data.schemas import (
    DataMapping,
    DatasetDesign,
    ReportSummary,
    RequirementSpec,
    ReviewReport,
)


def current_citations(
    records: list[dict],
    spec: RequirementSpec,
    mapping: DataMapping,
    design: DatasetDesign,
    review: ReviewReport | None,
) -> list[dict]:
    """State의 인용 기록은 재작업·수정 요청 때 버려진 시도의 것까지 쌓인다.
    최종 산출물에 남아 있는 인용만 남긴다."""
    outputs = {
        "requirement_analyst": spec,
        "data_explorer": mapping,
        "dataset_designer": design,
        "reviewer": review,
    }
    current = {(agent, c.doc_id) for agent, out in outputs.items() if out for c in out.citations}
    return [r for r in records if (r["agent"], r["doc_id"]) in current]


class ReportNode(BaseAgent):
    """Agent가 아닌 템플릿 노드. 사용자 유형별 요약만 LLM으로 쓴다."""

    name = "report"

    def __call__(self, state: Req2DataState) -> dict:
        spec = load(state, "requirement_spec", RequirementSpec)
        mapping = load(state, "data_mapping", DataMapping)
        design = load(state, "dataset_design", DatasetDesign)
        review = load(state, "review_report", ReviewReport)
        assert spec and mapping and design, "명세·매핑·설계가 모두 필요합니다."

        result = self.structured(
            self.system(state),
            self.prompt.task(
                "summarize",
                spec=as_json(spec.model_copy(update={"open_questions": []})),
                design=as_json(design.model_dump(exclude={"sql"})),
                review=as_json(review.model_dump(exclude={"eda"})) if review else "(없음)",
            ),
            ReportSummary,
            fallback=lambda: fallback_summary(design, review),
        )
        summary = result.value
        rows = render.traceability(spec, mapping, design, review)

        out = Path(self.deps.artifact_dir) / state["project_id"]
        out.mkdir(parents=True, exist_ok=True)
        files = {
            "summary.md": summary_md(summary).rstrip("\n")
            + "\n\n"
            + render.citations_md(
                current_citations(state.get("citations") or [], spec, mapping, design, review)
            ),
            "requirement_spec.md": render.spec_md(spec),
            "data_mapping.md": render.mapping_md(mapping),
            "dataset_design.md": render.design_md(design),
            "mart.sql": design.sql + "\n",
            "traceability.csv": render.traceability_csv(rows),
        }
        if review:
            files["review_report.md"] = render.review_md(review)
        artifacts = {}
        for name, content in files.items():
            path = out / name
            path.write_text(content, encoding="utf-8")
            artifacts[name] = str(path)
        return {
            "artifacts": artifacts,
            "trace": [self.result_event(result, files=list(artifacts))],
        }


def summary_md(s: ReportSummary) -> str:
    lines = [f"# {s.headline}", "", "## 핵심 내용", *[f"- {h}" for h in s.highlights]]
    lines += ["", "## 다음 할 일", *[f"- {a}" for a in s.next_actions]]
    return "\n".join(lines) + "\n"


def fallback_summary(design: DatasetDesign, review: ReviewReport | None) -> ReportSummary:
    status = "검증 통과" if review and review.passed else "검증 미통과"
    return ReportSummary(
        headline=f"{design.analysis_unit} 단위 데이터셋 설계 v{design.version} ({status})",
        highlights=[
            f"기준 시점 {design.reference_time}, 타깃: {design.target.description}",
            f"{design.row_count}행, 타깃 비율 {design.target_ratio}",
        ],
        next_actions=["검증 리포트의 경고 항목을 확인하세요."],
    )
