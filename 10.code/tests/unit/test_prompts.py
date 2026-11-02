import string

import pytest

from req2data.schemas import DatasetDesign, RequirementSpec
from tests.conftest import ROOT

AGENTS = [
    "supervisor",
    "requirement_analyst",
    "data_explorer",
    "dataset_designer",
    "reviewer",
    "report",
]


@pytest.mark.parametrize("agent", AGENTS)
def test_every_prompt_loads_and_tasks_format(deps, agent):
    prompt = deps.prompts.get(agent)
    assert prompt.persona and prompt.version >= 1
    for name, template in prompt.tasks.items():
        fields = {f[1] for f in string.Formatter().parse(template) if f[1]}
        prompt.task(name, **dict.fromkeys(fields, "x"))  # 리터럴 중괄호가 있으면 여기서 실패


def test_system_prompt_sections(deps):
    prompt = deps.prompts.get("requirement_analyst")
    practitioner = prompt.system(user_type="practitioner", feedback=["타깃 기간 누락"])
    planner = prompt.system(user_type="planner")
    assert "10년 차 비즈니스 분석가" in practitioner
    assert "추측해 채우지 말 것" in practitioner  # 금지 사항
    assert "이전 검증 피드백" in practitioner and "타깃 기간 누락" in practitioner
    assert "기술 용어" in planner and "이전 검증 피드백" not in planner


def test_fewshot_excludes_eval_domain_and_ranks_by_similarity(deps):
    chosen = deps.fewshots.select("requirement_spec", "다음 달 해지할 가입자에게 상담 전화", k=1)
    assert chosen[0]["id"] == "telecom_churn"
    for kind in ("requirement_spec", "dataset_design"):
        assert all(e["domain"] != "ecommerce" for e in deps.fewshots.examples(kind))


@pytest.mark.parametrize(
    ("kind", "schema"), [("requirement_spec", RequirementSpec), ("dataset_design", DatasetDesign)]
)
def test_fewshot_outputs_match_schema(deps, kind, schema):
    examples = deps.fewshots.examples(kind)
    assert examples
    for e in examples:
        schema.model_validate(e["output"])


def test_prompt_files_exist_for_all_agents():
    assert {p.stem for p in (ROOT / "prompts").glob("*.yaml")} >= set(AGENTS)
