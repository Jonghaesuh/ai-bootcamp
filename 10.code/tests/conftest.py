from pathlib import Path

import pytest

from req2data.agents import AgentDeps
from req2data.agents.fewshot import FewShotSelector
from req2data.agents.prompts import PromptRegistry
from req2data.config import load_policy
from tests.fixtures.fake_catalog import FakeCatalog
from tests.fixtures.scripted_llm import ScriptedChatModel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def catalog() -> FakeCatalog:
    return FakeCatalog()


@pytest.fixture
def llm() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
def policy() -> dict:
    return load_policy(ROOT / "config.yaml")


@pytest.fixture
def deps(llm, catalog, policy, tmp_path) -> AgentDeps:
    return AgentDeps(
        llm=llm,
        json_llm=llm,
        prompts=PromptRegistry(ROOT / "prompts"),
        fewshots=FewShotSelector(
            ROOT / "prompts" / "fewshot",
            k=policy["rag"]["fewshot_k"],
            exclude_domains=tuple(policy["fewshot"]["exclude_domains"]),
        ),
        policy=policy,
        catalog=catalog,
        artifact_dir=tmp_path / "artifacts",
    )
