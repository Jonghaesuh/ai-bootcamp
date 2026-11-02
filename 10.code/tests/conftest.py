from pathlib import Path

import pytest

from req2data.agents import AgentDeps
from req2data.agents.fewshot import FewShotSelector
from req2data.agents.prompts import PromptRegistry
from req2data.config import load_policy
from req2data.rag import KnowledgeBase, KnowledgePaths
from req2data.tools.local_catalog import LocalCatalogClient
from tests.fixtures.fake_catalog import FakeCatalog
from tests.fixtures.rag import HashEmbeddings, build_local_db
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


@pytest.fixture(scope="module")
def kb(tmp_path_factory) -> KnowledgeBase:
    """실제 카탈로그·용어집·가이드를 해시 임베딩으로 색인한 지식 베이스 (모듈마다 새로 만든다)."""
    data = KnowledgePaths.under(ROOT / "data")
    root = tmp_path_factory.mktemp("kb")
    paths = KnowledgePaths(
        data.catalog, root / "none.json", data.glossary, data.guides, root / "cases"
    )
    kb = KnowledgeBase(paths, HashEmbeddings(), persist_dir=root / "chroma")
    kb.rebuild()
    return kb


@pytest.fixture(scope="module")
def client(kb, tmp_path_factory) -> LocalCatalogClient:
    """실제 카탈로그 스키마 + 합성 데이터 DuckDB."""
    db = build_local_db(tmp_path_factory.mktemp("db") / "olist.duckdb", kb.spec)
    return LocalCatalogClient(db, kb, max_rows=50, timeout_seconds=5)
