"""서비스 구성 요소 조립: 지식 베이스, 카탈로그, LLM, 체크포인터, 프로젝트 저장소, 그래프 실행기."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from req2data.agents import AgentDeps
from req2data.config import Settings, get_settings
from req2data.llm.recording import wrap_for_demo
from req2data.rag.knowledge_base import KnowledgeBase
from req2data.service.projects import ProjectRepository
from req2data.service.runner import GraphRunner
from req2data.tools.local_catalog import LocalCatalogClient


@dataclass
class Services:
    kb: KnowledgeBase
    catalog: LocalCatalogClient
    deps: AgentDeps
    projects: ProjectRepository
    runner: GraphRunner
    artifact_dir: Path


def assemble(
    kb: KnowledgeBase,
    catalog: LocalCatalogClient,
    deps: AgentDeps,
    projects: ProjectRepository,
    checkpointer,
) -> Services:
    return Services(
        kb=kb,
        catalog=catalog,
        deps=deps,
        projects=projects,
        runner=GraphRunner(deps, checkpointer, projects),
        artifact_dir=Path(deps.artifact_dir),
    )


def build_services(settings: Settings | None = None) -> Services:
    s = settings or get_settings()
    kb = KnowledgeBase.from_settings(s)
    catalog = LocalCatalogClient.from_settings(s, kb=kb)
    deps = wrap_for_demo(AgentDeps.from_settings(s, catalog=catalog), s)
    s.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = SqliteSaver(sqlite3.connect(str(s.checkpoint_db), check_same_thread=False))
    projects = ProjectRepository(s.project_db)
    projects.mark_interrupted()
    return assemble(kb, catalog, deps, projects, checkpointer)
