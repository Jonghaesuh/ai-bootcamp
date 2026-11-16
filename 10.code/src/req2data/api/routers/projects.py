import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from req2data.api.deps import get_project, get_services
from req2data.graph.state import load
from req2data.reporting.render import case_markdown
from req2data.schemas import DatasetDesign, RequirementSpec, ReviewReport, UserType
from req2data.service import InvalidTransition, Project, ProjectBusy, Services

router = APIRouter(prefix="/projects", tags=["projects"])


class CreateProject(BaseModel):
    document: str = Field(min_length=10, description="요구사항 정의서 본문")
    user_type: UserType = UserType.practitioner
    title: str | None = None


class ResumeProject(BaseModel):
    answers: dict[str, str] | None = Field(
        default=None, description="질문 ID → 답변 (빈 값은 기본값)"
    )
    change_request: str | None = Field(default=None, description="대화형 수정 요청")


def _sse(events: Iterator[dict]) -> EventSourceResponse:
    """그래프는 동기 실행이라 스레드풀에서 돌리며 이벤트를 흘려보낸다."""

    async def gen():
        async for e in iterate_in_threadpool(events):
            yield {
                "event": e["event"],
                "data": json.dumps(e["data"], ensure_ascii=False, default=str),
            }

    return EventSourceResponse(gen(), ping=15)


def _start(fn, *args, **kwargs) -> EventSourceResponse:
    try:
        return _sse(fn(*args, **kwargs))
    except ProjectBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except InvalidTransition as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("", status_code=201)
def create_project(body: CreateProject, services: Services = Depends(get_services)) -> Project:
    return services.projects.create(body.document, body.user_type.value, body.title)


@router.get("")
def list_projects(services: Services = Depends(get_services)) -> list[Project]:
    return services.projects.list()


@router.get("/{project_id}")
def get_project_detail(
    project: Project = Depends(get_project), services: Services = Depends(get_services)
) -> dict[str, Any]:
    return {"project": project, "state": services.runner.state(project.id)}


@router.post("/{project_id}/run")
def run_project(
    project: Project = Depends(get_project), services: Services = Depends(get_services)
):
    """그래프를 시작하고 Agent 진행 상황을 SSE로 스트리밍한다 (trace / node / interrupt / done / error)."""
    return _start(services.runner.start, project)


@router.post("/{project_id}/resume")
def resume_project(
    body: ResumeProject,
    project: Project = Depends(get_project),
    services: Services = Depends(get_services),
):
    """확인 질문 답변 또는 수정 요청으로 재개하고 SSE로 스트리밍한다."""
    return _start(services.runner.resume, project, body.answers, body.change_request)


@router.get("/{project_id}/artifacts")
def list_artifacts(
    project: Project = Depends(get_project), services: Services = Depends(get_services)
) -> dict[str, str]:
    artifacts = services.runner.state(project.id).get("artifacts") or {}
    return {name: f"/projects/{project.id}/artifacts/{name}" for name in artifacts}


@router.get("/{project_id}/artifacts/{name}")
def download_artifact(
    name: str, project: Project = Depends(get_project), services: Services = Depends(get_services)
) -> FileResponse:
    artifacts = services.runner.state(project.id).get("artifacts") or {}
    if name not in artifacts:
        raise HTTPException(404, f"산출물이 없습니다: {name}")
    path = Path(artifacts[name]).resolve()
    if not path.is_relative_to(services.artifact_dir.resolve()) or not path.exists():
        raise HTTPException(404, "산출물 파일을 찾을 수 없습니다.")
    return FileResponse(path, filename=name)


@router.post("/{project_id}/approve")
def approve_project(
    project: Project = Depends(get_project), services: Services = Depends(get_services)
) -> dict[str, Any]:
    """설계를 승인해 과거 설계 사례로 저장한다 (장기 기억 → 다음 프로젝트의 RAG 검색 대상)."""
    state = services.runner.state(project.id)
    if not state.get("artifacts"):
        raise HTTPException(409, "산출물이 생성된 뒤에 승인할 수 있습니다.")
    spec = load(state, "requirement_spec", RequirementSpec)
    design = load(state, "dataset_design", DatasetDesign)
    review = load(state, "review_report", ReviewReport)
    if review is None or not review.passed:
        raise HTTPException(
            409, "검증을 통과한 설계만 승인할 수 있습니다. 대화로 수정한 뒤 다시 시도하세요."
        )
    docs = services.kb.add_case(
        f"case_{project.id}", case_markdown(project.title, spec, design, review)
    )
    services.projects.update(project.id, approved=True)
    return {"approved": True, "case_documents": [d.id for d in docs]}
