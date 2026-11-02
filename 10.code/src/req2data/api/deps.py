from fastapi import HTTPException, Request

from req2data.service import Project, Services


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_project(project_id: str, request: Request) -> Project:
    project = get_services(request).projects.get(project_id)
    if project is None:
        raise HTTPException(404, f"프로젝트가 없습니다: {project_id}")
    return project
