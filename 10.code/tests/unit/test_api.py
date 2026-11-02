"""FastAPI 서비스: 프로젝트 생성 → SSE 실행 → 확인 질문 → 재개 → 산출물 → 승인 → 대화형 수정."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from req2data.api.main import create_app
from req2data.service import ProjectRepository, assemble
from tests.fixtures import scenario as sc


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields.get("data", "{}"))))
    return events


@pytest.fixture
def api(deps, llm, kb, client):
    services = assemble(
        kb, client, replace(deps, catalog=client), ProjectRepository(":memory:"), InMemorySaver()
    )
    with TestClient(create_app(services)) as http:
        yield http


def _create(api) -> str:
    r = api.post("/projects", json={"document": sc.DOCUMENT, "user_type": "planner"})
    assert r.status_code == 201
    return r.json()["id"]


def test_project_lifecycle(api, llm):
    llm.script("RequirementSpec", sc.SPEC_ROUND1, sc.SPEC_ROUND2)
    llm.script("DataMapping", sc.MAPPING)
    llm.script("DesignPlan", sc.DESIGN_V1, sc.DESIGN_V2, sc.DESIGN_V2)
    llm.script(
        "ChangeScope",
        {
            "changed_fields": ["dataset_design.target.window_days"],
            "restart_from": "dataset_designer",
            "instruction": "타깃 기간 60일",
        },
    )
    pid = _create(api)

    # 1) 실행 → 확인 질문에서 멈춤
    events = parse_sse(api.post(f"/projects/{pid}/run").text)
    kinds = [e for e, _ in events]
    assert kinds[-1] == "interrupt" and "trace" in kinds and "node" in kinds
    questions = events[-1][1]["questions"]
    assert api.get(f"/projects/{pid}").json()["project"]["status"] == "waiting"

    assert api.post(f"/projects/{pid}/run").status_code == 409  # 이미 실행함
    assert api.post(f"/projects/{pid}/resume", json={"change_request": "바꿔줘"}).status_code == 409

    # 2) 답변 → 설계·검증·재작업 → 기준 시점 대안 확인
    events = parse_sse(
        api.post(f"/projects/{pid}/resume", json={"answers": {questions[0]["id"]: "90일"}}).text
    )
    assert events[-1][0] == "interrupt" and events[-1][1]["questions"][0]["asked_by"] == "reviewer"

    # 3) 기본값 유지 → 산출물
    events = parse_sse(api.post(f"/projects/{pid}/resume", json={"answers": {}}).text)
    assert events[-1][0] == "done"
    detail = api.get(f"/projects/{pid}").json()
    assert detail["project"]["status"] == "done"
    assert detail["state"]["review_report"]["passed"] and detail["state"]["interrupt"] is None

    artifacts = api.get(f"/projects/{pid}/artifacts").json()
    assert {"summary.md", "mart.sql", "traceability.csv"} <= set(artifacts)
    sql = api.get(artifacts["mart.sql"])
    assert sql.status_code == 200 and "customer_unique_id" in sql.text
    assert api.get(f"/projects/{pid}/artifacts/../../etc/passwd").status_code == 404

    # 4) 승인 → 과거 사례로 검색됨
    approved = api.post(f"/projects/{pid}/approve").json()
    assert approved["approved"] and approved["case_documents"]
    hits = api.get(
        "/catalog/search", params={"q": "첫 구매 고객 재구매 쿠폰", "source": "case"}
    ).json()
    assert hits and hits[0]["doc_id"].startswith(f"case:case_{pid}")

    # 5) 대화형 수정 → 설계부터 재실행, 승인 해제
    events = parse_sse(
        api.post(
            f"/projects/{pid}/resume", json={"change_request": "타깃 기간을 60일로 바꿔줘"}
        ).text
    )
    assert events[-1][0] == "done"
    routed = [d["next"] for e, d in events if e == "trace" and d.get("event") == "route"]
    assert "data_explorer" not in routed
    project = api.get(f"/projects/{pid}").json()["project"]
    assert project["approved"] is False and project["status"] == "done"
    assert llm.remaining() == {}


def test_validation_and_not_found(api):
    assert api.post("/projects", json={"document": "짧음"}).status_code == 422
    assert (
        api.post("/projects", json={"document": sc.DOCUMENT, "user_type": "ceo"}).status_code == 422
    )
    assert api.get("/projects/nope").status_code == 404
    pid = _create(api)
    assert api.post(f"/projects/{pid}/resume", json={"answers": {}}).status_code == 409  # 실행 전
    assert api.post(f"/projects/{pid}/approve").status_code == 409  # 산출물 전
    # 검증 미통과 설계는 사례로 저장하지 않는다 (test_project_lifecycle은 통과 후 승인)
    assert [p["id"] for p in api.get("/projects").json()][0] == pid


def test_catalog_and_admin(api):
    tables = {t["table"] for t in api.get("/catalog/tables").json()}
    assert len(tables) == 9
    orders = api.get("/catalog/tables/orders").json()
    assert {"left": "orders.customer_id", "right": "customers.customer_id"} in orders["joins"]
    assert api.get("/catalog/tables/nope").status_code == 404
    assert api.get("/catalog/glossary").json()[0]["term"] == "재구매"
    hits = api.get("/catalog/search", params={"q": "재구매", "source": "glossary", "k": 1}).json()
    assert hits[0]["doc_id"] == "glossary:재구매"
    assert api.get("/catalog/search", params={"q": "x", "source": "web"}).status_code == 422
    assert api.post("/admin/reindex").json()["indexed"]["catalog"] == 61
    assert api.get("/health").json()["status"] == "ok"


def test_llm_error_is_reported_as_event(api, deps, llm, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("LLM 서버 다운")

    monkeypatch.setattr("req2data.agents.requirement.RequirementAnalyst.__call__", boom)
    pid = _create(api)
    events = parse_sse(api.post(f"/projects/{pid}/run").text)
    assert events[-1] == ("error", {"message": "LLM 서버 다운"})
    project = api.get(f"/projects/{pid}").json()["project"]
    assert project["status"] == "error" and "LLM 서버 다운" in project["error"]


def test_failed_review_cannot_be_approved(api, llm):
    """재작업 한도를 다 써도 누수가 남으면 산출물은 만들되, 사례(장기 기억)로는 저장하지 않는다."""
    llm.script("RequirementSpec", sc.SPEC_ROUND2)
    llm.script("DataMapping", sc.MAPPING)
    llm.script("DesignPlan", sc.DESIGN_V1, sc.DESIGN_V1, sc.DESIGN_V1)  # 계속 누수
    pid = _create(api)
    events = parse_sse(api.post(f"/projects/{pid}/run").text)
    assert events[-1][0] == "interrupt"  # 누수 대안 확인 질문
    events = parse_sse(api.post(f"/projects/{pid}/resume", json={"answers": {}}).text)
    assert events[-1][0] == "done"
    state = api.get(f"/projects/{pid}").json()["state"]
    assert (
        state["rework_counts"] == {"dataset_designer": 2} and not state["review_report"]["passed"]
    )
    r = api.post(f"/projects/{pid}/approve")
    assert r.status_code == 409 and "검증을 통과한" in r.json()["detail"]


def test_running_projects_marked_interrupted_on_restart():
    repo = ProjectRepository(":memory:")
    p = repo.create(sc.DOCUMENT, "practitioner")
    repo.update(p.id, status="running")
    assert repo.mark_interrupted() == 1
    restored = repo.get(p.id)
    assert restored.status == "error" and "재개" in restored.error
    assert restored.title == "마케팅팀 요구사항 정의서"
