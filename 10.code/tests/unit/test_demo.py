"""데모 모드: 녹화 입력 기록·미리 채우기, 녹화에 없는 입력의 구분."""

from dataclasses import replace

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from req2data.api.main import create_app
from req2data.llm import invoke_structured
from req2data.llm.recording import REPLAY_MISS, RecordReplayChatModel
from req2data.schemas import RouteDecision
from req2data.service import ProjectRepository, assemble
from req2data.service.demo import DemoInputs
from tests.fixtures import scenario as sc


def _api(deps, kb, client, demo):
    services = assemble(
        kb,
        client,
        replace(deps, catalog=client),
        ProjectRepository(":memory:"),
        InMemorySaver(),
        demo,
    )
    return TestClient(create_app(services))


def test_record_mode_saves_document_and_answers(deps, llm, kb, client, tmp_path):
    llm.script("RequirementSpec", sc.SPEC_ROUND1, sc.SPEC_ROUND2)
    llm.script("DataMapping", sc.MAPPING)
    llm.script("DesignPlan", sc.DESIGN_V2)
    demo = DemoInputs(tmp_path / "demo_inputs.json", role="record")
    with _api(deps, kb, client, demo) as api:
        pid = api.post("/projects", json={"document": sc.DOCUMENT}).json()["id"]
        api.post(f"/projects/{pid}/run")
        q1 = api.get(f"/projects/{pid}").json()["state"]["interrupt"]["questions"][0]
        api.post(f"/projects/{pid}/resume", json={"answers": {q1["id"]: "60일"}})

    saved = demo.load()
    assert saved["projects"] == [{"document": sc.DOCUMENT, "user_type": "practitioner"}]
    assert saved["answers"][q1["question"]] == "60일"
    q2 = sc.SPEC_ROUND1["open_questions"][1]
    assert saved["answers"][q2["question"]] == q2["default_option"]  # 빈 답은 실제로 반영된 기본값


def test_replay_mode_prefills_recorded_answers(deps, llm, kb, client, tmp_path):
    q = sc.SPEC_ROUND1["open_questions"][0]
    (tmp_path / "demo_inputs.json").write_text(
        f'{{"projects": [{{"document": "doc", "user_type": "planner"}}], "answers": {{"{q["question"]}": "60일"}}}}',
        encoding="utf-8",
    )
    llm.script("RequirementSpec", sc.SPEC_ROUND1)
    with _api(deps, kb, client, DemoInputs(tmp_path / "demo_inputs.json", role="replay")) as api:
        assert api.get("/admin/demo").json() == {
            "demo_mode": True,
            "projects": [{"document": "doc", "user_type": "planner"}],
        }
        pid = api.post("/projects", json={"document": sc.DOCUMENT}).json()["id"]
        api.post(f"/projects/{pid}/run")
        state = api.get(f"/projects/{pid}").json()["state"]
    assert list(state["demo_answers"].values()) == ["60일"]  # 녹화에 있는 질문만


def test_demo_endpoint_off_outside_replay(deps, kb, client, tmp_path):
    with _api(deps, kb, client, DemoInputs(tmp_path / "x.json", role="record")) as api:
        assert api.get("/admin/demo").json() == {"demo_mode": False, "projects": []}


def test_replay_miss_falls_back_once_with_clear_message(tmp_path):
    model = RecordReplayChatModel(label="json", path=tmp_path / "rec.jsonl", mode="replay")
    result = invoke_structured(
        model,
        [SystemMessage(content="s"), HumanMessage(content="녹화에 없는 입력")],
        RouteDecision,
        retries=3,
        fallback=lambda: RouteDecision(next="end"),
    )
    assert result.used_fallback and result.attempts == 1  # 재시도하지 않음
    assert result.error == REPLAY_MISS


def test_dense_ranking_replays_without_embedding_server(kb, tmp_path):
    """임베딩 서버가 없어도 데모 재생 프롬프트가 녹화와 같도록, 의미 검색 순위를 기록·재생한다."""
    from req2data.rag.recording import DenseRecorder

    retriever = kb.retriever
    original = retriever.index.dense
    try:
        retriever.dense_recorder = DenseRecorder(tmp_path / "r.json", "record")
        recorded = kb.search("재구매 고객 키", k=3, source="catalog")

        def down(*args, **kwargs):
            raise ConnectionError("Ollama 없음")

        retriever.index.dense = down
        retriever.dense_recorder = DenseRecorder(tmp_path / "r.json", "replay")
        assert kb.search("재구매 고객 키", k=3, source="catalog") == recorded  # 기록으로 같은 순위
        unrecorded = kb.search("배송 지연", k=3, source="catalog")  # 기록 없음 → BM25만
        assert all(set(h["ranks"]) == {"bm25"} for h in unrecorded)
    finally:
        retriever.index.dense = original
        retriever.dense_recorder = None
