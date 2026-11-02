"""데모 모드: 기록한 LLM 응답으로 같은 시나리오가 LLM 없이 재현되는지."""

from dataclasses import replace
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from req2data.graph import build_graph, new_project_state
from req2data.llm.recording import RECORD_FILE, _store, wrap_for_demo
from tests.fixtures import scenario as sc


def _run(deps, thread):
    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread}}
    graph.invoke(new_project_state(thread, sc.DOCUMENT), config)
    q = graph.get_state(config).interrupts[0].value["questions"]
    graph.invoke(Command(resume={"answers": {q[0]["id"]: "90일"}}), config)
    graph.invoke(Command(resume={"answers": {}}), config)
    return graph.get_state(config).values


def test_record_then_replay_without_llm(deps, llm, tmp_path):
    llm.script("RequirementSpec", sc.SPEC_ROUND1, sc.SPEC_ROUND2)
    llm.script("DataMapping", sc.MAPPING)
    llm.script("DesignPlan", sc.DESIGN_V1, sc.DESIGN_V2)
    _store.cache_clear()

    recorded = _run(
        wrap_for_demo(
            deps, SimpleNamespace(demo_mode=False, llm_record=True, demo_recordings_dir=tmp_path)
        ),
        "p",
    )
    lines = (tmp_path / RECORD_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(llm.calls)  # LLM 호출마다 한 줄

    _store.cache_clear()
    calls_before = len(llm.calls)
    no_llm = replace(deps, llm=None, json_llm=None)
    replayed = _run(
        wrap_for_demo(
            no_llm, SimpleNamespace(demo_mode=True, llm_record=False, demo_recordings_dir=tmp_path)
        ),
        "p",
    )
    assert len(llm.calls) == calls_before  # 실제 LLM 호출 없음

    for key in ("requirement_spec", "data_mapping", "review_report"):
        assert replayed[key] == recorded[key]
    assert replayed["dataset_design"]["sql"] == recorded["dataset_design"]["sql"]
    assert [t["agent"] for t in replayed["trace"]] == [t["agent"] for t in recorded["trace"]]
