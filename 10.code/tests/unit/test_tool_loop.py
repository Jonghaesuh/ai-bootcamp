"""도구 호출 루프: 모델이 bind_tools로 묶인 도구 중 하나를 고르면 ToolNode가 실행해 결과를 돌려준다."""

from langchain_core.messages import AIMessage, ToolMessage

from req2data.agents.tool_loop import build_tool_loop, retrieved_from_tools, run_tool_loop
from req2data.tools.catalog import build_catalog_tools


def call(name: str, args: dict, i: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


def test_agent_selects_and_calls_tool(llm, catalog):
    tools = build_catalog_tools(catalog)
    llm.script_react(call("describe_table", {"table": "orders"}, 1))
    msgs = run_tool_loop(llm, tools, "시스템", "orders 구조 확인", max_steps=4)
    assert llm.bound_tools == [t.name for t in tools]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_msgs] == ["describe_table"] and "order_id" in tool_msgs[0].content
    assert isinstance(msgs[-1], AIMessage) and not msgs[-1].tool_calls


def test_tool_loop_graph_has_agent_and_tool_nodes(llm, catalog):
    g = build_tool_loop(llm, build_catalog_tools(catalog), "시스템").get_graph()
    assert {"agent", "tools"} <= set(g.nodes)


def test_unknown_tool_and_tool_error_return_to_model(llm, catalog):
    llm.script_react(call("없는_도구", {}, 1), call("describe_table", {"table": "없는_테이블"}, 2))
    msgs = run_tool_loop(llm, build_catalog_tools(catalog), "시스템", "과제", max_steps=4)
    assert len([m for m in msgs if isinstance(m, ToolMessage)]) == 2
    assert isinstance(msgs[-1], AIMessage) and not msgs[-1].tool_calls


def test_loop_stops_at_step_limit(llm, catalog):
    llm.script_react(*[call("describe_table", {"table": "orders"}, i) for i in range(10)])
    msgs = run_tool_loop(llm, build_catalog_tools(catalog), "시스템", "과제", max_steps=2)
    assert len([m for m in msgs if isinstance(m, ToolMessage)]) <= 2


def test_search_knowledge_tool_and_retrieved_docs(llm, client):
    tools = build_catalog_tools(client)
    assert [t.name for t in tools] == [
        "search_catalog",
        "search_knowledge",
        "describe_table",
        "profile_column",
        "find_join_path",
        "run_readonly_sql",
    ]
    llm.script_react(call("search_knowledge", {"query": "재구매", "source": "glossary"}, 1))
    docs = retrieved_from_tools(run_tool_loop(llm, tools, "시스템", "과제", max_steps=4))
    assert docs and all(d.id.startswith("glossary:") for d in docs)
