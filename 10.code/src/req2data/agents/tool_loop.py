"""도구 호출 루프 (LangGraph 하위 그래프).

agent 노드가 `bind_tools`로 묶인 도구 가운데 필요한 것을 고르면 tools 노드(ToolNode)가 실행하고,
결과를 다시 agent에 돌려준다. 도구 호출이 없는 응답이 나오면 끝난다.
"""

import json
import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from req2data.tools.catalog import hit_to_document

log = logging.getLogger(__name__)

SEARCH_TOOLS = ("search_catalog", "search_knowledge")


def build_tool_loop(llm: BaseChatModel, tools: list[BaseTool], system: str) -> CompiledStateGraph:
    model = llm.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        return {"messages": [model.invoke([SystemMessage(content=system), *state["messages"]])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools, handle_tool_errors=True))  # 도구 오류는 모델에 돌려준다
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)  # 도구 호출이 있으면 tools, 없으면 종료
    g.add_edge("tools", "agent")
    return g.compile()


def run_tool_loop(
    llm: BaseChatModel, tools: list[BaseTool], system: str, task: str, *, max_steps: int
) -> list[BaseMessage]:
    """루프를 실행해 주고받은 메시지를 돌려준다. 단계 한도에 닿으면 그때까지의 메시지로 끝낸다."""
    graph = build_tool_loop(llm, tools, system)
    messages: list[BaseMessage] = []
    try:
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=task)]},
            {"recursion_limit": max_steps * 2 + 1},
            stream_mode="values",
        ):
            messages = chunk["messages"]
    except GraphRecursionError:
        log.info("도구 호출 단계 한도(%d) 도달", max_steps)
    return messages


def retrieved_from_tools(messages: list[BaseMessage]) -> list[Document]:
    """검색 도구가 돌려준 문서. 인용 검증의 '실제로 검색된 문서' 목록에 더한다."""
    docs: list[Document] = []
    for m in messages:
        if not (isinstance(m, ToolMessage) and m.name in SEARCH_TOOLS):
            continue
        try:
            hits = json.loads(str(m.content))
        except ValueError:
            continue  # 도구 오류 메시지
        if isinstance(hits, list):
            docs += [hit_to_document(h) for h in hits if isinstance(h, dict) and h.get("doc_id")]
    return docs
