from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from req2data.agents import (
    AgentDeps,
    DataExplorer,
    DatasetDesigner,
    ReportNode,
    RequirementAnalyst,
    Reviewer,
    Supervisor,
    ask_user,
)
from req2data.graph.state import Req2DataState


def route_next(state: Req2DataState) -> str:
    """Supervisor가 State에 남긴 결정(route.next)을 다음 노드 이름으로 돌려준다."""
    return state["route"]["next"]


def build_graph(deps: AgentDeps, checkpointer: Checkpointer = None) -> CompiledStateGraph:
    """Supervisor 계층형 그래프. Supervisor에서 조건부 엣지로 갈라지고, 모든 노드는 끝나면 supervisor로 돌아온다."""
    workers = {
        "requirement_analyst": RequirementAnalyst(deps),
        "data_explorer": DataExplorer(deps),
        "dataset_designer": DatasetDesigner(deps),
        "reviewer": Reviewer(deps),
        "report": ReportNode(deps),
        "ask_user": ask_user,
    }
    g = StateGraph(Req2DataState)
    g.add_node("supervisor", Supervisor(deps))
    for name, node in workers.items():
        g.add_node(name, node)
        g.add_edge(name, "supervisor")
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor", route_next, {**{name: name for name in workers}, "end": END}
    )
    return g.compile(checkpointer=checkpointer)
