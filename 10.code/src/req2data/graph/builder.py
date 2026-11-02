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


def build_graph(deps: AgentDeps, checkpointer: Checkpointer = None) -> CompiledStateGraph:
    """Supervisor 계층형 그래프. 모든 노드는 끝나면 supervisor로 돌아온다."""
    workers = {
        "requirement_analyst": RequirementAnalyst(deps),
        "data_explorer": DataExplorer(deps),
        "dataset_designer": DatasetDesigner(deps),
        "reviewer": Reviewer(deps),
        "report": ReportNode(deps),
        "ask_user": ask_user,
    }
    g = StateGraph(Req2DataState)
    g.add_node("supervisor", Supervisor(deps), destinations=(*workers, END))
    for name, node in workers.items():
        g.add_node(name, node)
        g.add_edge(name, "supervisor")
    g.add_edge(START, "supervisor")
    return g.compile(checkpointer=checkpointer)
