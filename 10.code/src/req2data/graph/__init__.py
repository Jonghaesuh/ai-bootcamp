from req2data.graph.state import Req2DataState, new_project_state

__all__ = ["Req2DataState", "build_graph", "new_project_state"]


def __getattr__(name: str):
    # builder는 agents를 import하고 agents는 graph.state를 import하므로 지연 로딩으로 순환을 끊는다.
    if name == "build_graph":
        from req2data.graph.builder import build_graph

        return build_graph
    raise AttributeError(name)
