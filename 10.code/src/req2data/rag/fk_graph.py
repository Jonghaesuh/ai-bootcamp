"""FK 그래프 조인 경로. 검색된 테이블들을 최소한의 조인으로 잇는다 (중간 테이블 포함)."""

import networkx as nx
from networkx.algorithms.approximation import steiner_tree

from req2data.rag.sources import CatalogSpec


class FKGraph:
    def __init__(self, spec: CatalogSpec):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(spec.tables)
        for left, right in spec.foreign_keys():
            lt, rt = left.split(".")[0], right.split(".")[0]
            target = spec.column(right)
            # PK가 아닌 컬럼을 가리키는 FK(예: 우편번호)는 N:M 조인이라 다른 경로가 있으면 피한다.
            weight = 1 if target and target.pk else 5
            self.graph.add_edge(lt, rt, left=left, right=right, weight=weight)

    def join_path(self, tables: list[str]) -> list[dict[str, str]]:
        """[{"left": "orders.customer_id", "right": "customers.customer_id"}, ...]"""
        terminals = [t for t in dict.fromkeys(tables) if t in self.graph]
        if len(terminals) < 2:
            return []
        joins = []
        for component in nx.connected_components(self.graph):
            group = [t for t in terminals if t in component]
            if len(group) < 2:
                continue
            tree = steiner_tree(self.graph.subgraph(component), group, weight="weight")
            for a, b in sorted(tree.edges()):
                edge = self.graph.edges[a, b]
                joins.append({"left": edge["left"], "right": edge["right"]})
        return joins

    def via_tables(self, tables: list[str]) -> list[str]:
        """조인 경로에 필요하지만 입력에 없던 중간 테이블."""
        used = {ref.split(".")[0] for j in self.join_path(tables) for ref in j.values()}
        return sorted(used - set(tables))
