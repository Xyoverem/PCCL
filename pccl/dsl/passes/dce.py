"""Dead Code Elimination Pass - removes unreachable nodes preserving side effects."""

from typing import Set, List
from ..nodes import RdmaWriteNode, NotifyNode
from ..graph import PrimitiveIRGraph


class DeadCodeEliminationPass:
    def __init__(self, keep_side_effects: bool = True):
        self.keep_side_effects = keep_side_effects
        self._removed_nodes: List[str] = []

    def run(self, graph: PrimitiveIRGraph) -> PrimitiveIRGraph:
        if graph.is_empty():
            return graph
        self._removed_nodes = []

        reachable = self._mark_reachable(graph)
        to_remove = [nid for nid in graph.nodes if nid not in reachable]

        for node_id in to_remove:
            graph.remove_node(node_id)
            self._removed_nodes.append(node_id)

        if to_remove:
            graph._update_boundary_points()
        return graph

    def _mark_reachable(self, graph: PrimitiveIRGraph) -> Set[str]:
        reachable: Set[str] = set()
        # Start from exit nodes and side-effect nodes
        start_nodes = []
        for node_id, node in graph.nodes.items():
            if not node.next_ops or isinstance(node, (RdmaWriteNode, NotifyNode)):
                start_nodes.append(node_id)

        for start_id in start_nodes:
            self._reverse_dfs(start_id, reachable, graph)
        return reachable

    def _reverse_dfs(self, node_id: str, reachable: Set[str], graph: PrimitiveIRGraph) -> None:
        if node_id in reachable or node_id not in graph.nodes:
            return
        reachable.add(node_id)
        for dep_id in graph.nodes[node_id].dependencies:
            self._reverse_dfs(dep_id, reachable, graph)
