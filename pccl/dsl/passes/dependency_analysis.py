"""Dependency Analysis Pass - validates graph and establishes bidirectional edges."""

from typing import Dict, Set
from collections import defaultdict
from ..graph import PrimitiveIRGraph


class DependencyAnalysisPass:
    def __init__(self, strict: bool = True):
        self.strict = strict

    def run(self, graph: PrimitiveIRGraph) -> PrimitiveIRGraph:
        self._validate_graph(graph)
        self._establish_bidirectional_edges(graph)
        self._assign_topology_indices(graph)
        return graph

    def _validate_graph(self, graph: PrimitiveIRGraph) -> None:
        if graph.is_empty():
            if self.strict:
                raise ValueError("Cannot analyze empty graph")
            return

        for node_id, node in graph.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in graph.nodes:
                    msg = f"Node '{node_id}' has dependency on non-existent node '{dep_id}'"
                    if self.strict:
                        raise ValueError(msg)

            for next_id in node.next_ops:
                if next_id not in graph.nodes:
                    msg = f"Node '{node_id}' has next_op reference to non-existent node '{next_id}'"
                    if self.strict:
                        raise ValueError(msg)

        if graph._has_cycle():
            if self.strict:
                raise ValueError("Graph contains a cycle")

    def _establish_bidirectional_edges(self, graph: PrimitiveIRGraph) -> None:
        if graph.is_empty():
            return

        deps_to_add: Dict[str, Set[str]] = defaultdict(set)
        for node_id, node in graph.nodes.items():
            for dep_id in node.dependencies:
                deps_to_add[node_id].add(dep_id)

        for node in graph.nodes.values():
            node.next_ops.clear()

        for node_id, dep_ids in deps_to_add.items():
            for dep_id in dep_ids:
                if dep_id in graph.nodes:
                    dep_node = graph.nodes[dep_id]
                    if node_id not in dep_node.next_ops:
                        dep_node.add_next_op(node_id)

    def _assign_topology_indices(self, graph: PrimitiveIRGraph) -> None:
        if graph.is_empty():
            return
        try:
            topo_order = graph.topological_sort()
        except ValueError:
            if self.strict:
                raise
            return
        self._topological_order = {node.op_id: idx for idx, node in enumerate(topo_order)}
