"""
Simplified Primitive IR Graph Container.
Keeps: DAG container, validation, topological sort.
Dropped: clone, visualize, summary, get_nodes_by_device.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from collections import deque
import uuid

from .nodes import IRNode, IRNodeVariant, PrimitiveOpType


@dataclass
class PrimitiveIRGraph:
    nodes: Dict[str, IRNodeVariant] = field(default_factory=dict)
    graph_name: str = "unnamed_graph"
    graph_id: str = field(default="")
    entry_points: List[str] = field(default_factory=list)
    exit_points: List[str] = field(default_factory=list)
    collective_type: str = ""

    def __post_init__(self):
        if not self.graph_id:
            self.graph_id = f"{self.graph_name}_{uuid.uuid4().hex[:8]}"

    def add_node(self, node: IRNodeVariant) -> IRNodeVariant:
        if node.op_id in self.nodes:
            raise ValueError(f"Node with id '{node.op_id}' already exists in graph")
        self.nodes[node.op_id] = node
        return node

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Node with id '{node_id}' not found in graph")
        for other_node in self.nodes.values():
            if node_id in other_node.dependencies:
                other_node.dependencies.remove(node_id)
            if node_id in other_node.next_ops:
                other_node.next_ops.remove(node_id)
        del self.nodes[node_id]

    def add_edge(self, src_id: str, dst_id: str) -> None:
        if src_id not in self.nodes:
            raise KeyError(f"Source node '{src_id}' not found in graph")
        if dst_id not in self.nodes:
            raise KeyError(f"Destination node '{dst_id}' not found in graph")
        if src_id == dst_id:
            raise ValueError(f"Cannot add self-loop on node '{src_id}'")

        src_node = self.nodes[src_id]
        dst_node = self.nodes[dst_id]

        if dst_id in src_node.next_ops:
            return

        dst_node.add_dependency(src_id)
        src_node.add_next_op(dst_id)

        if self._has_cycle():
            dst_node.dependencies.remove(src_id)
            src_node.next_ops.remove(dst_id)
            raise ValueError(f"Adding edge {src_id} -> {dst_id} would create a cycle")

    def get_node(self, node_id: str) -> IRNodeVariant:
        if node_id not in self.nodes:
            raise KeyError(f"Node with id '{node_id}' not found in graph")
        return self.nodes[node_id]

    def get_nodes_by_type(self, op_type: PrimitiveOpType) -> List[IRNodeVariant]:
        return [n for n in self.nodes.values() if n.op_type == op_type]

    def _has_cycle(self) -> bool:
        visited = set()
        rec_stack = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            for next_id in self.nodes[node_id].next_ops:
                if next_id not in visited:
                    if dfs(next_id):
                        return True
                elif next_id in rec_stack:
                    return True
            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        return False

    def validate(self) -> bool:
        if not self.nodes:
            raise ValueError("Graph is empty (no nodes)")

        for node_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    raise ValueError(f"Node '{node_id}' has dependency on non-existent node '{dep_id}'")
            for next_id in node.next_ops:
                if next_id not in self.nodes:
                    raise ValueError(f"Node '{node_id}' has next_op reference to non-existent node '{next_id}'")

        if self._has_cycle():
            raise ValueError("Graph contains a cycle")

        for node in self.nodes.values():
            if hasattr(node, 'validate'):
                node.validate()

        self._update_boundary_points()
        return True

    def _update_boundary_points(self) -> None:
        self.entry_points = [nid for nid, node in self.nodes.items() if not node.dependencies]
        self.exit_points = [nid for nid, node in self.nodes.items() if not node.next_ops]

    def topological_sort(self) -> List[IRNodeVariant]:
        in_degree = {nid: len(node.dependencies) for nid, node in self.nodes.items()}
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(self.nodes[node_id])
            for next_id in self.nodes[node_id].next_ops:
                in_degree[next_id] -= 1
                if in_degree[next_id] == 0:
                    queue.append(next_id)

        if len(result) != len(self.nodes):
            raise ValueError("Cannot perform topological sort on graph with cycles")
        return result

    def size(self) -> int:
        return len(self.nodes)

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    def __repr__(self) -> str:
        return f"PrimitiveIRGraph(name={self.graph_name}, nodes={len(self.nodes)})"
