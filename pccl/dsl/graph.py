"""
Simplified Primitive IR Graph Container.
Keeps: DAG container, validation, topological sort.
Dropped: clone, visualize, summary, get_nodes_by_device.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
import uuid

from .nodes import IRNode, IRNodeVariant, OcsBarrierNode, PrimitiveOpType


@dataclass
class OcsPhase:
    """One data-execution phase followed by an optional OCS control barrier."""

    index: int
    nodes: List[IRNodeVariant]
    barrier: Optional[OcsBarrierNode] = None


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
        self._validate_ocs_barrier_boundaries()
        return True

    def has_ocs_barriers(self) -> bool:
        return any(isinstance(node, OcsBarrierNode) for node in self.nodes.values())

    def split_ocs_phases(self) -> List[OcsPhase]:
        """Split a graph at OCS barriers after validating global phase cuts."""
        if not self.has_ocs_barriers():
            return [OcsPhase(index=0, nodes=self.topological_sort())]

        self._validate_ocs_barrier_boundaries()
        phases: List[OcsPhase] = []
        current_nodes: List[IRNodeVariant] = []

        for node in self.topological_sort():
            if isinstance(node, OcsBarrierNode):
                phases.append(OcsPhase(
                    index=len(phases), nodes=current_nodes, barrier=node))
                current_nodes = []
            else:
                current_nodes.append(node)

        phases.append(OcsPhase(index=len(phases), nodes=current_nodes))
        return phases

    def _validate_ocs_barrier_boundaries(self) -> None:
        """Ensure every OCS barrier is a graph-wide phase cut.

        A host control barrier cannot safely be embedded in the current CUDA
        persistent kernel. All data nodes before it must reach it, and every
        entry node after it must wait on it. This makes phase splitting lossless.
        """
        if not self.has_ocs_barriers():
            return

        topo = self.topological_sort()
        phase = 0
        phase_by_id: Dict[str, int] = {}
        phase_node_ids: List[List[str]] = [[]]
        preceding_barrier: Optional[str] = None

        for node in topo:
            if isinstance(node, OcsBarrierNode):
                required = set(phase_node_ids[phase])
                missing = sorted(required.difference(node.dependencies))
                if missing:
                    raise ValueError(
                        f"OCS barrier '{node.op_id}' must depend on every node in phase "
                        f"{phase}; missing {missing}")
                preceding_barrier = node.op_id
                phase += 1
                phase_node_ids.append([])
                continue

            same_phase_deps = [
                dep_id for dep_id in node.dependencies
                if phase_by_id.get(dep_id) == phase
            ]
            if phase > 0 and not same_phase_deps and preceding_barrier not in node.dependencies:
                raise ValueError(
                    f"phase-{phase} entry node '{node.op_id}' must depend on OCS barrier "
                    f"'{preceding_barrier}'")

            for dep_id in node.dependencies:
                dep_phase = phase_by_id.get(dep_id)
                if dep_phase is not None and dep_phase < phase:
                    raise ValueError(
                        f"data dependency '{dep_id}' -> '{node.op_id}' crosses an OCS barrier; "
                        "depend on the barrier node instead")

            phase_by_id[node.op_id] = phase
            phase_node_ids[phase].append(node.op_id)

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
