"""Runtime graph generation for normal PCCL graphs and phased OCS graphs."""

import json
from typing import Any, Dict, List

from ..graph import OcsPhase, PrimitiveIRGraph
from ..nodes import IRNodeVariant
from .mapping import get_executor_name, get_primitive_name


class RuntimeGraphGenerator:
    """Generate JSON v2 device graphs or JSON v3 phased OCS manifests.

    The current C++ engine accepts JSON v2 ``operations`` only. A graph with
    an OCS barrier needs a host callback between CUDA phases, so it is emitted
    as a v3 manifest without top-level ``operations``. Passing that manifest
    to the old engine intentionally fails registration instead of silently
    running work from both sides of the barrier in one kernel launch.
    """

    def __init__(self, indent: int = 2):
        self.indent = indent

    def generate(self, graph: PrimitiveIRGraph) -> Dict[str, Any]:
        if graph.is_empty():
            raise ValueError("Cannot generate JSON from empty graph")

        graph.validate()
        if graph.has_ocs_barriers():
            return self._generate_phased_ocs(graph)
        return self._generate_runtime_v2(graph)

    def _base_fields(self, graph: PrimitiveIRGraph) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for node in graph.nodes.values():
            if node.tensor_info is not None:
                result["tensor_info"] = {
                    "dtype": str(node.tensor_info.dtype),
                    "shape": list(node.tensor_info.shape),
                }
                break
        if graph.collective_type:
            result["collective_type"] = graph.collective_type
        return result

    def _generate_operations(self, nodes: List[IRNodeVariant]) -> Dict[str, Any]:
        node_ids = {node.op_id for node in nodes}
        node_to_index = {node.op_id: index for index, node in enumerate(nodes)}
        executors_seen: Dict[str, bool] = {}
        operations = []

        for node in nodes:
            name = get_executor_name(node)
            executors_seen[name] = True
            dependencies = [
                node_to_index[dep_id] for dep_id in node.dependencies
                if dep_id in node_ids
            ]
            next_ops = [
                node_to_index[next_id] for next_id in node.next_ops
                if next_id in node_ids
            ]
            operations.append({
                "index": node_to_index[node.op_id],
                "executor": name,
                "primitive": get_primitive_name(node),
                "channel": node.channel,
                "dependencies": dependencies,
                "next_ops": next_ops,
                "params": node.to_params(),
            })

        return {
            "executors": list(executors_seen.keys()),
            "operations": operations,
        }

    def _generate_runtime_v2(self, graph: PrimitiveIRGraph) -> Dict[str, Any]:
        topo_nodes = graph.topological_sort()
        result = {"version": 2}
        result.update(self._base_fields(graph))
        result.update(self._generate_operations(topo_nodes))
        return result

    def _generate_phased_ocs(self, graph: PrimitiveIRGraph) -> Dict[str, Any]:
        phases = graph.split_ocs_phases()
        phase_json = []
        control_operations = []
        executors_seen: Dict[str, bool] = {}

        for phase in phases:
            phase_ops = self._generate_operations(phase.nodes)
            for executor in phase_ops["executors"]:
                executors_seen[executor] = True
            phase_json.append({
                "index": phase.index,
                "executors": phase_ops["executors"],
                "operations": phase_ops["operations"],
            })

            if phase.barrier is not None:
                control_operations.append({
                    "index": len(control_operations),
                    "primitive": get_primitive_name(phase.barrier),
                    "executor": get_executor_name(phase.barrier),
                    "after_phase": phase.index,
                    "before_phase": phase.index + 1,
                    "params": phase.barrier.to_params(),
                })

        result = {
            "version": 3,
            "execution_model": "phased_ocs",
            "executors": list(executors_seen.keys()),
            "phases": phase_json,
            "control_operations": control_operations,
        }
        result.update(self._base_fields(graph))
        return result

    def generate_string(self, graph: PrimitiveIRGraph) -> str:
        return json.dumps(self.generate(graph), indent=self.indent)

    def generate_to_file(self, graph: PrimitiveIRGraph, filename: str) -> str:
        json_str = self.generate_string(graph)
        with open(filename, 'w') as f:
            f.write(json_str)
        return json_str
