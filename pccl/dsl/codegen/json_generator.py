"""JSON v2 Generation from Primitive IR graphs. Simplified: v2 only, uses node.to_params()."""

import json
from typing import Dict, Any
from ..graph import PrimitiveIRGraph
from .mapping import get_primitive_name, get_executor_name


class RuntimeGraphGenerator:
    def __init__(self, indent: int = 2):
        self.indent = indent

    def generate(self, graph: PrimitiveIRGraph) -> Dict[str, Any]:
        if graph.is_empty():
            raise ValueError("Cannot generate JSON from empty graph")

        # Extract tensor info from first node that has it
        tensor_info_dict = None
        for node in graph.nodes.values():
            if node.tensor_info is not None:
                tensor_info_dict = {
                    "dtype": str(node.tensor_info.dtype),
                    "shape": list(node.tensor_info.shape),
                }
                break

        try:
            topo_nodes = graph.topological_sort()
        except ValueError as e:
            raise ValueError(f"Cannot generate JSON from graph with cycles: {e}")

        node_to_index = {node.op_id: idx for idx, node in enumerate(topo_nodes)}

        # Collect unique executor names
        executors_seen = {}
        for node in topo_nodes:
            name = get_executor_name(node)
            if name not in executors_seen:
                executors_seen[name] = True

        operations = []
        for node in topo_nodes:
            dependencies = [node_to_index[dep_id] for dep_id in node.dependencies]
            next_ops = [node_to_index[nid] for nid in node.next_ops]
            op_dict = {
                "index": node_to_index[node.op_id],
                "executor": get_executor_name(node),
                "primitive": get_primitive_name(node),
                "channel": node.channel,
                "dependencies": dependencies,
                "next_ops": next_ops,
                "params": node.to_params(),
            }
            operations.append(op_dict)

        result = {"version": 2}
        if tensor_info_dict is not None:
            result["tensor_info"] = tensor_info_dict
        if graph.collective_type:
            result["collective_type"] = graph.collective_type
        result["executors"] = list(executors_seen.keys())
        result["operations"] = operations
        return result

    def generate_string(self, graph: PrimitiveIRGraph) -> str:
        return json.dumps(self.generate(graph), indent=self.indent)

    def generate_to_file(self, graph: PrimitiveIRGraph, filename: str) -> str:
        json_str = self.generate_string(graph)
        with open(filename, 'w') as f:
            f.write(json_str)
        return json_str
