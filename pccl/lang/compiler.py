from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

from .config import OperatorConfig, TopologyConfig, AllreduceConfig, BroadcastConfig, AllgatherConfig, ReduceScatterConfig
# CollectiveIR is now defined in the IR system
from ..ir.primitive_ir import *
from .topology import TopologyBuilder

@dataclass
class ExecutionPlan:
    """Compiled execution plan for communication operators"""
    operations: List[Dict[str, Any]]
    topology: Dict[str, Any]
    dependencies: List[Tuple[int, int]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operations': self.operations,
            'topology': self.topology,
            'dependencies': [[src, dst] for src, dst in self.dependencies],
            'metadata': self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

class DSLCompiler:
    """Compiles declarative configurations to C++ execution plans"""

    def __init__(self):
        self.topology_builder = TopologyBuilder()
        self._op_counter = 0

    def compile(self, config: OperatorConfig, participants: Optional[List[int]] = None) -> ExecutionPlan:
        """Compile a single operator configuration to execution plan"""
        if not config.validate():
            raise ValueError(f"Invalid configuration: {config}")

        if participants is None:
            participants = getattr(config, 'participants', [])

        operations = []
        dependencies = []

        if isinstance(config, AllreduceConfig):
            op_data = self._compile_allreduce(config, participants)
            operations.append(op_data)
        elif isinstance(config, BroadcastConfig):
            op_data = self._compile_broadcast(config, participants)
            operations.append(op_data)
        elif isinstance(config, AllgatherConfig):
            op_data = self._compile_allgather(config, participants)
            operations.append(op_data)
        elif isinstance(config, ReduceScatterConfig):
            op_data = self._compile_reduce_scatter(config, participants)
            operations.append(op_data)
        else:
            raise ValueError(f"Unsupported operator type: {type(config)}")

        topology_data = self._compile_topology(config, participants)

        metadata = {
            'buffer_size': config.buffer_size,
            'enable_overlap': config.enable_overlap,
            'pipeline_depth': config.pipeline_depth,
            'device_type': config.device_type.name
        }

        return ExecutionPlan(
            operations=operations,
            topology=topology_data,
            dependencies=dependencies,
            metadata=metadata
        )

    def compile_multiple(self, configs: List[OperatorConfig],
                        participants: Optional[List[int]] = None) -> ExecutionPlan:
        """Compile multiple operator configurations with dependencies"""
        all_operations = []
        all_dependencies = []
        combined_topology = None

        op_id_map = {}

        for i, config in enumerate(configs):
            if participants is None:
                current_participants = getattr(config, 'participants', [])
            else:
                current_participants = participants

            plan = self.compile(config, current_participants)

            for j, op in enumerate(plan.operations):
                op_id = self._next_op_id()
                op['id'] = op_id
                all_operations.append(op)
                op_id_map[(i, j)] = op_id

            if combined_topology is None:
                combined_topology = plan.topology
            else:
                combined_topology = self._merge_topologies(combined_topology, plan.topology)

            for src_idx, dst_idx in plan.dependencies:
                all_dependencies.append((src_idx, dst_idx))

        return ExecutionPlan(
            operations=all_operations,
            topology=combined_topology or {},
            dependencies=all_dependencies,
            metadata={'compiled_from': 'multiple_configs'}
        )

    def optimize(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Apply optimizations to the execution plan"""
        optimized_ops = self._apply_topology_optimizations(plan.operations, plan.topology)

        if plan.metadata.get('enable_overlap', False):
            optimized_ops = self._apply_overlap_optimizations(optimized_ops)

        optimized_ops = self._apply_operator_fusion(optimized_ops)

        return ExecutionPlan(
            operations=optimized_ops,
            topology=plan.topology,
            dependencies=plan.dependencies,
            metadata={**plan.metadata, 'optimized': True}
        )

    def estimate_cost(self, plan: ExecutionPlan) -> Dict[str, float]:
        """Estimate execution cost of the plan"""
        total_cost = 0.0
        bandwidth_cost = 0.0
        latency_cost = 0.0

        topology_info = plan.topology

        for op in plan.operations:
            op_cost = self._estimate_operation_cost(op, topology_info)
            total_cost += op_cost
            bandwidth_cost += op_cost * 0.7
            latency_cost += op_cost * 0.3

        return {
            'total_cost_ms': total_cost,
            'bandwidth_cost_ms': bandwidth_cost,
            'latency_cost_ms': latency_cost,
            'estimated_bandwidth_utilization': min(bandwidth_cost / total_cost * 100, 100.0) if total_cost > 0 else 0.0
        }

    def _compile_allreduce(self, config: AllreduceConfig, participants: List[int]) -> Dict[str, Any]:
        """Compile allreduce operation"""
        op_data = {
            'type': 'allreduce',
            'reduce_op': config.reduce_op.name.lower(),
            'algorithm': config.algorithm.name.lower(),
            'participants': participants,
            'buffer_size': config.buffer_size,
            'enable_overlap': config.enable_overlap,
            'pipeline_depth': config.pipeline_depth
        }

        if config.topology:
            op_data['topology'] = self._compile_topology_config(config.topology)

        return op_data

    def _compile_broadcast(self, config: BroadcastConfig, participants: List[int]) -> Dict[str, Any]:
        """Compile broadcast operation"""
        return {
            'type': 'broadcast',
            'root_rank': config.root_rank,
            'participants': participants,
            'buffer_size': config.buffer_size,
            'enable_overlap': config.enable_overlap
        }

    def _compile_allgather(self, config: AllgatherConfig, participants: List[int]) -> Dict[str, Any]:
        """Compile allgather operation"""
        return {
            'type': 'allgather',
            'participants': participants,
            'input_size': config.input_size,
            'buffer_size': config.buffer_size,
            'enable_overlap': config.enable_overlap
        }

    def _compile_reduce_scatter(self, config: ReduceScatterConfig, participants: List[int]) -> Dict[str, Any]:
        """Compile reduce-scatter operation"""
        return {
            'type': 'reduce_scatter',
            'reduce_op': config.reduce_op.name.lower(),
            'participants': participants,
            'input_size': config.input_size,
            'output_size': config.output_size,
            'buffer_size': config.buffer_size,
            'enable_overlap': config.enable_overlap
        }

    def _compile_topology(self, config: OperatorConfig, participants: List[int]) -> Dict[str, Any]:
        """Compile topology configuration"""
        if config.topology:
            return self._compile_topology_config(config.topology, participants)
        else:
            return self._infer_optimal_topology(config, participants)

    def _compile_topology_config(self, topology_config: TopologyConfig,
                                participants: Optional[List[int]] = None) -> Dict[str, Any]:
        """Compile specific topology configuration"""
        result = {
            'type': topology_config.topology_type.name.lower(),
            'interconnect': topology_config.interconnect_type.name.lower(),
            'bandwidth': topology_config.bandwidth,
            'latency': topology_config.latency
        }

        if hasattr(topology_config, 'branching_factor'):
            result['branching_factor'] = topology_config.branching_factor

        if hasattr(topology_config, 'node_size'):
            result['node_size'] = topology_config.node_size

        if hasattr(topology_config, 'intra_interconnect'):
            result['intra_interconnect'] = topology_config.intra_interconnect.name.lower()

        if hasattr(topology_config, 'inter_interconnect'):
            result['inter_interconnect'] = topology_config.inter_interconnect.name.lower()

        if participants:
            result['participants'] = participants

        return result

    def _infer_optimal_topology(self, config: OperatorConfig, participants: List[int]) -> Dict[str, Any]:
        """Infer optimal topology based on configuration and participants"""
        num_participants = len(participants)

        if num_participants <= 2:
            topology_type = 'fully_connected'
        elif num_participants <= 8:
            topology_type = 'ring'
        elif num_participants <= 16:
            topology_type = 'tree'
        else:
            topology_type = 'hierarchical'

        return {
            'type': topology_type,
            'interconnect': 'pcie',
            'bandwidth': 10.0,
            'latency': 1.0,
            'participants': participants
        }

    def _apply_topology_optimizations(self, operations: List[Dict[str, Any]],
                                    topology: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply topology-aware optimizations"""
        optimized_ops = operations.copy()

        topology_type = topology.get('type', 'ring')
        bandwidth = topology.get('bandwidth', 10.0)

        if topology_type == 'hierarchical' and bandwidth > 25.0:
            for op in optimized_ops:
                if op['type'] == 'allreduce':
                    if op.get('algorithm') == 'ring':
                        op['algorithm'] = 'rabenseifner'

        elif topology_type == 'ring' and bandwidth > 50.0:
            for op in optimized_ops:
                if op['type'] == 'allreduce':
                    op['enable_overlap'] = True
                    op['pipeline_depth'] = max(op.get('pipeline_depth', 2), 4)

        return optimized_ops

    def _apply_overlap_optimizations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply compute-communication overlap optimizations"""
        optimized_ops = []

        for i, op in enumerate(operations):
            optimized_op = op.copy()

            if op['type'] in ['allreduce', 'reduce_scatter']:
                optimized_op['enable_overlap'] = True
                optimized_op['pipeline_depth'] = max(op.get('pipeline_depth', 2), 4)

                if op['type'] == 'allreduce':
                    optimized_op['compute_chunks'] = 4
                    optimized_op['communication_chunks'] = 2

            optimized_ops.append(optimized_op)

        return optimized_ops

    def _apply_operator_fusion(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply operator fusion optimizations"""
        if len(operations) <= 1:
            return operations

        fused_ops = []
        i = 0

        while i < len(operations):
            current_op = operations[i]

            if i + 1 < len(operations):
                next_op = operations[i + 1]

                if (current_op['type'] == 'allreduce' and
                    next_op['type'] == 'allreduce' and
                    current_op['participants'] == next_op['participants']):

                    fused_op = {
                        'type': 'fused_allreduce',
                        'operations': [current_op, next_op],
                        'participants': current_op['participants'],
                        'enable_overlap': True,
                        'pipeline_depth': max(current_op.get('pipeline_depth', 2),
                                             next_op.get('pipeline_depth', 2))
                    }
                    fused_ops.append(fused_op)
                    i += 2
                    continue

            fused_ops.append(current_op)
            i += 1

        return fused_ops

    def _estimate_operation_cost(self, op: Dict[str, Any], topology: Dict[str, Any]) -> float:
        """Estimate execution cost of a single operation"""
        bandwidth = topology.get('bandwidth', 10.0)
        latency = topology.get('latency', 1.0)
        num_participants = len(op.get('participants', [0, 1]))
        buffer_size = op.get('buffer_size', 128 * 1024 * 1024)

        if op['type'] == 'allreduce':
            algorithm = op.get('algorithm', 'ring')
            if algorithm == 'ring':
                comm_steps = 2 * (num_participants - 1)
                data_per_step = buffer_size / num_participants
            elif algorithm == 'tree':
                tree_depth = int((num_participants - 1) ** 0.5) + 1
                comm_steps = 2 * tree_depth
                data_per_step = buffer_size
            else:
                comm_steps = num_participants
                data_per_step = buffer_size / num_participants

            data_time = (comm_steps * data_per_step) / (bandwidth * 1024 * 1024 * 1024) * 1000
            latency_time = comm_steps * latency
            return data_time + latency_time

        elif op['type'] == 'broadcast':
            tree_depth = int((num_participants - 1) ** 0.5) + 1
            data_time = buffer_size / (bandwidth * 1024 * 1024 * 1024) * 1000
            latency_time = tree_depth * latency
            return data_time + latency_time

        elif op['type'] == 'allgather':
            data_time = (num_participants - 1) * buffer_size / (bandwidth * 1024 * 1024 * 1024) * 1000
            latency_time = (num_participants - 1) * latency
            return data_time + latency_time

        return 100.0

    def _merge_topologies(self, topo1: Dict[str, Any], topo2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two topology configurations"""
        merged = topo1.copy()

        if topo2.get('type') == topo1.get('type'):
            merged['bandwidth'] = min(topo1.get('bandwidth', 10.0), topo2.get('bandwidth', 10.0))
            merged['latency'] = max(topo1.get('latency', 1.0), topo2.get('latency', 1.0))
        else:
            merged['type'] = 'hierarchical'
            merged['interconnect'] = 'mixed'

        participants1 = set(topo1.get('participants', []))
        participants2 = set(topo2.get('participants', []))
        merged['participants'] = list(participants1.union(participants2))

        return merged

    def _next_op_id(self) -> int:
        """Get next operation ID"""
        self._op_counter += 1
        return self._op_counter