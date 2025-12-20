"""
Collective to Primitive Lowering Pass

Implements lowering of high-level collective operations (AllReduce, Broadcast, etc.)
to primitive operations (Write, Reduce, Copy, Signal, Wait).
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .base import Pass, PassResult, PassContext, PassType, PassCapabilities
from ..ir.json_serializer import (
    IRGraph, IRValue, IROperation, IRType,
    PrimitiveOpType, ReduceOpType, DeviceType
)
from ..ir.primitive_ir import (
    PrimitiveIRBuilder, WriteOp, ReduceOp, CopyOp,
    SignalOp, WaitSignalOp, CollectiveOperation, CollectiveOpType
)


class CollectiveToPrimitiveLowering(Pass):


    def __init__(self):
        super().__init__(
            name="collective_to_primitive",
            pass_type=PassType.LAYER1_TO_LAYER2,
            description="Lower collective operations to primitive IR"
        )

    def declare_capabilities(self) -> PassCapabilities:
    
        return PassCapabilities(
            input_ir_types=["collective"],
            output_ir_types=["primitive"],
            supported_devices=["cpu", "cuda", "rdma", "auto"],
            required_features=[],
            optional_features=["topology_aware", "algorithm_selection"]
        )

    def validate_input(self, ir: Any, context: PassContext) -> bool:
    
        if not isinstance(ir, IRGraph):
            return False

        if ir.ir_type != IRType.COLLECTIVE:
            return False

        return len(ir.operations) > 0

    def execute(self, ir: IRGraph, context: PassContext) -> PassResult:
    
        try:
            primitive_graph = self._lower_collective_graph(ir, context)
            return PassResult(
                success=True,
                ir=primitive_graph,
                metadata={"original_collective_ops": len(ir.operations)},
                diagnostics=[f"Successfully lowered {len(ir.operations)} collective operations"]
            )
        except Exception as e:
            return PassResult(
                success=False,
                ir=ir,
                diagnostics=[f"Lowering failed: {str(e)}"]
            )

    def _lower_collective_graph(self, collective_graph: IRGraph, context: PassContext) -> IRGraph:
    
        primitive_graph = IRGraph(ir_type=IRType.PRIMITIVE, values={}, operations={})
        primitive_graph.metadata = collective_graph.metadata.copy()
        primitive_graph.metadata["lowering_info"] = {
            "pass": "collective_to_primitive",
            "target_device": context.target_device,
            "optimization_level": context.optimization_level
        }

        value_mapping = {}

        for op_id, operation in collective_graph.operations.items():
            collective_op = self._parse_collective_operation(operation)
            lowered_ops = self._lower_collective_operation(collective_op, context)

            # Add lowered operations and values to primitive graph
            for lowered_op in lowered_ops:
                primitive_graph.add_operation(lowered_op)

                # Update value mapping
                for i, output_id in enumerate(lowered_op.outputs):
                    if i < len(collective_op.outputs):
                        value_mapping[collective_op.outputs[i]] = output_id

        for value_id, value in collective_graph.values.items():
            if value_id not in value_mapping:
                # This is an input value, copy it directly
                primitive_value = IRValue(
                    id=value.id + "_prim",
                    dtype=value.dtype,
                    shape=value.shape,
                    device_id=value.device_id,
                    device_type=value.device_type,
                    metadata=value.metadata.copy()
                )
                primitive_graph.add_value(primitive_value)
                value_mapping[value_id] = primitive_value.id

        for op_id, operation in primitive_graph.operations.items():
            new_inputs = []
            for input_id in operation.inputs:
                if input_id in value_mapping:
                    new_inputs.append(value_mapping[input_id])
                else:
                    new_inputs.append(input_id)
            operation.inputs = new_inputs

        return primitive_graph

    def _parse_collective_operation(self, operation: IROperation) -> CollectiveOperation:
    
        try:
            op_type = CollectiveOpType(operation.op_type)
        except ValueError:
            raise ValueError(f"Unknown collective operation type: {operation.op_type}")

        return CollectiveOperation(
            id=operation.id,
            op_type=op_type,
            inputs=operation.inputs,
            outputs=operation.outputs,
            attributes=operation.attributes.copy()
        )

    def _lower_collective_operation(self, collective_op: CollectiveOperation, context: PassContext) -> List[IROperation]:
    
        if collective_op.op_type == CollectiveOpType.ALLREDUCE:
            return self._lower_allreduce(collective_op, context)
        elif collective_op.op_type == CollectiveOpType.BROADCAST:
            return self._lower_broadcast(collective_op, context)
        elif collective_op.op_type == CollectiveOpType.ALLGATHER:
            return self._lower_allgather(collective_op, context)
        elif collective_op.op_type == CollectiveOpType.REDUCE_SCATTER:
            return self._lower_reducescatter(collective_op, context)
        elif collective_op.op_type == CollectiveOpType.REDUCE:
            return self._lower_reduce(collective_op, context)
        elif collective_op.op_type == CollectiveOpType.ALLTOALL:
            return self._lower_alltoall(collective_op, context)
        else:
            raise ValueError(f"Unsupported collective operation: {collective_op.op_type}")

    def _lower_allreduce(self, collective_op: CollectiveOperation, context: PassContext) -> List[IROperation]:
    
        world_size = collective_op.attributes.get("world_size", 2)
        rank = collective_op.attributes.get("rank", 0)
        reduce_op = ReduceOpType(collective_op.attributes.get("reduce_op", "sum"))

        algorithm = collective_op.attributes.get("algorithm", "ring")
        if algorithm == "auto":
            algorithm = self._select_allreduce_algorithm(world_size, collective_op.inputs, context)

        if algorithm == "ring":
            return self._lower_allreduce_ring(collective_op, world_size, rank, reduce_op, context)
        elif algorithm == "tree":
            return self._lower_allreduce_tree(collective_op, world_size, rank, reduce_op, context)
        elif algorithm == "rabenseifner":
            return self._lower_allreduce_rabenseifner(collective_op, world_size, rank, reduce_op, context)
        else:
            raise ValueError(f"Unknown AllReduce algorithm: {algorithm}")

    def _lower_allreduce_ring(self, collective_op: CollectiveOperation, world_size: int, rank: int,
                            reduce_op: ReduceOpType, context: PassContext) -> List[IROperation]:
    
        operations = []

        if not collective_op.inputs:
            return operations

        input_id = collective_op.inputs[0]
        output_id = collective_op.outputs[0] if collective_op.outputs else f"{input_id}_output"

        current_data = input_id
        for step in range(world_size - 1):
            # Send to next rank
            target_rank = (rank + 1) % world_size
            signal_op = SignalOp(
                id=f"scatter_signal_{step}",
                signal_id=f"scatter_sig_{step}",
                target_ranks=[target_rank]
            )
            operations.append(signal_op)

            # Receive from previous rank
            source_rank = (rank - 1) % world_size
            received_data = f"received_scatter_{step}"
            wait_op = WaitSignalOp(
                id=f"scatter_wait_{step}",
                signal_id=f"scatter_sig_{step}",
                source_ranks=[source_rank]
            )
            operations.append(wait_op)

            # Reduce with received data
            reduce_op_primitive = ReduceOp(
                id=f"reduce_scatter_{step}",
                input_values=[current_data, received_data],
                output_value=f"reduced_scatter_{step}",
                reduce_op=reduce_op
            )
            operations.append(reduce_op_primitive)

            current_data = f"reduced_scatter_{step}"

        final_data = current_data
        for step in range(world_size - 1):
            # Send current data to next rank
            target_rank = (rank + 1) % world_size
            signal_op = SignalOp(
                id=f"gather_signal_{step}",
                signal_id=f"gather_sig_{step}",
                target_ranks=[target_rank]
            )
            operations.append(signal_op)

            # Receive data from previous rank
            source_rank = (rank - 1) % world_size
            wait_op = WaitSignalOp(
                id=f"gather_wait_{step}",
                signal_id=f"gather_sig_{step}",
                source_ranks=[source_rank]
            )
            operations.append(wait_op)

            # Copy received data (simplified - would need proper chunk management)
            copy_op = CopyOp(
                id=f"copy_gather_{step}",
                input_value=f"received_gather_{step}",
                output_value=output_id
            )
            operations.append(copy_op)

        return operations

    def _lower_allreduce_tree(self, collective_op: CollectiveOperation, world_size: int, rank: int,
                            reduce_op: ReduceOpType, context: PassContext) -> List[IROperation]:
    
        operations = []

        if not collective_op.inputs:
            return operations

        input_id = collective_op.inputs[0]
        output_id = collective_op.outputs[0] if collective_op.outputs else f"{input_id}_output"

        if rank == 0:
            # Root: reduce all inputs
            all_inputs = collective_op.inputs
            reduce_op_primitive = ReduceOp(
                id="tree_reduce",
                input_values=all_inputs,
                output_value=output_id,
                reduce_op=reduce_op
            )
            operations.append(reduce_op_primitive)

        return operations

    def _lower_allreduce_rabenseifner(self, collective_op: CollectiveOperation, world_size: int, rank: int,
                                    reduce_op: ReduceOpType, context: PassContext) -> List[IROperation]:
    
        return self._lower_allreduce_ring(collective_op, world_size, rank, reduce_op, context)

    def _lower_broadcast(self, collective_op: CollectiveOperation, context: PassContext) -> List[IROperation]:
    
        operations = []

        if not collective_op.inputs or not collective_op.outputs:
            return operations

        root_rank = collective_op.attributes.get("root_rank", 0)
        rank = collective_op.attributes.get("rank", 0)

        input_id = collective_op.inputs[0]
        output_id = collective_op.outputs[0]

        if rank == root_rank:
            # Root: copy input to output
            copy_op = CopyOp(
                id="broadcast_copy_root",
                input_value=input_id,
                output_value=output_id
            )
            operations.append(copy_op)
        else:
            # Non-root: wait for signal and copy
            signal_id = f"broadcast_sig_{root_rank}"
            wait_op = WaitSignalOp(
                id="broadcast_wait",
                signal_id=signal_id,
                source_ranks=[root_rank]
            )
            operations.append(wait_op)

            copy_op = CopyOp(
                id="broadcast_copy",
                input_value=input_id,
                output_value=output_id
            )
            operations.append(copy_op)

        signal_op = SignalOp(
            id="broadcast_complete",
            signal_id=f"broadcast_complete_{rank}",
            target_ranks=[]
        )
        operations.append(signal_op)

        return operations

    def _lower_allgather(self, collective_op: CollectiveOperation, context: PassContext) -> List[IROperation]:
    
        operations = []
        return operations

    def _lower_reducescatter(self, collective_op: CollectiveOperation, context: PassContext) -> List[IROperation]:
    
        operations = []
        return operations

    def _lower_reduce(self, collective_op: CollectiveOperation, context: PassContext) -> List[IROperation]:
    
        operations = []

        if not collective_op.inputs:
            return operations

        root_rank = collective_op.attributes.get("root_rank", 0)
        rank = collective_op.attributes.get("rank", 0)
        reduce_op = ReduceOpType(collective_op.attributes.get("reduce_op", "sum"))

        if rank == root_rank:
            # Root: reduce all inputs
            reduce_op_primitive = ReduceOp(
                id="reduce_root",
                input_values=collective_op.inputs,
                output_value=collective_op.outputs[0] if collective_op.outputs else "reduce_output",
                reduce_op=reduce_op
            )
            operations.append(reduce_op_primitive)

        return operations

    def _lower_alltoall(self, collective_op: CollectiveOperation, context: PassContext) -> List[IROperation]:
    
        operations = []
        return operations

    def _select_allreduce_algorithm(self, world_size: int, inputs: List[str], context: PassContext) -> str:
    
        if world_size <= 2:
            return "ring"
        elif context.optimization_level == "performance":
            return "tree"
        else:
            return "ring"


# Register the pass
from ..passes.registry import register_pass

@register_pass("collective_to_primitive")
class RegisteredCollectiveToPrimitiveLowering(CollectiveToPrimitiveLowering):
    pass