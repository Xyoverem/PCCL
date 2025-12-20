"""
L2 to L3 lowering pass from primitive IR to hardware-specific primitives.

This pass transforms the five basic operations (Write, Reduce, Copy, Signal, WaitSignal)
into hardware-specific operations like CUDA multi-memory operations or RDMA verbs.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging

from .base import Pass, PassResult, PassContext
from .registry import register_pass
from ..ir.primitive_ir import *
from ..ir.json_serializer import IRGraph, IROperation, IRValue
from ..plugins.hardware_primitives import (
    HardwareType,
    HardwarePrimitiveRegistry,
    PrimitiveToHardwareLoweringPass,
    get_hardware_primitives_for_device,
    create_lowering_pass_for_device
)

logger = logging.getLogger(__name__)


class HardwareOptimizationLevel(Enum):
    """Hardware optimization levels for L2 to L3 lowering."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    DEVICE_SPECIFIC = "device_specific"


@register_pass("primitive_to_cuda")
class PrimitiveToCUDAPass(PrimitiveToHardwareLoweringPass):
    """Specialized pass for lowering to CUDA hardware primitives."""

    def __init__(self, optimization_level: HardwareOptimizationLevel = HardwareOptimizationLevel.BASIC):
        super().__init__(HardwareType.CUDA)
        self.optimization_level = optimization_level

        # Update capabilities based on optimization level
        self.capabilities.supported_operations = {"write", "reduce", "copy", "signal", "wait_signal"}
        self.capabilities.requires_analysis = optimization_level != HardwareOptimizationLevel.NONE


@register_pass("primitive_to_rdma")
class PrimitiveToRDMAPass(PrimitiveToHardwareLoweringPass):
    """Specialized pass for lowering to RDMA hardware primitives."""

    def __init__(self, optimization_level: HardwareOptimizationLevel = HardwareOptimizationLevel.BASIC):
        super().__init__(HardwareType.RDMA)
        self.optimization_level = optimization_level

        self.capabilities.supported_operations = {"write", "reduce", "copy", "signal", "wait_signal"}
        self.capabilities.requires_analysis = optimization_level != HardwareOptimizationLevel.NONE


@register_pass("hardware_fusion")
class HardwareFusionPass(Pass):
    """Hardware-specific operation fusion pass."""

    def __init__(self, hardware_type: HardwareType):
        super().__init__(
            name="hardware_fusion",
            description=f"Fuse hardware primitives for {hardware_type.value}"
        )
        self.hardware_type = hardware_type
        self.capabilities.supported_operations = {"copy", "reduce"}
        self.capabilities.preserves_structure = False
        self.capabilities.can_change_execution_order = True

    def apply(self, ir_graph: IRGraph, context: PassContext) -> PassResult:
        """Apply hardware-specific fusion optimizations."""
        fused_graph = IRGraph(
            values=dict(ir_graph.values),
            operations={},
            metadata={
                **ir_graph.metadata,
                "hardware_fusion_applied": True,
                "hardware_type": self.hardware_type.value
            }
        )

        fusion_stats = {
            "original_operations": len(ir_graph.operations),
            "fused_operations": 0,
            "fusion_patterns": []
        }

        if self.hardware_type == HardwareType.CUDA:
            fused_ops, stats = self._apply_cuda_fusion(ir_graph, context)
        elif self.hardware_type == HardwareType.RDMA:
            fused_ops, stats = self._apply_rdma_fusion(ir_graph, context)
        else:
            # No fusion available for this hardware type
            fused_ops = dict(ir_graph.operations)
            stats = {"fused_operations": 0, "fusion_patterns": []}

        fused_graph.operations.update(fused_ops)
        fusion_stats.update(stats)

        return PassResult(
            success=True,
            ir_graph=fused_graph,
            message=f"Applied hardware fusion for {self.hardware_type.value}",
            transformation_stats=fusion_stats
        )

    def _apply_cuda_fusion(self, ir_graph: IRGraph, context: PassContext) -> tuple[Dict[str, IROperation], Dict[str, Any]]:
        """Apply CUDA-specific fusion patterns."""
        fused_ops = {}
        fusion_patterns = []

        # Pattern 1: Fuse consecutive memcpy operations
        memcpy_groups = self._group_consecutive_memcpy(ir_graph)
        for group_id, op_ids in enumerate(memcpy_groups):
            if len(op_ids) > 1:
                # Create fused memcpy operation
                first_op = ir_graph.operations[op_ids[0]]
                last_op = ir_graph.operations[op_ids[-1]]

                fused_op = IROperation(
                    op_type="fused_memcpy",
                    inputs=first_op.inputs,
                    outputs=last_op.outputs,
                    attributes={
                        "original_operations": op_ids,
                        "total_size": sum(
                            ir_graph.operations[op_id].attributes.get("size", 0)
                            for op_id in op_ids
                        ),
                        "fusion_type": "consecutive_memcpy"
                    }
                )

                fused_ops[f"fused_memcpy_{group_id}"] = fused_op
                fusion_patterns.append(f"Consecutive memcpy fusion: {len(op_ids)} ops")

        # Pattern 2: Fuse reduction operations with same output
        reduce_groups = self._group_reducible_operations(ir_graph)
        for group_id, op_ids in enumerate(reduce_groups):
            if len(op_ids) > 1:
                # Create fused reduction operation
                ops = [ir_graph.operations[op_id] for op_id in op_ids]
                all_inputs = []
                for op in ops:
                    all_inputs.extend(op.inputs)

                fused_op = IROperation(
                    op_type="fused_reduction",
                    inputs=all_inputs,
                    outputs=ops[0].outputs,
                    attributes={
                        "original_operations": op_ids,
                        "fusion_type": "reduction_fusion"
                    }
                )

                fused_ops[f"fused_reduce_{group_id}"] = fused_op
                fusion_patterns.append(f"Reduction fusion: {len(op_ids)} ops")

        # Keep non-fused operations as-is
        processed_ops = set()
        for groups in [memcpy_groups, reduce_groups]:
            for op_ids in groups:
                processed_ops.update(op_ids)

        for op_id, operation in ir_graph.operations.items():
            if op_id not in processed_ops:
                fused_ops[op_id] = operation

        stats = {
            "fused_operations": len(fusion_patterns),
            "fusion_patterns": fusion_patterns
        }

        return fused_ops, stats

    def _apply_rdma_fusion(self, ir_graph: IRGraph, context: PassContext) -> tuple[Dict[str, IROperation], Dict[str, Any]]:
        """Apply RDMA-specific fusion patterns."""
        fused_ops = {}
        fusion_patterns = []

        # Pattern 1: Batch multiple RDMA writes to same target
        write_groups = self._group_rdma_writes_by_target(ir_graph)
        for group_id, op_ids in enumerate(write_groups):
            if len(op_ids) > 1:
                # Create batched RDMA write
                ops = [ir_graph.operations[op_id] for op_id in op_ids]
                batched_op = IROperation(
                    op_type="batched_rdma_write",
                    inputs=[input for op in ops for input in op.inputs],
                    outputs=ops[0].outputs,
                    attributes={
                        "original_operations": op_ids,
                        "batch_size": len(op_ids),
                        "fusion_type": "rdma_write_batching"
                    }
                )

                fused_ops[f"batched_write_{group_id}"] = batched_op
                fusion_patterns.append(f"RDMA write batching: {len(op_ids)} ops")

        # Keep non-fused operations
        processed_ops = set()
        for op_ids in write_groups:
            processed_ops.update(op_ids)

        for op_id, operation in ir_graph.operations.items():
            if op_id not in processed_ops:
                fused_ops[op_id] = operation

        stats = {
            "fused_operations": len(fusion_patterns),
            "fusion_patterns": fusion_patterns
        }

        return fused_ops, stats

    def _group_consecutive_memcpy(self, ir_graph: IRGraph) -> List[List[str]]:
        """Group consecutive memcpy operations."""
        groups = []
        current_group = []

        # Simple implementation - group memcpy operations that appear consecutively
        for op_id, operation in ir_graph.operations.items():
            if operation.op_type == "device_memcpy" or operation.op_type == "memcpy_h2d" or operation.op_type == "memcpy_d2h":
                current_group.append(op_id)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []

        if current_group:
            groups.append(current_group)

        return groups

    def _group_reducible_operations(self, ir_graph: IRGraph) -> List[List[str]]:
        """Group operations that can be fused into a single reduction."""
        groups = []

        # Group multi-memory reduce operations with same output
        output_to_ops = {}
        for op_id, operation in ir_graph.operations.items():
            if operation.op_type == "multimem_reduce":
                output = operation.outputs[0] if operation.outputs else None
                if output:
                    if output not in output_to_ops:
                        output_to_ops[output] = []
                    output_to_ops[output].append(op_id)

        # Create groups from operations with same output
        for output, op_ids in output_to_ops.items():
            if len(op_ids) > 1:
                groups.append(op_ids)

        return groups

    def _group_rdma_writes_by_target(self, ir_graph: IRGraph) -> List[List[str]]:
        """Group RDMA write operations by target address."""
        target_to_ops = {}

        for op_id, operation in ir_graph.operations.items():
            if operation.op_type == "rdma_write":
                target = operation.attributes.get("remote_addr", "")
                if target:
                    if target not in target_to_ops:
                        target_to_ops[target] = []
                    target_to_ops[target].append(op_id)

        return list(target_to_ops.values())


@register_pass("hardware_memory_layout_optimization")
class HardwareMemoryLayoutPass(Pass):
    """Hardware-specific memory layout optimization pass."""

    def __init__(self, hardware_type: HardwareType):
        super().__init__(
            name="hardware_memory_layout_optimization",
            description=f"Optimize memory layout for {hardware_type.value}"
        )
        self.hardware_type = hardware_type
        self.capabilities.supported_operations = {"copy", "reduce", "write"}
        self.capabilities.preserves_structure = False

    def apply(self, ir_graph: IRGraph, context: PassContext) -> PassResult:
        """Apply memory layout optimizations."""
        optimized_graph = IRGraph(
            values=dict(ir_graph.values),
            operations=dict(ir_graph.operations),
            metadata={
                **ir_graph.metadata,
                "memory_layout_optimized": True,
                "hardware_type": self.hardware_type.value
            }
        )

        optimization_stats = {
            "optimized_operations": 0,
            "layout_transforms": []
        }

        if self.hardware_type == HardwareType.CUDA:
            optimization_stats = self._apply_cuda_memory_optimizations(optimized_graph, context)
        elif self.hardware_type == HardwareType.RDMA:
            optimization_stats = self._apply_rdma_memory_optimizations(optimized_graph, context)

        return PassResult(
            success=True,
            ir_graph=optimized_graph,
            message=f"Applied memory layout optimizations for {self.hardware_type.value}",
            transformation_stats=optimization_stats
        )

    def _apply_cuda_memory_optimizations(self, ir_graph: IRGraph, context: PassContext) -> Dict[str, Any]:
        """Apply CUDA-specific memory optimizations."""
        optimized_ops = 0
        layout_transforms = []

        # Add memory coalescing hints to copy operations
        for op_id, operation in ir_graph.operations.items():
            if operation.op_type == "device_memcpy":
                # Add coalescing hint if size is multiple of warp size
                size = operation.attributes.get("size", 0)
                if size % 32 == 0:  # Warp size alignment
                    operation.attributes["memory_coalescing"] = True
                    operation.attributes["alignment"] = 32
                    optimized_ops += 1
                    layout_transforms.append(f"Added coalescing hint to {op_id}")

        return {
            "optimized_operations": optimized_ops,
            "layout_transforms": layout_transforms
        }

    def _apply_rdma_memory_optimizations(self, ir_graph: IRGraph, context: PassContext) -> Dict[str, Any]:
        """Apply RDMA-specific memory optimizations."""
        optimized_ops = 0
        layout_transforms = []

        # Add registration hints for memory regions
        for op_id, operation in ir_graph.operations.items():
            if operation.op_type == "rdma_write":
                # Add memory registration optimization
                operation.attributes["memory_registration"] = "optimized"
                operation.attributes["inline_data"] = operation.attributes.get("size", 0) < 64
                optimized_ops += 1
                layout_transforms.append(f"Optimized memory registration for {op_id}")

        return {
            "optimized_operations": optimized_ops,
            "layout_transforms": layout_transforms
        }


@register_pass("hardware_specific_optimization")
class HardwareSpecificOptimizationPass(Pass):
    """Apply hardware-specific optimizations beyond lowering and fusion."""

    def __init__(self, hardware_type: HardwareType, optimization_passes: Optional[List[str]] = None):
        super().__init__(
            name="hardware_specific_optimization",
            description=f"Apply {hardware_type.value}-specific optimizations"
        )
        self.hardware_type = hardware_type
        self.optimization_passes = optimization_passes or []

        self.capabilities.supported_operations = {"write", "reduce", "copy", "signal", "wait_signal"}
        self.capabilities.preserves_structure = False

    def apply(self, ir_graph: IRGraph, context: PassContext) -> PassResult:
        """Apply hardware-specific optimizations."""
        optimized_graph = IRGraph(
            values=dict(ir_graph.values),
            operations=dict(ir_graph.operations),
            metadata={
                **ir_graph.metadata,
                "hardware_specific_optimized": True,
                "hardware_type": self.hardware_type.value,
                "applied_optimizations": self.optimization_passes
            }
        )

        optimization_stats = {
            "applied_passes": [],
            "optimization_results": {}
        }

        if self.hardware_type == HardwareType.CUDA:
            optimization_stats = self._apply_cuda_specific_optimizations(optimized_graph, context)
        elif self.hardware_type == HardwareType.RDMA:
            optimization_stats = self._apply_rdma_specific_optimizations(optimized_graph, context)

        return PassResult(
            success=True,
            ir_graph=optimized_graph,
            message=f"Applied {self.hardware_type.value}-specific optimizations",
            transformation_stats=optimization_stats
        )

    def _apply_cuda_specific_optimizations(self, ir_graph: IRGraph, context: PassContext) -> Dict[str, Any]:
        """Apply CUDA-specific optimizations."""
        applied_passes = []
        results = {}

        # Optimization 1: Shared memory usage
        if "shared_memory" in self.optimization_passes:
            shared_ops = self._optimize_shared_memory_usage(ir_graph)
            results["shared_memory_optimization"] = shared_ops
            applied_passes.append("shared_memory")

        # Optimization 2: Warp-level operations
        if "warp_level" in self.optimization_passes:
            warp_ops = self._optimize_warp_level_operations(ir_graph)
            results["warp_level_optimization"] = warp_ops
            applied_passes.append("warp_level")

        return {
            "applied_passes": applied_passes,
            "optimization_results": results
        }

    def _apply_rdma_specific_optimizations(self, ir_graph: IRGraph, context: PassContext) -> Dict[str, Any]:
        """Apply RDMA-specific optimizations."""
        applied_passes = []
        results = {}

        # Optimization 1: Zero-copy transfers
        if "zero_copy" in self.optimization_passes:
            zerocopy_ops = self._optimize_zero_copy_transfers(ir_graph)
            results["zero_copy_optimization"] = zerocopy_ops
            applied_passes.append("zero_copy")

        # Optimization 2: Queue pair management
        if "queue_pair" in self.optimization_passes:
            qp_ops = self._optimize_queue_pair_usage(ir_graph)
            results["queue_pair_optimization"] = qp_ops
            applied_passes.append("queue_pair")

        return {
            "applied_passes": applied_passes,
            "optimization_results": results
        }

    def _optimize_shared_memory_usage(self, ir_graph: IRGraph) -> int:
        """Optimize shared memory usage in CUDA operations."""
        optimized_ops = 0

        for op_id, operation in ir_graph.operations.items():
            if operation.op_type in ["multimem_reduce", "warp_level_reduce"]:
                # Add shared memory optimization hints
                operation.attributes["use_shared_memory"] = True
                operation.attributes["shared_memory_size"] = operation.attributes.get("elements", 0) * 4  # Assume float32
                optimized_ops += 1

        return optimized_ops

    def _optimize_warp_level_operations(self, ir_graph: IRGraph) -> int:
        """Optimize warp-level operations."""
        optimized_ops = 0

        for op_id, operation in ir_graph.operations.items():
            if operation.op_type == "multimem_reduce":
                elements = operation.attributes.get("elements", 0)
                if elements <= 32:  # Warp size
                    # Convert to warp-level operation
                    operation.op_type = "warp_level_reduce"
                    operation.attributes["warp_optimized"] = True
                    optimized_ops += 1

        return optimized_ops

    def _optimize_zero_copy_transfers(self, ir_graph: IRGraph) -> int:
        """Optimize RDMA zero-copy transfers."""
        optimized_ops = 0

        for op_id, operation in ir_graph.operations.items():
            if operation.op_type == "rdma_write":
                # Add zero-copy optimization if size is appropriate
                size = operation.attributes.get("length", 0)
                if size <= 4096:  # Small transfers benefit from zero-copy
                    operation.attributes["zero_copy"] = True
                    optimized_ops += 1

        return optimized_ops

    def _optimize_queue_pair_usage(self, ir_graph: IRGraph) -> int:
        """Optimize RDMA queue pair usage."""
        optimized_ops = 0
        qp_usage = {}

        # Count operations per queue pair
        for op_id, operation in ir_graph.operations.items():
            if "qp_id" in operation.attributes:
                qp_id = operation.attributes["qp_id"]
                qp_usage[qp_id] = qp_usage.get(qp_id, 0) + 1

        # Rebalance operations across queue pairs if needed
        max_ops_per_qp = 10
        for qp_id, op_count in qp_usage.items():
            if op_count > max_ops_per_qp:
                # Mark for rebalancing
                for op_id, operation in ir_graph.operations.items():
                    if operation.attributes.get("qp_id") == qp_id:
                        operation.attributes["load_balance"] = True
                        optimized_ops += 1

        return optimized_ops