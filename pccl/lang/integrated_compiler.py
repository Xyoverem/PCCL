"""
Integrated compiler that connects DSL with IR lowering pass system.

This bridges the gap between the high-level DSL API and the three-layer IR
lowering system, enabling complete L1→L2→L3 lowering.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

from .config import OperatorConfig, TopologyConfig, AllreduceConfig, BroadcastConfig, AllgatherConfig, ReduceScatterConfig
from .compiler import ExecutionPlan
from ..ir.primitive_ir import *
from ..ir.json_serializer import IRSerializer
from ..passes.collective_to_primitive import CollectiveToPrimitivePass
from ..passes.primitive_to_hardware import *
from ..plugins.hardware_primitives import HardwareType, create_lowering_pass_for_device
from ..passes.base import PassContext
from ..passes.manager import PassManager
from ..passes.pipeline import PassPipeline


@dataclass
class IntegratedExecutionPlan:
    """Enhanced execution plan that includes IR lowering information"""

    # Original execution plan
    execution_plan: ExecutionPlan

    # IR lowering results
    primitive_ir_graph: Optional[Any] = None
    hardware_ir_graph: Optional[Any] = None
    lowering_stats: Optional[Dict[str, Any]] = None

    # Hardware-specific information
    hardware_type: Optional[str] = None
    json_output: Optional[str] = None


class IntegratedCompiler:
    """Integrated compiler that connects DSL with IR lowering pass system"""

    def __init__(self, hardware_type: HardwareType = HardwareType.CUDA):
        self.hardware_type = hardware_type

        # Initialize pass system
        self.pass_manager = PassManager()

        # Create lowering pipeline
        self.lowering_pipeline = PassPipeline()
        self.lowering_pipeline.add_pass("collective_to_primitive")
        self.lowering_pipeline.add_pass(f"primitive_to_{hardware_type.value}")

        # Add optimization passes
        self.lowering_pipeline.add_pass("hardware_fusion")
        self.lowering_pipeline.add_pass("hardware_memory_layout_optimization")

    def compile(self, config: OperatorConfig, participants: Optional[List[int]] = None) -> IntegratedExecutionPlan:
        """
        Compile DSL configuration to hardware-executable plan using full IR lowering.
        """
        print(f"🔄 Compiling {config.__class__.__name__} with integrated lowering pipeline...")

        # Step 1: Convert DSL config to collective operation
        collective_op = self._config_to_collective(config, participants)
        print(f"   ✅ Created L1 collective: {collective_op.name}")

        # Step 2: L1 → L2 lowering (Collective to Primitive)
        primitive_builder = PrimitiveIRBuilder()
        primitive_ir_graph = primitive_builder.from_collective(collective_op)
        print(f"   ✅ Lowered to L2: {len(primitive_ir_graph.operations)} primitive operations")

        # Step 3: L2 → L3 lowering (Primitive to Hardware)
        pass_context = PassContext()

        # Apply L2 → L3 lowering pass
        hardware_pass = create_lowering_pass_for_device(self.hardware_type)
        hardware_result = hardware_pass.apply(primitive_ir_graph, pass_context)

        if not hardware_result.success:
            raise RuntimeError(f"L2→L3 lowering failed: {hardware_result.message}")

        hardware_ir_graph = hardware_result.ir_graph
        lowering_stats = hardware_result.transformation_stats
        print(f"   ✅ Lowered to L3: {lowering_stats['hardware_operations']} {self.hardware_type.value} operations")
        print(f"   ✅ Expansion ratio: {lowering_stats['expansion_ratio']:.2f}")

        # Step 4: Apply optimization passes
        optimized_result = self._apply_optimizations(hardware_ir_graph, pass_context)
        if optimized_result.success:
            final_ir_graph = optimized_result.ir_graph
            print(f"   ✅ Applied optimizations: {len(optimized_result.transformation_stats.get('applied_passes', []))} passes")
        else:
            final_ir_graph = hardware_ir_graph
            print(f"   ⚠️  Optimization skipped: {optimized_result.message}")

        # Step 5: Serialize to JSON for C++ runtime
        serializer = IRSerializer()
        json_output = serializer.serialize_graph(final_ir_graph)
        print(f"   ✅ Serialized to JSON: {len(json_output)} characters")

        # Step 6: Generate execution plan for compatibility
        execution_plan = self._ir_graph_to_execution_plan(final_ir_graph, config, participants)

        # Step 7: Create integrated execution plan
        integrated_plan = IntegratedExecutionPlan(
            execution_plan=execution_plan,
            primitive_ir_graph=primitive_ir_graph,
            hardware_ir_graph=final_ir_graph,
            lowering_stats=lowering_stats,
            hardware_type=self.hardware_type.value,
            json_output=json_output
        )

        print(f"🎉 Integrated compilation completed successfully!")
        return integrated_plan

    def _config_to_collective(self, config: OperatorConfig, participants: Optional[List[int]]) -> CollectiveOperation:
        """Convert DSL configuration to collective operation"""

        if isinstance(config, AllreduceConfig):
            # Create chunks for participants
            devices = [Device(i, DeviceType.CUDA) for i in participants or range(4)]
            input_chunks = [
                Chunk(f"rank_{i}_data", config.buffer_size, DataType.FLOAT32, devices[i])
                for i in range(len(devices))
            ]

            # Map reduce operation
            reduce_type_map = {
                "sum": ReduceOp.REDUCE_SUM,
                "avg": ReduceOp.REDUCE_AVG,
                "max": ReduceOp.REDUCE_MAX,
                "min": ReduceOp.REDUCE_MIN
            }

            # Map algorithm
            algorithm_map = {
                "ring": CollectiveAlgorithm.RING,
                "tree": CollectiveAlgorithm.TREE,
                "rabenseifner": CollectiveAlgorithm.RABENSEIFNER
            }

            return CollectiveOperation(
                name="allreduce_integrated",
                op_type=CollectiveOpType.ALLREDUCE,
                inputs=input_chunks,
                output=Chunk("allreduce_result", config.buffer_size, DataType.FLOAT32, devices[0]),
                algorithm=algorithm_map.get(getattr(config, 'algorithm', 'ring'), CollectiveAlgorithm.RING),
                reduce_type=reduce_type_map.get(getattr(config, 'reduce_op', 'sum'), ReduceOp.REDUCE_SUM)
            )

        elif isinstance(config, BroadcastConfig):
            devices = [Device(i, DeviceType.CUDA) for i in participants or range(4)]
            input_chunk = Chunk("broadcast_data", config.buffer_size, DataType.FLOAT32, devices[config.root_rank])
            output_chunks = [
                Chunk(f"rank_{i}_output", config.buffer_size, DataType.FLOAT32, devices[i])
                for i in range(len(devices))
            ]

            return CollectiveOperation(
                name="broadcast_integrated",
                op_type=CollectiveOpType.BROADCAST,
                inputs=[input_chunk],
                outputs=output_chunks,
                algorithm=CollectiveAlgorithm.TREE,
                reduce_type=None,
                root_rank=config.root_rank
            )

        else:
            raise NotImplementedError(f"Configuration type {config.__class__.__name__} not yet supported")

    def _apply_optimizations(self, ir_graph, pass_context) -> Any:
        """Apply hardware-specific optimization passes"""
        try:
            # Create optimization pipeline
            optimization_pipeline = PassPipeline()
            optimization_pipeline.add_pass("hardware_fusion")
            optimization_pipeline.add_pass("hardware_memory_layout_optimization")

            # Apply optimizations
            return optimization_pipeline.apply(ir_graph, pass_context)

        except Exception as e:
            # Return original graph if optimization fails
            from ..passes.base import PassResult
            return PassResult(
                success=False,
                ir_graph=ir_graph,
                message=f"Optimization failed: {e}"
            )

    def _ir_graph_to_execution_plan(self, ir_graph, config: OperatorConfig, participants: Optional[List[int]]) -> ExecutionPlan:
        """Convert IR graph back to execution plan for compatibility"""

        operations = []
        dependencies = []

        # Convert IR operations to execution plan format
        for op_id, operation in ir_graph.operations.items():
            op_data = {
                'id': op_id,
                'type': operation.op_type,
                'inputs': operation.inputs,
                'outputs': operation.outputs,
                'attributes': operation.attributes,
                'device_type': ir_graph.metadata.get('hardware_type', 'unknown'),
                'ir_layer': ir_graph.metadata.get('ir_layer', 'hardware_primitives')
            }
            operations.append(op_data)

        # Create basic topology
        topology = {
            'type': 'mesh',
            'devices': participants or list(range(4)),
            'hardware_type': self.hardware_type.value
        }

        return ExecutionPlan(
            operations=operations,
            topology=topology,
            dependencies=dependencies,
            metadata={
                'compiled_with': 'integrated_lowering',
                'hardware_type': self.hardware_type.value,
                'ir_operations': len(operations),
                'config_type': config.__class__.__name__
            }
        )

    def get_lowering_statistics(self, plan: IntegratedExecutionPlan) -> Dict[str, Any]:
        """Get detailed lowering statistics"""
        stats = {
            'lowering_successful': True,
            'hardware_type': plan.hardware_type,
            'l2_operations': len(plan.primitive_ir_graph.operations) if plan.primitive_ir_graph else 0,
            'l3_operations': len(plan.hardware_ir_graph.operations) if plan.hardware_ir_graph else 0,
            'expansion_ratio': 0.0,
            'json_size': len(plan.json_output) if plan.json_output else 0
        }

        if plan.lowering_stats:
            stats.update(plan.lowering_stats)

        if stats['l2_operations'] > 0:
            stats['expansion_ratio'] = stats['l3_operations'] / stats['l2_operations']

        return stats


# Factory functions for creating integrated compilers
def create_cuda_compiler() -> IntegratedCompiler:
    """Create integrated compiler targeting CUDA hardware"""
    return IntegratedCompiler(HardwareType.CUDA)


def create_rdma_compiler() -> IntegratedCompiler:
    """Create integrated compiler targeting RDMA hardware"""
    return IntegratedCompiler(HardwareType.RDMA)


def create_cpu_compiler() -> IntegratedCompiler:
    """Create integrated compiler targeting CPU hardware"""
    return IntegratedCompiler(HardwareType.CPU)


# Enhanced DSL functions that use integrated compiler
def allreduce_integrated(reduce_op="sum", algorithm="ring", participants=None,
                        buffer_size=128*1024*1024, hardware_type="cuda", enable_optimizations=True):
    """Create AllReduce with integrated IR lowering"""
    from .config import AllreduceConfig, ReduceOp, AlgorithmType

    hardware_map = {
        "cuda": HardwareType.CUDA,
        "rdma": HardwareType.RDMA,
        "cpu": HardwareType.CPU
    }

    reduce_op_map = {
        "sum": ReduceOp.SUM,
        "avg": ReduceOp.AVG,
        "max": ReduceOp.MAX,
        "min": ReduceOp.MIN
    }

    algorithm_map = {
        "ring": AlgorithmType.RING,
        "tree": AlgorithmType.TREE,
        "rabenseifner": AlgorithmType.RABENSEIFNER
    }

    config = AllreduceConfig(
        reduce_op=reduce_op_map.get(reduce_op, ReduceOp.SUM),
        algorithm=algorithm_map.get(algorithm, AlgorithmType.RING),
        participants=participants or [],
        buffer_size=buffer_size,
        enable_overlap=False
    )

    compiler = IntegratedCompiler(hardware_map.get(hardware_type, HardwareType.CUDA))
    return compiler.compile(config, participants)


def broadcast_integrated(root_rank=0, participants=None, buffer_size=128*1024*1024, hardware_type="cuda"):
    """Create Broadcast with integrated IR lowering"""
    from .config import BroadcastConfig

    hardware_map = {
        "cuda": HardwareType.CUDA,
        "rdma": HardwareType.RDMA,
        "cpu": HardwareType.CPU
    }

    config = BroadcastConfig(
        root_rank=root_rank,
        participants=participants or [root_rank],
        buffer_size=buffer_size
    )

    compiler = IntegratedCompiler(hardware_map.get(hardware_type, HardwareType.CUDA))
    return compiler.compile(config, participants)