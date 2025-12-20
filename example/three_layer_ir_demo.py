#!/usr/bin/env python3
"""
Three-Layer IR Architecture Demo

This example demonstrates the complete L1 → L2 → L3 lowering pipeline
from high-level collective operations to hardware-specific primitives.

The demo shows:
1. L1: High-level AllReduce collective operation
2. L2: Lowered to Write/Reduce/Copy/Signal/Wait primitive operations
3. L3: Further lowered to CUDA/RDMA hardware primitives
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
from typing import Dict, Any

from pccl.ir.primitive_ir import *
from pccl.ir.json_serializer import IRSerializer
from pccl.passes.manager import PassManager
from pccl.passes.pipeline import PassPipeline
from pccl.passes.collective_to_primitive import CollectiveToPrimitivePass
from pccl.passes.primitive_to_hardware import *
from pccl.plugins.hardware_primitives import HardwareType


def create_allreduce_collective() -> CollectiveOperation:
    """Create a high-level AllReduce collective operation."""

    # Create input chunks on different devices
    devices = [Device(i, DeviceType.CUDA) for i in range(4)]
    input_chunks = [
        Chunk(f"rank_{i}_data", 1024 * 1024, DataType.FLOAT32, devices[i])
        for i in range(4)
    ]

    # Create AllReduce operation
    allreduce = CollectiveOperation(
        name="allreduce_0",
        op_type=CollectiveOpType.ALLREDUCE,
        inputs=input_chunks,
        output=Chunk("allreduce_result", 1024 * 1024, DataType.FLOAT32, Device(0, DeviceType.CUDA)),
        algorithm=CollectiveAlgorithm.RING,
        reduce_type=ReduceOp.REDUCE_SUM
    )

    return allreduce


def demonstrate_l1_to_l2_lowering():
    """Demonstrate L1 (Collective) to L2 (Primitive) lowering."""

    print("=" * 80)
    print("L1 → L2 Lowering: Collective to Primitive Operations")
    print("=" * 80)

    # Create high-level collective operation
    allreduce = create_allreduce_collective()
    print(f"L1 Input: {allreduce}")
    print(f"Input chunks: {len(allreduce.inputs)} devices")

    # Apply L1 → L2 lowering
    pass_context = PassContext()
    lowering_pass = CollectiveToPrimitivePass()

    # Create IR graph from collective
    primitive_builder = PrimitiveIRBuilder()
    ir_graph = primitive_builder.from_collective(allreduce)

    print(f"\nL2 Output: {len(ir_graph.operations)} primitive operations")
    for op_id, op in ir_graph.operations.items():
        print(f"  {op_id}: {op.op_type} -> {op.outputs}")


def demonstrate_l2_to_l3_lowering():
    """Demonstrate L2 (Primitive) to L3 (Hardware) lowering for different backends."""

    print("\n" + "=" * 80)
    print("L2 → L3 Lowering: Primitive to Hardware Primitives")
    print("=" * 80)

    # Create primitive IR graph (simplified example)
    allreduce = create_allreduce_collective()
    primitive_builder = PrimitiveIRBuilder()
    ir_graph = primitive_builder.from_collective(allreduce)

    # Test different hardware backends
    backends = [
        (HardwareType.CUDA, "CUDA Multi-Memory Operations"),
        (HardwareType.RDMA, "RDMA Verbs Operations")
    ]

    for hardware_type, description in backends:
        print(f"\n--- {description} ---")

        # Create L2 → L3 lowering pass
        hardware_pass = create_lowering_pass_for_device(hardware_type)

        # Apply lowering
        pass_context = PassContext()
        result = hardware_pass.apply(ir_graph, pass_context)

        if result.success:
            hw_graph = result.ir_graph
            stats = result.transformation_stats

            print(f"L3 Output: {stats['hardware_operations']} hardware operations")
            print(f"Expansion ratio: {stats['expansion_ratio']:.2f}")

            # Show sample hardware operations
            sample_ops = list(hw_graph.operations.keys())[:5]
            for op_id in sample_ops:
                hw_op = hw_graph.operations[op_id]
                print(f"  {op_id}: {hw_op.op_type} [{len(hw_op.inputs)} inputs -> {len(hw_op.outputs)} outputs]")
        else:
            print(f"Lowering failed: {result.message}")


def demonstrate_complete_lowering_pipeline():
    """Demonstrate the complete L1 → L2 → L3 lowering pipeline with optimizations."""

    print("\n" + "=" * 80)
    print("Complete L1 → L2 → L3 Lowering Pipeline with Optimizations")
    print("=" * 80)

    # Create high-level collective operation
    allreduce = create_allreduce_collective()
    print(f"L1 Input: {allreduce.op_type} on {len(allreduce.inputs)} devices")

    # Create pass pipeline
    pipeline = PassPipeline()

    # Stage 1: L1 → L2 lowering
    pipeline.add_pass("collective_to_primitive")

    # Stage 2: L2 → L3 lowering (CUDA)
    pipeline.add_pass("primitive_to_cuda")

    # Stage 3: Hardware-specific optimizations
    pipeline.add_pass("hardware_fusion")
    pipeline.add_pass("hardware_memory_layout_optimization")

    # Execute pipeline
    pass_context = PassContext()

    # Start with primitive IR
    primitive_builder = PrimitiveIRBuilder()
    ir_graph = primitive_builder.from_collective(allreduce)

    print(f"\nInitial L2 graph: {len(ir_graph.operations)} primitive operations")

    # Apply pipeline stages
    start_time = time.time()

    for i, stage_result in enumerate(pipeline.execute_stages(ir_graph, pass_context)):
        stage_name = pipeline.passes[i].name
        if stage_result.success:
            current_graph = stage_result.ir_graph
            stats = stage_result.transformation_stats

            print(f"Stage {i+1} ({stage_name}): {len(current_graph.operations)} operations")
            if 'expansion_ratio' in stats:
                print(f"  Expansion ratio: {stats['expansion_ratio']:.2f}")
            if 'fusion_patterns' in stats and stats['fusion_patterns']:
                print(f"  Fusion patterns: {', '.join(stats['fusion_patterns'])}")
        else:
            print(f"Stage {i+1} ({stage_name}) failed: {stage_result.message}")
            break

    total_time = time.time() - start_time
    print(f"\nPipeline completed in {total_time:.4f} seconds")

    # Serialize final hardware IR to JSON
    if 'current_graph' in locals():
        serializer = IRSerializer()
        json_output = serializer.serialize_graph(current_graph)

        print(f"\nFinal Hardware IR (JSON): {len(json_output)} characters")
        print("Sample JSON structure:")

        # Parse and show sample of JSON
        parsed = json.loads(json_output)
        print(f"  Metadata: {parsed.get('metadata', {}).__class__.__name__}")
        print(f"  Values: {len(parsed.get('values', []))}")
        print(f"  Operations: {len(parsed.get('operations', []))}")

        # Show sample hardware operation
        operations = parsed.get('operations', {})
        if operations:
            sample_op_key = list(operations.keys())[0]
            sample_op = operations[sample_op_key]
            print(f"  Sample operation:")
            print(f"    Type: {sample_op.get('op_type', 'unknown')}")
            print(f"    Attributes: {list(sample_op.get('attributes', {}).keys())}")


def demonstrate_different_user_levels():
    """Demonstrate the three-tier user API design."""

    print("\n" + "=" * 80)
    print("Three-Tier User API Demonstration")
    print("=" * 80)

    # Level 1: Beginner API (High-level collectives)
    print("\n--- Level 1: Beginner API ---")
    print("# Simple high-level collective call")
    print("pccl.allreduce(tensor_data, op='sum', algorithm='ring')")

    # Level 2: Developer API (Primitive IR access)
    print("\n--- Level 2: Developer API ---")
    print("# Access to primitive IR for custom optimization")

    allreduce = create_allreduce_collective()
    primitive_builder = PrimitiveIRBuilder()
    ir_graph = primitive_builder.from_collective(allreduce)

    print(f"ir = pccl.lower_to_primitive(allreduce_op)")
    print(f"ir = pccl.optimize(ir, ['fusion', 'memory_layout'])")
    print(f"# Result: {len(ir_graph.operations)} primitive operations")

    # Level 3: Engineer API (Hardware primitives)
    print("\n--- Level 3: Engineer API ---")
    print("# Fine-grained hardware primitive control")

    cuda_pass = create_lowering_pass_for_device(HardwareType.CUDA)
    pass_context = PassContext()
    result = cuda_pass.apply(ir_graph, pass_context)

    if result.success:
        hw_graph = result.ir_graph
        print(f"hw_ir = pccl.lower_to_hardware(ir, target='cuda')")
        print(f"hw_ir = pccl.apply_hardware_optimizations(hw_ir, ['multimem_reduce', 'warp_fusion'])")
        print(f"# Result: {len(hw_graph.operations)} hardware operations")

        # Show hardware-specific operations
        cuda_ops = [op for op in hw_graph.operations.values() if 'cuda' in op.op_type or 'multimem' in op.op_type]
        print(f"  CUDA-specific operations: {len(cuda_ops)}")
        for op in cuda_ops[:3]:
            print(f"    - {op.op_type}")


def benchmark_lowering_pipeline():
    """Benchmark the lowering pipeline performance."""

    print("\n" + "=" * 80)
    print("Lowering Pipeline Performance Benchmark")
    print("=" * 80)

    # Test different problem sizes
    sizes = [1024, 1024*1024, 4*1024*1024]  # 1KB, 1MB, 4MB
    num_ranks = 4

    print(f"{'Size':<10} {'L1→L2 (ms)':<12} {'L2→L3 (ms)':<12} {'Total (ms)':<12} {'L3 Ops':<8}")
    print("-" * 60)

    for size in sizes:
        # Create test collective
        devices = [Device(i, DeviceType.CUDA) for i in range(num_ranks)]
        input_chunks = [
            Chunk(f"rank_{i}_data", size, DataType.FLOAT32, devices[i])
            for i in range(num_ranks)
        ]

        allreduce = CollectiveOperation(
            name=f"allreduce_{size}",
            op_type=CollectiveOpType.ALLREDUCE,
            inputs=input_chunks,
            output=Chunk(f"result_{size}", size, DataType.FLOAT32, Device(0, DeviceType.CUDA)),
            algorithm=CollectiveAlgorithm.RING,
            reduce_type=ReduceOp.REDUCE_SUM
        )

        # Benchmark L1 → L2
        primitive_builder = PrimitiveIRBuilder()
        start_time = time.time()
        ir_graph = primitive_builder.from_collective(allreduce)
        l1_to_l2_time = (time.time() - start_time) * 1000

        # Benchmark L2 → L3
        cuda_pass = create_lowering_pass_for_device(HardwareType.CUDA)
        pass_context = PassContext()
        start_time = time.time()
        result = cuda_pass.apply(ir_graph, pass_context)
        l2_to_l3_time = (time.time() - start_time) * 1000

        # Count hardware operations
        if result.success:
            hw_ops = len(result.ir_graph.operations)
        else:
            hw_ops = 0

        total_time = l1_to_l2_time + l2_to_l3_time

        # Format size
        if size >= 1024*1024:
            size_str = f"{size//(1024*1024)}MB"
        elif size >= 1024:
            size_str = f"{size//1024}KB"
        else:
            size_str = f"{size}B"

        print(f"{size_str:<10} {l1_to_l2_time:<12.4f} {l2_to_l3_time:<12.4f} {total_time:<12.4f} {hw_ops:<8}")


def main():
    """Main demonstration function."""

    print("PCCL Three-Layer IR Architecture Demonstration")
    print("==============================================")

    try:
        # Run demonstrations
        demonstrate_l1_to_l2_lowering()
        demonstrate_l2_to_l3_lowering()
        demonstrate_complete_lowering_pipeline()
        demonstrate_different_user_levels()
        benchmark_lowering_pipeline()

        print("\n" + "=" * 80)
        print("✅ All demonstrations completed successfully!")
        print("=" * 80)

        print("\nKey Features Demonstrated:")
        print("  ✅ L1 → L2: Collective to primitive lowering")
        print("  ✅ L2 → L3: Primitive to hardware primitives (CUDA/RDMA)")
        print("  ✅ Hardware-specific optimizations (fusion, memory layout)")
        print("  ✅ Complete lowering pipeline with JSON serialization")
        print("  ✅ Three-tier user API design")
        print("  ✅ Performance benchmarking")

        print("\nArchitecture Benefits:")
        print("  🎯 Progressive abstraction levels for different users")
        print("  🚀 Hardware-specific optimizations and primitives")
        print("  🔧 Extensible pass system for custom optimizations")
        print("  📊 JSON-based Python-C++ integration")
        print("  ⚡ Plugin-based hardware support")

    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())