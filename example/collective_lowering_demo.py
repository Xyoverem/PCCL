#!/usr/bin/env python3
"""
Demo of Collective to Primitive Lowering

Shows how high-level collective operations are lowered to primitive IR.
"""

import sys
import os

# Add the pccl module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pccl.ir.json_serializer import (
    IRGraph, IRValue, IROperation, IRType,
    ReduceOpType, DeviceType
)
from pccl.passes.base import PassContext
from pccl.passes.collective_to_primitive import CollectiveOpType
from pccl.passes.manager import PassManager
from pccl.passes.pipeline import StandardPipelines
import json


def create_collective_allreduce_graph() -> IRGraph:
    """Create a simple AllReduce collective IR graph"""
    graph = IRGraph(ir_type=IRType.COLLECTIVE, values={}, operations={})
    graph.metadata["graph_id"] = "allreduce_demo"

    # Create input value
    input_value = IRValue(
        id="input_tensor",
        dtype="float32",
        shape=[1024],
        device_id=0,
        device_type=DeviceType.CUDA
    )
    graph.add_value(input_value)

    # Create AllReduce operation
    allreduce_op = IROperation(
        id="allreduce_0",
        op_type="allreduce",
        inputs=["input_tensor"],
        outputs=["output_tensor"],
        attributes={
            "reduce_op": "sum",
            "world_size": 4,
            "rank": 0,
            "algorithm": "ring"
        }
    )
    graph.add_operation(allreduce_op)

    # Create output value
    output_value = IRValue(
        id="output_tensor",
        dtype="float32",
        shape=[1024],
        device_id=0,
        device_type=DeviceType.CUDA
    )
    graph.add_value(output_value)

    return graph


def create_collective_broadcast_graph() -> IRGraph:
    """Create a simple Broadcast collective IR graph"""
    graph = IRGraph(ir_type=IRType.COLLECTIVE, values={}, operations={})
    graph.metadata["graph_id"] = "broadcast_demo"

    # Create input value
    input_value = IRValue(
        id="input_data",
        dtype="float32",
        shape=[512],
        device_id=1,
        device_type=DeviceType.CUDA
    )
    graph.add_value(input_value)

    # Create Broadcast operation
    broadcast_op = IROperation(
        id="broadcast_0",
        op_type="broadcast",
        inputs=["input_data"],
        outputs=["output_data"],
        attributes={
            "root_rank": 0,
            "world_size": 4,
            "rank": 1
        }
    )
    graph.add_operation(broadcast_op)

    # Create output value
    output_value = IRValue(
        id="output_data",
        dtype="float32",
        shape=[512],
        device_id=1,
        device_type=DeviceType.CUDA
    )
    graph.add_value(output_value)

    return graph


def demo_collective_lowering():
    """Demonstrate collective to primitive lowering"""
    print("=== PCCL Collective to Primitive Lowering Demo ===\n")

    # Create pass manager
    manager = PassManager(enable_profiling=True, enable_caching=True)

    # Demo 1: AllReduce lowering
    print("1. AllReduce Lowering")
    print("-" * 40)

    allreduce_graph = create_collective_allreduce_graph()
    print(f"Original collective graph:")
    print(f"  Type: {allreduce_graph.ir_type.value}")
    print(f"  Operations: {len(allreduce_graph.operations)}")
    print(f"  Values: {len(allreduce_graph.values)}")

    for op_id, op in allreduce_graph.operations.items():
        print(f"  Operation {op_id}: {op.op_type}({', '.join(op.inputs)}) -> {', '.join(op.outputs)}")

    # Execute lowering
    context = PassContext(
        target_device="cuda",
        optimization_level="default",
        enable_profiling=True
    )

    result = manager.execute_pass("collective_to_primitive", allreduce_graph, context)

    if result.success:
        primitive_graph = result.ir
        print(f"\nLowered primitive graph:")
        print(f"  Type: {primitive_graph.ir_type.value}")
        print(f"  Operations: {len(primitive_graph.operations)}")
        print(f"  Values: {len(primitive_graph.values)}")
        print(f"  Execution time: {result.execution_time:.4f}s")

        # Show lowered operations
        for op_id, op in primitive_graph.operations.items():
            attrs = ', '.join(f"{k}={v}" for k, v in op.attributes.items() if v is not None)
            print(f"  Operation {op_id}: {op.op_type}({', '.join(op.inputs)}) -> {', '.join(op.outputs)} [{attrs}]")

        # Serialize to JSON
        from pccl.ir.json_serializer import serialize_graph
        json_output = serialize_graph(primitive_graph)
        print(f"\nJSON Output (first 500 chars):")
        print(json_output[:500] + "..." if len(json_output) > 500 else json_output)

        # Save to file
        output_file = "allreduce_primitive.json"
        from pccl.ir.json_serializer import serialize_to_file
        serialize_to_file(primitive_graph, output_file)
        print(f"\nSaved to: {output_file}")

    else:
        print(f"Lowering failed: {result.diagnostics}")

    print("\n" + "="*60 + "\n")

    # Demo 2: Broadcast lowering
    print("2. Broadcast Lowering")
    print("-" * 40)

    broadcast_graph = create_collective_broadcast_graph()
    print(f"Original collective graph:")
    print(f"  Type: {broadcast_graph.ir_type.value}")
    print(f"  Operations: {len(broadcast_graph.operations)}")

    # Execute lowering
    result = manager.execute_pass("collective_to_primitive", broadcast_graph, context)

    if result.success:
        primitive_graph = result.ir
        print(f"\nLowered primitive graph:")
        print(f"  Type: {primitive_graph.ir_type.value}")
        print(f"  Operations: {len(primitive_graph.operations)}")

        # Show lowered operations
        for op_id, op in primitive_graph.operations.items():
            attrs = ', '.join(f"{k}={v}" for k, v in op.attributes.items() if v is not None)
            print(f"  Operation {op_id}: {op.op_type} -> [{attrs}]")

    else:
        print(f"Lowering failed: {result.diagnostics}")

    print("\n" + "="*60 + "\n")

    # Demo 3: Pipeline execution
    print("3. Pipeline Execution")
    print("-" * 40)

    pipeline = StandardPipelines.collective_to_primitive_pipeline()
    print(f"Pipeline: {pipeline.name}")
    print(f"Description: {pipeline.description}")
    print(f"Stages: {[stage.name for stage in pipeline.stages]}")

    # Validate pipeline
    errors = pipeline.validate()
    if errors:
        print(f"Pipeline validation errors: {errors}")
    else:
        print("Pipeline validation: PASSED")

    # Execute pipeline
    result = pipeline.execute(allreduce_graph, context)

    if result.success:
        print(f"\nPipeline execution: SUCCESS")
        print(f"  Executed stages: {result.metadata.get('executed_stages', [])}")
        print(f"  Final operations: {len(result.ir.operations)}")
        print(f"  Total execution time: {result.metadata.get('execution_time', 0):.4f}s")
    else:
        print(f"Pipeline execution: FAILED")
        for diagnostic in result.diagnostics:
            print(f"  {diagnostic}")

    # Show execution statistics
    stats = manager.get_execution_statistics()
    print(f"\nExecution Statistics:")
    for pass_name, pass_stats in stats.items():
        print(f"  {pass_name}:")
        print(f"    Executions: {pass_stats['executions']}")
        print(f"    Successes: {pass_stats['successes']}")
        print(f"    Failures: {pass_stats['failures']}")
        print(f"    Avg time: {pass_stats['avg_time']:.4f}s")


if __name__ == "__main__":
    demo_collective_lowering()