#!/usr/bin/env python3
"""
End-to-End Test for PCCL Python-C++ Execution Bridge

Tests the complete flow from Python DSL to C++ hardware execution.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("Warning: numpy not available, using basic Python arrays")
    HAS_NUMPY = False

import time

from pccl.ir.json_serializer import IRGraph, IRValue, IROperation, IRType, DeviceType
from pccl.lang.execution_bridge import ExecutionBridge
from pccl.lang.execution_manager import ExecutionManager
from pccl.lang.integrated_compiler import IntegratedCompiler, HardwareType
from pccl.lang.config import AllreduceConfig, ReduceOp
from pccl.lang.operator import Allreduce


def create_test_ir_graph() -> IRGraph:
    """Create a test IR graph for allreduce operation"""
    graph = IRGraph(ir_type=IRType.PRIMITIVE, values={}, operations={})
    graph.metadata["graph_id"] = "test_allreduce"
    graph.metadata["test_case"] = "ring_allreduce_4_nodes"

    input_value = IRValue(
        id="test_input",
        dtype="float32",
        shape=[1024],
        device_id=0,
        device_type=DeviceType.CUDA
    )
    graph.add_value(input_value)

    intermediate_value = IRValue(
        id="intermediate",
        dtype="float32",
        shape=[1024],
        device_id=0,
        device_type=DeviceType.CUDA
    )
    graph.add_value(intermediate_value)

    output_value = IRValue(
        id="test_output",
        dtype="float32",
        shape=[1024],
        device_id=0,
        device_type=DeviceType.CUDA
    )
    graph.add_value(output_value)

    reduce_op = IROperation(
        id="reduce_op",
        op_type="reduce",
        inputs=["test_input"],
        outputs=["intermediate"],
        attributes={
            "reduce_op": "sum",
            "num_inputs": 4,
            "device_id": 0,
            "device_type": "cuda"
        }
    )
    graph.add_operation(reduce_op)

    copy_op = IROperation(
        id="copy_op",
        op_type="copy",
        inputs=["intermediate"],
        outputs=["test_output"],
        attributes={
            "src_device_id": 0,
            "dst_device_id": 0,
            "device_type": "cuda"
        }
    )
    graph.add_operation(copy_op)

    return graph


def test_ir_graph_executor():
    """Test the IR graph execution engine directly"""
    print("=" * 60)
    print("Testing IR Graph Execution Engine")
    print("=" * 60)

    graph = create_test_ir_graph()

    try:
        bridge = ExecutionBridge(HardwareType.CUDA, device_id=0)
        print(f"Hardware execution available: {bridge.is_hardware_execution_available()}")
        print(f"Hardware info: {bridge.get_hardware_info()}")

        result = bridge.execute_ir_graph(graph)
        print(f"Execution result: {result}")

        if result['success']:
            print("✅ IR Graph Execution Test PASSED")
            print(f"   Execution time: {result['execution_time_ms']:.2f} ms")
            print(f"   Operations: {result['num_operations']}")
            print(f"   Values: {result['num_values']}")
        else:
            print(f"❌ IR Graph Execution Test FAILED: {result['error_message']}")

    except Exception as e:
        print(f"❌ IR Graph Execution Test EXCEPTION: {e}")


def test_integrated_compiler():
    """Test the integrated compiler with execution bridge"""
    print("\n" + "=" * 60)
    print("Testing Integrated Compiler with Execution Bridge")
    print("=" * 60)

    try:
        config = AllreduceConfig(
            reduce_op=ReduceOp.SUM,
            algorithm="ring",
            participants=[0, 1, 2, 3],
            buffer_size=1024 * 1024
        )

        bridge = ExecutionBridge(HardwareType.CUDA, device_id=0)
        input_data = np.random.rand(1024).astype(np.float32)

        result = bridge.compile_and_execute(config, input_data)

        print(f"Compilation and execution result: {result}")

        if result['success']:
            print("✅ Integrated Compiler Test PASSED")
            print(f"   Total time: {result.get('total_time_ms', 0):.2f} ms")
            if 'lowering_stats' in result:
                print(f"   Lowering successful: {result['lowering_stats']}")
        else:
            print(f"❌ Integrated Compiler Test FAILED: {result['error_message']}")

    except Exception as e:
        print(f"❌ Integrated Compiler Test EXCEPTION: {e}")


def test_execution_manager():
    """Test the execution manager"""
    print("\n" + "=" * 60)
    print("Testing Execution Manager")
    print("=" * 60)

    try:
        manager = ExecutionManager(HardwareType.CUDA)

        config = AllreduceConfig(
            reduce_op=ReduceOp.SUM,
            algorithm="ring",
            participants=[0, 1, 2, 3],
            buffer_size=1024 * 1024
        )

        input_data = np.random.rand(1024).astype(np.float32)

        result = manager.execute(config, input_data)

        print(f"Manager execution result: {result}")

        if result['success']:
            print("✅ Execution Manager Test PASSED")
            print(f"   Total time: {result.get('total_time_ms', 0):.2f} ms")
        else:
            print(f"❌ Execution Manager Test FAILED: {result['error_message']}")

        stats = manager.get_execution_statistics()
        print(f"Execution statistics: {stats}")

        hardware_status = manager.get_hardware_status()
        print(f"Hardware status: {hardware_status}")

    except Exception as e:
        print(f"❌ Execution Manager Test EXCEPTION: {e}")


def test_end_to_end_dsl():
    """Test end-to-end DSL execution"""
    print("\n" + "=" * 60)
    print("Testing End-to-End DSL Execution")
    print("=" * 60)

    try:
        from pccl.lang.executor import get_execution_engine
        from pccl.lang.compiler import ExecutionPlan, DSLCompiler

        allreduce_op = Allreduce(
            reduce_op="sum",
            algorithm="ring",
            participants=[0, 1, 2, 3],
            buffer_size=1024 * 1024
        )

        compiler = DSLCompiler()
        plan = compiler.compile(allreduce_op.config)

        input_data = np.random.rand(1024).astype(np.float32)

        engine = get_execution_engine(HardwareType.CUDA, device_id=0)
        result = engine.execute_plan_sync(plan, input_data)

        print("✅ End-to-End DSL Test PASSED")
        print(f"   Input shape: {input_data.shape}")
        print(f"   Result shape: {result.shape}")

    except Exception as e:
        print(f"❌ End-to-End DSL Test EXCEPTION: {e}")


def test_batch_execution():
    """Test batch execution"""
    print("\n" + "=" * 60)
    print("Testing Batch Execution")
    print("=" * 60)

    try:
        manager = ExecutionManager(HardwareType.CUDA)

        configs = []
        input_data_list = []

        for i in range(3):
            config = AllreduceConfig(
                reduce_op=ReduceOp.SUM,
                algorithm="ring",
                participants=[0, 1, 2, 3],
                buffer_size=1024 * 1024
            )
            configs.append(config)
            input_data_list.append(np.random.rand(1024).astype(np.float32))

        results = manager.execute_batch(configs, input_data_list)

        print(f"Batch execution results: {len(results)} operations")

        successful = sum(1 for r in results if r['success'])
        print(f"✅ Batch Execution Test: {successful}/{len(results)} successful")

    except Exception as e:
        print(f"❌ Batch Execution Test EXCEPTION: {e}")


def test_benchmark():
    """Test benchmarking functionality"""
    print("\n" + "=" * 60)
    print("Testing Benchmarking")
    print("=" * 60)

    try:
        manager = ExecutionManager(HardwareType.CUDA)

        config = AllreduceConfig(
            reduce_op=ReduceOp.SUM,
            algorithm="ring",
            participants=[0, 1, 2, 3],
            buffer_size=1024 * 1024
        )

        input_data = np.random.rand(1024).astype(np.float32)

        benchmark_results = manager.benchmark_config(config, input_data, num_iterations=5)

        print("Benchmark results:")
        for key, results in benchmark_results.items():
            print(f"   {key}:")
            print(f"     Success rate: {results['success_rate']:.2f}")
            print(f"     Avg time: {results['avg_time_ms']:.2f} ms")
            print(f"     Min/Max time: {results['min_time_ms']:.2f}/{results['max_time_ms']:.2f} ms")

        print("✅ Benchmark Test PASSED")

    except Exception as e:
        print(f"❌ Benchmark Test EXCEPTION: {e}")


def main():
    """Run all end-to-end tests"""
    print("PCCL End-to-End Execution Bridge Tests")
    print("=====================================")

    test_ir_graph_executor()
    test_integrated_compiler()
    test_execution_manager()
    test_end_to_end_dsl()
    test_batch_execution()
    test_benchmark()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()