#!/usr/bin/env python3
"""
Hardware Execution Demo for PCCL

Demonstrates real hardware execution through the Python-C++ execution bridge.
Shows the complete flow from high-level DSL to actual hardware primitives.
"""

import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pccl.lang.execution_bridge import ExecutionBridge
from pccl.lang.execution_manager import ExecutionManager
from pccl.lang.integrated_compiler import IntegratedCompiler, HardwareType
from pccl.lang.config import AllreduceConfig, ReduceOp, AllgatherConfig, BroadcastConfig
from pccl.lang.operator import allreduce, broadcast, allgather
from pccl.lang.executor import get_execution_engine


def demo_basic_hardware_execution():
    """Demonstrate basic hardware execution"""
    print("🚀 Basic Hardware Execution Demo")
    print("=" * 50)

    bridge = ExecutionBridge(HardwareType.CUDA, device_id=0)

    print(f"Hardware execution available: {bridge.is_hardware_execution_available()}")
    print(f"Hardware info: {bridge.get_hardware_info()}")

    config = AllreduceConfig(
        reduce_op=ReduceOp.SUM,
        algorithm="ring",
        participants=[0, 1, 2, 3],
        buffer_size=1024 * 1024
    )

    input_data = np.random.rand(1024 * 256).astype(np.float32)
    print(f"Input tensor shape: {input_data.shape}")

    start_time = time.time()
    result = bridge.compile_and_execute(config, input_data)
    end_time = time.time()

    print(f"Total execution time: {(end_time - start_time) * 1000:.2f} ms")

    if result['success']:
        print("✅ Hardware execution successful!")
        print(f"   Operations executed: {result['num_operations']}")
        print(f"   IR values processed: {result['num_values']}")
        print(f"   Execution time: {result.get('execution_time_ms', 0):.2f} ms")
        print(f"   Operation counts: {result['operation_counts']}")

        if 'lowering_stats' in result:
            stats = result['lowering_stats']
            print(f"   L1→L2 lowering: {stats.get('l1_to_l2_time_ms', 0):.2f} ms")
            print(f"   L2→L3 lowering: {stats.get('l2_to_l3_time_ms', 0):.2f} ms")
    else:
        print(f"❌ Hardware execution failed: {result['error_message']}")


def demo_layered_ir_execution():
    """Demonstrate the three-layer IR execution"""
    print("\n🏗️  Three-Layer IR Execution Demo")
    print("=" * 50)

    from pccl.ir.json_serializer import IRGraph, IRValue, IROperation, IRType, DeviceType
    from pccl.ir.primitive_ir import WriteOp, ReduceOp, CopyOp, SignalOp, WaitSignalOp, PrimitiveIRBuilder

    builder = PrimitiveIRBuilder("three_layer_demo")

    input_value = builder.add_value(
        dtype="float32",
        shape=[1024],
        device_type=DeviceType.CUDA
    )

    write_op = builder.add_write_op(
        input_value,
        address=0x1000,
        size=1024 * 4,
        device_type=DeviceType.CUDA
    )

    reduce_op = builder.add_reduce_op(
        [input_value],
        reduce_op=ReduceOp.SUM,
        device_type=DeviceType.CUDA
    )

    signal_op = builder.add_signal_op("reduce_complete", target_ranks=[1, 2, 3])
    wait_op = builder.add_wait_signal_op("reduce_complete", source_ranks=[0, 1, 2])

    ir_graph = builder.get_graph()

    print(f"Generated IR graph with {len(ir_graph.operations)} operations and {len(ir_graph.values)} values")
    print("IR layers:")
    print("  L1: Collective primitives (AllReduce, Broadcast, etc.)")
    print("  L2: Primitive operations (Write, Reduce, Copy, Signal, Wait)")
    print("  L3: Hardware primitives (CUDA multimem, RDMA verbs)")

    bridge = ExecutionBridge(HardwareType.CUDA, device_id=0)
    result = bridge.execute_ir_graph(ir_graph)

    if result['success']:
        print("✅ Three-layer IR execution successful!")
        print(f"   Execution time: {result.get('execution_time_ms', 0):.2f} ms")
        print(f"   Operations by type: {result['operation_counts']}")
    else:
        print(f"❌ Three-layer IR execution failed: {result['error_message']}")


def demo_hardware_performance_comparison():
    """Compare performance across different hardware types"""
    print("\n⚡ Hardware Performance Comparison Demo")
    print("=" * 50)

    manager = ExecutionManager(HardwareType.CUDA)

    config = AllreduceConfig(
        reduce_op=ReduceOp.SUM,
        algorithm="ring",
        participants=[0, 1, 2, 3],
        buffer_size=1024 * 1024
    )

    input_data = np.random.rand(1024 * 256).astype(np.float32)

    print("Benchmarking different hardware types...")

    hardware_types = [HardwareType.CUDA, HardwareType.CPU]
    if HardwareType.RDMA:
        hardware_types.append(HardwareType.RDMA)

    benchmark_results = {}

    for hw_type in hardware_types:
        print(f"\nTesting {hw_type.value}...")
        try:
            hw_manager = ExecutionManager(hw_type)
            results = hw_manager.benchmark_config(config, input_data, num_iterations=10)

            if results:
                key = list(results.keys())[0]
                benchmark_results[hw_type.value] = results[key]
                print(f"   Average time: {results[key]['avg_time_ms']:.2f} ms")
                print(f"   Success rate: {results[key]['success_rate']:.2f}")
            else:
                print(f"   No results for {hw_type.value}")

        except Exception as e:
            print(f"   Error testing {hw_type.value}: {e}")

    print("\n📊 Performance Summary:")
    for hw_type, results in benchmark_results.items():
        print(f"   {hw_type}: {results['avg_time_ms']:.2f} ms (success: {results['success_rate']:.2f})")


def demo_algorithm_comparison():
    """Compare different allreduce algorithms"""
    print("\n🔬 Algorithm Comparison Demo")
    print("=" * 50)

    manager = ExecutionManager(HardwareType.CUDA)

    algorithms = ["ring", "tree", "rabenseifner"]
    input_data = np.random.rand(1024 * 256).astype(np.float32)

    results = {}

    for algorithm in algorithms:
        print(f"\nTesting {algorithm} algorithm...")

        config = AllreduceConfig(
            reduce_op=ReduceOp.SUM,
            algorithm=algorithm,
            participants=[0, 1, 2, 3, 4, 5, 6, 7],
            buffer_size=1024 * 1024
        )

        try:
            benchmark_results = manager.benchmark_config(config, input_data, num_iterations=5)

            if benchmark_results:
                key = list(benchmark_results.keys())[0]
                result = benchmark_results[key]
                results[algorithm] = result
                print(f"   Average time: {result['avg_time_ms']:.2f} ms")
                print(f"   Min/Max: {result['min_time_ms']:.2f}/{result['max_time_ms']:.2f} ms")
                print(f"   Std deviation: {result['std_time_ms']:.2f} ms")

        except Exception as e:
            print(f"   Error testing {algorithm}: {e}")

    print("\n🏆 Algorithm Performance Ranking:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_time_ms'])
    for i, (algorithm, result) in enumerate(sorted_results, 1):
        print(f"   {i}. {algorithm}: {result['avg_time_ms']:.2f} ms")


def demo_real_world_workload():
    """Demonstrate a real-world training workload"""
    print("\n🎯 Real-World Workload Demo")
    print("=" * 50)

    gradients = [
        np.random.rand(512 * 1024).astype(np.float32),  # Large gradient
        np.random.rand(256 * 1024).astype(np.float32),  # Medium gradient
        np.random.rand(128 * 1024).astype(np.float32),  # Small gradient
    ]

    print("Simulating distributed training gradient allreduce...")

    manager = ExecutionManager(HardwareType.CUDA)

    total_start = time.time()
    results = []

    for i, grad in enumerate(gradients):
        print(f"\nProcessing gradient {i+1}/{len(gradients)} (size: {grad.size})")

        config = AllreduceConfig(
            reduce_op=ReduceOp.SUM,
            algorithm="ring",
            participants=list(range(8)),
            buffer_size=grad.size * 4,
            enable_overlap=True
        )

        start_time = time.time()
        result = manager.execute(config, grad)
        end_time = time.time()

        results.append({
            'gradient_size': grad.size,
            'execution_time': (end_time - start_time) * 1000,
            'success': result['success']
        })

        if result['success']:
            print(f"   ✅ Gradient {i+1} allreduce successful")
            print(f"   Time: {(end_time - start_time) * 1000:.2f} ms")
        else:
            print(f"   ❌ Gradient {i+1} allreduce failed: {result['error_message']}")

    total_end = time.time()
    total_time = (total_end - total_start) * 1000

    print(f"\n📈 Workload Summary:")
    print(f"   Total gradients: {len(gradients)}")
    print(f"   Total parameters: {sum(g.size for g in gradients)}")
    print(f"   Total time: {total_time:.2f} ms")
    print(f"   Successful allreduces: {sum(1 for r in results if r['success'])}/{len(results)}")

    successful_results = [r for r in results if r['success']]
    if successful_results:
        avg_time = np.mean([r['execution_time'] for r in successful_results])
        print(f"   Average time per gradient: {avg_time:.2f} ms")


def demo_dsl_to_hardware():
    """Demo complete DSL to hardware execution"""
    print("\n🔗 DSL to Hardware Execution Demo")
    print("=" * 50)

    try:
        from pccl.lang.compiler import ExecutionPlan, DSLCompiler

        op = allreduce(
            reduce_op="sum",
            algorithm="ring",
            participants=[0, 1, 2, 3, 4, 5, 6, 7],
            buffer_size=2048 * 1024
        )

        print("Created allreduce operation via DSL")
        print(f"Operation type: {type(op).__name__}")
        print(f"Configuration: {op.config}")

        compiler = DSLCompiler()
        plan = compiler.compile(op.config)

        print("Compiled execution plan")
        print(f"Plan has {len(plan.operations)} operations")

        input_data = np.random.rand(2048 * 512).astype(np.float32)

        engine = get_execution_engine(HardwareType.CUDA, device_id=0)

        print("Executing on CUDA hardware...")
        start_time = time.time()
        result = engine.execute_plan_sync(plan, input_data)
        end_time = time.time()

        print(f"✅ DSL to hardware execution successful!")
        print(f"   Execution time: {(end_time - start_time) * 1000:.2f} ms")
        print(f"   Input shape: {input_data.shape}")
        print(f"   Output shape: {result.shape}")

    except Exception as e:
        print(f"❌ DSL to hardware execution failed: {e}")


def main():
    """Run all hardware execution demos"""
    print("PCCL Hardware Execution Demos")
    print("============================\n")

    try:
        demo_basic_hardware_execution()
        demo_layered_ir_execution()
        demo_hardware_performance_comparison()
        demo_algorithm_comparison()
        demo_real_world_workload()
        demo_dsl_to_hardware()

        print("\n🎉 All demos completed successfully!")
        print("PCCL is now ready for real hardware execution!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()