"""
AllReduce Communication Example

This example demonstrates different AllReduce algorithms and configurations
available in PCCL's Python DSL.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import pccl
from pccl.lang import (
    allreduce, ring_topology, tree_topology, hierarchical_topology,
    ConfigBuilder, AllreduceAlgorithm, ReduceOp, TopologyOptimizer
)

def benchmark_allreduce(algorithm_name, allreduce_op, data_size=1024*1024, num_iterations=10):
    """Benchmark an AllReduce operation"""
    print(f"\n--- Benchmarking {algorithm_name} ---")

    if TORCH_AVAILABLE:
        input_tensor = torch.randn(data_size // 4)
        output_tensor = torch.zeros_like(input_tensor)
    else:
        input_tensor = np.random.randn(data_size // 4).astype(np.float32)
        output_tensor = np.zeros_like(input_tensor)

    print(f"Data size: {data_size} bytes")
    print(f"Input tensor shape: {input_tensor.shape}")

    try:
        plan = allreduce_op.compile()
        print(f"Compiled {algorithm_name} plan")

        cost_estimate = allreduce_op.estimate_cost()
        print(f"Estimated cost: {cost_estimate}")

        times = []
        for i in range(num_iterations):
            start_time = time.perf_counter()
            result = allreduce_op.execute(input_tensor)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)

        bandwidth_gb_s = (data_size / avg_time) / (1024**3) if avg_time > 0 else 0

        print(f"Average time: {avg_time:.3f} ms")
        print(f"Min time: {min_time:.3f} ms")
        print(f"Max time: {max_time:.3f} ms")
        print(f"Bandwidth: {bandwidth_gb_s:.3f} GB/s")

        return avg_time, bandwidth_gb_s

    except Exception as e:
        print(f"Failed to execute {algorithm_name}: {e}")
        return None, None

def demonstrate_ring_allreduce():
    """Demonstrate Ring AllReduce algorithm"""
    print("=== Ring AllReduce Demo ===")

    devices = [0, 1, 2, 3]
    ring_allreduce = allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=devices,
        buffer_size=64 * 1024 * 1024
    )

    return benchmark_allreduce("Ring AllReduce", ring_allreduce)

def demonstrate_tree_allreduce():
    """Demonstrate Tree AllReduce algorithm"""
    print("=== Tree AllReduce Demo ===")

    devices = [0, 1, 2, 3, 4, 5, 6, 7]
    tree_allreduce = allreduce(
        reduce_op="sum",
        algorithm="tree",
        participants=devices,
        buffer_size=64 * 1024 * 1024
    )

    return benchmark_allreduce("Tree AllReduce", tree_allreduce)

def demonstrate_hierarchical_allreduce():
    """Demonstrate Hierarchical AllReduce algorithm"""
    print("=== Hierarchical AllReduce Demo ===")

    devices = [0, 1, 2, 3, 4, 5, 6, 7]

    hierarchical_config = ConfigBuilder.hierarchical_allreduce(
        reduce_op="sum",
        participants=devices,
        node_size=4,
        intra_interconnect=pccl.lang.InterconnectType.NVLINK,
        inter_interconnect=pccl.lang.InterconnectType.RDMA,
        buffer_size=128 * 1024 * 1024
    )

    hierarchical_allreduce = pccl.Allreduce(
        reduce_op="sum",
        algorithm="rabenseifner",
        participants=devices,
        topology=hierarchical_config.topology,
        buffer_size=128 * 1024 * 1024
    )

    return benchmark_allreduce("Hierarchical AllReduce", hierarchical_allreduce)

def demonstrate_pipeline_allreduce():
    """Demonstrate Pipeline AllReduce with compute overlap"""
    print("=== Pipeline AllReduce Demo ===")

    devices = [0, 1, 2, 3]
    pipeline_allreduce = pccl.PipelineAllreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=devices,
        compute_chunks=4,
        communication_chunks=2,
        buffer_size=64 * 1024 * 1024
    )

    return benchmark_allreduce("Pipeline AllReduce", pipeline_allreduce)

def demonstrate_reduce_operations():
    """Demonstrate different reduction operations"""
    print("\n=== Reduction Operations Demo ===")

    devices = [0, 1, 2, 3]
    reduce_ops = ["sum", "avg", "max", "min"]

    for reduce_op in reduce_ops:
        print(f"\n--- {reduce_op.upper()} Reduction ---")
        allreduce_op = allreduce(
            reduce_op=reduce_op,
            algorithm="ring",
            participants=devices,
            buffer_size=1024 * 1024
        )

        if TORCH_AVAILABLE:
            input_tensor = torch.ones(256) * (devices.index(0) + 1)
        else:
            input_tensor = np.ones(256, dtype=np.float32) * (devices.index(0) + 1)

        try:
            result = allreduce_op.execute(input_tensor)

            if TORCH_AVAILABLE:
                print(f"Input sum: {input_tensor.sum().item():.1f}")
                print(f"Result sum: {result.sum().item():.1f}")
            else:
                print(f"Input sum: {input_tensor.sum():.1f}")
                print(f"Result sum: {result.sum():.1f}")

        except Exception as e:
            print(f"Failed to execute {reduce_op} reduction: {e}")

def demonstrate_scalability():
    """Demonstrate AllReduce scalability across different data sizes"""
    print("\n=== Scalability Demo ===")

    devices = [0, 1, 2, 3]
    data_sizes = [1024, 64*1024, 1024*1024, 16*1024*1024]  # 1KB to 16MB

    ring_allreduce = allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=devices
    )

    print("Data Size\tTime (ms)\tBandwidth (GB/s)")
    print("-" * 40)

    for data_size in data_sizes:
        if TORCH_AVAILABLE:
            input_tensor = torch.randn(data_size // 4)
        else:
            input_tensor = np.random.randn(data_size // 4).astype(np.float32)

        try:
            plan = ring_allreduce.compile()
            start_time = time.perf_counter()
            result = ring_allreduce.execute(input_tensor)
            end_time = time.perf_counter()

            time_ms = (end_time - start_time) * 1000
            bandwidth_gb_s = (data_size / time_ms) / (1024**3) * 1000

            if data_size >= 1024*1024:
                size_str = f"{data_size // (1024*1024)}MB"
            elif data_size >= 1024:
                size_str = f"{data_size // 1024}KB"
            else:
                size_str = f"{data_size}B"

            print(f"{size_str}\t\t{time_ms:.3f}\t\t{bandwidth_gb_s:.3f}")

        except Exception as e:
            print(f"{data_size}\tFailed: {e}")

def demonstrate_topology_optimization():
    """Demonstrate topology-aware AllReduce optimization"""
    print("\n=== Topology Optimization Demo ===")

    devices = [0, 1, 2, 3, 4, 5, 6, 7]

    ring_topo = ring_topology(devices, bandwidth=25.0, latency=0.8)
    tree_topo = tree_topology(devices, branching_factor=2, bandwidth=25.0, latency=0.8)

    ring_metrics = ring_topo['metrics']
    tree_metrics = tree_topo['metrics']

    print(f"Ring Topology:")
    print(f"  Total bandwidth: {ring_metrics.total_bandwidth:.1f} GB/s")
    print(f"  Network diameter: {ring_metrics.network_diameter}")
    print(f"  Connectivity: {ring_metrics.connectivity:.1f}%")

    print(f"Tree Topology:")
    print(f"  Total bandwidth: {tree_metrics.total_bandwidth:.1f} GB/s")
    print(f"  Network diameter: {tree_metrics.network_diameter}")
    print(f"  Connectivity: {tree_metrics.connectivity:.1f}%")

    data_size = 4 * 1024 * 1024  # 4MB

    ring_allreduce = allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=devices,
        buffer_size=data_size
    )

    tree_allreduce = allreduce(
        reduce_op="sum",
        algorithm="tree",
        participants=devices,
        buffer_size=data_size
    )

    if TORCH_AVAILABLE:
        input_tensor = torch.randn(data_size // 4)
    else:
        input_tensor = np.random.randn(data_size // 4).astype(np.float32)

    print(f"\nData size: {data_size // (1024*1024)}MB")

    try:
        ring_plan = ring_allreduce.compile()
        tree_plan = tree_allreduce.compile()

        ring_cost = ring_allreduce.estimate_cost()
        tree_cost = tree_allreduce.estimate_cost()

        print(f"Ring AllReduce estimated cost: {ring_cost}")
        print(f"Tree AllReduce estimated cost: {tree_cost}")

    except Exception as e:
        print(f"Topology comparison failed: {e}")

def main():
    """Main demonstration function"""
    print("PCCL AllReduce Communication Example")
    print("=" * 50)

    results = {}

    try:
        ring_time, ring_bw = demonstrate_ring_allreduce()
        if ring_time is not None:
            results['ring'] = (ring_time, ring_bw)

        tree_time, tree_bw = demonstrate_tree_allreduce()
        if tree_time is not None:
            results['tree'] = (tree_time, tree_bw)

        hier_time, hier_bw = demonstrate_hierarchical_allreduce()
        if hier_time is not None:
            results['hierarchical'] = (hier_time, hier_bw)

        pipe_time, pipe_bw = demonstrate_pipeline_allreduce()
        if pipe_time is not None:
            results['pipeline'] = (pipe_time, pipe_bw)

        demonstrate_reduce_operations()
        demonstrate_scalability()
        demonstrate_topology_optimization()

        print("\n" + "=" * 50)
        print("AllReduce Comparison Summary:")
        print("-" * 30)
        for algo, (time_ms, bw) in results.items():
            print(f"{algo.capitalize():12} {time_ms:8.3f}ms {bw:8.3f}GB/s")

        print("\nAll demonstrations completed successfully!")

    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()