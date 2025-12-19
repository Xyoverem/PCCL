"""
Python DSL Demonstration

This example shows how to use PCCL's Python DSL for communication patterns.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import List

import pccl
from pccl.lang import (
    communication, compile, execute,
    ConfigBuilder, AllreduceAlgorithm, ReduceOp,
    ring_topology, hierarchical_topology, TopologyDiscovery
)

@communication
class SimpleAllReduce:
    """Simple AllReduce communication pattern"""
    allreduce_op: pccl.Allreduce = pccl.lang.allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=[0, 1, 2, 3],
        buffer_size=64 * 1024 * 1024
    )

@communication
class TrainingCommunication:
    """Communication pattern for distributed training"""
    gradient_allreduce: pccl.Allreduce = pccl.lang.allreduce(
        reduce_op="sum",
        algorithm="hierarchical",
        participants=[0, 1, 2, 3, 4, 5, 6, 7],
        buffer_size=128 * 1024 * 1024,
        enable_overlap=True
    )

    param_broadcast: pccl.Broadcast = pccl.lang.broadcast(
        root_rank=0,
        participants=[0, 1, 2, 3, 4, 5, 6, 7]
    )

def demonstrate_basic_dsl():
    """Demonstrate basic DSL usage"""
    print("=== Basic DSL Demo ===")

    simple_comm = SimpleAllReduce()
    print(f"Created communication pattern: {simple_comm.name}")

    input_data = np.random.randn(1024, 1024).astype(np.float32)
    print(f"Input data shape: {input_data.shape}")

    try:
        plan = compile(simple_comm, participants=[0, 1, 2, 3])
        print(f"Compiled plan with {len(plan.operations)} operations")

        result = execute(simple_comm, input_data, participants=[0, 1, 2, 3])
        print("DSL execution completed successfully")

    except Exception as e:
        print(f"DSL execution failed: {e}")

def demonstrate_config_builder():
    """Demonstrate configuration builder pattern"""
    print("\n=== Configuration Builder Demo ===")

    ring_config = ConfigBuilder.ring_allreduce(
        reduce_op=ReduceOp.SUM,
        participants=[0, 1, 2, 3],
        buffer_size=64 * 1024 * 1024,
        enable_overlap=True
    )

    print(f"Ring AllReduce config:")
    print(f"  Algorithm: {ring_config.algorithm.name}")
    print(f"  Reduce op: {ring_config.reduce_op.name}")
    print(f"  Buffer size: {ring_config.buffer_size // (1024*1024)}MB")
    print(f"  Overlap: {ring_config.enable_overlap}")

    hierarchical_config = ConfigBuilder.hierarchical_allreduce(
        reduce_op=ReduceOp.SUM,
        participants=[0, 1, 2, 3, 4, 5, 6, 7],
        node_size=4,
        buffer_size=128 * 1024 * 1024
    )

    print(f"\nHierarchical AllReduce config:")
    print(f"  Algorithm: {hierarchical_config.algorithm.name}")
    print(f"  Node size: {hierarchical_config.topology.node_size}")
    print(f"  Intra-bandwidth: {hierarchical_config.topology.intra_bandwidth}GB/s")
    print(f"  Inter-bandwidth: {hierarchical_config.topology.inter_bandwidth}GB/s")

def demonstrate_direct_operators():
    """Demonstrate direct operator creation"""
    print("\n=== Direct Operators Demo ===")

    allreduce_op = pccl.Allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=[0, 1, 2, 3],
        buffer_size=32 * 1024 * 1024
    )

    broadcast_op = pccl.Broadcast(
        root_rank=0,
        participants=[0, 1, 2, 3],
        buffer_size=16 * 1024 * 1024
    )

    allgather_op = pccl.Allgather(
        participants=[0, 1, 2, 3],
        input_size=1024,
        buffer_size=16 * 1024 * 1024
    )

    print("Created operators:")
    print(f"  AllReduce: {allreduce_op.name}")
    print(f"  Broadcast: {broadcast_op.name}")
    print(f"  AllGather: {allgather_op.name}")

    input_data = np.random.randn(512, 512).astype(np.float32)

    try:
        allreduce_result = allreduce_op.execute(input_data, participants=[0, 1, 2, 3])
        print("AllReduce execution completed")

        broadcast_result = broadcast_op.execute(input_data, participants=[0, 1, 2, 3])
        print("Broadcast execution completed")

    except Exception as e:
        print(f"Direct operator execution failed: {e}")

def demonstrate_topology_discovery():
    """Demonstrate automatic topology discovery"""
    print("\n=== Topology Discovery Demo ===")

    discovery = TopologyDiscovery()
    devices = discovery.discover_devices()

    print(f"Discovered {len(devices)} devices:")
    for device in devices:
        print(f"  Device {device.device_id}: {device.device_type}")
        print(f"    Memory bandwidth: {device.memory_bandwidth}GB/s")
        print(f"    Compute capability: {device.compute_capability}")

    topology = discovery.discover_topology(num_devices=4)
    print(f"\nAuto-discovered topology:")
    print(f"  Type: {topology['type'].name}")
    print(f"  Devices: {len(topology['devices'])}")
    print(f"  Links: {len(topology['links'])}")

def demonstrate_manual_topology():
    """Demonstrate manual topology construction"""
    print("\n=== Manual Topology Demo ===")

    devices = [0, 1, 2, 3]

    ring_topo = ring_topology(
        devices=devices,
        bandwidth=25.0,
        latency=0.5
    )

    print(f"Ring topology:")
    print(f"  Devices: {ring_topo['devices']}")
    print(f"  Links: {len(ring_topo['links'])}")
    print(f"  Bandwidth: {ring_topo['metrics'].total_bandwidth}GB/s")
    print(f"  Latency: {ring_topo['metrics'].average_latency}μs")

    node_groups = [[0, 1], [2, 3]]
    hier_topo = hierarchical_topology(
        node_groups=node_groups,
        intra_bandwidth=50.0,
        inter_bandwidth=10.0
    )

    print(f"\nHierarchical topology:")
    print(f"  Node groups: {hier_topo['node_groups']}")
    print(f"  Total devices: {len(hier_topo['devices'])}")
    print(f"  Total bandwidth: {hier_topo['metrics'].total_bandwidth}GB/s")

def demonstrate_compilation_optimization():
    """Demonstrate compilation and optimization"""
    print("\n=== Compilation & Optimization Demo ===")

    config = ConfigBuilder.tree_allreduce(
        reduce_op=ReduceOp.SUM,
        participants=[0, 1, 2, 3, 4, 5, 6, 7],
        branching_factor=2,
        buffer_size=64 * 1024 * 1024,
        enable_overlap=True
    )

    allreduce_op = pccl.Allreduce(
        reduce_op="sum",
        algorithm="tree",
        participants=[0, 1, 2, 3, 4, 5, 6, 7],
        buffer_size=config.buffer_size
    )

    try:
        plan = compile(allreduce_op)
        print(f"Compiled plan with {len(plan.operations)} operations")

        cost_estimate = plan.metadata
        print(f"Plan metadata:")
        print(f"  Buffer size: {cost_estimate.get('buffer_size', 0) // (1024*1024)}MB")
        print(f"  Overlap enabled: {cost_estimate.get('enable_overlap', False)}")
        print(f"  Pipeline depth: {cost_estimate.get('pipeline_depth', 0)}")

    except Exception as e:
        print(f"Compilation failed: {e}")

def demonstrate_composite_operators():
    """Demonstrate composite operators"""
    print("\n=== Composite Operators Demo ===")

    allreduce_op = pccl.Allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=[0, 1, 2, 3]
    )

    broadcast_op = pccl.Broadcast(
        root_rank=0,
        participants=[0, 1, 2, 3]
    )

    composite = pccl.CompositeOperator(
        operators=[allreduce_op, broadcast_op],
        name="training_composite"
    )

    print(f"Created composite operator: {composite.name}")
    print(f"Contains {len(composite.operators)} operations")

    input_data = np.random.randn(256, 256).astype(np.float32)

    try:
        plan = composite.compile(participants=[0, 1, 2, 3])
        print(f"Compiled composite plan with {len(plan.operations)} operations")

        result = composite.execute(input_data, participants=[0, 1, 2, 3])
        print("Composite execution completed")

    except Exception as e:
        print(f"Composite execution failed: {e}")

def demonstrate_registry():
    """Demonstrate operator registry"""
    print("\n=== Operator Registry Demo ===")

    print(f"Available operators: {pccl.lang.registry.list_operators()}")

    pattern = pccl.lang.registry.get_pattern("ring_allreduce")
    if pattern:
        print(f"Ring AllReduce pattern found with {len(pattern.default_params)} default params")

    allreduce_op = pccl.lang.registry.create_operator(
        "allreduce",
        reduce_op="sum",
        algorithm="ring",
        participants=[0, 1, 2, 3]
    )

    if allreduce_op:
        print(f"Created AllReduce via registry: {allreduce_op.name}")

def demonstrate_async_execution():
    """Demonstrate asynchronous execution"""
    print("\n=== Async Execution Demo ===")

    allreduce_op = pccl.Allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=[0, 1, 2, 3]
    )

    input_data = np.random.randn(512, 512).astype(np.float32)

    try:
        handle = allreduce_op.async_execute(input_data, participants=[0, 1, 2, 3])
        print("Started async execution")

        while not handle.is_completed():
            print("  Waiting for completion...")
            import time
            time.sleep(0.1)

        result = handle.wait()
        print("Async execution completed")

    except Exception as e:
        print(f"Async execution failed: {e}")

def main():
    """Main demonstration function"""
    print("PCCL Python DSL Demonstration")
    print("=" * 50)

    try:
        demonstrate_basic_dsl()
        demonstrate_config_builder()
        demonstrate_direct_operators()
        demonstrate_topology_discovery()
        demonstrate_manual_topology()
        demonstrate_compilation_optimization()
        demonstrate_composite_operators()
        demonstrate_registry()
        demonstrate_async_execution()

        print("\n" + "=" * 50)
        print("All DSL demonstrations completed successfully!")

    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()