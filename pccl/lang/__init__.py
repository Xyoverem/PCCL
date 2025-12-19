"""
PCCL Python DSL - High-level communication operators and topology management.

This module provides a declarative Python interface for defining and executing
communication operations in the PCCL library.
"""

from .config import (
    ReduceOp,
    DeviceType,
    InterconnectType,
    TopologyType,
    AlgorithmType,
    OperatorConfig,
    TopologyConfig,
    AllreduceConfig,
    BroadcastConfig,
    AllgatherConfig,
    ReduceScatterConfig,
    ConfigBuilder
)

from .operator import (
    CommunicationOperator,
    Allreduce,
    Broadcast,
    Allgather,
    ReduceScatter,
    Send,
    Recv,
    PipelineAllreduce,
    CompositeOperator,
    CommunicationPattern,
    communication,
    registry,
    compile,
    execute
)

from .compiler import (
    ExecutionPlan,
    DSLCompiler
)

from .topology import (
    DeviceInfo,
    LinkInfo,
    TopologyMetrics,
    TopologyBuilder,
    TopologyDiscovery,
    TopologyOptimizer
)

from .executor import (
    ExecutionHandle,
    ExecutionEngine,
    execute_plan,
    execute_plan_async,
    wait_for_completion,
    get_execution_engine,
    PerformanceProfiler,
    get_profiler,
    profile,
    shutdown_execution_engine
)

from .chunk import (
    PrimitiveOpType,
    Chunk,
    Device,
    Link,
    PrimitiveOp,
    CollectiveIR
)

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "ReduceOp",
    "DeviceType",
    "InterconnectType",
    "TopologyType",
    "AlgorithmType",
    "OperatorConfig",
    "TopologyConfig",
    "AllreduceConfig",
    "BroadcastConfig",
    "AllgatherConfig",
    "ReduceScatterConfig",
    "ConfigBuilder",

    # Operators
    "CommunicationOperator",
    "Allreduce",
    "Broadcast",
    "Allgather",
    "ReduceScatter",
    "Send",
    "Recv",
    "PipelineAllreduce",
    "CompositeOperator",
    "CommunicationPattern",
    "communication",
    "registry",
    "compile",
    "execute",

    # Compiler
    "ExecutionPlan",
    "DSLCompiler",

    # Topology
    "DeviceInfo",
    "LinkInfo",
    "TopologyMetrics",
    "TopologyBuilder",
    "TopologyDiscovery",
    "TopologyOptimizer",

    # Execution
    "ExecutionHandle",
    "ExecutionEngine",
    "execute_plan",
    "execute_plan_async",
    "wait_for_completion",
    "get_execution_engine",
    "PerformanceProfiler",
    "get_profiler",
    "profile",
    "shutdown_execution_engine",

    # IR
    "PrimitiveOpType",
    "Chunk",
    "Device",
    "Link",
    "PrimitiveOp",
    "CollectiveIR",
]

# Convenience functions for common operations
def allreduce(reduce_op="sum", algorithm="ring", participants=None,
              buffer_size=128*1024*1024, enable_overlap=False):
    """Create an AllReduce operator"""
    return Allreduce(reduce_op=reduce_op, algorithm=algorithm,
                    participants=participants, buffer_size=buffer_size,
                    enable_overlap=enable_overlap)

def broadcast(root_rank=0, participants=None, buffer_size=128*1024*1024):
    """Create a Broadcast operator"""
    return Broadcast(root_rank=root_rank, participants=participants,
                    buffer_size=buffer_size)

def allgather(participants=None, input_size=0, buffer_size=128*1024*1024):
    """Create an AllGather operator"""
    return Allgather(participants=participants, input_size=input_size,
                    buffer_size=buffer_size)

def ring_topology(devices, bandwidth=10.0, latency=1.0):
    """Create a ring topology"""
    return TopologyBuilder.build_ring_topology(devices, bandwidth=bandwidth, latency=latency)

def tree_topology(devices, branching_factor=2, bandwidth=10.0, latency=1.0):
    """Create a tree topology"""
    return TopologyBuilder.build_tree_topology(devices, branching_factor=branching_factor,
                                               bandwidth=bandwidth, latency=latency)

def hierarchical_topology(node_groups, intra_bandwidth=50.0, inter_bandwidth=10.0):
    """Create a hierarchical topology"""
    return TopologyBuilder.build_hierarchical_topology(node_groups,
                                                      intra_bandwidth=intra_bandwidth,
                                                      inter_bandwidth=inter_bandwidth)

# Auto-discovery functionality
def discover_topology(num_devices=None):
    """Auto-discover system topology"""
    discovery = TopologyDiscovery()
    return discovery.discover_topology(num_devices)

def profile_execution(plan, input_data, num_iterations=10):
    """Profile execution of a plan"""
    profiler = get_profiler()
    return profiler.profile_execution(plan, input_data, num_iterations)