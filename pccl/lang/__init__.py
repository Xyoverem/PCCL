
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

# Import actual IR components
from ..ir.primitive_ir import *
from ..ir.json_serializer import *

__version__ = "0.1.0"
__all__ = [
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

    "ExecutionPlan",
    "DSLCompiler",

    "DeviceInfo",
    "LinkInfo",
    "TopologyMetrics",
    "TopologyBuilder",
    "TopologyDiscovery",
    "TopologyOptimizer",

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

    # Hardware execution functions
    "execute_on_hardware",
    "allreduce_hardware",
    "benchmark_hardware",
    "get_hardware_info",

    # IR components are imported with *
]
def allreduce(reduce_op="sum", algorithm="ring", participants=None,
              buffer_size=128*1024*1024, enable_overlap=False, input_data=None):
    """Execute AllReduce on real hardware"""
    reduce_op_map = {
        "sum": ReduceOp.SUM,
        "avg": ReduceOp.AVG,
        "max": ReduceOp.MAX,
        "min": ReduceOp.MIN
    }

    config = AllreduceConfig(
        reduce_op=reduce_op_map.get(reduce_op, ReduceOp.SUM),
        algorithm=algorithm,
        participants=participants,
        buffer_size=buffer_size,
        enable_overlap=enable_overlap
    )

    return execute_on_hardware(config, input_data, "cuda", 0)

def broadcast(root_rank=0, participants=None, buffer_size=128*1024*1024, input_data=None):
    """Execute Broadcast on real hardware"""
    config = BroadcastConfig(
        root_rank=root_rank,
        participants=participants,
        buffer_size=buffer_size
    )

    return execute_on_hardware(config, input_data, "cuda", 0)

def allgather(participants=None, input_size=0, buffer_size=128*1024*1024, input_data=None):
    """Execute AllGather on real hardware"""
    config = AllgatherConfig(
        participants=participants,
        input_size=input_size,
        buffer_size=buffer_size
    )

    return execute_on_hardware(config, input_data, "cuda", 0)

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

def discover_topology(num_devices=None):
    """Auto-discover system topology"""
    discovery = TopologyDiscovery()
    return discovery.discover_topology(num_devices)

def profile_execution(plan, input_data, num_iterations=10):
    """Profile execution of a plan"""
    profiler = get_profiler()
    return profiler.profile_execution(plan, input_data, num_iterations)

# Hardware execution functions
def execute_on_hardware(config, input_data, hardware_type="cuda", device_id=0):
    """Execute communication operation on real hardware"""
    from .execution_manager import ExecutionManager
    from .integrated_compiler import HardwareType

    hw_type = HardwareType.CUDA if hardware_type == "cuda" else HardwareType.CPU
    if hardware_type == "rdma" and HardwareType.RDMA:
        hw_type = HardwareType.RDMA

    manager = ExecutionManager(hw_type)
    result = manager.execute(config, input_data, hw_type, device_id)
    return result

def allreduce_hardware(reduce_op="sum", algorithm="ring", participants=None,
                      buffer_size=128*1024*1024, enable_overlap=False,
                      hardware_type="cuda", device_id=0, input_data=None):
    """Execute AllReduce on real hardware"""
    from .config import AllreduceConfig, ReduceOp

    reduce_op_map = {
        "sum": ReduceOp.SUM,
        "avg": ReduceOp.AVG,
        "max": ReduceOp.MAX,
        "min": ReduceOp.MIN
    }

    config = AllreduceConfig(
        reduce_op=reduce_op_map.get(reduce_op, ReduceOp.SUM),
        algorithm=algorithm,
        participants=participants,
        buffer_size=buffer_size,
        enable_overlap=enable_overlap
    )

    return execute_on_hardware(config, input_data, hardware_type, device_id)

def benchmark_hardware(config, input_data, num_iterations=10):
    """Benchmark hardware execution performance"""
    from .execution_manager import ExecutionManager

    manager = ExecutionManager()
    return manager.benchmark_config(config, input_data, num_iterations)

def get_hardware_info():
    """Get available hardware information"""
    from .execution_manager import ExecutionManager
    from .integrated_compiler import HardwareType

    manager = ExecutionManager()
    return manager.get_hardware_status()