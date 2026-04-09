"""Configurable GPU cost model for DAG pattern evaluation.

Computes critical-path latency through a 2-rank DAG, factoring in:
  - Per-op kernel launch overhead (default 1us)
  - Data transfer time (size / executor bandwidth)
  - Reduce compute time (count / executor throughput)
  - Multi-channel parallelism: ops on different channels run on
    independent HW paths with dedicated per-channel bandwidth.
  - Topology-aware link bandwidth (NVLink vs RDMA vs PCIe)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque

from ..nodes import PrimitiveOpType, ExecutorType, infer_executor
from .rule import PatternNode, PatternEdge


@dataclass
class TopologyProfile:
    link_type: str = "nvlink"
    bandwidth_gb_s: float = 450.0
    latency_us: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_type": self.link_type,
            "bandwidth_gb_s": self.bandwidth_gb_s,
            "latency_us": self.latency_us,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TopologyProfile":
        return cls(
            link_type=d.get("link_type", "nvlink"),
            bandwidth_gb_s=d.get("bandwidth_gb_s", 450.0),
            latency_us=d.get("latency_us", 1.0),
        )


NVLINK_TOPOLOGY = TopologyProfile(link_type="nvlink", bandwidth_gb_s=450.0, latency_us=1.0)
RDMA_TOPOLOGY = TopologyProfile(link_type="rdma", bandwidth_gb_s=25.0, latency_us=5.0)
PCIE_TOPOLOGY = TopologyProfile(link_type="pcie", bandwidth_gb_s=32.0, latency_us=2.0)


@dataclass
class GpuProfile:
    name: str = "h100"
    launch_overhead_us: float = 1.0
    bandwidth_gb_s: Dict[str, float] = field(default_factory=lambda: {
        "sm": 50.0,
        "tma": 60.0,
        "ce": 40.0,
        "rdma": 25.0,
        "multimem": 180.0,
        "host": 10.0,
    })
    reduce_throughput_gops: Dict[str, float] = field(default_factory=lambda: {
        "sm": 30.0,
        "tma": 35.0,
        "multimem": 100.0,
    })
    num_channels: int = 4
    per_channel_bw_gb_s: Dict[str, float] = field(default_factory=lambda: {
        "sm": 50.0,
        "tma": 60.0,
        "ce": 40.0,
        "rdma": 25.0,
        "multimem": 180.0,
        "host": 10.0,
    })
    channel_launch_overhead_us: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "launch_overhead_us": self.launch_overhead_us,
            "bandwidth_gb_s": self.bandwidth_gb_s,
            "reduce_throughput_gops": self.reduce_throughput_gops,
            "num_channels": self.num_channels,
            "per_channel_bw_gb_s": self.per_channel_bw_gb_s,
            "channel_launch_overhead_us": self.channel_launch_overhead_us,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GpuProfile":
        return cls(
            name=d.get("name", "custom"),
            launch_overhead_us=d.get("launch_overhead_us", 1.0),
            bandwidth_gb_s=d.get("bandwidth_gb_s", {}),
            reduce_throughput_gops=d.get("reduce_throughput_gops", {}),
            num_channels=d.get("num_channels", 4),
            per_channel_bw_gb_s=d.get("per_channel_bw_gb_s", {}),
            channel_launch_overhead_us=d.get("channel_launch_overhead_us", 0.5),
        )


H100_PROFILE = GpuProfile(name="h100")

_COPY_OPS = {
    PrimitiveOpType.SM_COPY, PrimitiveOpType.TMA_COPY,
    PrimitiveOpType.CE_COPY, PrimitiveOpType.MULTIMEM_STORE,
    PrimitiveOpType.RDMA_WRITE, PrimitiveOpType.RDMA_READ,
}
_REDUCE_OPS = {
    PrimitiveOpType.SM_REDUCE, PrimitiveOpType.TMA_REDUCE,
    PrimitiveOpType.MULTIMEM_REDUCE,
}
_SYNC_OPS = {PrimitiveOpType.NOTIFY, PrimitiveOpType.WAIT_NOTIFY, PrimitiveOpType.NOOP}


def op_latency(
    op_type: PrimitiveOpType,
    profile: GpuProfile,
    data_size_bytes: int = 0,
) -> float:
    executor = infer_executor(op_type)
    executor_key = executor.value if executor else "host"

    if op_type in _SYNC_OPS:
        return profile.launch_overhead_us

    launch = profile.launch_overhead_us

    if op_type in _COPY_OPS:
        bw = profile.bandwidth_gb_s.get(executor_key, 10.0)
        transfer_us = (data_size_bytes / (bw * 1e3)) if bw > 0 else 0.0
        return launch + transfer_us

    if op_type in _REDUCE_OPS:
        bw = profile.bandwidth_gb_s.get(executor_key, 10.0)
        transfer_us = (data_size_bytes / (bw * 1e3)) if bw > 0 else 0.0
        tp = profile.reduce_throughput_gops.get(executor_key, 30.0)
        elem_count = data_size_bytes // 4
        compute_us = (elem_count / (tp * 1e3)) if tp > 0 else 0.0
        return launch + max(transfer_us, compute_us)

    return launch


def critical_path_cost(
    nodes: List[PatternNode],
    edges: List[PatternEdge],
    profile: GpuProfile,
    data_size_bytes: int = 0,
) -> float:
    n = len(nodes)
    if n == 0:
        return 0.0

    channels_used = set(nd.channel for nd in nodes)
    is_multi_channel = len(channels_used) > 1

    latencies = [op_latency(node.op_type, profile, data_size_bytes) for node in nodes]

    adj = defaultdict(list)
    for e in edges:
        adj[e.src_idx].append(e.dst_idx)

    if is_multi_channel:
        for ch in channels_used:
            ch_indices = [i for i in range(n) if nodes[i].channel == ch]
            for a, b in zip(ch_indices, ch_indices[1:]):
                if b not in adj[a]:
                    adj[a].append(b)

    in_degree = [0] * n
    for u in range(n):
        for v in adj[u]:
            in_degree[v] += 1

    earliest_finish = [0.0] * n
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            earliest_finish[i] = latencies[i]
            queue.append(i)

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            start_time = earliest_finish[u]
            finish = start_time + latencies[v]
            if finish > earliest_finish[v]:
                earliest_finish[v] = finish
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    base_cost = max(earliest_finish) if earliest_finish else 0.0

    if is_multi_channel:
        base_cost += profile.channel_launch_overhead_us * (len(channels_used) - 1)

    return base_cost


def _executor_key_from_op_type_str(op_type_str: str) -> str:
    parts = op_type_str.split(".", 1)
    return parts[0] if parts else "host"


def egraph_node_cost(
    enode: "ENode",
    child_costs: Dict[int, float],
    device: GpuProfile,
    topology: TopologyProfile,
) -> float:
    executor_key = _executor_key_from_op_type_str(enode.op_type)

    if enode.op_type in ("notify", "wait_notify", "noop"):
        return device.launch_overhead_us + max(child_costs.values(), default=0.0)

    params_dict = dict(enode.params)
    data_size = params_dict.get("size", params_dict.get("count", 0))
    if not isinstance(data_size, (int, float)):
        data_size = 0

    launch = device.launch_overhead_us
    bw = device.bandwidth_gb_s.get(executor_key, 10.0)
    transfer_us = (data_size / (bw * 1e3)) if bw > 0 and data_size > 0 else 0.0

    op_cost = launch + transfer_us
    dep_cost = max(child_costs.values(), default=0.0)
    return dep_cost + op_cost


def cost_delta(
    source_nodes: List[PatternNode],
    source_edges: List[PatternEdge],
    replacement_nodes: List[PatternNode],
    replacement_edges: List[PatternEdge],
    profile: GpuProfile,
    data_size_bytes: int = 0,
) -> float:
    src_cost = critical_path_cost(source_nodes, source_edges, profile, data_size_bytes)
    repl_cost = critical_path_cost(replacement_nodes, replacement_edges, profile, data_size_bytes)
    return src_cost - repl_cost
