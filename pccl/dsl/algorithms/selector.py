"""Topology-aware algorithm selector.

Picks the best allreduce algorithm for given (world_size, data_bytes, topology)
using cost-model estimates.
"""

from typing import Optional
from ..superopt.cost_model import (
    TopologyProfile, NVLINK_TOPOLOGY,
    GpuProfile, H100_PROFILE,
    critical_path_cost,
)
from .base import CollectiveAlgorithm
from .ring import RingAllreduce
from .recursive_hd import RecursiveHalvingDoubling
from .tree import TreeAllreduce


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def select_algorithm(
    world_size: int,
    data_bytes: int,
    topology: TopologyProfile = NVLINK_TOPOLOGY,
    device: GpuProfile = H100_PROFILE,
) -> CollectiveAlgorithm:
    """Pick the best allreduce algorithm for given parameters.

    Uses a heuristic based on message size and GPU count.
    Ring is bandwidth-optimal but O(N) steps.
    Recursive halving-doubling is O(log N) steps and bandwidth-optimal for
    power-of-2 world sizes.
    Tree is O(log N) steps but NOT bandwidth-optimal (root bottleneck).

    Heuristic:
    - Large messages (>= 1MB): Ring (bandwidth-optimal, step overhead amortized)
    - Medium messages (64KB - 1MB) with N >= 4 and power-of-2: RHD
    - Small messages (< 64KB) with N >= 4 and power-of-2: RHD (fewer steps)
    - Non-power-of-2 world sizes: Ring (always works)
    """
    po2 = _is_power_of_2(world_size)

    if not po2 or world_size <= 2:
        return RingAllreduce()

    if data_bytes >= 1024 * 1024:
        return RingAllreduce()

    if data_bytes < 1024 * 1024 and world_size >= 4:
        return RecursiveHalvingDoubling()

    return RingAllreduce()


def select_algorithm_cost_based(
    rank: int,
    world_size: int,
    tensor_size: int,
    data_bytes: int,
    topology: TopologyProfile = NVLINK_TOPOLOGY,
    device: GpuProfile = H100_PROFILE,
    dtype: str = "bfloat16",
    executor: str = "tma",
) -> CollectiveAlgorithm:
    """Pick algorithm by building each candidate and estimating cost.

    More expensive than the heuristic but considers actual graph structure.
    Falls back to heuristic for non-power-of-2 world sizes.
    """
    candidates = [RingAllreduce()]
    if _is_power_of_2(world_size) and world_size >= 2:
        candidates.append(RecursiveHalvingDoubling())
        candidates.append(TreeAllreduce())

    if len(candidates) == 1:
        return candidates[0]

    from ..superopt.rule import PatternNode, PatternEdge

    best_alg = candidates[0]
    best_cost = float("inf")

    for alg in candidates:
        try:
            graph = alg.build_allreduce(
                rank=rank, world_size=world_size,
                tensor_size=tensor_size, dtype=dtype,
                executor=executor,
            )
            nodes_list = []
            edges_list = []
            id_to_idx = {}
            for idx, (nid, node) in enumerate(graph.nodes.items()):
                from ..nodes import PrimitiveOpType
                op_type = node.op_type if node.op_type else PrimitiveOpType.NOOP
                nodes_list.append(PatternNode(
                    op_type=op_type, rank=0, channel=node.channel,
                ))
                id_to_idx[nid] = idx
            for nid, node in graph.nodes.items():
                src_idx = id_to_idx[nid]
                for dep_id in node.next_ops:
                    if dep_id in id_to_idx:
                        edges_list.append(PatternEdge(src_idx, id_to_idx[dep_id]))
            cost = critical_path_cost(nodes_list, edges_list, device, data_bytes)
            if cost < best_cost:
                best_cost = cost
                best_alg = alg
        except Exception:
            continue

    return best_alg
