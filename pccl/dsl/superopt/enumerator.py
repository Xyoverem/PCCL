"""Enumerate communication DAG skeletons for superoptimization.

Design: enumerate only *data operations* (copy/reduce), with topology
derived from chunk assignments. Two independent op sets for link types:
  - intra-node (NVLink): sm_copy, ce_copy, sm_reduce
  - inter-node (RDMA):   rdma_write, rdma_read

Multi-channel support: each op can be assigned to a hardware channel.
Ops on different channels execute on independent HW paths.

Executor upgrades (no z3 needed):
  - sm_copy  -> tma_copy   (intra only, same resource, always faster)
  - sm_reduce -> tma_reduce (intra only, same resource, always faster)
"""

from typing import List, Tuple, Generator, Set, Dict
from itertools import product, combinations
from collections import defaultdict
from enum import Enum

from ..nodes import PrimitiveOpType
from .rule import PatternNode, PatternEdge


class LinkType(Enum):
    INTRA = "intra"
    INTER = "inter"


CANONICAL_OPS = {
    LinkType.INTRA: [
        PrimitiveOpType.SM_COPY,
        PrimitiveOpType.CE_COPY,
        PrimitiveOpType.SM_REDUCE,
        PrimitiveOpType.MULTIMEM_REDUCE,
        PrimitiveOpType.MULTIMEM_STORE,
    ],
    LinkType.INTER: [
        PrimitiveOpType.RDMA_WRITE,
        PrimitiveOpType.RDMA_READ,
    ],
}

EXECUTOR_UPGRADES = {
    PrimitiveOpType.SM_COPY: PrimitiveOpType.TMA_COPY,
    PrimitiveOpType.SM_REDUCE: PrimitiveOpType.TMA_REDUCE,
}

Skeleton = Tuple[List[PatternNode], List[PatternEdge]]


def _bell_partitions(n: int) -> List[List[int]]:
    if n == 0:
        return [[]]
    results = []

    def _gen(pos: int, assign: list, next_label: int):
        if pos == n:
            results.append(list(assign))
            return
        for label in range(next_label):
            assign.append(label)
            _gen(pos + 1, assign, next_label)
            assign.pop()
        assign.append(next_label)
        _gen(pos + 1, assign, next_label + 1)
        assign.pop()

    _gen(0, [], 0)
    return results


def _channel_assignments(k: int, max_channels: int) -> List[List[int]]:
    if max_channels <= 1:
        return [[0] * k]
    partitions = _bell_partitions(k)
    return [p for p in partitions if max(p) < max_channels]


def _canonical_hash(nodes: List[PatternNode], edges: List[PatternEdge]) -> int:
    node_keys = tuple((n.op_type.value, n.chunk_id, n.channel) for n in nodes)
    edge_keys = tuple(sorted((e.src_idx, e.dst_idx) for e in edges))
    return hash((node_keys, edge_keys))


def _derive_topology(chunk_ids: List[int]) -> List[Tuple[int, int]]:
    chunk_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, cid in enumerate(chunk_ids):
        chunk_to_indices[cid].append(i)
    edges = []
    for indices in chunk_to_indices.values():
        for a, b in zip(indices, indices[1:]):
            edges.append((a, b))
    return edges


def _channel_topology(
    channel_ids: List[int],
    base_edges: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    existing = set()
    for s, d in base_edges:
        existing.add((s, d))
        existing.add((d, s))

    ch_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, ch in enumerate(channel_ids):
        ch_to_indices[ch].append(i)

    extra = []
    for indices in ch_to_indices.values():
        for a, b in zip(indices, indices[1:]):
            if (a, b) not in existing and (b, a) not in existing:
                extra.append((a, b))
                existing.add((a, b))
                existing.add((b, a))
    return extra


def _extra_orderings(
    k: int,
    base_edges: List[Tuple[int, int]],
) -> Generator[List[Tuple[int, int]], None, None]:
    existing_pairs = set()
    for s, d in base_edges:
        existing_pairs.add((s, d))
        existing_pairs.add((d, s))

    candidates = [
        (i, j)
        for i in range(k) for j in range(i + 1, k)
        if (i, j) not in existing_pairs and (j, i) not in existing_pairs
    ]

    yield []
    max_extra = min(1 if k <= 3 else 2, len(candidates))
    for r in range(1, max_extra + 1):
        for combo in combinations(candidates, r):
            yield list(combo)


def enumerate_skeletons(
    k: int,
    link_type: LinkType = LinkType.INTRA,
    max_channels: int = 1,
) -> Generator[Skeleton, None, None]:
    if k <= 0:
        return

    ops = CANONICAL_OPS[link_type]
    seen: Set[int] = set()
    chunk_partitions = _bell_partitions(k)
    ch_assigns = _channel_assignments(k, max_channels)

    for op_combo in product(ops, repeat=k):
        for chunk_assign in chunk_partitions:
            for ch_assign in ch_assigns:
                base_edges = _derive_topology(chunk_assign)
                ch_edges = _channel_topology(ch_assign, base_edges)
                combined_base = base_edges + ch_edges

                for extra in _extra_orderings(k, combined_base):
                    all_edges = combined_base + extra
                    nodes = [
                        PatternNode(
                            op_type=op_combo[i], rank=0,
                            chunk_id=chunk_assign[i],
                            link_type=link_type.value,
                            channel=ch_assign[i],
                        )
                        for i in range(k)
                    ]
                    edges = [PatternEdge(s, d) for s, d in all_edges]
                    h = _canonical_hash(nodes, edges)
                    if h in seen:
                        continue
                    seen.add(h)
                    yield nodes, edges


def trivial_executor_rules():
    """Generate executor-upgrade rules without z3.

    TMA always dominates SM on the same resource, so only upgrade direction.
    Only applies to intra-node link type.
    """
    from .rule import RewriteRule

    rules = []
    for src_op, repl_op in EXECUTOR_UPGRADES.items():
        rules.append(RewriteRule(
            source_nodes=[PatternNode(
                op_type=src_op, rank=0, chunk_id=0,
                link_type=LinkType.INTRA.value,
            )],
            source_edges=[],
            replacement_nodes=[PatternNode(
                op_type=repl_op, rank=0, chunk_id=0,
                link_type=LinkType.INTRA.value,
            )],
            replacement_edges=[],
            param_constraints=[
                "repl_n0_src_offset == src_n0_src_offset",
                "repl_n0_dst_offset == src_n0_dst_offset",
            ],
            link_type=LinkType.INTRA.value,
            rule_id=f"executor_upgrade_{src_op.value}_to_{repl_op.value}",
        ))
    return rules
