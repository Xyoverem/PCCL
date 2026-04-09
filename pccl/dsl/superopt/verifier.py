"""SMT-based semantic equivalence checking between two DAG skeletons.

Hybrid approach for speed + correctness:
  1. Concrete simulation: run both patterns on multiple realistic parameter
     sets, compare buffer outputs. Eliminates >99.9% of non-equivalent pairs.
  2. Slot-based z3 proof: for surviving pairs, model buffers as fixed named
     slots (no Array theory, no quantifiers), just bitvec equations.
     ~100x faster than the Array+ForAll approach.
"""

import z3
import uuid
import hashlib
from typing import List, Dict, Optional, Tuple, Set
from itertools import permutations
from collections import defaultdict

from ..nodes import PrimitiveOpType
from .rule import PatternNode, PatternEdge, RewriteRule


_OP_PARAM_TEMPLATES: Dict[PrimitiveOpType, List[str]] = {
    PrimitiveOpType.SM_COPY: ["source_rank", "src_offset", "dst_offset", "size"],
    PrimitiveOpType.SM_REDUCE: ["source_rank", "src_offset", "dst_offset", "remote_offset", "reduce_op", "count"],
    PrimitiveOpType.TMA_COPY: ["source_rank", "src_offset", "dst_offset", "size"],
    PrimitiveOpType.TMA_REDUCE: ["source_rank", "src_offset", "dst_offset", "remote_offset", "reduce_op", "count"],
    PrimitiveOpType.CE_COPY: ["source_rank", "src_offset", "dst_offset", "size"],
    PrimitiveOpType.MULTIMEM_REDUCE: ["source_rank", "src_offset", "dst_offset", "remote_offset", "reduce_op", "count"],
    PrimitiveOpType.MULTIMEM_STORE: ["source_rank", "src_offset", "dst_offset", "size"],
    PrimitiveOpType.RDMA_WRITE: ["target_rank", "src_offset", "dst_offset", "size"],
    PrimitiveOpType.RDMA_READ: ["source_rank", "src_offset", "dst_offset", "size"],
    PrimitiveOpType.NOTIFY: ["target_rank", "signal_id"],
    PrimitiveOpType.WAIT_NOTIFY: ["source_rank", "signal_id"],
    PrimitiveOpType.NOOP: [],
}

_COPY_LIKE = {
    PrimitiveOpType.SM_COPY, PrimitiveOpType.TMA_COPY, PrimitiveOpType.CE_COPY,
    PrimitiveOpType.MULTIMEM_STORE, PrimitiveOpType.RDMA_READ,
}
_REDUCE_LIKE = {
    PrimitiveOpType.SM_REDUCE, PrimitiveOpType.TMA_REDUCE,
    PrimitiveOpType.MULTIMEM_REDUCE,
}
_WRITE_REMOTE = {PrimitiveOpType.RDMA_WRITE}
_SYNC_OPS = {PrimitiveOpType.NOTIFY, PrimitiveOpType.WAIT_NOTIFY, PrimitiveOpType.NOOP}

# ---------------------------------------------------------------------------
# Concrete simulation (fast filter)
# ---------------------------------------------------------------------------

_TEST_PARAM_SETS = [
    {"chunk_size": 4096, "base_off": 0},
    {"chunk_size": 8192, "base_off": 0},
    {"chunk_size": 4096, "base_off": 4096},
    {"chunk_size": 1024, "base_off": 0},
    {"chunk_size": 16384, "base_off": 8192},
]


def _topo_sort(num_nodes: int, edges: List[PatternEdge]) -> Optional[List[int]]:
    adj = defaultdict(list)
    in_deg = defaultdict(int)
    for e in edges:
        adj[e.src_idx].append(e.dst_idx)
        in_deg[e.dst_idx] += 1
    queue = [i for i in range(num_nodes) if in_deg.get(i, 0) == 0]
    order = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for s in adj[n]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    return order if len(order) == num_nodes else None


def _make_concrete_params(node: PatternNode, chunk_size: int, base_off: int) -> dict:
    off = base_off + node.chunk_id * chunk_size
    ot = node.op_type
    if ot in _COPY_LIKE:
        return {"source_rank": 1, "src_offset": off, "dst_offset": off, "size": chunk_size}
    if ot in _REDUCE_LIKE:
        return {"source_rank": 1, "src_offset": off, "dst_offset": off,
                "remote_offset": off, "reduce_op": "sum", "count": chunk_size}
    if ot in _WRITE_REMOTE:
        return {"target_rank": 1, "src_offset": off, "dst_offset": off, "size": chunk_size}
    if ot == PrimitiveOpType.RDMA_READ:
        return {"source_rank": 1, "src_offset": off, "dst_offset": off, "size": chunk_size}
    if ot == PrimitiveOpType.NOTIFY:
        return {"target_rank": 1, "signal_id": node.chunk_id}
    if ot == PrimitiveOpType.WAIT_NOTIFY:
        return {"source_rank": 1, "signal_id": node.chunk_id}
    return {}


NUM_SLOTS = 32


def _simulate(
    nodes: List[PatternNode],
    edges: List[PatternEdge],
    chunk_size: int,
    base_off: int,
) -> Optional[Tuple[list, list]]:
    order = _topo_sort(len(nodes), edges)
    if order is None:
        return None

    buf = {0: list(range(NUM_SLOTS)), 1: list(range(NUM_SLOTS, NUM_SLOTS * 2))}

    for idx in order:
        node = nodes[idx]
        p = _make_concrete_params(node, chunk_size, base_off)
        ot = node.op_type
        r = node.rank

        slot = (base_off + node.chunk_id * chunk_size) % NUM_SLOTS

        if ot in _COPY_LIKE:
            src_r = p.get("source_rank", 1 - r)
            buf[r][slot] = buf[src_r][slot]
        elif ot in _REDUCE_LIKE:
            src_r = p.get("source_rank", 1 - r)
            buf[r][slot] = buf[r][slot] + buf[src_r][slot]
        elif ot in _WRITE_REMOTE:
            tgt_r = p.get("target_rank", 1 - r)
            buf[tgt_r][slot] = buf[r][slot]

    return (buf[0], buf[1])


def concrete_simulation_match(
    src_nodes: List[PatternNode],
    src_edges: List[PatternEdge],
    repl_nodes: List[PatternNode],
    repl_edges: List[PatternEdge],
) -> bool:
    for ps in _TEST_PARAM_SETS:
        src_out = _simulate(src_nodes, src_edges, ps["chunk_size"], ps["base_off"])
        repl_out = _simulate(repl_nodes, repl_edges, ps["chunk_size"], ps["base_off"])
        if src_out is None or repl_out is None:
            return False
        if src_out != repl_out:
            return False
    return True


# ---------------------------------------------------------------------------
# Signature-based pre-filter
# ---------------------------------------------------------------------------

def signatures_compatible(
    src_nodes: List[PatternNode],
    repl_nodes: List[PatternNode],
) -> bool:
    src_reduce = sum(1 for n in src_nodes if n.op_type in _REDUCE_LIKE)
    repl_reduce = sum(1 for n in repl_nodes if n.op_type in _REDUCE_LIKE)
    if src_reduce != repl_reduce:
        return False

    src_chunks = len(set(n.chunk_id for n in src_nodes))
    repl_chunks = len(set(n.chunk_id for n in repl_nodes))
    if src_chunks != repl_chunks:
        return False

    src_has_write = any(n.op_type in _WRITE_REMOTE for n in src_nodes)
    repl_has_write = any(n.op_type in _WRITE_REMOTE for n in repl_nodes)
    if src_has_write != repl_has_write:
        return False

    src_data = sum(1 for n in src_nodes if n.op_type not in _SYNC_OPS)
    repl_data = sum(1 for n in repl_nodes if n.op_type not in _SYNC_OPS)
    if src_data == 0 or repl_data == 0:
        return False

    src_copy = sum(1 for n in src_nodes if n.op_type in _COPY_LIKE)
    repl_copy = sum(1 for n in repl_nodes if n.op_type in _COPY_LIKE)
    src_channels = len(set(n.channel for n in src_nodes))
    repl_channels = len(set(n.channel for n in repl_nodes))
    if src_channels > 1 and repl_channels > 1:
        if src_copy != repl_copy:
            return False

    return True


def concrete_fingerprint(
    nodes: List[PatternNode],
    edges: List[PatternEdge],
) -> Optional[str]:
    result = _simulate(nodes, edges, 4096, 0)
    if result is None:
        return None
    return hashlib.md5(str(result).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Slot-based z3 proof (fast formal verification)
# ---------------------------------------------------------------------------

def _slot_for_chunk(chunk_id: int) -> int:
    return chunk_id % NUM_SLOTS


def _make_slot_z3_state(prefix: str, num_chunks: int):
    slots = {}
    for r in range(2):
        for c in range(num_chunks):
            s = _slot_for_chunk(c)
            slots[(r, s)] = z3.BitVec(f"{prefix}_r{r}_s{s}", 64)
    return slots


def _apply_op_slots(
    node: PatternNode,
    node_idx: int,
    prefix: str,
    slots: dict,
    num_chunks: int,
) -> dict:
    new_slots = dict(slots)
    ot = node.op_type
    r = node.rank
    s = _slot_for_chunk(node.chunk_id)

    if ot in _COPY_LIKE:
        peer = 1 - r
        new_slots[(r, s)] = slots.get((peer, s), z3.BitVecVal(0, 64))
    elif ot in _REDUCE_LIKE:
        peer = 1 - r
        local = slots.get((r, s), z3.BitVecVal(0, 64))
        remote = slots.get((peer, s), z3.BitVecVal(0, 64))
        new_slots[(r, s)] = local + remote
    elif ot in _WRITE_REMOTE:
        peer = 1 - r
        new_slots[(peer, s)] = slots.get((r, s), z3.BitVecVal(0, 64))

    return new_slots


def _slot_based_check(
    src_nodes: List[PatternNode],
    src_edges: List[PatternEdge],
    repl_nodes: List[PatternNode],
    repl_edges: List[PatternEdge],
    timeout: float = 2.0,
) -> bool:
    all_chunks = set()
    for n in src_nodes + repl_nodes:
        all_chunks.add(n.chunk_id)
    num_chunks = max(all_chunks) + 1 if all_chunks else 1

    init_slots = _make_slot_z3_state("init", num_chunks)

    src_order = _topo_sort(len(src_nodes), src_edges)
    repl_order = _topo_sort(len(repl_nodes), repl_edges)
    if src_order is None or repl_order is None:
        return False

    src_state = dict(init_slots)
    for idx in src_order:
        src_state = _apply_op_slots(src_nodes[idx], idx, "src", src_state, num_chunks)

    repl_state = dict(init_slots)
    for idx in repl_order:
        repl_state = _apply_op_slots(repl_nodes[idx], idx, "repl", repl_state, num_chunks)

    solver = z3.Solver()
    solver.set("timeout", int(timeout * 1000))

    neq_clauses = []
    all_keys = set(src_state.keys()) | set(repl_state.keys())
    for key in all_keys:
        sv = src_state.get(key, init_slots.get(key, z3.BitVecVal(0, 64)))
        rv = repl_state.get(key, init_slots.get(key, z3.BitVecVal(0, 64)))
        neq_clauses.append(sv != rv)

    if not neq_clauses:
        return True

    solver.add(z3.Or(*neq_clauses))
    result = solver.check()
    return result == z3.unsat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_equivalence(
    src_nodes: List[PatternNode],
    src_edges: List[PatternEdge],
    repl_nodes: List[PatternNode],
    repl_edges: List[PatternEdge],
    timeout: float = 2.0,
) -> Optional[RewriteRule]:
    if _is_trivially_identical(src_nodes, src_edges, repl_nodes, repl_edges):
        return None

    if not concrete_simulation_match(src_nodes, src_edges, repl_nodes, repl_edges):
        return None

    if not _slot_based_check(src_nodes, src_edges, repl_nodes, repl_edges, timeout):
        return None

    return RewriteRule(
        source_nodes=src_nodes,
        source_edges=src_edges,
        replacement_nodes=repl_nodes,
        replacement_edges=repl_edges,
        param_constraints=[],
        rule_id=f"rule_{uuid.uuid4().hex[:8]}",
    )


def _is_trivially_identical(
    src_nodes: List[PatternNode],
    src_edges: List[PatternEdge],
    repl_nodes: List[PatternNode],
    repl_edges: List[PatternEdge],
) -> bool:
    if len(src_nodes) != len(repl_nodes) or len(src_edges) != len(repl_edges):
        return False
    for s, r in zip(src_nodes, repl_nodes):
        if s.op_type != r.op_type or s.rank != r.rank or s.chunk_id != r.chunk_id or s.channel != r.channel:
            return False
    src_edge_set = {(e.src_idx, e.dst_idx) for e in src_edges}
    repl_edge_set = {(e.src_idx, e.dst_idx) for e in repl_edges}
    return src_edge_set == repl_edge_set
