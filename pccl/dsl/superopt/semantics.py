"""Formal op semantics encoded as z3 state transformers.

State model (region-level abstraction):
  - buf[rank] : z3 Array(BitVec(32) -> BitVec(64))  per-rank buffer
  - signals   : Dict[(src, dst, sig_id)] -> z3 Bool
Each op reads/writes symbolic regions identified by (offset, size) pairs.
Reduce ops use uninterpreted functions with algebraic axioms.
"""

import z3
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from ..nodes import PrimitiveOpType


@dataclass
class SymbolicState:
    bufs: Dict[int, z3.ArrayRef]
    signals: Dict[Tuple[int, int, int], z3.BoolRef] = field(default_factory=dict)
    constraints: List[z3.BoolRef] = field(default_factory=list)

    def copy(self) -> "SymbolicState":
        return SymbolicState(
            bufs=dict(self.bufs),
            signals=dict(self.signals),
            constraints=list(self.constraints),
        )


_reduce_fn_cache: Dict[str, z3.FuncDeclRef] = {}


def _get_reduce_fn(reduce_op: str) -> z3.FuncDeclRef:
    if reduce_op not in _reduce_fn_cache:
        fn = z3.Function(
            f"reduce_{reduce_op}",
            z3.BitVecSort(64), z3.BitVecSort(64), z3.BitVecSort(64),
        )
        _reduce_fn_cache[reduce_op] = fn
    return _reduce_fn_cache[reduce_op]


def get_reduce_axioms(reduce_op: str) -> List[z3.BoolRef]:
    fn = _get_reduce_fn(reduce_op)
    a = z3.BitVec("_ax_a", 64)
    b = z3.BitVec("_ax_b", 64)
    c = z3.BitVec("_ax_c", 64)
    axioms = []
    if reduce_op in ("sum", "max", "min"):
        axioms.append(z3.ForAll([a, b], fn(a, b) == fn(b, a)))
        axioms.append(z3.ForAll([a, b, c], fn(fn(a, b), c) == fn(a, fn(b, c))))
    return axioms


def make_initial_state(num_ranks: int = 2, prefix: str = "init") -> SymbolicState:
    bufs = {}
    for r in range(num_ranks):
        bufs[r] = z3.Array(f"{prefix}_buf_r{r}", z3.BitVecSort(32), z3.BitVecSort(64))
    return SymbolicState(bufs=bufs)


def encode_op(
    op_type: PrimitiveOpType,
    self_rank: int,
    params: Dict[str, z3.ExprRef],
    state: SymbolicState,
) -> SymbolicState:
    new_state = state.copy()

    if op_type in (PrimitiveOpType.SM_COPY, PrimitiveOpType.TMA_COPY, PrimitiveOpType.CE_COPY,
                    PrimitiveOpType.MULTIMEM_STORE):
        _encode_copy(new_state, self_rank, params)
    elif op_type in (PrimitiveOpType.SM_REDUCE, PrimitiveOpType.TMA_REDUCE,
                     PrimitiveOpType.MULTIMEM_REDUCE):
        _encode_reduce(new_state, self_rank, params)
    elif op_type == PrimitiveOpType.RDMA_WRITE:
        _encode_rdma_write(new_state, self_rank, params)
    elif op_type == PrimitiveOpType.RDMA_READ:
        _encode_rdma_read(new_state, self_rank, params)
    elif op_type == PrimitiveOpType.NOTIFY:
        _encode_notify(new_state, self_rank, params)
    elif op_type == PrimitiveOpType.WAIT_NOTIFY:
        _encode_wait_notify(new_state, self_rank, params)
    elif op_type == PrimitiveOpType.NOOP:
        pass
    else:
        raise ValueError(f"Unknown op type: {op_type}")

    return new_state


def _encode_copy(state: SymbolicState, self_rank: int, params: Dict[str, z3.ExprRef]):
    src_rank_val = params["source_rank"]
    src_off = params["src_offset"]
    dst_off = params["dst_offset"]
    size = params["size"]

    src_buf = state.bufs[self_rank]
    for r, buf in state.bufs.items():
        if isinstance(src_rank_val, int):
            if r == src_rank_val:
                src_buf = buf
                break
        else:
            break

    if not isinstance(src_rank_val, int):
        src_buf = z3.If(
            src_rank_val == z3.BitVecVal(0, 32),
            state.bufs[0],
            state.bufs[1],
        )

    value = z3.Select(src_buf, src_off)
    state.bufs[self_rank] = z3.Store(state.bufs[self_rank], dst_off, value)


def _encode_reduce(state: SymbolicState, self_rank: int, params: Dict[str, z3.ExprRef]):
    src_rank_val = params["source_rank"]
    dst_off = params["dst_offset"]
    remote_off = params["remote_offset"]
    reduce_op = params.get("reduce_op", "sum")

    if isinstance(reduce_op, str):
        reduce_op_str = reduce_op
    else:
        reduce_op_str = "sum"

    fn = _get_reduce_fn(reduce_op_str)

    local_val = z3.Select(state.bufs[self_rank], dst_off)

    if isinstance(src_rank_val, int):
        remote_val = z3.Select(state.bufs[src_rank_val], remote_off)
    else:
        remote_val = z3.If(
            src_rank_val == z3.BitVecVal(0, 32),
            z3.Select(state.bufs[0], remote_off),
            z3.Select(state.bufs[1], remote_off),
        )

    result = fn(local_val, remote_val)
    state.bufs[self_rank] = z3.Store(state.bufs[self_rank], dst_off, result)


def _encode_rdma_write(state: SymbolicState, self_rank: int, params: Dict[str, z3.ExprRef]):
    target_rank_val = params["target_rank"]
    src_off = params["src_offset"]
    dst_off = params["dst_offset"]

    value = z3.Select(state.bufs[self_rank], src_off)

    if isinstance(target_rank_val, int):
        state.bufs[target_rank_val] = z3.Store(state.bufs[target_rank_val], dst_off, value)
    else:
        for r in list(state.bufs.keys()):
            if r != self_rank:
                state.bufs[r] = z3.If(
                    target_rank_val == z3.BitVecVal(r, 32),
                    z3.Store(state.bufs[r], dst_off, value),
                    state.bufs[r],
                )


def _encode_rdma_read(state: SymbolicState, self_rank: int, params: Dict[str, z3.ExprRef]):
    src_rank_val = params["source_rank"]
    src_off = params["src_offset"]
    dst_off = params["dst_offset"]

    if isinstance(src_rank_val, int):
        value = z3.Select(state.bufs[src_rank_val], src_off)
    else:
        value = z3.If(
            src_rank_val == z3.BitVecVal(0, 32),
            z3.Select(state.bufs[0], src_off),
            z3.Select(state.bufs[1], src_off),
        )

    state.bufs[self_rank] = z3.Store(state.bufs[self_rank], dst_off, value)


def _encode_notify(state: SymbolicState, self_rank: int, params: Dict[str, z3.ExprRef]):
    target_rank_val = params["target_rank"]
    signal_id_val = params["signal_id"]

    if isinstance(target_rank_val, int) and isinstance(signal_id_val, int):
        key = (self_rank, target_rank_val, signal_id_val)
        state.signals[key] = z3.BoolVal(True)
    else:
        sig_name = f"sig_{self_rank}_to_{target_rank_val}_{signal_id_val}"
        state.signals[(self_rank, -1, -1)] = z3.Bool(sig_name)


def _encode_wait_notify(state: SymbolicState, self_rank: int, params: Dict[str, z3.ExprRef]):
    src_rank_val = params["source_rank"]
    signal_id_val = params["signal_id"]

    if isinstance(src_rank_val, int) and isinstance(signal_id_val, int):
        key = (src_rank_val, self_rank, signal_id_val)
        if key in state.signals:
            state.constraints.append(state.signals[key] == z3.BoolVal(True))
        else:
            state.constraints.append(z3.BoolVal(False))


def encode_ordering(edges: List[Tuple[int, int]]) -> List[z3.BoolRef]:
    time_vars = {}
    constraints = []
    for src_idx, dst_idx in edges:
        if src_idx not in time_vars:
            time_vars[src_idx] = z3.Int(f"t_{src_idx}")
        if dst_idx not in time_vars:
            time_vars[dst_idx] = z3.Int(f"t_{dst_idx}")
        constraints.append(time_vars[src_idx] < time_vars[dst_idx])
    for v in time_vars.values():
        constraints.append(v >= 0)
    return constraints
