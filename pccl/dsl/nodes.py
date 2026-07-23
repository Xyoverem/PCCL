"""PCCL DSL IR Nodes"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
from enum import Enum
import uuid


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    RDMA = "rdma"


class ExecutorType(Enum):
    SM = "sm"
    TMA = "tma"
    CE = "ce"
    HOST = "host"
    RDMA = "rdma"
    MULTIMEM = "multimem"


class PrimitiveOpType(Enum):
    SM_REDUCE = "sm.reduce"
    SM_COPY = "sm.copy"
    TMA_COPY = "tma.copy"
    TMA_REDUCE = "tma.reduce"
    CE_COPY = "ce.copy"
    MULTIMEM_REDUCE = "multimem.reduce"
    MULTIMEM_STORE = "multimem.store"
    RDMA_WRITE = "rdma.write"
    RDMA_READ = "rdma.read"
    NOTIFY = "notify"
    WAIT_NOTIFY = "wait_notify"
    OCS_BARRIER = "ocs.barrier"
    NOOP = "noop"


_OP_TYPE_TO_EXECUTOR: Dict[PrimitiveOpType, ExecutorType] = {
    PrimitiveOpType.SM_REDUCE: ExecutorType.SM,
    PrimitiveOpType.SM_COPY: ExecutorType.SM,
    PrimitiveOpType.TMA_COPY: ExecutorType.TMA,
    PrimitiveOpType.TMA_REDUCE: ExecutorType.TMA,
    PrimitiveOpType.CE_COPY: ExecutorType.CE,
    PrimitiveOpType.MULTIMEM_REDUCE: ExecutorType.MULTIMEM,
    PrimitiveOpType.MULTIMEM_STORE: ExecutorType.MULTIMEM,
    PrimitiveOpType.RDMA_WRITE: ExecutorType.RDMA,
    PrimitiveOpType.RDMA_READ: ExecutorType.RDMA,
    PrimitiveOpType.OCS_BARRIER: ExecutorType.HOST,
    PrimitiveOpType.NOOP: ExecutorType.SM,
}


def infer_executor(op_type: PrimitiveOpType) -> Optional[ExecutorType]:
    return _OP_TYPE_TO_EXECUTOR.get(op_type, None)


class ReduceOp(Enum):
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    AVG = "avg"


VALID_DTYPES = {"float32", "float16", "bfloat16", "float8_e4m3", "float8_e5m2"}
OCS_ALGORITHMS = {
    "ring", "rhd", "tree", "direct", "hierarchical", "auto",
    "torch_native", "custom",
}
OCS_BACKENDS = {"torch", "pccl"}
OCS_ROUTE_MODES = {"STATIC_PLAN", "ID_ROUTE", "SEGMENT_ROUTE", "USER_PLAN"}


@dataclass
class TensorInfo:
    dtype: Any
    shape: tuple

    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            raise ValueError(f"shape must be a tuple, got {type(self.shape)}")
        dtype_str = str(self.dtype)
        if dtype_str not in VALID_DTYPES:
            raise ValueError(
                f"unsupported dtype '{dtype_str}', must be one of {sorted(VALID_DTYPES)}")

    def numel(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result


@dataclass(repr=False)
class IRNode:
    op_id: str = ""
    op_type: Optional[PrimitiveOpType] = None
    device: DeviceType = DeviceType.CUDA
    executor: Optional[ExecutorType] = None
    tensor_info: Optional[TensorInfo] = None
    dependencies: List[str] = field(default_factory=list)
    next_ops: List[str] = field(default_factory=list)
    channel: int = 0

    def __post_init__(self):
        if not self.op_id and self.op_type is not None:
            self.op_id = f"{self.op_type.value.replace('.', '_')}_{uuid.uuid4().hex[:8]}"
        if self.executor is None and self.op_type is not None:
            self.executor = infer_executor(self.op_type)

    def add_dependency(self, dep_id: str) -> None:
        if dep_id not in self.dependencies:
            self.dependencies.append(dep_id)

    def add_next_op(self, next_id: str) -> None:
        if next_id not in self.next_ops:
            self.next_ops.append(next_id)

    def to_params(self) -> Dict[str, Any]:
        return {}

    def validate(self) -> bool:
        return True

    def __repr__(self) -> str:
        executor_str = self.executor.value if self.executor else "none"
        op_str = self.op_type.value if self.op_type else "none"
        return f"IRNode(id={self.op_id}, type={op_str}, executor={executor_str})"


# --- SM Operations ---

@dataclass
class SmReduceNode(IRNode):
    reduce_op: str = "sum"
    source_rank: int = -1
    src_offset: int = 0
    dst_offset: int = 0
    remote_offset: int = 0
    count: int = 0

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.SM_REDUCE
        IRNode.__post_init__(self)

    def validate(self) -> bool:
        if self.reduce_op not in [e.value for e in ReduceOp]:
            raise ValueError(f"SmReduceNode: invalid reduce_op '{self.reduce_op}'")
        if self.source_rank < 0:
            raise ValueError(f"SmReduceNode: source_rank must be non-negative, got {self.source_rank}")
        if self.count < 0:
            raise ValueError(f"SmReduceNode: count must be non-negative, got {self.count}")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {
            "reduce_op": self.reduce_op, "source_rank": self.source_rank,
            "src_offset": self.src_offset, "dst_offset": self.dst_offset,
            "remote_offset": self.remote_offset, "count": self.count,
        }


@dataclass
class SmCopyNode(IRNode):
    source_rank: int = -1
    src_offset: int = 0
    dst_offset: int = 0
    size: int = 0

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.SM_COPY
        IRNode.__post_init__(self)

    def validate(self) -> bool:
        if self.source_rank < 0:
            raise ValueError(f"SmCopyNode: source_rank must be non-negative, got {self.source_rank}")
        if self.size < 0:
            raise ValueError(f"SmCopyNode: size must be non-negative, got {self.size}")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {
            "source_rank": self.source_rank, "src_offset": self.src_offset,
            "dst_offset": self.dst_offset, "size": self.size,
        }


# --- TMA Operations ---

@dataclass
class TmaCopyNode(IRNode):
    source_rank: int = -1
    src_offset: int = 0
    dst_offset: int = 0
    size: int = 0

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.TMA_COPY
        IRNode.__post_init__(self)

    def validate(self) -> bool:
        if self.source_rank < 0:
            raise ValueError(f"TmaCopyNode: source_rank must be non-negative, got {self.source_rank}")
        if self.size < 0:
            raise ValueError(f"TmaCopyNode: size must be non-negative, got {self.size}")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {
            "source_rank": self.source_rank, "src_offset": self.src_offset,
            "dst_offset": self.dst_offset, "size": self.size,
        }


@dataclass
class TmaReduceNode(IRNode):
    reduce_op: str = "sum"
    source_rank: int = -1
    src_offset: int = 0
    dst_offset: int = 0
    remote_offset: int = 0
    count: int = 0

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.TMA_REDUCE
        IRNode.__post_init__(self)

    def validate(self) -> bool:
        if self.reduce_op not in [e.value for e in ReduceOp]:
            raise ValueError(f"TmaReduceNode: invalid reduce_op '{self.reduce_op}'")
        if self.source_rank < 0:
            raise ValueError(f"TmaReduceNode: source_rank must be non-negative, got {self.source_rank}")
        if self.count < 0:
            raise ValueError(f"TmaReduceNode: count must be non-negative, got {self.count}")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {
            "reduce_op": self.reduce_op, "source_rank": self.source_rank,
            "src_offset": self.src_offset, "dst_offset": self.dst_offset,
            "remote_offset": self.remote_offset, "count": self.count,
        }


# --- Multimem (NVLS) Operations ---

@dataclass
class MultimemReduceNode(IRNode):
    reduce_op: str = "sum"
    source_rank: int = -1
    src_offset: int = 0
    dst_offset: int = 0
    remote_offset: int = 0
    count: int = 0

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.MULTIMEM_REDUCE
        IRNode.__post_init__(self)

    def validate(self) -> bool:
        if self.reduce_op not in [e.value for e in ReduceOp]:
            raise ValueError(f"MultimemReduceNode: invalid reduce_op '{self.reduce_op}'")
        if self.source_rank < 0:
            raise ValueError(f"MultimemReduceNode: source_rank must be non-negative, got {self.source_rank}")
        if self.count < 0:
            raise ValueError(f"MultimemReduceNode: count must be non-negative, got {self.count}")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {
            "reduce_op": self.reduce_op, "source_rank": self.source_rank,
            "src_offset": self.src_offset, "dst_offset": self.dst_offset,
            "remote_offset": self.remote_offset, "count": self.count,
        }


@dataclass
class MultimemStoreNode(IRNode):
    source_rank: int = -1
    src_offset: int = 0
    dst_offset: int = 0
    size: int = 0

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.MULTIMEM_STORE
        IRNode.__post_init__(self)

    def validate(self) -> bool:
        if self.source_rank < 0:
            raise ValueError(f"MultimemStoreNode: source_rank must be non-negative, got {self.source_rank}")
        if self.size < 0:
            raise ValueError(f"MultimemStoreNode: size must be non-negative, got {self.size}")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {
            "source_rank": self.source_rank, "src_offset": self.src_offset,
            "dst_offset": self.dst_offset, "size": self.size,
        }


# --- Copy Engine Operations ---

@dataclass
class CeCopyNode(IRNode):
    source_rank: int = -1
    src_offset: int = 0
    dst_offset: int = 0
    size: int = 0

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.CE_COPY
        IRNode.__post_init__(self)

    def validate(self) -> bool:
        if self.source_rank < 0:
            raise ValueError(f"CeCopyNode: source_rank must be non-negative, got {self.source_rank}")
        if self.size < 0:
            raise ValueError(f"CeCopyNode: size must be non-negative, got {self.size}")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {
            "source_rank": self.source_rank, "src_offset": self.src_offset,
            "dst_offset": self.dst_offset, "size": self.size,
        }


# --- RDMA Operations ---

@dataclass
class RdmaWriteNode(IRNode):
    target_rank: int = -1
    src_offset: int = 0
    dst_offset: int = 0
    size: int = 0

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.RDMA_WRITE
        IRNode.__post_init__(self)

    def validate(self) -> bool:
        if self.target_rank < 0:
            raise ValueError(f"RdmaWriteNode: target_rank must be non-negative, got {self.target_rank}")
        if self.size < 0:
            raise ValueError(f"RdmaWriteNode: size must be non-negative, got {self.size}")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {
            "target_rank": self.target_rank, "src_offset": self.src_offset,
            "dst_offset": self.dst_offset, "size": self.size,
        }


@dataclass
class RdmaReadNode(IRNode):
    source_rank: int = -1
    src_offset: int = 0
    dst_offset: int = 0
    size: int = 0

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.RDMA_READ
        IRNode.__post_init__(self)

    def validate(self) -> bool:
        if self.source_rank < 0:
            raise ValueError(f"RdmaReadNode: source_rank must be non-negative, got {self.source_rank}")
        if self.size < 0:
            raise ValueError(f"RdmaReadNode: size must be non-negative, got {self.size}")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {
            "source_rank": self.source_rank, "src_offset": self.src_offset,
            "dst_offset": self.dst_offset, "size": self.size,
        }


# --- Synchronization Operations ---

@dataclass
class NotifyNode(IRNode):
    signal_id: int = 0
    target_rank: int = -1

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.NOTIFY
        IRNode.__post_init__(self)
        if self.executor is None:
            self.executor = ExecutorType.SM if self.device == DeviceType.CUDA else ExecutorType.HOST

    def validate(self) -> bool:
        if self.target_rank < 0:
            raise ValueError(f"NotifyNode: target_rank must be non-negative, got {self.target_rank}")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {"signal_id": self.signal_id, "target_rank": self.target_rank}


@dataclass
class WaitNotifyNode(IRNode):
    signal_id: int = 0
    source_rank: int = -1

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.WAIT_NOTIFY
        IRNode.__post_init__(self)
        if self.executor is None:
            self.executor = ExecutorType.SM if self.device == DeviceType.CUDA else ExecutorType.HOST

    def validate(self) -> bool:
        if self.source_rank < 0:
            raise ValueError(f"WaitNotifyNode: source_rank must be non-negative, got {self.source_rank}")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {"signal_id": self.signal_id, "source_rank": self.source_rank}


@dataclass
class OcsBarrierNode(IRNode):
    """Host-control boundary for an OCS topology reconfiguration.

    This is deliberately not lowered to a CUDA primitive. The compiler emits
    an OCS phased manifest, and the host runtime must obtain a controller
    release before launching the next data phase.
    """

    group_id: int = 0
    barrier_id: int = 0
    epoch_id: int = 0
    next_epoch_id: int = 1
    participant_ranks: Tuple[int, ...] = ()
    topology_id: int = 0
    route_mode: str = "STATIC_PLAN"
    route_plan_id: int = 0
    algorithm: str = "auto"
    backend: str = "pccl"
    payload: bytes = b""
    timeout_ms: int = 0

    def __post_init__(self):
        if self.op_type is None:
            self.op_type = PrimitiveOpType.OCS_BARRIER
        if self.device == DeviceType.CUDA:
            self.device = DeviceType.CPU
        IRNode.__post_init__(self)
        if self.executor is None:
            self.executor = ExecutorType.HOST

        self.participant_ranks = tuple(int(rank) for rank in self.participant_ranks)
        if self.payload is None:
            self.payload = b""
        elif not isinstance(self.payload, (bytes, bytearray, memoryview)):
            raise TypeError("OcsBarrierNode: payload must be bytes-like")
        else:
            self.payload = bytes(self.payload)

    @property
    def participant_bitmap(self) -> int:
        bitmap = 0
        for rank in self.participant_ranks:
            bitmap |= 1 << rank
        return bitmap

    def validate(self) -> bool:
        if self.group_id < 0:
            raise ValueError(f"OcsBarrierNode: group_id must be non-negative, got {self.group_id}")
        if self.barrier_id < 0:
            raise ValueError(
                f"OcsBarrierNode: barrier_id must be non-negative, got {self.barrier_id}")
        if self.epoch_id < 0 or self.next_epoch_id <= self.epoch_id:
            raise ValueError(
                "OcsBarrierNode: next_epoch_id must be greater than a non-negative epoch_id")
        if not self.participant_ranks:
            raise ValueError("OcsBarrierNode: participant_ranks must not be empty")
        if any(rank < 0 for rank in self.participant_ranks):
            raise ValueError("OcsBarrierNode: participant_ranks must be non-negative")
        if len(set(self.participant_ranks)) != len(self.participant_ranks):
            raise ValueError("OcsBarrierNode: participant_ranks must not contain duplicates")
        if self.topology_id < 0 or self.route_plan_id < 0:
            raise ValueError("OcsBarrierNode: topology_id and route_plan_id must be non-negative")
        if self.route_mode not in OCS_ROUTE_MODES:
            raise ValueError(f"OcsBarrierNode: unsupported route_mode '{self.route_mode}'")
        if self.algorithm not in OCS_ALGORITHMS:
            raise ValueError(f"OcsBarrierNode: unsupported algorithm '{self.algorithm}'")
        if self.backend not in OCS_BACKENDS:
            raise ValueError(f"OcsBarrierNode: unsupported backend '{self.backend}'")
        if self.timeout_ms < 0:
            raise ValueError("OcsBarrierNode: timeout_ms must be non-negative")
        return True

    def to_params(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "barrier_id": self.barrier_id,
            "epoch_id": self.epoch_id,
            "next_epoch_id": self.next_epoch_id,
            "participant_ranks": list(self.participant_ranks),
            "participant_bitmap": self.participant_bitmap,
            "topology_id": self.topology_id,
            "route_mode": self.route_mode,
            "route_plan_id": self.route_plan_id,
            "algorithm": self.algorithm,
            "backend": self.backend,
            "payload_hex": self.payload.hex(),
            "timeout_ms": self.timeout_ms,
        }

    def to_ocs_plan(self):
        """Convert to the runtime plan without coupling DSL import order to OCS."""
        from ..ocs.plan import OCSPlan

        return OCSPlan(
            group_id=self.group_id,
            barrier_id=self.barrier_id,
            epoch_id=self.epoch_id,
            next_epoch_id=self.next_epoch_id,
            participant_ranks=self.participant_ranks,
            topology_id=self.topology_id,
            route_mode=self.route_mode,
            route_plan_id=self.route_plan_id,
            algorithm=self.algorithm,
            backend=self.backend,
            payload=self.payload,
        )


IRNodeVariant = Union[
    SmReduceNode, SmCopyNode, TmaCopyNode, TmaReduceNode,
    MultimemReduceNode, MultimemStoreNode,
    CeCopyNode, RdmaWriteNode, RdmaReadNode,
    NotifyNode, WaitNotifyNode, OcsBarrierNode,
]
