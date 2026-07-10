"""Plan objects passed from an external OCS controller into PCCL wrappers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Tuple
import time


SUPPORTED_ALGORITHMS = {"ring", "rhd", "tree", "auto", "torch_native"}
SUPPORTED_BACKENDS = {"torch", "pccl"}
KNOWN_ROUTE_MODES = {"STATIC_PLAN", "ID_ROUTE", "SEGMENT_ROUTE", "USER_PLAN"}


def _time_us() -> int:
    return time.time_ns() // 1000


@dataclass(frozen=True)
class OCSPlan:
    """A precomputed OCS reconfiguration plan.

    The v0 runtime treats this as controller output. It does not compute the
    topology or algorithm; it only validates that all ranks commit the same
    plan before releasing the next collective.
    """

    job_id: str = "default"
    group_id: int = 0
    barrier_id: int = 0
    epoch_id: int = 0
    next_epoch_id: int = 1
    participant_ranks: Tuple[int, ...] = ()
    topology_id: int = 0
    route_mode: str = "STATIC_PLAN"
    route_plan_id: int = 0
    algorithm: str = "torch_native"
    backend: str = "torch"
    payload: bytes = b""

    def __post_init__(self) -> None:
        algorithm = str(self.algorithm)
        backend = str(self.backend)
        route_mode = str(self.route_mode)

        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"unsupported OCS algorithm '{algorithm}', expected one of "
                f"{sorted(SUPPORTED_ALGORITHMS)}")
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"unsupported OCS backend '{backend}', expected one of "
                f"{sorted(SUPPORTED_BACKENDS)}")
        if route_mode not in KNOWN_ROUTE_MODES:
            raise ValueError(
                f"unsupported route_mode '{route_mode}', expected one of "
                f"{sorted(KNOWN_ROUTE_MODES)}")

        participants = tuple(int(rank) for rank in self.participant_ranks)
        if any(rank < 0 for rank in participants):
            raise ValueError("participant_ranks must be non-negative")
        if len(set(participants)) != len(participants):
            raise ValueError("participant_ranks must not contain duplicates")

        payload = self.payload
        if payload is None:
            payload_bytes = b""
        elif isinstance(payload, (bytes, bytearray, memoryview)):
            payload_bytes = bytes(payload)
        else:
            raise TypeError("payload must be bytes-like")

        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "route_mode", route_mode)
        object.__setattr__(self, "participant_ranks", participants)
        object.__setattr__(self, "payload", payload_bytes)

    @property
    def participant_bitmap(self) -> int:
        bitmap = 0
        for rank in self.participant_ranks:
            bitmap |= 1 << rank
        return bitmap

    def with_default_participants(self, world_size: int) -> "OCSPlan":
        if self.participant_ranks:
            return self
        return replace(self, participant_ranks=tuple(range(int(world_size))))

    def consistency_fields(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "group_id": self.group_id,
            "barrier_id": self.barrier_id,
            "epoch_id": self.epoch_id,
            "next_epoch_id": self.next_epoch_id,
            "participant_bitmap": self.participant_bitmap,
            "topology_id": self.topology_id,
            "route_mode": self.route_mode,
            "route_plan_id": self.route_plan_id,
            "algorithm": self.algorithm,
            "backend": self.backend,
        }

    def ready_record(
        self,
        src_rank: int,
        world_size: int,
        arrive_seq: int = 0,
        arrival_time_us: int = None,
    ) -> Dict[str, Any]:
        plan = self.with_default_participants(world_size)
        record = {
            "msg_type": "OCS_BARRIER_READY",
            "version": 1,
            "src_rank": int(src_rank),
            "src_gpu_id": int(src_rank),
            "arrive_seq": int(arrive_seq),
            "payload_len": len(plan.payload),
            "arrival_time_us": _time_us() if arrival_time_us is None else int(arrival_time_us),
        }
        record.update(plan.consistency_fields())
        return record


def normalize_plan_sequence(plans: Iterable[OCSPlan]) -> Tuple[OCSPlan, ...]:
    result = tuple(plans)
    if not all(isinstance(plan, OCSPlan) for plan in result):
        raise TypeError("all static plans must be OCSPlan instances")
    return result
