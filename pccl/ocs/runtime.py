"""Runtime glue for OCS-aware PyTorch/PCCL collectives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable
import time

import torch
import torch.distributed as dist

from .controller import OCSPlanController, StaticPlanController
from .exceptions import OCSBarrierTimeout, OCSPlanMismatchError
from .plan import OCSPlan


CONSISTENCY_FIELDS = (
    "job_id",
    "group_id",
    "barrier_id",
    "epoch_id",
    "next_epoch_id",
    "topology_id",
    "route_mode",
    "route_plan_id",
    "algorithm",
    "backend",
)


def _time_us() -> int:
    return time.time_ns() // 1000


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank_world_size(group: Optional[dist.ProcessGroup] = None) -> Tuple[int, int]:
    if not _dist_ready():
        return 0, 1
    return dist.get_rank(group), dist.get_world_size(group)


@runtime_checkable
class SwitchConnector(Protocol):
    """Southbound READY/RELEASE transport contract for an OCS barrier."""

    def exchange_ready(
        self,
        ready_record: Dict[str, Any],
        group: Optional[dist.ProcessGroup] = None,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Collect READY records and return the controller-visible set."""


class TorchDistributedSwitchConnector:
    """Switch simulation backed by torch.distributed object collectives."""

    def exchange_ready(
        self,
        ready_record: Dict[str, Any],
        group: Optional[dist.ProcessGroup] = None,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not _dist_ready():
            return [ready_record]

        world_size = dist.get_world_size(group)
        records: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
        dist.all_gather_object(records, ready_record, group=group)
        return [record for record in records if record is not None]


class OCSRuntime:
    """Coordinates OCS barrier commit before running a collective."""

    def __init__(
        self,
        controller: Optional[OCSPlanController] = None,
        connector: Optional[SwitchConnector] = None,
    ) -> None:
        self.controller = controller if controller is not None else StaticPlanController()
        self.connector = connector if connector is not None else TorchDistributedSwitchConnector()
        self.history: List[Dict[str, Any]] = []
        self._arrive_seq = 0

    def next_plan(
        self,
        event_key: str,
        group: Optional[dist.ProcessGroup] = None,
    ) -> OCSPlan:
        rank, world_size = _rank_world_size(group)
        return self.controller.next_plan(event_key, rank=rank, world_size=world_size)

    def barrier_switch(
        self,
        plan: OCSPlan,
        group: Optional[dist.ProcessGroup] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        if plan.route_mode != "STATIC_PLAN":
            raise NotImplementedError(
                f"route_mode={plan.route_mode!r} is reserved; v0 only executes STATIC_PLAN")

        rank, world_size = _rank_world_size(group)
        normalized = plan.with_default_participants(world_size)
        if rank not in normalized.participant_ranks:
            raise OCSPlanMismatchError(
                f"rank {rank} is not listed in participant_ranks={normalized.participant_ranks}")

        arrive_seq = self._arrive_seq
        self._arrive_seq += 1
        arrival_time_us = _time_us()
        ready = normalized.ready_record(
            src_rank=rank,
            world_size=world_size,
            arrive_seq=arrive_seq,
            arrival_time_us=arrival_time_us,
        )

        records = self.connector.exchange_ready(ready, group=group, timeout=timeout)
        release_time_us = _time_us()
        self._validate_ready_records(normalized, records)

        release = {
            "msg_type": "OCS_BARRIER_RELEASE",
            "version": 1,
            "job_id": normalized.job_id,
            "group_id": normalized.group_id,
            "barrier_id": normalized.barrier_id,
            "epoch_id": normalized.epoch_id,
            "next_epoch_id": normalized.next_epoch_id,
            "status": "OK",
            "error_code": 0,
            "participant_bitmap": normalized.participant_bitmap,
            "topology_id": normalized.topology_id,
            "route_plan_id": normalized.route_plan_id,
            "algorithm": normalized.algorithm,
            "backend": normalized.backend,
            "arrival_time_us": arrival_time_us,
            "release_time_us": release_time_us,
            "latency_us": release_time_us - arrival_time_us,
            "ready_records": records,
        }
        self.history.append(release)
        return release

    def _validate_ready_records(
        self,
        plan: OCSPlan,
        records: Sequence[Dict[str, Any]],
    ) -> None:
        expected = plan.consistency_fields()
        expected_ranks = set(plan.participant_ranks)
        arrived: Dict[int, Dict[str, Any]] = {}

        for record in records:
            if record.get("msg_type") != "OCS_BARRIER_READY":
                raise OCSPlanMismatchError(f"unexpected OCS message: {record!r}")

            src_rank = int(record.get("src_rank", -1))
            if src_rank not in expected_ranks:
                continue
            if src_rank in arrived:
                raise OCSPlanMismatchError(
                    f"duplicate READY from rank {src_rank} for barrier {plan.barrier_id}")

            arrived[src_rank] = record
            for field in CONSISTENCY_FIELDS:
                if record.get(field) != expected[field]:
                    raise OCSPlanMismatchError(
                        f"rank {src_rank} has inconsistent {field}: "
                        f"{record.get(field)!r} != {expected[field]!r}")

            if record.get("participant_bitmap") != expected["participant_bitmap"]:
                raise OCSPlanMismatchError(
                    f"rank {src_rank} has inconsistent participant_bitmap: "
                    f"{record.get('participant_bitmap')!r} != {expected['participant_bitmap']!r}")

        missing = sorted(expected_ranks.difference(arrived))
        if missing:
            raise OCSBarrierTimeout(
                f"barrier {plan.barrier_id} missing READY from ranks {missing}")


_default_runtime: Optional[OCSRuntime] = None


def get_default_runtime() -> OCSRuntime:
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = OCSRuntime()
    return _default_runtime


def ocs_barrier_switch(
    group: Optional[dist.ProcessGroup],
    plan: OCSPlan,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    return get_default_runtime().barrier_switch(plan, group=group, timeout=timeout)


def ocs_all_reduce(
    tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    runtime: Optional[OCSRuntime] = None,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    async_op: bool = False,
    timeout: Optional[float] = None,
) -> torch.Tensor:
    if async_op:
        raise NotImplementedError("OCS all_reduce is blocking in v0; async_op is reserved")

    rt = runtime if runtime is not None else get_default_runtime()
    plan = rt.next_plan("all_reduce", group=group)
    if plan.backend != "torch":
        raise NotImplementedError(
            f"backend={plan.backend!r} is reserved; v0 only executes the torch backend")

    rt.barrier_switch(plan, group=group, timeout=timeout)
    dist.all_reduce(tensor, op=op, group=group)
    return tensor
