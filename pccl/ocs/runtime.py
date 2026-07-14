"""Runtime glue for OCS-aware PyTorch/PCCL collectives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable
import time

import torch
import torch.distributed as dist

from .controller import OCSPlanController, StaticPlanController
from .exceptions import OCSBarrierTimeout, OCSLinkNotReady, OCSPlanMismatchError
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


class OCSBarrierState(str, Enum):
    """Host-visible OCS commit states for one communication phase boundary."""

    READY = "READY"
    ALL_ARRIVED = "ALL_ARRIVED"
    SWITCHING = "SWITCHING"
    WAIT_LINK_ALIGN = "WAIT_LINK_ALIGN"
    RELEASED = "RELEASED"
    FAILED = "FAILED"


class OCSLinkState(str, Enum):
    """Minimal hardware-facing link states required by the software contract."""

    OCS_CONFIG_ACCEPTED = "OCS_CONFIG_ACCEPTED"
    OCS_SWITCH_DONE = "OCS_SWITCH_DONE"
    LINK_ALIGNED = "LINK_ALIGNED"
    LINK_NOT_READY = "LINK_NOT_READY"


@dataclass(frozen=True)
class OCSSwitchResult:
    """Switch-controller result consumed before a barrier release is issued."""

    link_state: OCSLinkState
    switch_start_time_us: int
    switch_done_time_us: int
    link_ready_time_us: Optional[int]
    error_code: int = 0
    error_message: str = ""

    @property
    def is_link_ready(self) -> bool:
        return self.link_state is OCSLinkState.LINK_ALIGNED


def _state_event(state: OCSBarrierState, timestamp_us: int) -> Dict[str, Any]:
    return {"state": state.value, "time_us": int(timestamp_us)}


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

    def commit_switch(
        self,
        plan: OCSPlan,
        ready_records: Sequence[Dict[str, Any]],
        group: Optional[dist.ProcessGroup] = None,
        timeout: Optional[float] = None,
    ) -> OCSSwitchResult:
        """Commit the topology and report whether the resulting links are aligned."""


class TorchDistributedSwitchConnector:
    """Torch-distributed mock of READY aggregation and switch/link completion."""

    def __init__(
        self,
        switch_delay_s: float = 0.0,
        link_ready_delay_s: float = 0.0,
        link_ready: bool = True,
    ) -> None:
        if switch_delay_s < 0 or link_ready_delay_s < 0:
            raise ValueError("mock switch and link-ready delays must be non-negative")
        self.switch_delay_s = float(switch_delay_s)
        self.link_ready_delay_s = float(link_ready_delay_s)
        self.link_ready = bool(link_ready)

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

    def commit_switch(
        self,
        plan: OCSPlan,
        ready_records: Sequence[Dict[str, Any]],
        group: Optional[dist.ProcessGroup] = None,
        timeout: Optional[float] = None,
    ) -> OCSSwitchResult:
        del ready_records, timeout

        def simulate_commit() -> OCSSwitchResult:
            switch_start_time_us = _time_us()
            if self.switch_delay_s:
                time.sleep(self.switch_delay_s)
            switch_done_time_us = _time_us()
            if self.link_ready_delay_s:
                time.sleep(self.link_ready_delay_s)
            if self.link_ready:
                return OCSSwitchResult(
                    link_state=OCSLinkState.LINK_ALIGNED,
                    switch_start_time_us=switch_start_time_us,
                    switch_done_time_us=switch_done_time_us,
                    link_ready_time_us=_time_us(),
                )
            return OCSSwitchResult(
                link_state=OCSLinkState.LINK_NOT_READY,
                switch_start_time_us=switch_start_time_us,
                switch_done_time_us=switch_done_time_us,
                link_ready_time_us=None,
                error_code=1,
                error_message="mock link did not reach alignment",
            )

        if not _dist_ready():
            return simulate_commit()

        rank, _ = _rank_world_size(group)
        leader_rank = min(plan.participant_ranks)
        result_box: List[Optional[OCSSwitchResult]] = [
            simulate_commit() if rank == leader_rank else None
        ]
        dist.broadcast_object_list(result_box, src=leader_rank, group=group)
        result = result_box[0]
        if not isinstance(result, OCSSwitchResult):
            raise RuntimeError("switch connector did not broadcast an OCSSwitchResult")
        return result


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

        state_trace = [_state_event(OCSBarrierState.READY, arrival_time_us)]
        records = self.connector.exchange_ready(ready, group=group, timeout=timeout)
        self._validate_ready_records(normalized, records)
        all_arrived_time_us = _time_us()
        state_trace.append(_state_event(OCSBarrierState.ALL_ARRIVED, all_arrived_time_us))

        switch_result = self.connector.commit_switch(
            normalized,
            records,
            group=group,
            timeout=timeout,
        )
        state_trace.append(
            _state_event(OCSBarrierState.SWITCHING, switch_result.switch_start_time_us))
        state_trace.append(
            _state_event(OCSBarrierState.WAIT_LINK_ALIGN, switch_result.switch_done_time_us))

        if not switch_result.is_link_ready:
            failed_time_us = _time_us()
            state_trace.append(_state_event(OCSBarrierState.FAILED, failed_time_us))
            self.history.append({
                "msg_type": "OCS_BARRIER_ABORT",
                "version": 1,
                "job_id": normalized.job_id,
                "group_id": normalized.group_id,
                "barrier_id": normalized.barrier_id,
                "epoch_id": normalized.epoch_id,
                "next_epoch_id": normalized.next_epoch_id,
                "status": "LINK_NOT_READY",
                "error_code": switch_result.error_code,
                "error_message": switch_result.error_message,
                "link_state": switch_result.link_state.value,
                "arrival_time_us": arrival_time_us,
                "switch_start_time_us": switch_result.switch_start_time_us,
                "switch_done_time_us": switch_result.switch_done_time_us,
                "state_trace": state_trace,
                "ready_records": records,
            })
            raise OCSLinkNotReady(
                f"barrier {normalized.barrier_id} link state is "
                f"{switch_result.link_state.value}: {switch_result.error_message}")

        release_time_us = _time_us()
        state_trace.append(_state_event(OCSBarrierState.RELEASED, release_time_us))

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
            "switch_start_time_us": switch_result.switch_start_time_us,
            "switch_done_time_us": switch_result.switch_done_time_us,
            "link_ready_time_us": switch_result.link_ready_time_us,
            "link_state": switch_result.link_state.value,
            "release_time_us": release_time_us,
            "latency_us": release_time_us - arrival_time_us,
            "state_trace": state_trace,
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
