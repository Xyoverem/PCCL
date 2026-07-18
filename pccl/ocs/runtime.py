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


def _monotonic_ns() -> int:
    """Return a process-local monotonic timestamp for latency accounting."""
    return time.perf_counter_ns()


def _duration_us(start_ns: int, end_ns: int) -> float:
    return max(0.0, (int(end_ns) - int(start_ns)) / 1_000.0)


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
    switch_start_mono_ns: Optional[int] = None
    switch_done_mono_ns: Optional[int] = None
    link_ready_mono_ns: Optional[int] = None

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
        delay_mode: str = "sleep",
    ) -> None:
        if switch_delay_s < 0 or link_ready_delay_s < 0:
            raise ValueError("mock switch and link-ready delays must be non-negative")
        if delay_mode not in {"sleep", "spin"}:
            raise ValueError("delay_mode must be 'sleep' or 'spin'")
        self.switch_delay_s = float(switch_delay_s)
        self.link_ready_delay_s = float(link_ready_delay_s)
        self.link_ready = bool(link_ready)
        self.delay_mode = delay_mode

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
            switch_start_mono_ns = _monotonic_ns()
            switch_start_time_us = _time_us()
            self._wait_delay(self.switch_delay_s)
            switch_done_mono_ns = _monotonic_ns()
            switch_done_time_us = _time_us()
            self._wait_delay(self.link_ready_delay_s)
            if self.link_ready:
                link_ready_mono_ns = _monotonic_ns()
                return OCSSwitchResult(
                    link_state=OCSLinkState.LINK_ALIGNED,
                    switch_start_time_us=switch_start_time_us,
                    switch_done_time_us=switch_done_time_us,
                    link_ready_time_us=_time_us(),
                    switch_start_mono_ns=switch_start_mono_ns,
                    switch_done_mono_ns=switch_done_mono_ns,
                    link_ready_mono_ns=link_ready_mono_ns,
                )
            return OCSSwitchResult(
                link_state=OCSLinkState.LINK_NOT_READY,
                switch_start_time_us=switch_start_time_us,
                switch_done_time_us=switch_done_time_us,
                link_ready_time_us=None,
                error_code=1,
                error_message="mock link did not reach alignment",
                switch_start_mono_ns=switch_start_mono_ns,
                switch_done_mono_ns=switch_done_mono_ns,
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

    def _wait_delay(self, delay_s: float) -> None:
        if delay_s <= 0:
            return
        if self.delay_mode == "sleep":
            time.sleep(delay_s)
            return

        deadline_ns = _monotonic_ns() + int(delay_s * 1_000_000_000)
        while _monotonic_ns() < deadline_ns:
            pass


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
        arrival_mono_ns = _monotonic_ns()
        arrival_time_us = _time_us()
        ready = normalized.ready_record(
            src_rank=rank,
            world_size=world_size,
            arrive_seq=arrive_seq,
            arrival_time_us=arrival_time_us,
        )

        state_trace = [_state_event(OCSBarrierState.READY, arrival_time_us)]
        records = self.connector.exchange_ready(ready, group=group, timeout=timeout)
        ready_exchange_done_mono_ns = _monotonic_ns()
        self._validate_ready_records(normalized, records)
        ready_validation_done_mono_ns = _monotonic_ns()
        all_arrived_time_us = _time_us()
        state_trace.append(_state_event(OCSBarrierState.ALL_ARRIVED, all_arrived_time_us))

        commit_start_mono_ns = _monotonic_ns()
        switch_result = self.connector.commit_switch(
            normalized,
            records,
            group=group,
            timeout=timeout,
        )
        commit_done_mono_ns = _monotonic_ns()
        state_trace.append(
            _state_event(OCSBarrierState.SWITCHING, switch_result.switch_start_time_us))
        state_trace.append(
            _state_event(OCSBarrierState.WAIT_LINK_ALIGN, switch_result.switch_done_time_us))

        if not switch_result.is_link_ready:
            failed_time_us = _time_us()
            failed_mono_ns = _monotonic_ns()
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
                "timing": self._latency_breakdown(
                    arrival_mono_ns,
                    ready_exchange_done_mono_ns,
                    ready_validation_done_mono_ns,
                    commit_start_mono_ns,
                    commit_done_mono_ns,
                    failed_mono_ns,
                    switch_result,
                ),
                "state_trace": state_trace,
                "ready_records": records,
            })
            raise OCSLinkNotReady(
                f"barrier {normalized.barrier_id} link state is "
                f"{switch_result.link_state.value}: {switch_result.error_message}")

        release_time_us = _time_us()
        release_mono_ns = _monotonic_ns()
        state_trace.append(_state_event(OCSBarrierState.RELEASED, release_time_us))
        timing = self._latency_breakdown(
            arrival_mono_ns,
            ready_exchange_done_mono_ns,
            ready_validation_done_mono_ns,
            commit_start_mono_ns,
            commit_done_mono_ns,
            release_mono_ns,
            switch_result,
        )

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
            "latency_us": int(round(timing["total_us"])),
            "timing": timing,
            "state_trace": state_trace,
            "ready_records": records,
        }
        self.history.append(release)
        return release

    @staticmethod
    def _latency_breakdown(
        arrival_mono_ns: int,
        ready_exchange_done_mono_ns: int,
        ready_validation_done_mono_ns: int,
        commit_start_mono_ns: int,
        commit_done_mono_ns: int,
        completion_mono_ns: int,
        switch_result: OCSSwitchResult,
    ) -> Dict[str, float]:
        """Return local control-path timings plus controller-internal delays.

        ``*_local_us`` and ``controller_commit_us`` are process-local monotonic
        measurements. ``controller_switch_us`` and ``controller_link_align_us``
        originate from the connector leader and are therefore reported
        separately rather than summed into a cross-host timeline.
        """
        if (
            switch_result.switch_start_mono_ns is not None
            and switch_result.switch_done_mono_ns is not None
        ):
            controller_switch_us = _duration_us(
                switch_result.switch_start_mono_ns,
                switch_result.switch_done_mono_ns,
            )
        else:
            controller_switch_us = max(
                0.0,
                float(switch_result.switch_done_time_us - switch_result.switch_start_time_us),
            )

        if (
            switch_result.switch_done_mono_ns is not None
            and switch_result.link_ready_mono_ns is not None
        ):
            controller_link_align_us = _duration_us(
                switch_result.switch_done_mono_ns,
                switch_result.link_ready_mono_ns,
            )
        elif switch_result.link_ready_time_us is not None:
            controller_link_align_us = max(
                0.0,
                float(switch_result.link_ready_time_us - switch_result.switch_done_time_us),
            )
        else:
            controller_link_align_us = 0.0

        return {
            "ready_exchange_us": _duration_us(
                arrival_mono_ns, ready_exchange_done_mono_ns),
            "ready_validation_us": _duration_us(
                ready_exchange_done_mono_ns, ready_validation_done_mono_ns),
            "controller_commit_us": _duration_us(
                commit_start_mono_ns, commit_done_mono_ns),
            "release_local_us": _duration_us(commit_done_mono_ns, completion_mono_ns),
            "total_us": _duration_us(arrival_mono_ns, completion_mono_ns),
            "controller_switch_us": controller_switch_us,
            "controller_link_align_us": controller_link_align_us,
        }

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
