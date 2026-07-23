"""Torch-distributed execution plans with OCS barriers between collectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from .plan import OCSPlan
from .runtime import OCSRuntime


_SUPPORTED_COLLECTIVES = {"all_reduce", "all_to_all_single"}


@dataclass(frozen=True)
class TorchCollectivePhase:
    """One blocking torch.distributed collective and an optional following barrier."""

    collective: str
    barrier_after: Optional[OCSPlan] = None

    def __post_init__(self) -> None:
        if self.collective not in _SUPPORTED_COLLECTIVES:
            raise ValueError(
                f"unsupported torch collective {self.collective!r}; "
                f"expected one of {sorted(_SUPPORTED_COLLECTIVES)}")


@dataclass(frozen=True)
class TorchCollectivePlan:
    """A fixed torch collective sequence with OCS boundaries between phases.

    Every non-final phase requires a barrier.  The final phase may also carry a
    barrier when the plan commits the topology for the next epoch/iteration.
    """

    phases: Tuple[TorchCollectivePhase, ...]

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("torch collective plan must contain at least one phase")
        for phase in self.phases[:-1]:
            if phase.barrier_after is None:
                raise ValueError("every non-final torch collective phase requires an OCS barrier")


class OcsTorchPlanRunner:
    """Execute blocking torch collectives around link-aligned OCS barriers."""

    def __init__(self, runtime: Optional[OCSRuntime] = None) -> None:
        self.runtime = runtime if runtime is not None else OCSRuntime()

    def execute(
        self,
        plan: TorchCollectivePlan,
        input_tensor: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
        group: Optional[dist.ProcessGroup] = None,
        timeout: Optional[float] = None,
        async_op: bool = False,
    ) -> torch.Tensor:
        if async_op:
            raise NotImplementedError("OCS torch collective plans only support blocking execution")

        current_tensor = input_tensor
        final_index = len(plan.phases) - 1
        for index, phase in enumerate(plan.phases):
            if index == final_index and output_tensor is not None:
                phase_output = output_tensor
            else:
                phase_output = torch.empty_like(current_tensor)

            if phase.collective == "all_reduce":
                phase_output.copy_(current_tensor)
                dist.all_reduce(phase_output, group=group)
            else:
                dist.all_to_all_single(phase_output, current_tensor, group=group)

            current_tensor = phase_output
            if phase.barrier_after is not None:
                self.runtime.barrier_switch(phase.barrier_after, group=group, timeout=timeout)

        return current_tensor


def build_torch_allreduce_alltoall_plan(
    world_size: int,
    job_id: str = "ocs_torch_collective_plan",
    group_id: int = 0,
    first_barrier_id: int = 0,
    first_epoch_id: int = 0,
    include_final_barrier: bool = False,
) -> TorchCollectivePlan:
    """Build an AllReduce/AllToAll/AllReduce phased plan.

    ``include_final_barrier`` adds the phase-2 boundary that commits the next
    epoch before another plan iteration starts.
    """
    if world_size < 2:
        raise ValueError("the fixed torch collective plan requires at least two ranks")

    participants = tuple(range(world_size))

    def barrier(offset: int, topology_id: int) -> OCSPlan:
        epoch_id = first_epoch_id + offset
        return OCSPlan(
            job_id=job_id,
            group_id=group_id,
            barrier_id=first_barrier_id + offset,
            epoch_id=epoch_id,
            next_epoch_id=epoch_id + 1,
            participant_ranks=participants,
            topology_id=topology_id,
            route_plan_id=first_barrier_id + offset,
            algorithm="torch_native",
            backend="torch",
        )

    return TorchCollectivePlan(phases=(
        TorchCollectivePhase("all_reduce", barrier_after=barrier(0, topology_id=1)),
        TorchCollectivePhase("all_to_all_single", barrier_after=barrier(1, topology_id=2)),
        TorchCollectivePhase(
            "all_reduce",
            barrier_after=barrier(2, topology_id=0) if include_final_barrier else None,
        ),
    ))
