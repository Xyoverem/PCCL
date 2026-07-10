"""Controller-facing plan providers for OCS-aware collectives."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

from .plan import OCSPlan, normalize_plan_sequence


PlanInput = Union[Iterable[OCSPlan], Mapping[str, Iterable[OCSPlan]]]


class StaticPlanController:
    """A deterministic stand-in for the external OCS controller.

    v0 assumes the real controller has already computed when to switch, which
    topology to use, and which collective algorithm should run. This class
    returns those precomputed plans without doing optimization itself.
    """

    def __init__(
        self,
        plans: Optional[PlanInput] = None,
        default_algorithm: str = "torch_native",
        default_backend: str = "torch",
    ) -> None:
        self.default_algorithm = default_algorithm
        self.default_backend = default_backend
        self._plans: Dict[str, Tuple[OCSPlan, ...]] = {}
        self._cursors: Dict[str, int] = {}

        if plans is None:
            return

        if isinstance(plans, Mapping):
            for event_key, event_plans in plans.items():
                self._plans[str(event_key)] = normalize_plan_sequence(event_plans)
        else:
            self._plans["all_reduce"] = normalize_plan_sequence(plans)

    def next_plan(self, event_key: str, rank: int, world_size: int) -> OCSPlan:
        event = str(event_key)
        cursor = self._cursors.get(event, 0)
        self._cursors[event] = cursor + 1

        plans = self._plans.get(event)
        if plans:
            plan = plans[cursor % len(plans)]
        else:
            plan = OCSPlan(
                barrier_id=cursor,
                epoch_id=cursor,
                next_epoch_id=cursor + 1,
                participant_ranks=tuple(range(int(world_size))),
                algorithm=self.default_algorithm,
                backend=self.default_backend,
            )

        if not plan.participant_ranks:
            plan = plan.with_default_participants(world_size)

        return replace(plan)
