"""Execution plans that place OCS barriers between distinct collectives."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ..dsl.algorithms import RingAllreduce
from ..dsl.codegen import RuntimeGraphGenerator
from ..dsl.compiler import Compiler
from ..dsl.graph import PrimitiveIRGraph
from .plan import OCSPlan
from .runtime import OCSRuntime


@dataclass(frozen=True)
class OCSCollectivePhase:
    """One independently compiled collective followed by an optional barrier."""

    name: str
    graph: PrimitiveIRGraph
    barrier_after: Optional[OCSPlan] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("OCS collective phase name must not be empty")
        if self.graph.has_ocs_barriers():
            raise ValueError("an OCS collective phase must not contain an OCS barrier node")
        if not self.graph.collective_type:
            raise ValueError("an OCS collective phase must declare collective_type")


@dataclass(frozen=True)
class OCSCollectivePlan:
    """A fixed sequence of collective graphs and intervening OCS barriers."""

    phases: Tuple[OCSCollectivePhase, ...]

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("OCS collective plan must contain at least one phase")
        if self.phases[-1].barrier_after is not None:
            raise ValueError("the final collective phase must not have a following barrier")
        for phase in self.phases[:-1]:
            if phase.barrier_after is None:
                raise ValueError("every non-final collective phase requires an OCS barrier")


@dataclass
class PreparedOcsCollectivePlan:
    """Materialized PCCL operations for a concrete OCS collective plan."""

    operation_names: Tuple[str, ...]
    phase_files: Tuple[Path, ...]
    barriers_after_phase: Tuple[Optional[OCSPlan], ...]
    collective_types: Tuple[str, ...]
    json_dir: Path
    owns_json_dir: bool
    closed: bool = False

    def close(self) -> None:
        """Remove generated JSON after plan execution is complete."""
        if not self.closed and self.owns_json_dir:
            shutil.rmtree(self.json_dir, ignore_errors=True)
        self.closed = True


class OcsCollectivePlanRunner:
    """Execute separate PCCL collectives with link-aligned OCS boundaries."""

    def __init__(
        self,
        engine: Any = None,
        runtime: Optional[OCSRuntime] = None,
        compiler: Optional[Compiler] = None,
        json_dir: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.runtime = runtime if runtime is not None else OCSRuntime()
        self.compiler = compiler if compiler is not None else Compiler()
        self.json_dir = Path(json_dir) if json_dir is not None else None

    def prepare(
        self,
        plan: OCSCollectivePlan,
        operation_name: str = "ocs_collective_plan",
    ) -> PreparedOcsCollectivePlan:
        """Compile and materialize each collective as a JSON v2 operation.

        Registration is deliberately deferred until execution.  PCCL v0 maps
        every registered operation to the same runtime buffer and signal
        regions, so several live phase workspaces would alias each other.
        """
        json_dir, owns_json_dir = self._create_json_dir()
        operation_names: List[str] = []
        phase_files: List[Path] = []
        barriers: List[Optional[OCSPlan]] = []
        collective_types: List[str] = []

        try:
            for index, phase in enumerate(plan.phases):
                compiled = self.compiler.compile(phase.graph)
                manifest = RuntimeGraphGenerator().generate(compiled)
                if manifest.get("version") != 2:
                    raise RuntimeError(
                        f"collective phase '{phase.name}' did not compile to JSON v2")

                phase_name = f"{operation_name}_{index}_{phase.name}"
                phase_file = json_dir / f"{phase_name}.json"
                phase_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

                operation_names.append(phase_name)
                phase_files.append(phase_file)
                barriers.append(phase.barrier_after)
                collective_types.append(phase.graph.collective_type)
        except Exception:
            if owns_json_dir:
                shutil.rmtree(json_dir, ignore_errors=True)
            raise

        return PreparedOcsCollectivePlan(
            operation_names=tuple(operation_names),
            phase_files=tuple(phase_files),
            barriers_after_phase=tuple(barriers),
            collective_types=tuple(collective_types),
            json_dir=json_dir,
            owns_json_dir=owns_json_dir,
        )

    def execute(
        self,
        prepared: PreparedOcsCollectivePlan,
        input_tensor: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
        group: Optional[dist.ProcessGroup] = None,
        timeout: Optional[float] = None,
        async_op: bool = False,
    ) -> torch.Tensor:
        """Run every collective, releasing the next only after link alignment."""
        if async_op:
            raise NotImplementedError("OCS collective plans only support blocking execution")
        if prepared.closed:
            raise RuntimeError("prepared OCS collective plan has been closed")

        current_tensor = input_tensor
        final_index = len(prepared.operation_names) - 1
        for index, (operation_name, phase_file) in enumerate(
                zip(prepared.operation_names, prepared.phase_files)):
            if index == final_index and output_tensor is not None:
                phase_output = output_tensor
            else:
                phase_output = torch.empty_like(current_tensor)

            if not self._engine().register_operation(operation_name, str(phase_file)):
                raise RuntimeError(f"failed to register collective phase '{operation_name}'")
            self._synchronize_phase_registration(group)
            self._engine().execute_operation(operation_name, current_tensor, phase_output)
            current_tensor = phase_output

            barrier = prepared.barriers_after_phase[index]
            if barrier is not None:
                self.runtime.barrier_switch(barrier, group=group, timeout=timeout)
                self._reset_signals(operation_name)

        return current_tensor

    def _create_json_dir(self) -> Tuple[Path, bool]:
        if self.json_dir is not None:
            self.json_dir.mkdir(parents=True, exist_ok=True)
            return self.json_dir, False
        return Path(tempfile.mkdtemp(prefix="pccl-ocs-plan-")), True

    def _engine(self) -> Any:
        if self.engine is None:
            from .. import engine as pccl_engine
            self.engine = pccl_engine
        return self.engine

    def _reset_signals(self, operation_name: str) -> None:
        reset = getattr(self._engine(), "reset_signals", None)
        if callable(reset):
            reset(operation_name)

    @staticmethod
    def _synchronize_phase_registration(group: Optional[dist.ProcessGroup]) -> None:
        """Align ranks after local ``regOp`` and before peer-signal traffic.

        PCCL v0 has no distributed registration lifecycle.  Its static launch
        path performs this fence explicitly, so phased execution must preserve
        the same invariant until registration is moved into the engine.
        """
        if dist.is_initialized():
            dist.barrier(group=group)


def build_ring_allreduce_alltoall_plan(
    rank: int,
    world_size: int,
    tensor_size: int,
    dtype: str = "float32",
    executor: str = "sm",
    job_id: str = "ocs_collective_plan",
    group_id: int = 0,
    first_barrier_id: int = 0,
    first_epoch_id: int = 0,
) -> OCSCollectivePlan:
    """Build ``AllReduce -> Barrier -> AllToAll -> Barrier -> AllReduce``."""
    if world_size < 2:
        raise ValueError("the fixed collective plan requires at least two ranks")
    if tensor_size <= 0 or tensor_size % world_size:
        raise ValueError("tensor_size must be positive and divisible by world_size")

    ring = RingAllreduce()
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
            algorithm="ring",
            backend="pccl",
        )

    return OCSCollectivePlan(phases=(
        OCSCollectivePhase(
            name="allreduce_0",
            graph=ring.build_allreduce(rank, world_size, tensor_size, dtype, executor),
            barrier_after=barrier(0, topology_id=1),
        ),
        OCSCollectivePhase(
            name="alltoall_1",
            graph=ring.build_alltoall(rank, world_size, tensor_size, dtype, executor),
            barrier_after=barrier(1, topology_id=2),
        ),
        OCSCollectivePhase(
            name="allreduce_2",
            graph=ring.build_allreduce(rank, world_size, tensor_size, dtype, executor),
        ),
    ))
