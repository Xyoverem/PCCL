"""Host orchestration for JSON v3 phased OCS graphs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ..dsl.codegen import RuntimeGraphGenerator
from ..dsl.compiler import Compiler
from ..dsl.graph import PrimitiveIRGraph
from .plan import OCSPlan
from .runtime import OCSRuntime


@dataclass
class PreparedOcsGraph:
    """Materialized PCCL phase operations and their control barriers."""

    operation_names: Tuple[Optional[str], ...]
    phase_files: Tuple[Optional[Path], ...]
    barriers_after_phase: Tuple[Optional[OCSPlan], ...]
    manifest: Dict[str, Any]
    json_dir: Path
    owns_json_dir: bool
    closed: bool = False

    def close(self) -> None:
        """Remove generated JSON files after the engine has loaded them."""
        if not self.closed and self.owns_json_dir:
            shutil.rmtree(self.json_dir, ignore_errors=True)
        self.closed = True


class OcsPhaseRunner:
    """Execute PCCL data phases around OCS controller commit points.

    The existing PCCL engine executes one JSON v2 graph at a time. This runner
    bridges a JSON v3 ``phased_ocs`` manifest to that engine by registering each
    data phase separately, synchronously executing it, then obtaining release
    for the barrier that follows the phase.
    """

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
        graph: PrimitiveIRGraph,
        operation_name: Optional[str] = None,
    ) -> PreparedOcsGraph:
        """Compile and materialize every data phase in ``graph``.

        PCCL v0 uses one shared runtime workspace.  Deferring registration to
        ``execute`` ensures that only the phase currently running owns it.
        """
        compiled = self.compiler.compile(graph)
        manifest = RuntimeGraphGenerator().generate(compiled)
        if manifest.get("execution_model") != "phased_ocs":
            raise ValueError("OcsPhaseRunner requires a graph containing an OCS barrier")

        phases = compiled.split_ocs_phases()
        manifest_phases = manifest["phases"]
        if len(phases) != len(manifest_phases):
            raise RuntimeError("OCS phase manifest does not match the compiled IR graph")

        json_dir, owns_json_dir = self._create_json_dir()
        base_name = operation_name or compiled.graph_id
        operation_names: List[Optional[str]] = []
        phase_files: List[Optional[Path]] = []
        barriers: List[Optional[OCSPlan]] = []

        try:
            for phase, phase_manifest in zip(phases, manifest_phases):
                barriers.append(
                    phase.barrier.to_ocs_plan() if phase.barrier is not None else None)

                if not phase_manifest["operations"]:
                    operation_names.append(None)
                    phase_files.append(None)
                    continue

                phase_name = "{}_phase_{}".format(base_name, phase.index)
                phase_file = json_dir / "{}.json".format(phase_name)
                phase_file.write_text(
                    json.dumps(self._to_runtime_v2(manifest, phase_manifest), indent=2),
                    encoding="utf-8",
                )
                operation_names.append(phase_name)
                phase_files.append(phase_file)
        except Exception:
            if owns_json_dir:
                shutil.rmtree(json_dir, ignore_errors=True)
            raise

        return PreparedOcsGraph(
            operation_names=tuple(operation_names),
            phase_files=tuple(phase_files),
            barriers_after_phase=tuple(barriers),
            manifest=manifest,
            json_dir=json_dir,
            owns_json_dir=owns_json_dir,
        )

    def execute(
        self,
        prepared: PreparedOcsGraph,
        input_tensor: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
        group: Optional[dist.ProcessGroup] = None,
        timeout: Optional[float] = None,
        async_op: bool = False,
    ) -> torch.Tensor:
        """Run phase data operations and commit each following OCS barrier."""
        if async_op:
            raise NotImplementedError("OcsPhaseRunner only supports blocking execution")
        if prepared.closed:
            raise RuntimeError("prepared OCS graph has been closed")

        data_phase_indices = [
            index for index, name in enumerate(prepared.operation_names)
            if name is not None
        ]
        if not data_phase_indices:
            if output_tensor is None:
                result = input_tensor
            else:
                output_tensor.copy_(input_tensor)
                result = output_tensor
            for barrier in prepared.barriers_after_phase:
                if barrier is not None:
                    self.runtime.barrier_switch(barrier, group=group, timeout=timeout)
            return result

        last_data_phase = data_phase_indices[-1]
        current_tensor = input_tensor
        for phase_index, (operation_name, phase_file) in enumerate(
                zip(prepared.operation_names, prepared.phase_files)):
            if operation_name is not None:
                if phase_file is None:
                    raise RuntimeError("data phase is missing its generated JSON")
                if phase_index == last_data_phase and output_tensor is not None:
                    phase_output = output_tensor
                else:
                    phase_output = torch.empty_like(current_tensor)
                # Engine.exeOp synchronizes its PCCL stream before returning, so the
                # following host-control barrier cannot overtake this data phase.
                if not self._engine().register_operation(operation_name, str(phase_file)):
                    raise RuntimeError("failed to register OCS phase '{}'".format(operation_name))
                self._synchronize_phase_registration(group)
                self._engine().execute_operation(
                    operation_name, current_tensor, phase_output)
                current_tensor = phase_output

            barrier = prepared.barriers_after_phase[phase_index]
            if barrier is not None:
                self.runtime.barrier_switch(barrier, group=group, timeout=timeout)
                if operation_name is not None:
                    self._reset_signals(operation_name)

        return current_tensor

    def _create_json_dir(self) -> Tuple[Path, bool]:
        if self.json_dir is not None:
            self.json_dir.mkdir(parents=True, exist_ok=True)
            return self.json_dir, False
        return Path(tempfile.mkdtemp(prefix="pccl-ocs-")), True

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
        """Align ranks after local ``regOp`` and before peer-signal traffic."""
        if dist.is_initialized():
            dist.barrier(group=group)

    @staticmethod
    def _to_runtime_v2(
        manifest: Dict[str, Any],
        phase_manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = {
            "version": 2,
            "executors": phase_manifest["executors"],
            "operations": phase_manifest["operations"],
        }
        for field in ("tensor_info", "collective_type"):
            if field in manifest:
                result[field] = manifest[field]
        return result
