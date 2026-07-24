"""Lower controller Execution Plans into existing PCCL/Torch phase runners."""

from __future__ import annotations

import hashlib
import json
from os import PathLike
from typing import Callable, List, Optional, Tuple, Union, cast

from ..dsl.algorithms import (
    ALGORITHMS,
    AlgorithmIRCollectives,
    RingAllreduce,
    select_algorithm,
)
from ..dsl.algorithms.base import CollectiveAlgorithm
from ..dsl.codegen import RuntimeGraphGenerator
from ..dsl.compiler import Compiler
from ..dsl.graph import PrimitiveIRGraph
from ..dsl.msccl_xml import MSCCLCompatibilityError, MSCCLXMLAlgorithm
from .collective_plan import OCSCollectivePhase, OCSCollectivePlan
from .exceptions import OCSExecutionPlanError
from .execution_plan import OCSExecutionPhase, OCSExecutionPlan
from .torch_plan import TorchCollectivePhase, TorchCollectivePlan


CompiledExecutionPlan = Union[OCSCollectivePlan, TorchCollectivePlan]
MSCCLArtifact = Union[MSCCLXMLAlgorithm, str, bytes, PathLike]
MSCCLArtifactResolver = Callable[[str], MSCCLArtifact]

_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3": 1,
    "float8_e5m2": 1,
}
_TORCH_COLLECTIVES = {
    "allreduce": "all_reduce",
    "alltoall": "all_to_all_single",
}
_ALGORITHM_METHODS = {
    "allreduce": "build_allreduce",
    "alltoall": "build_alltoall",
    "allgather": "build_allgather",
    "reducescatter": "build_reduce_scatter",
}


class ExecutionPlanCompiler:
    """Compile a validated controller plan into a concrete backend plan.

    Tensor metadata stays outside the controller schema because the same plan
    may be materialized for several bucket sizes. The compiler binds rank,
    tensor size, dtype, executor, and channel count to create PCCL artifacts.
    """

    def __init__(
        self,
        algorithm_lowering: str = "template",
        artifact_resolver: Optional[MSCCLArtifactResolver] = None,
    ) -> None:
        if algorithm_lowering not in {"template", "generated", "msccl"}:
            raise ValueError("algorithm_lowering must be 'template', 'generated', or 'msccl'")
        self.algorithm_lowering = algorithm_lowering
        self.artifact_resolver = artifact_resolver

    def compile(
        self,
        plan: OCSExecutionPlan,
        rank: int,
        tensor_size: Optional[int] = None,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> CompiledExecutionPlan:
        if not isinstance(plan, OCSExecutionPlan):
            raise TypeError("plan must be an OCSExecutionPlan")
        plan.validate()
        backends = {phase.backend for phase in plan.phases}
        if len(backends) != 1:
            raise OCSExecutionPlanError(
                "mixed backend plans are unsupported; split the plan at a backend boundary"
            )
        backend = next(iter(backends))
        if backend == "torch":
            return self.compile_torch(plan)
        if tensor_size is None:
            raise OCSExecutionPlanError("tensor_size is required for the pccl backend")
        return self.compile_pccl(
            plan,
            rank=rank,
            tensor_size=tensor_size,
            dtype=dtype,
            executor=executor,
            num_channels=num_channels,
        )

    def compile_torch(self, plan: OCSExecutionPlan) -> TorchCollectivePlan:
        if not isinstance(plan, OCSExecutionPlan):
            raise TypeError("plan must be an OCSExecutionPlan")
        plan.validate()
        phases = []
        for phase in plan.phases:
            if phase.backend != "torch":
                raise OCSExecutionPlanError(
                    "phase {} uses backend {!r}; expected 'torch'".format(
                        phase.phase_id, phase.backend
                    )
                )
            collective = _TORCH_COLLECTIVES.get(phase.op_type)
            if collective is None:
                raise OCSExecutionPlanError(
                    "torch plan compiler does not support op_type {!r}".format(phase.op_type)
                )
            phases.append(
                TorchCollectivePhase(
                    collective=collective,
                    barrier_after=plan.barrier_plan(
                        phase.phase_id, target_algorithm="torch_native"
                    ),
                )
            )
        return TorchCollectivePlan(phases=tuple(phases))

    def compile_pccl(
        self,
        plan: OCSExecutionPlan,
        rank: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> OCSCollectivePlan:
        if not isinstance(plan, OCSExecutionPlan):
            raise TypeError("plan must be an OCSExecutionPlan")
        plan.validate()
        if rank not in plan.rank_list:
            raise OCSExecutionPlanError(
                "rank {} is not in rank_list {}".format(rank, plan.rank_list)
            )
        if isinstance(tensor_size, bool) or not isinstance(tensor_size, int) or tensor_size <= 0:
            raise OCSExecutionPlanError("tensor_size must be a positive integer")
        if dtype not in _DTYPE_BYTES:
            raise OCSExecutionPlanError("unsupported dtype {!r}".format(dtype))
        if isinstance(num_channels, bool) or not isinstance(num_channels, int) or num_channels <= 0:
            raise OCSExecutionPlanError("num_channels must be a positive integer")
        if self.algorithm_lowering != "msccl" and tensor_size % (
            len(plan.rank_list) * num_channels
        ):
            raise OCSExecutionPlanError(
                "tensor_size must be divisible by world_size * num_channels"
            )

        local_rank = plan.rank_list.index(rank)
        world_size = len(plan.rank_list)
        materialized: List[Tuple[OCSExecutionPhase, PrimitiveIRGraph, str]] = []
        for phase in plan.phases:
            if phase.backend != "pccl":
                raise OCSExecutionPlanError(
                    "phase {} uses backend {!r}; expected 'pccl'".format(
                        phase.phase_id, phase.backend
                    )
                )
            graph, resolved_algorithm = self._build_graph(
                phase,
                local_rank=local_rank,
                world_size=world_size,
                tensor_size=tensor_size,
                dtype=dtype,
                executor=executor,
                num_channels=num_channels,
            )
            self._remap_group_ranks(graph, plan.rank_list)
            self._verify_graph_digest(phase, graph)
            materialized.append((phase, graph, resolved_algorithm))

        phases = []
        for phase, graph, _resolved_algorithm in materialized:
            barrier = phase.barrier_after
            target_algorithm = None
            if barrier is not None:
                target_algorithm = materialized[barrier.next_phase_id][2]
            phases.append(
                OCSCollectivePhase(
                    name="phase_{}_{}".format(phase.phase_id, phase.op_type),
                    graph=graph,
                    barrier_after=plan.barrier_plan(
                        phase.phase_id, target_algorithm=target_algorithm
                    ),
                )
            )
        return OCSCollectivePlan(phases=tuple(phases))

    def _build_graph(
        self,
        phase: OCSExecutionPhase,
        local_rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str,
        executor: str,
        num_channels: int,
    ) -> Tuple[PrimitiveIRGraph, str]:
        if self.algorithm_lowering == "msccl":
            return self._build_msccl_graph(
                phase,
                local_rank=local_rank,
                world_size=world_size,
                tensor_size=tensor_size,
                dtype=dtype,
                executor=executor,
            )
        if self.algorithm_lowering == "generated":
            return self._build_generated_graph(
                phase,
                local_rank=local_rank,
                world_size=world_size,
                tensor_size=tensor_size,
                dtype=dtype,
                executor=executor,
                num_channels=num_channels,
            )

        method_name = _ALGORITHM_METHODS.get(phase.op_type)
        if method_name is None:
            raise OCSExecutionPlanError(
                "pccl plan compiler does not support op_type {!r}".format(phase.op_type)
            )

        if phase.algorithm_type == "auto":
            if phase.op_type == "allreduce":
                algorithm = select_algorithm(world_size, tensor_size * _DTYPE_BYTES[dtype])
            else:
                algorithm = RingAllreduce()
        elif phase.algorithm_type == "direct":
            if phase.op_type != "alltoall":
                raise OCSExecutionPlanError(
                    "algorithm 'direct' is currently valid only for alltoall"
                )
            # RingAllreduce.build_alltoall implements pairwise direct exchanges.
            algorithm = RingAllreduce()
        else:
            algorithm_factory = ALGORITHMS.get(phase.algorithm_type)
            if algorithm_factory is None:
                raise OCSExecutionPlanError(
                    "algorithm_type {!r} has no PCCL lowering".format(phase.algorithm_type)
                )
            factory = cast(Callable[[], CollectiveAlgorithm], algorithm_factory)
            algorithm = factory()

        method = getattr(algorithm, method_name)
        try:
            graph = method(
                rank=local_rank,
                world_size=world_size,
                tensor_size=tensor_size,
                dtype=dtype,
                executor=executor,
                num_channels=num_channels,
            )
            resolved_algorithm = "direct" if phase.algorithm_type == "direct" else algorithm.name
            return graph, resolved_algorithm
        except (NotImplementedError, ValueError) as exc:
            raise OCSExecutionPlanError(
                "cannot lower phase {} ({}/{}): {}".format(
                    phase.phase_id, phase.op_type, phase.algorithm_type, exc
                )
            ) from exc

    @staticmethod
    def _build_generated_graph(
        phase: OCSExecutionPhase,
        local_rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str,
        executor: str,
        num_channels: int,
    ) -> Tuple[PrimitiveIRGraph, str]:
        generated = AlgorithmIRCollectives()
        try:
            if phase.op_type == "allreduce" and phase.algorithm_type == "ring":
                graph = generated.build_allreduce(
                    rank=local_rank,
                    world_size=world_size,
                    tensor_size=tensor_size,
                    dtype=dtype,
                    executor=executor,
                    num_channels=num_channels,
                )
                return graph, "ring"
            if phase.op_type == "alltoall" and phase.algorithm_type == "direct":
                graph = generated.build_alltoall(
                    rank=local_rank,
                    world_size=world_size,
                    tensor_size=tensor_size,
                    dtype=dtype,
                    executor=executor,
                    num_channels=num_channels,
                )
                return graph, "direct"
        except ValueError as exc:
            raise OCSExecutionPlanError(
                "cannot lower generated phase {} ({}/{}): {}".format(
                    phase.phase_id, phase.op_type, phase.algorithm_type, exc
                )
            ) from exc

        raise OCSExecutionPlanError(
            "generated Algorithm IR lowering does not support phase {} ({}/{})".format(
                phase.phase_id, phase.op_type, phase.algorithm_type
            )
        )

    def _build_msccl_graph(
        self,
        phase: OCSExecutionPhase,
        local_rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str,
        executor: str,
    ) -> Tuple[PrimitiveIRGraph, str]:
        if phase.artifact_id is None:
            raise OCSExecutionPlanError(
                "phase {} requires artifact_id for MSCCL lowering".format(phase.phase_id)
            )
        if self.artifact_resolver is None:
            raise OCSExecutionPlanError("MSCCL lowering requires an artifact_resolver")
        try:
            artifact = self.artifact_resolver(phase.artifact_id)
            algorithm = MSCCLXMLAlgorithm.resolve(artifact)
            if algorithm.num_gpus != world_size:
                raise MSCCLCompatibilityError(
                    "MSCCL ngpus={} does not match plan group size {}".format(
                        algorithm.num_gpus, world_size
                    )
                )
            if not algorithm.matches_collective(phase.op_type):
                raise MSCCLCompatibilityError(
                    "MSCCL collective {!r} does not match phase op_type {!r}".format(
                        algorithm.collective, phase.op_type
                    )
                )
            graph = algorithm.lower(
                rank=local_rank,
                tensor_size=tensor_size,
                dtype=dtype,
                executor=executor,
            )
        except (KeyError, OSError, TypeError, MSCCLCompatibilityError) as exc:
            raise OCSExecutionPlanError(
                "cannot lower MSCCL phase {} from artifact {!r}: {}".format(
                    phase.phase_id, phase.artifact_id, exc
                )
            ) from exc
        return graph, phase.algorithm_type

    @staticmethod
    def _remap_group_ranks(graph: PrimitiveIRGraph, rank_list: tuple) -> None:
        """Map algorithm-local ranks onto the plan's possibly sparse group."""
        for node in graph.nodes.values():
            for field in ("source_rank", "target_rank"):
                value = getattr(node, field, None)
                if isinstance(value, int) and value >= 0:
                    if value >= len(rank_list):
                        raise OCSExecutionPlanError(
                            "generated {}={} is outside rank_list".format(field, value)
                        )
                    setattr(node, field, rank_list[value])

    @staticmethod
    def _verify_graph_digest(
        phase: OCSExecutionPhase,
        graph: PrimitiveIRGraph,
    ) -> None:
        if phase.graph_digest is None:
            return
        manifest = RuntimeGraphGenerator().generate(Compiler().compile(graph))
        encoded = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        actual = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if actual != phase.graph_digest:
            raise OCSExecutionPlanError(
                "phase {} graph_digest mismatch: expected {}, generated {}".format(
                    phase.phase_id, phase.graph_digest, actual
                )
            )
