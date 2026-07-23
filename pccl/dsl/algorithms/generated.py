"""Collective schedules generated through the hardware-independent Algorithm IR."""

from typing import Optional

from ..algorithm_ir import (
    AlgorithmBuffer,
    AlgorithmIRBuilder,
    AlgorithmIRLowerer,
    AlgorithmIRError,
    CollectiveAlgorithmIR,
)
from ..graph import PrimitiveIRGraph
from .base import CollectiveAlgorithm


def build_ring_allreduce_ir(world_size: int) -> CollectiveAlgorithmIR:
    """Describe ring reduce-scatter + allgather without PCCL executor details."""
    builder = AlgorithmIRBuilder(
        name="generated_ring_allreduce",
        collective_type="allreduce",
        world_size=world_size,
        chunks_per_rank=world_size,
    )

    for round_index in range(world_size - 1):
        step = builder.step("reduce_scatter_{}".format(round_index))
        for dst_rank in range(world_size):
            src_rank = (dst_rank - 1) % world_size
            chunk_index = (dst_rank - round_index - 1) % world_size
            step.reduce(
                builder.chunk(src_rank, chunk_index, AlgorithmBuffer.INPUT),
                builder.chunk(dst_rank, chunk_index, AlgorithmBuffer.INPUT),
            )

    for round_index in range(world_size - 1):
        step = builder.step("allgather_{}".format(round_index))
        for dst_rank in range(world_size):
            src_rank = (dst_rank - 1) % world_size
            chunk_index = (dst_rank - round_index) % world_size
            step.copy(
                builder.chunk(src_rank, chunk_index, AlgorithmBuffer.INPUT),
                builder.chunk(dst_rank, chunk_index, AlgorithmBuffer.INPUT),
            )

    completion = builder.step("completion_sync")
    for src_rank in range(world_size):
        completion.sync(src_rank, (src_rank + 1) % world_size)
    return builder.build()


def build_direct_alltoall_ir(world_size: int) -> CollectiveAlgorithmIR:
    """Describe direct pairwise all-to-all after a ring input rendezvous."""
    builder = AlgorithmIRBuilder(
        name="generated_direct_alltoall",
        collective_type="alltoall",
        world_size=world_size,
        chunks_per_rank=world_size,
    )

    # Match the existing PCCL template: N-1 signal rounds make every input
    # buffer visible before destinations start remote reads.
    for round_index in range(world_size - 1):
        sync = builder.step("input_sync_{}".format(round_index))
        for src_rank in range(world_size):
            sync.sync(src_rank, (src_rank + 1) % world_size)

    for round_index in range(world_size - 1):
        step = builder.step("exchange_{}".format(round_index))
        for dst_rank in range(world_size):
            src_rank = (dst_rank + round_index + 1) % world_size
            step.copy(
                builder.chunk(src_rank, dst_rank, AlgorithmBuffer.INPUT),
                builder.chunk(dst_rank, src_rank, AlgorithmBuffer.SCRATCH),
                requires_signal=False,
            )
    return builder.build()


class AlgorithmIRCollectives(CollectiveAlgorithm):
    """PCCL-compatible facade for collectives generated from Algorithm IR."""

    name = "algorithm_ir"

    def __init__(self, lowerer: Optional[AlgorithmIRLowerer] = None) -> None:
        self.lowerer = lowerer if lowerer is not None else AlgorithmIRLowerer()

    def build_allreduce(
        self,
        rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> PrimitiveIRGraph:
        self._require_single_channel(num_channels)
        return self.lowerer.lower(
            build_ring_allreduce_ir(world_size),
            rank=rank,
            tensor_size=tensor_size,
            dtype=dtype,
            executor=executor,
        )

    def build_alltoall(
        self,
        rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> PrimitiveIRGraph:
        self._require_single_channel(num_channels)
        return self.lowerer.lower(
            build_direct_alltoall_ir(world_size),
            rank=rank,
            tensor_size=tensor_size,
            dtype=dtype,
            executor=executor,
        )

    @staticmethod
    def _require_single_channel(num_channels: int) -> None:
        if num_channels != 1:
            raise AlgorithmIRError(
                "Algorithm IR v1 generated collectives currently support one channel"
            )
