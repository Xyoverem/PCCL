"""Recursive halving-doubling (Rabenseifner) allreduce algorithm.

O(2 log N) steps. Bandwidth-optimal for power-of-2 world sizes.
Phase 1 (reduce-scatter): recursive halving -- each rank exchanges half-data
  with a partner at distance 2^k, reducing the portion it's responsible for.
Phase 2 (allgather): recursive doubling -- reverse of phase 1, each rank
  sends its reduced chunk to partners at increasing distances.

Best for medium messages at higher GPU counts (N=8: 6 steps vs ring's 14).
Requires power-of-2 world_size.
"""

import math

from ..graph import PrimitiveIRGraph
from ..nodes import DeviceType
from ..decorators import CommunicationOp, Stream
from .base import CollectiveAlgorithm


class RecursiveHalvingDoubling(CollectiveAlgorithm):
    name: str = "rhd"

    @property
    def step_count(self) -> str:
        return "O(2 log N)"

    @property
    def bandwidth_optimal(self) -> bool:
        return True

    def build_allreduce(
        self,
        rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> PrimitiveIRGraph:
        if world_size & (world_size - 1) != 0:
            raise ValueError(
                f"RecursiveHalvingDoubling requires power-of-2 world_size, got {world_size}")

        log_n = int(math.log2(world_size))
        use_tma = (executor == "tma")
        name = f"rhd_ar_{executor}_rank{rank}"

        # Signal isolation: use signal_id = source_rank so each partner gets
        # its own slot in the target's signal array (runtime indexes by
        # signal_id only, NOT by source_rank).
        with CommunicationOp(name=name, device=DeviceType.CUDA) as op:
            op.tensor(dtype=dtype, shape=(tensor_size,))

            with Stream("main"):
                op.set_channel(0)

                def _base_offset(k):
                    """Start of rank's active region at step k."""
                    base = 0
                    for j in range(k):
                        if rank & (1 << j):
                            base += tensor_size >> (j + 1)
                    return base

                # Phase 1: Reduce-scatter via recursive halving.
                # At step k, split the active region [base, base+2*seg) into
                # two halves; keep the half indicated by bit k of rank.
                # remote_offset == recv_offset so each partner reads a
                # non-overlapping region — no TMA race.
                for k in range(log_n):
                    partner = rank ^ (1 << k)
                    seg_size = tensor_size >> (k + 1)
                    base = _base_offset(k)

                    if rank & (1 << k):
                        recv_offset = base + seg_size
                    else:
                        recv_offset = base

                    op.notify(signal_id=rank, target_rank=partner)
                    op.wait_notify(signal_id=partner, source_rank=partner)

                    if use_tma:
                        op.tma_reduce(
                            reduce_op="sum",
                            source_rank=partner,
                            src_offset=recv_offset,
                            dst_offset=recv_offset,
                            remote_offset=recv_offset,
                            count=seg_size,
                        )
                    else:
                        op.sm_reduce(
                            reduce_op="sum",
                            source_rank=partner,
                            src_offset=recv_offset,
                            dst_offset=recv_offset,
                            remote_offset=recv_offset,
                            count=seg_size,
                        )

                # Phase 2: Allgather via recursive doubling (reverse order).
                # Copy the partner's reduced chunk that we don't have yet.
                # "ready" notify at the start ensures the partner has finished
                # all higher-k Phase 2 steps before we read its buffer.
                # Only the final step (k=0) sends a "done" ack so that all
                # signal channels stay balanced across repeated executions.
                for k in range(log_n - 1, -1, -1):
                    partner = rank ^ (1 << k)
                    seg_size = tensor_size >> (k + 1)
                    base = _base_offset(k)

                    if rank & (1 << k):
                        copy_offset = base
                    else:
                        copy_offset = base + seg_size

                    op.notify(signal_id=rank, target_rank=partner)
                    op.wait_notify(signal_id=partner, source_rank=partner)

                    if use_tma:
                        op.tma_copy(
                            source_rank=partner,
                            src_offset=copy_offset,
                            dst_offset=copy_offset,
                            size=seg_size,
                        )
                    else:
                        op.sm_copy(
                            source_rank=partner,
                            src_offset=copy_offset,
                            dst_offset=copy_offset,
                            size=seg_size,
                        )
                    if k == 0:
                        op.notify(signal_id=rank, target_rank=partner)

                # Final wait for the last partner's Phase 2 done
                partner = rank ^ 1
                op.wait_notify(signal_id=partner, source_rank=partner)

            graph = op.get_graph()
            graph.collective_type = "allreduce"
            return graph
