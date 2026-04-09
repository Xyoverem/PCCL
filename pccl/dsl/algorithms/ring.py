"""Ring collective algorithm templates.

Allreduce: O(2(N-1)) steps. Bandwidth-optimal for large messages.
  Reduce-scatter phase + allgather phase.
Reduce-scatter: O(N-1) steps. First half of ring allreduce.
Allgather: O(N-1) steps. Second half of ring allreduce.
Alltoall: O(N-1) steps. Pairwise copy exchanges along the ring.
"""

from ..graph import PrimitiveIRGraph
from ..nodes import DeviceType
from ..decorators import CommunicationOp, Stream
from .base import CollectiveAlgorithm


class RingAllreduce(CollectiveAlgorithm):
    name: str = "ring"

    @property
    def step_count(self) -> str:
        return "O(2(N-1))"

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
        chunk_per_ch = tensor_size // num_channels
        chunk_per_rank = chunk_per_ch // world_size
        name = f"ring_ar_{executor}_c{num_channels}_rank{rank}"
        use_tma = (executor == "tma")
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size

        with CommunicationOp(name=name, device=DeviceType.CUDA) as op:
            op.tensor(dtype=dtype, shape=(tensor_size,))

            for c in range(num_channels):
                base = c * chunk_per_ch
                sig_base = c * 1000

                with Stream(f"ch{c}"):
                    op.set_channel(c)

                    op.notify(signal_id=sig_base + 100, target_rank=next_rank)
                    op.wait_notify(signal_id=sig_base + 100, source_rank=prev_rank)

                    for step in range(world_size - 1):
                        chunk_idx = (rank - step - 1) % world_size
                        off = base + chunk_idx * chunk_per_rank
                        if step > 0:
                            op.wait_notify(signal_id=sig_base, source_rank=prev_rank)
                        if use_tma:
                            op.tma_reduce(reduce_op="sum", source_rank=prev_rank,
                                          src_offset=off, dst_offset=off,
                                          remote_offset=off, count=chunk_per_rank)
                        else:
                            op.sm_reduce(reduce_op="sum", source_rank=prev_rank,
                                         src_offset=off, dst_offset=off,
                                         remote_offset=off, count=chunk_per_rank)
                        op.notify(signal_id=sig_base, target_rank=next_rank)

                    for step in range(world_size - 1):
                        chunk_idx = (rank - step) % world_size
                        off = base + chunk_idx * chunk_per_rank
                        op.wait_notify(signal_id=sig_base, source_rank=prev_rank)
                        if use_tma:
                            op.tma_copy(source_rank=prev_rank,
                                        src_offset=off, dst_offset=off,
                                        size=chunk_per_rank)
                        else:
                            op.sm_copy(source_rank=prev_rank,
                                       src_offset=off, dst_offset=off,
                                       size=chunk_per_rank)
                        op.notify(signal_id=sig_base, target_rank=next_rank)

                    op.wait_notify(signal_id=sig_base, source_rank=prev_rank)

            graph = op.get_graph()
            graph.collective_type = "allreduce"
            return graph

    def build_reduce_scatter(
        self,
        rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> PrimitiveIRGraph:
        chunk_per_ch = tensor_size // num_channels
        chunk_per_rank = chunk_per_ch // world_size
        name = f"ring_rs_{executor}_c{num_channels}_rank{rank}"
        use_tma = (executor == "tma")
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size

        with CommunicationOp(name=name, device=DeviceType.CUDA) as op:
            op.tensor(dtype=dtype, shape=(tensor_size,))

            for c in range(num_channels):
                base = c * chunk_per_ch
                sig_base = c * 1000

                with Stream(f"ch{c}"):
                    op.set_channel(c)

                    op.notify(signal_id=sig_base + 100, target_rank=next_rank)
                    op.wait_notify(signal_id=sig_base + 100, source_rank=prev_rank)

                    for step in range(world_size - 1):
                        # Offset by -2 (vs allreduce's -1) so the fully-reduced
                        # chunk lands at position `rank`, not `(rank+1)%N`.
                        chunk_idx = (rank - step - 2) % world_size
                        off = base + chunk_idx * chunk_per_rank
                        if step > 0:
                            op.wait_notify(signal_id=sig_base, source_rank=prev_rank)
                        if use_tma:
                            op.tma_reduce(reduce_op="sum", source_rank=prev_rank,
                                          src_offset=off, dst_offset=off,
                                          remote_offset=off, count=chunk_per_rank)
                        else:
                            op.sm_reduce(reduce_op="sum", source_rank=prev_rank,
                                         src_offset=off, dst_offset=off,
                                         remote_offset=off, count=chunk_per_rank)
                        op.notify(signal_id=sig_base, target_rank=next_rank)

                    op.wait_notify(signal_id=sig_base, source_rank=prev_rank)

            graph = op.get_graph()
            graph.collective_type = "reduce_scatter"
            return graph

    def build_allgather(
        self,
        rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> PrimitiveIRGraph:
        chunk_per_ch = tensor_size // num_channels
        chunk_per_rank = chunk_per_ch // world_size
        name = f"ring_ag_{executor}_c{num_channels}_rank{rank}"
        use_tma = (executor == "tma")
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size

        with CommunicationOp(name=name, device=DeviceType.CUDA) as op:
            op.tensor(dtype=dtype, shape=(tensor_size,))

            for c in range(num_channels):
                base = c * chunk_per_ch
                sig_base = c * 1000

                with Stream(f"ch{c}"):
                    op.set_channel(c)

                    op.notify(signal_id=sig_base + 100, target_rank=next_rank)
                    op.wait_notify(signal_id=sig_base + 100, source_rank=prev_rank)

                    for step in range(world_size - 1):
                        # (rank-step-1) so step 0 reads prev_rank's own
                        # chunk (at position prev_rank), not position rank.
                        chunk_idx = (rank - step - 1) % world_size
                        off = base + chunk_idx * chunk_per_rank
                        if step > 0:
                            op.wait_notify(signal_id=sig_base, source_rank=prev_rank)
                        if use_tma:
                            op.tma_copy(source_rank=prev_rank,
                                        src_offset=off, dst_offset=off,
                                        size=chunk_per_rank)
                        else:
                            op.sm_copy(source_rank=prev_rank,
                                       src_offset=off, dst_offset=off,
                                       size=chunk_per_rank)
                        op.notify(signal_id=sig_base, target_rank=next_rank)

                    op.wait_notify(signal_id=sig_base, source_rank=prev_rank)

            graph = op.get_graph()
            graph.collective_type = "allgather"
            return graph

    def build_alltoall(
        self,
        rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> PrimitiveIRGraph:
        chunk_per_ch = tensor_size // num_channels
        chunk_per_rank = chunk_per_ch // world_size
        name = f"ring_a2a_sm_c{num_channels}_rank{rank}"
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size

        with CommunicationOp(name=name, device=DeviceType.CUDA) as op:
            op.tensor(dtype=dtype, shape=(tensor_size,))

            for c in range(num_channels):
                base = c * chunk_per_ch
                sig_base = c * 1000

                with Stream(f"ch{c}"):
                    op.set_channel(c)

                    # Full ring barrier (N-1 rounds) to ensure every
                    # rank's input data is in its IPC buffer before any
                    # rank starts reading from peers.
                    for i in range(world_size - 1):
                        op.notify(signal_id=sig_base + 100 + i, target_rank=next_rank)
                        op.wait_notify(signal_id=sig_base + 100 + i, source_rank=prev_rank)

                    # Direct P2P copies: read chunk[rank] from each
                    # partner and write to scratch area beyond the tensor
                    # (offset tensor_size + partner*chunk).  This avoids
                    # any read/write race because reads target the
                    # original data region [0, tensor_size) and writes go
                    # to the scratch region [tensor_size, 2*tensor_size).
                    # Always use SM copies (TMA descriptors are bounded
                    # by tensor_size so cannot address the scratch area).
                    for step in range(world_size - 1):
                        partner = (rank + step + 1) % world_size
                        src_off = base + rank * chunk_per_rank
                        dst_off = base + tensor_size + partner * chunk_per_rank
                        op.sm_copy(source_rank=partner,
                                   src_offset=src_off,
                                   dst_offset=dst_off,
                                   size=chunk_per_rank)

            graph = op.get_graph()
            graph.collective_type = "alltoall"
            return graph
