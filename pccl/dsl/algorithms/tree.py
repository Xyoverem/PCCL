"""Binary tree allreduce algorithm template.

O(2 log N) steps. NOT bandwidth-optimal (root sees full data).
Phase 1: Binary tree reduce to root -- leaves send to parents, parents reduce.
Phase 2: Binary tree broadcast from root -- root sends to children.

Best for small messages where latency (step count) dominates and per-step
bandwidth waste is acceptable. For N=8 this uses 6 steps vs ring's 14.
"""

import math

from ..graph import PrimitiveIRGraph
from ..nodes import DeviceType
from ..decorators import CommunicationOp, Stream
from .base import CollectiveAlgorithm


class TreeAllreduce(CollectiveAlgorithm):
    name: str = "tree"

    @property
    def step_count(self) -> str:
        return "O(2 log N)"

    @property
    def bandwidth_optimal(self) -> bool:
        return False

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
                f"TreeAllreduce requires power-of-2 world_size, got {world_size}")

        log_n = int(math.log2(world_size))
        use_tma = (executor == "tma")
        name = f"tree_ar_{executor}_rank{rank}"

        # Signal isolation: use signal_id = source_rank so each partner gets
        # its own slot in the target's signal array (runtime indexes by
        # signal_id only, NOT by source_rank).
        with CommunicationOp(name=name, device=DeviceType.CUDA) as op:
            op.tensor(dtype=dtype, shape=(tensor_size,))

            with Stream("main"):
                op.set_channel(0)

                # Phase 1: Reduce to root (rank 0)
                for k in range(log_n):
                    stride = 1 << k
                    partner = rank ^ stride

                    if rank & stride:
                        op.notify(signal_id=rank, target_rank=partner)
                        op.wait_notify(signal_id=partner, source_rank=partner)
                    elif partner < world_size:
                        op.wait_notify(signal_id=partner, source_rank=partner)

                        if use_tma:
                            op.tma_reduce(
                                reduce_op="sum",
                                source_rank=partner,
                                src_offset=0,
                                dst_offset=0,
                                remote_offset=0,
                                count=tensor_size,
                            )
                        else:
                            op.sm_reduce(
                                reduce_op="sum",
                                source_rank=partner,
                                src_offset=0,
                                dst_offset=0,
                                remote_offset=0,
                                count=tensor_size,
                            )
                        op.notify(signal_id=rank, target_rank=partner)

                    if rank & stride:
                        break

                # Phase 2: Broadcast from root (rank 0)
                # Only ranks whose lower k bits are all zero participate at
                # level k (same set that was active after Phase 1 level k).
                for k in range(log_n - 1, -1, -1):
                    stride = 1 << k
                    if rank % stride != 0:
                        continue
                    partner = rank ^ stride

                    if rank & stride:
                        op.wait_notify(signal_id=partner, source_rank=partner)

                        if use_tma:
                            op.tma_copy(
                                source_rank=partner,
                                src_offset=0,
                                dst_offset=0,
                                size=tensor_size,
                            )
                        else:
                            op.sm_copy(
                                source_rank=partner,
                                src_offset=0,
                                dst_offset=0,
                                size=tensor_size,
                            )
                        op.notify(signal_id=rank, target_rank=partner)
                    elif partner < world_size:
                        op.notify(signal_id=rank, target_rank=partner)
                        op.wait_notify(signal_id=partner, source_rank=partner)

            graph = op.get_graph()
            graph.collective_type = "allreduce"
            return graph
