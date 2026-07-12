"""NCCL smoke test for the engine-backed OCS phase runner.

Run with two compatible GPUs, for example:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      tests/ocs_phase_runner_smoke.py
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from pccl import DeviceType, OCSRuntime, OcsPhaseRunner, Stream, build_graph
from pccl.engine import initialize_engine


def build_two_phase_graph(rank: int, world_size: int, elements: int):
    def build(op):
        op.tensor(dtype="float32", shape=(elements,))
        with Stream("phase0"):
            op.sm_copy(source_rank=rank, src_offset=0, dst_offset=0, size=elements)
        op.ocs_barrier(
            barrier_id=101,
            epoch_id=0,
            next_epoch_id=1,
            participant_ranks=tuple(range(world_size)),
            topology_id=1,
            route_plan_id=101,
            algorithm="ring",
            backend="pccl",
        )
        with Stream("phase1"):
            op.sm_copy(source_rank=rank, src_offset=0, dst_offset=0, size=elements)

    return build_graph("ocs_phase_runner_smoke", build, device=DeviceType.CUDA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=4096)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    try:
        initialize_engine(dist.group.WORLD)
        runtime = OCSRuntime()
        runner = OcsPhaseRunner(runtime=runtime)
        prepared = runner.prepare(
            build_two_phase_graph(rank, world_size, args.elements),
            operation_name="ocs_phase_runner_smoke_rank{}".format(rank),
        )
        input_tensor = torch.full(
            (args.elements,), float(rank + 1), dtype=torch.float32, device="cuda")
        output_tensor = torch.empty_like(input_tensor)

        started_ns = time.perf_counter_ns()
        result = runner.execute(prepared, input_tensor, output_tensor=output_tensor)
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter_ns() - started_ns) // 1000

        if not torch.equal(result, input_tensor):
            raise RuntimeError("phase runner output does not match this rank's input")
        if len(runtime.history) != 1:
            raise RuntimeError("expected exactly one OCS barrier release")

        release = runtime.history[0]
        if release["status"] != "OK" or release["barrier_id"] != 101:
            raise RuntimeError("unexpected OCS release: {}".format(release))
        arrived_ranks = sorted(record["src_rank"] for record in release["ready_records"])
        if arrived_ranks != list(range(world_size)):
            raise RuntimeError("OCS release is missing READY records: {}".format(arrived_ranks))

        dist.barrier()
        print(
            "OCS_PHASE_RUNNER_SMOKE_PASS "
            + json.dumps(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "phase_runner_elapsed_us": elapsed_us,
                    "barrier_latency_us": release["latency_us"],
                    "ready_ranks": arrived_ranks,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        prepared.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
