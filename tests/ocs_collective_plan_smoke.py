"""NCCL smoke test for an OCS plan spanning allreduce and alltoall phases.

Run with two compatible GPUs, for example:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      tests/ocs_collective_plan_smoke.py --iterations 3
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import (
    OCSRuntime,
    OcsCollectivePlanRunner,
    build_ring_allreduce_alltoall_plan,
)
from pccl.engine import get_engine, initialize_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.elements <= 0:
        parser.error("--elements must be positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.elements % world_size:
        parser.error("--elements must be divisible by WORLD_SIZE")

    def report(stage: str) -> None:
        print("OCS_COLLECTIVE_PLAN_STAGE rank={} {}".format(rank, stage), flush=True)

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    report("process_group_ready")

    try:
        preflight = torch.tensor([rank + 1], dtype=torch.int32, device="cuda")
        dist.all_reduce(preflight)
        if preflight.item() != sum(range(1, world_size + 1)):
            raise RuntimeError("NCCL all_reduce preflight returned an unexpected value")

        get_engine()
        initialize_engine(dist.group.WORLD)
        report("endpoints_ready")

        runtime = OCSRuntime()
        runner = OcsCollectivePlanRunner(runtime=runtime)
        reduced_value = float(sum(range(1, world_size + 1)))
        expected_value = reduced_value * world_size

        for iteration in range(args.iterations):
            first_barrier_id = 401 + iteration * 2
            prepared = runner.prepare(
                build_ring_allreduce_alltoall_plan(
                    rank=rank,
                    world_size=world_size,
                    tensor_size=args.elements,
                    executor="sm",
                    job_id="ocs_collective_plan_smoke",
                    first_barrier_id=first_barrier_id,
                    first_epoch_id=iteration * 2,
                ),
                operation_name="ocs_collective_plan_rank{}_iter{}".format(rank, iteration),
            )
            input_tensor = torch.full(
                (args.elements,), float(rank + 1), dtype=torch.float32, device="cuda")
            output_tensor = torch.empty_like(input_tensor)
            result = runner.execute(prepared, input_tensor, output_tensor=output_tensor)
            torch.cuda.synchronize()

            expected = torch.full_like(result, expected_value)
            if not torch.equal(result, expected):
                raise RuntimeError(
                    "mixed collective plan output does not match expected allreduce result")

            releases = runtime.history[-2:]
            if [release["barrier_id"] for release in releases] != [
                first_barrier_id,
                first_barrier_id + 1,
            ]:
                raise RuntimeError("collective plan barrier ids are out of sequence")
            if any(release["link_state"] != "LINK_ALIGNED" for release in releases):
                raise RuntimeError("collective plan did not receive LINK_ALIGNED")
            prepared.close()

        dist.barrier()
        print(
            "OCS_COLLECTIVE_PLAN_SMOKE_PASS "
            + json.dumps(
                {
                    "barrier_ids": [release["barrier_id"] for release in runtime.history],
                    "iterations": args.iterations,
                    "rank": rank,
                    "result_value": expected_value,
                    "world_size": world_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if dist.is_initialized():
            report("process_group_destroying")
            dist.destroy_process_group()
            report("process_group_destroyed")


if __name__ == "__main__":
    main()
