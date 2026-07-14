"""NCCL smoke test for a torch allreduce/alltoall OCS execution plan.

Run with two compatible GPUs, for example:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      tests/ocs_torch_collective_plan_smoke.py --iterations 3
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
    OcsTorchPlanRunner,
    build_torch_allreduce_alltoall_plan,
)


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
        print("OCS_TORCH_PLAN_STAGE rank={} {}".format(rank, stage), flush=True)

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    report("process_group_ready")

    try:
        runtime = OCSRuntime()
        runner = OcsTorchPlanRunner(runtime=runtime)
        expected_value = float(sum(range(1, world_size + 1)) * world_size)

        for iteration in range(args.iterations):
            first_barrier_id = 601 + iteration * 2
            report("iteration_{}_executing".format(iteration))
            result = runner.execute(
                build_torch_allreduce_alltoall_plan(
                    world_size=world_size,
                    job_id="ocs_torch_collective_plan_smoke",
                    first_barrier_id=first_barrier_id,
                    first_epoch_id=iteration * 2,
                ),
                torch.full(
                    (args.elements,), float(rank + 1), dtype=torch.float32, device="cuda"),
                group=dist.group.WORLD,
            )
            torch.cuda.synchronize()

            if not torch.equal(result, torch.full_like(result, expected_value)):
                raise RuntimeError("torch collective plan output does not match expected value")
            releases = runtime.history[-2:]
            if [release["barrier_id"] for release in releases] != [
                first_barrier_id,
                first_barrier_id + 1,
            ]:
                raise RuntimeError("torch collective plan barrier ids are out of sequence")
            if any(release["link_state"] != "LINK_ALIGNED" for release in releases):
                raise RuntimeError("torch collective plan did not receive LINK_ALIGNED")
            report("iteration_{}_complete".format(iteration))

        dist.barrier()
        print(
            "OCS_TORCH_PLAN_SMOKE_PASS "
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
