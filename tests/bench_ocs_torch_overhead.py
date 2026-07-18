"""Measure the control-plane overhead of an OCS torch collective plan.

Run each variant in a fresh process group, for example:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      tests/bench_ocs_torch_overhead.py --variant native
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      tests/bench_ocs_torch_overhead.py --variant ocs

This is not a PCCL engine bandwidth benchmark. It compares the same native
torch collective sequence with and without OCS control boundaries.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, Dict, List, Sequence

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import (
    OCSRuntime,
    OcsTorchPlanRunner,
    TorchDistributedSwitchConnector,
    build_torch_allreduce_alltoall_plan,
)


def _percentile(samples: Sequence[float], percentile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = int(round((len(ordered) - 1) * percentile / 100.0))
    return float(ordered[index])


def _summary(samples: Sequence[float]) -> Dict[str, float]:
    return {
        "median_us": float(statistics.median(samples)) if samples else 0.0,
        "p95_us": _percentile(samples, 95.0),
    }


def _run_native_sequence(input_tensor: torch.Tensor) -> torch.Tensor:
    reduced = torch.empty_like(input_tensor)
    reduced.copy_(input_tensor)
    dist.all_reduce(reduced)

    exchanged = torch.empty_like(reduced)
    dist.all_to_all_single(exchanged, reduced)

    result = torch.empty_like(exchanged)
    result.copy_(exchanged)
    dist.all_reduce(result)
    return result


def _gather_rank_samples(local_samples: List[float]) -> List[List[float]]:
    world_size = dist.get_world_size()
    gathered: List[List[float]] = [[] for _ in range(world_size)]
    dist.all_gather_object(gathered, local_samples)
    return gathered


def _max_per_iteration(samples_by_rank: Sequence[Sequence[float]]) -> List[float]:
    if not samples_by_rank:
        return []
    sample_count = len(samples_by_rank[0])
    if any(len(samples) != sample_count for samples in samples_by_rank):
        raise RuntimeError("rank sample counts do not match")
    return [max(samples[index] for samples in samples_by_rank) for index in range(sample_count)]


def _measure(
    run_once: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    expected_value: float,
) -> List[float]:
    output = None
    for _ in range(warmup):
        dist.barrier()
        output = run_once()
        torch.cuda.synchronize()

    durations_us = []
    for _ in range(iterations):
        dist.barrier()
        start_ns = time.perf_counter_ns()
        output = run_once()
        torch.cuda.synchronize()
        durations_us.append((time.perf_counter_ns() - start_ns) / 1_000.0)

    if output is None or not torch.equal(output, torch.full_like(output, expected_value)):
        raise RuntimeError("collective sequence produced an unexpected result")
    return durations_us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("native", "ocs"), required=True)
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--switch-delay-us", type=float, default=0.0)
    parser.add_argument("--link-ready-delay-us", type=float, default=0.0)
    args = parser.parse_args()
    if args.elements <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("elements must be positive, warmup non-negative, and iterations positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.elements % world_size:
        parser.error("--elements must be divisible by WORLD_SIZE")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        expected_value = float(sum(range(1, world_size + 1)) * world_size)
        input_tensor = torch.full(
            (args.elements,), float(rank + 1), dtype=torch.float32, device="cuda")
        index = 0
        runtime = None

        if args.variant == "native":

            def run_once() -> torch.Tensor:
                return _run_native_sequence(input_tensor)

        else:
            connector = TorchDistributedSwitchConnector(
                switch_delay_s=args.switch_delay_us / 1_000_000.0,
                link_ready_delay_s=args.link_ready_delay_us / 1_000_000.0,
            )
            runtime = OCSRuntime(connector=connector)
            runner = OcsTorchPlanRunner(runtime=runtime)
            plans = [
                build_torch_allreduce_alltoall_plan(
                    world_size=world_size,
                    job_id="ocs_torch_overhead_benchmark",
                    first_barrier_id=10_000 + iteration * 2,
                    first_epoch_id=iteration * 2,
                )
                for iteration in range(args.warmup + args.iterations)
            ]

            def run_once() -> torch.Tensor:
                nonlocal index
                plan = plans[index]
                index += 1
                return runner.execute(plan, input_tensor)

        local_durations = _measure(
            run_once,
            warmup=args.warmup,
            iterations=args.iterations,
            expected_value=expected_value,
        )
        durations_by_rank = _gather_rank_samples(local_durations)

        barrier_by_rank: List[List[float]] = []
        if runtime is not None:
            measured_releases = runtime.history[args.warmup * 2:]
            local_barriers = [float(release["latency_us"]) for release in measured_releases]
            barrier_by_rank = _gather_rank_samples(local_barriers)

        if rank == 0:
            completion_us = _max_per_iteration(durations_by_rank)
            result = {
                "completion": _summary(completion_us),
                "elements": args.elements,
                "iterations": args.iterations,
                "variant": args.variant,
                "world_size": world_size,
            }
            if barrier_by_rank:
                barrier_completion_us = _max_per_iteration(barrier_by_rank)
                result["barriers_per_iteration"] = 2
                result["barrier_completion"] = _summary(barrier_completion_us)
                result["barrier_total_per_iteration"] = _summary([
                    barrier_completion_us[index] + barrier_completion_us[index + 1]
                    for index in range(0, len(barrier_completion_us), 2)
                ])
                result["switch_delay_us"] = args.switch_delay_us
                result["link_ready_delay_us"] = args.link_ready_delay_us
            print("OCS_TORCH_OVERHEAD_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
