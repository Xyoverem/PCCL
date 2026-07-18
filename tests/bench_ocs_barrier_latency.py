"""Measure OCS barrier-control latency and its phase breakdown.

The ``ocs`` variant measures the v0 host control path.  With ``--delay-mode
spin``, configured switch and link delays are held with a monotonic busy wait
so 1--100 us mock targets are not rounded by ``time.sleep``.  This validates
the latency accounting only; it does not emulate a hardware OCS or controller.

Examples:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      tests/bench_ocs_barrier_latency.py --variant torch_barrier
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      tests/bench_ocs_barrier_latency.py --variant ocs --delay-mode spin \
      --switch-delay-us 10 --link-ready-delay-us 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, List, Sequence

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import OCSPlan, OCSRuntime, TorchDistributedSwitchConnector


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


def _gather(local_samples: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
    world_size = dist.get_world_size()
    result: List[List[Dict[str, float]]] = [[] for _ in range(world_size)]
    dist.all_gather_object(result, local_samples)
    return result


def _rank_max(samples_by_rank: Sequence[Sequence[Dict[str, float]]], key: str) -> List[float]:
    if not samples_by_rank:
        return []
    count = len(samples_by_rank[0])
    if any(len(samples) != count for samples in samples_by_rank):
        raise RuntimeError("rank sample counts do not match")
    return [
        max(float(samples[index][key]) for samples in samples_by_rank)
        for index in range(count)
    ]


def _make_plan(iteration: int, world_size: int) -> OCSPlan:
    return OCSPlan(
        job_id="ocs_barrier_latency",
        group_id=0,
        barrier_id=20_000 + iteration,
        epoch_id=iteration,
        next_epoch_id=iteration + 1,
        participant_ranks=tuple(range(world_size)),
        topology_id=iteration + 1,
        route_plan_id=20_000 + iteration,
        algorithm="torch_native",
        backend="torch",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("torch_barrier", "ocs"), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--switch-delay-us", type=float, default=0.0)
    parser.add_argument("--link-ready-delay-us", type=float, default=0.0)
    parser.add_argument("--delay-mode", choices=("sleep", "spin"), default="spin")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")
    if args.switch_delay_us < 0 or args.link_ready_delay_us < 0:
        parser.error("configured delays must be non-negative")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    try:
        runtime = None
        if args.variant == "ocs":
            runtime = OCSRuntime(connector=TorchDistributedSwitchConnector(
                switch_delay_s=args.switch_delay_us / 1_000_000.0,
                link_ready_delay_s=args.link_ready_delay_us / 1_000_000.0,
                delay_mode=args.delay_mode,
            ))

        samples: List[Dict[str, float]] = []
        for iteration in range(args.warmup + args.iterations):
            # Align the measurement start; this fence is outside the sample.
            dist.barrier()
            start_ns = time.perf_counter_ns()
            if runtime is None:
                dist.barrier()
                timing = {"total_us": (time.perf_counter_ns() - start_ns) / 1_000.0}
            else:
                release = runtime.barrier_switch(_make_plan(iteration, world_size))
                timing = dict(release["timing"])
                timing["call_us"] = (time.perf_counter_ns() - start_ns) / 1_000.0

            if iteration >= args.warmup:
                samples.append({key: float(value) for key, value in timing.items()})

        gathered = _gather(samples)
        if rank == 0:
            result = {
                "variant": args.variant,
                "world_size": world_size,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "completion": _summary(_rank_max(gathered, "total_us")),
            }
            if runtime is not None:
                local_keys = (
                    "ready_exchange_us",
                    "ready_validation_us",
                    "controller_commit_us",
                    "release_local_us",
                    "call_us",
                )
                controller_keys = ("controller_switch_us", "controller_link_align_us")
                result["configured_switch_delay_us"] = args.switch_delay_us
                result["configured_link_ready_delay_us"] = args.link_ready_delay_us
                result["delay_mode"] = args.delay_mode
                result["local_stages"] = {
                    key: _summary(_rank_max(gathered, key)) for key in local_keys
                }
                # These are measured by the leader and broadcast with the release.
                result["controller_internal"] = {
                    key: _summary([float(sample[key]) for sample in gathered[0]])
                    for key in controller_keys
                }
            print("OCS_BARRIER_LATENCY_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
