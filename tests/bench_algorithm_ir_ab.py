"""A/B benchmark for hand-written PCCL templates and generated Algorithm IR.

The timed region contains only PCCL data-plane execution:
``execute_operation_async`` followed by ``sync_operation``.  Graph building,
JSON compilation, operation registration, OCS barriers, and controller work are
deliberately outside the timed region.

Example:
  torchrun --standalone --nproc_per_node=2 tests/bench_algorithm_ir_ab.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Dict, Iterable, List

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import compile_to_json_file
from pccl.dsl.algorithms import AlgorithmIRCollectives, RingAllreduce
from pccl.engine import (
    execute_operation_async,
    get_engine,
    initialize_engine,
    register_operation,
    reset_signals,
    sync_operation,
)


MODES = ("template", "generated")
COLLECTIVES = ("allreduce", "alltoall")


def _parse_csv_ints(value: str) -> List[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _parse_csv_choices(value: str, choices: Iterable[str]) -> List[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(values) - set(choices))
    if not values or invalid:
        raise argparse.ArgumentTypeError(
            "expected comma-separated values from {}; invalid={}".format(sorted(choices), invalid)
        )
    return values


def _build_graph(
    mode: str,
    collective: str,
    rank: int,
    world_size: int,
    tensor_size: int,
    executor: str,
):
    algorithm = RingAllreduce() if mode == "template" else AlgorithmIRCollectives()
    builder = getattr(algorithm, "build_{}".format(collective))
    return builder(
        rank=rank,
        world_size=world_size,
        tensor_size=tensor_size,
        dtype="float32",
        executor=executor,
    )


def _run_once(operation_name: str, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
    execute_operation_async(operation_name, input_tensor, output_tensor)
    sync_operation(operation_name)


def _check_output(
    collective: str,
    output_tensor: torch.Tensor,
    world_size: int,
) -> None:
    expected = float(sum(range(1, world_size + 1))) if collective == "allreduce" else 1.0
    if not torch.equal(output_tensor, torch.full_like(output_tensor, expected)):
        mismatches = int(torch.count_nonzero(output_tensor != expected).item())
        raise RuntimeError(
            "{} correctness check failed on rank {}: {} mismatched elements".format(
                collective, dist.get_rank(), mismatches
            )
        )


def _measure(
    operation_name: str,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    iterations: int,
) -> float:
    dist.barrier()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _run_once(operation_name, input_tensor, output_tensor)
    end.record()
    torch.cuda.synchronize()
    local_us = start.elapsed_time(end) * 1000.0 / iterations

    # Collective completion is determined by the slowest participating rank.
    latency = torch.tensor(local_us, dtype=torch.float64, device=input_tensor.device)
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return float(latency.item())


def _bench_case(
    collective: str,
    data_bytes: int,
    rank: int,
    world_size: int,
    executor: str,
    warmup: int,
    iterations: int,
    repeats: int,
    output_dir: Path,
) -> Dict[str, object]:
    element_count = data_bytes // torch.tensor([], dtype=torch.float32).element_size()
    element_count = (element_count // world_size) * world_size
    if element_count <= 0:
        raise ValueError("data size is too small for the process group")
    actual_bytes = element_count * 4

    operation_names: Dict[str, str] = {}
    for mode in MODES:
        operation_name = "alg_ir_ab_{}_{}_{}b_{}".format(collective, mode, actual_bytes, rank)
        graph = _build_graph(mode, collective, rank, world_size, element_count, executor)
        json_file = output_dir / "{}_rank{}.json".format(operation_name, rank)
        compile_to_json_file(graph, str(json_file))
        if not register_operation(operation_name, str(json_file)):
            raise RuntimeError("failed to register {}".format(operation_name))
        operation_names[mode] = operation_name

    dist.barrier()
    if collective == "allreduce":
        input_tensor = torch.full(
            (element_count,), float(rank + 1), dtype=torch.float32, device="cuda"
        )
    else:
        # Equal values make both local and remote all-to-all slots directly checkable.
        input_tensor = torch.ones(element_count, dtype=torch.float32, device="cuda")
    outputs = {mode: torch.empty_like(input_tensor) for mode in MODES}

    for mode in MODES:
        _run_once(operation_names[mode], input_tensor, outputs[mode])
        _check_output(collective, outputs[mode], world_size)
        for _ in range(warmup):
            _run_once(operation_names[mode], input_tensor, outputs[mode])

    samples: Dict[str, List[float]] = {mode: [] for mode in MODES}
    for repeat in range(repeats):
        # Reverse order every repeat to reduce thermal and first-run bias.
        order = MODES if repeat % 2 == 0 else tuple(reversed(MODES))
        for mode in order:
            samples[mode].append(
                _measure(operation_names[mode], input_tensor, outputs[mode], iterations)
            )

    for mode in MODES:
        reset_signals(operation_names[mode])
    dist.barrier()

    medians = {mode: statistics.median(samples[mode]) for mode in MODES}
    return {
        "collective": collective,
        "data_bytes": actual_bytes,
        "executor": executor,
        "iterations": iterations,
        "repeats": repeats,
        "latency_us": medians,
        "samples_us": samples,
        "generated_over_template": medians["generated"] / medians["template"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=_parse_csv_ints,
        default=_parse_csv_ints("1048576,16777216,67108864"),
        help="comma-separated payload sizes in bytes",
    )
    parser.add_argument(
        "--collectives",
        type=lambda value: _parse_csv_choices(value, COLLECTIVES),
        default=list(COLLECTIVES),
    )
    parser.add_argument("--executor", choices=("sm", "tma"), default="sm")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        default="tests/generated_json_algorithm_ir/algorithm_ir_ab_results.json",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative; iterations and repeats must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        parser.error("the A/B benchmark requires at least two ranks")

    os.environ.setdefault("PCCL_DISABLE_FUSED", "1")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        get_engine()
        dist.barrier()
        initialize_engine(dist.group.WORLD)
        dist.barrier()

        output_file = Path(args.output)
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for collective in args.collectives:
            for data_bytes in args.sizes:
                case = _bench_case(
                    collective=collective,
                    data_bytes=data_bytes,
                    rank=rank,
                    world_size=world_size,
                    executor=args.executor,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                    output_dir=output_dir,
                )
                if rank == 0:
                    results.append(case)
                    print("ALGORITHM_IR_AB " + json.dumps(case, sort_keys=True), flush=True)

        if rank == 0:
            document = {
                "world_size": world_size,
                "device": torch.cuda.get_device_name(local_rank),
                "fused_disabled": os.environ.get("PCCL_DISABLE_FUSED") == "1",
                "timed_region": "execute_operation_async + sync_operation",
                "results": results,
            }
            output_file.write_text(json.dumps(document, indent=2), encoding="utf-8")
            print(
                "ALGORITHM_IR_AB_PASS "
                + json.dumps(
                    {
                        "cases": len(results),
                        "output": str(output_file),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
