"""GPU smoke for Execution Plan template/generated/MSCCL lowering.

Example:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      tests/ocs_algorithm_ir_smoke.py --lowering generated --iterations 3
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import ExecutionPlanCompiler, OCSExecutionPlan, OCSRuntime, OcsCollectivePlanRunner
from pccl.engine import get_engine, initialize_engine


def build_plan_dict(world_size, iteration, lowering):
    first = iteration * 3
    topologies = [10, 20, 30]
    operations = ["allreduce", "alltoall", "allreduce"]
    algorithms = ["ring", "direct", "ring"]
    phases = []
    for phase_id in range(3):
        next_phase_id = (phase_id + 1) % 3
        phases.append(
            {
                "phase_id": phase_id,
                "epoch": first + phase_id,
                "op_type": operations[phase_id],
                "algorithm_type": algorithms[phase_id],
                "backend": "pccl",
                "topology_id": topologies[phase_id],
                "artifact_id": (
                    "msccl:ring2"
                    if lowering == "msccl" and operations[phase_id] == "allreduce"
                    else ("msccl:direct-a2a-2" if lowering == "msccl" else None)
                ),
                "graph_digest": None,
                "barrier_after": {
                    "barrier_id": first + phase_id,
                    "next_epoch": first + phase_id + 1,
                    "next_phase_id": next_phase_id,
                    "switch_action": "APPLY_ROUTE",
                    "route_plan": {
                        "route_plan_id": "smoke-route-{}".format(first + phase_id),
                        "route_mode": "STATIC_PLAN",
                        "source_topology_id": topologies[phase_id],
                        "target_topology_id": topologies[next_phase_id],
                        "payload": {"iteration": iteration, "phase_id": phase_id},
                    },
                },
            }
        )
    return {
        "schema_version": "ocs-pccl.execution-plan.v1",
        "job_id": "algorithm-ir-gpu-smoke",
        "plan_id": "{}-round-{}".format(os.environ.get("LOWERING", "unknown"), iteration),
        "group_id": 0,
        "rank_list": list(range(world_size)),
        "participant_bitmap": hex((1 << world_size) - 1),
        "phases": phases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--lowering", choices=("template", "generated", "msccl"), required=True)
    args = parser.parse_args()
    if args.elements <= 0:
        parser.error("--elements must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.elements % world_size:
        parser.error("--elements must be divisible by WORLD_SIZE")
    if args.lowering == "msccl" and world_size != 2:
        parser.error("MSCCL smoke fixtures require WORLD_SIZE=2")
    os.environ["LOWERING"] = args.lowering

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        preflight = torch.tensor([rank + 1], dtype=torch.int32, device="cuda")
        dist.all_reduce(preflight)
        if preflight.item() != sum(range(1, world_size + 1)):
            raise RuntimeError("NCCL preflight returned an unexpected value")

        get_engine()
        initialize_engine(dist.group.WORLD)
        runtime = OCSRuntime()
        runner = OcsCollectivePlanRunner(runtime=runtime)
        fixture_dir = Path(__file__).parent / "fixtures"
        artifacts = {
            "msccl:ring2": fixture_dir / "msccl_ring_allreduce_2_fused.xml",
            "msccl:direct-a2a-2": fixture_dir / "msccl_direct_alltoall_2.xml",
        }
        compiler = ExecutionPlanCompiler(
            algorithm_lowering=args.lowering,
            artifact_resolver=(artifacts.__getitem__ if args.lowering == "msccl" else None),
        )
        expected_value = float(sum(range(1, world_size + 1)) * world_size)

        for iteration in range(args.iterations):
            plan = OCSExecutionPlan.from_dict(build_plan_dict(world_size, iteration, args.lowering))
            compiled = compiler.compile(
                plan,
                rank=rank,
                tensor_size=args.elements,
                dtype="float32",
                executor="sm",
            )
            prepared = runner.prepare(
                compiled,
                operation_name="{}_rank{}_round{}".format(args.lowering, rank, iteration),
            )
            input_tensor = torch.full(
                (args.elements,), float(rank + 1), dtype=torch.float32, device="cuda"
            )
            output_tensor = torch.empty_like(input_tensor)
            result = runner.execute(prepared, input_tensor, output_tensor=output_tensor)
            torch.cuda.synchronize()
            if not torch.equal(result, torch.full_like(result, expected_value)):
                mismatches = int(torch.count_nonzero(result != expected_value).item())
                raise RuntimeError(
                    "{} output mismatch in round {}: {} elements".format(
                        args.lowering, iteration, mismatches
                    )
                )
            prepared.close()

        expected_barriers = list(range(args.iterations * 3))
        actual_barriers = [release["barrier_id"] for release in runtime.history]
        if actual_barriers != expected_barriers:
            raise RuntimeError(
                "barrier sequence mismatch: {} != {}".format(actual_barriers, expected_barriers)
            )
        if any(release["link_state"] != "LINK_ALIGNED" for release in runtime.history):
            raise RuntimeError("one or more barriers did not reach LINK_ALIGNED")

        dist.barrier()
        print(
            "OCS_ALGORITHM_IR_SMOKE_PASS "
            + json.dumps(
                {
                    "barrier_ids": actual_barriers,
                    "iterations": args.iterations,
                    "lowering": args.lowering,
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
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
