"""NCCL smoke test for the engine-backed OCS phase runner.

Run with two compatible GPUs, for example:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      tests/ocs_phase_runner_smoke.py
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import DeviceType, OCSRuntime, OcsPhaseRunner, Stream, build_graph
from pccl.engine import get_engine, initialize_engine


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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--endpoint-only", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--barrier-only", action="store_true")
    parser.add_argument("--first-phase-only", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    def report(stage: str) -> None:
        print("OCS_PHASE_RUNNER_STAGE rank={} {}".format(rank, stage), flush=True)

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    report("process_group_ready")

    try:
        reduce_input = torch.tensor([rank + 1], dtype=torch.int32, device="cuda")
        dist.all_reduce(reduce_input)
        if reduce_input.item() != sum(range(1, world_size + 1)):
            raise RuntimeError("NCCL all_reduce preflight returned an unexpected value")
        report("nccl_all_reduce_ready")
        preflight_input = torch.tensor([rank + 1], dtype=torch.int32, device="cuda")
        preflight_output = torch.empty(world_size, dtype=torch.int32, device="cuda")
        dist.all_gather_into_tensor(preflight_output, preflight_input)
        expected_preflight = torch.arange(
            1, world_size + 1, dtype=torch.int32, device="cuda")
        if not torch.equal(preflight_output, expected_preflight):
            raise RuntimeError("NCCL all_gather preflight returned unexpected values")
        report("nccl_all_gather_ready")
        if args.preflight_only:
            return
        report("engine_constructing")
        get_engine()
        report("engine_constructed")
        report("endpoints_initializing")
        initialize_engine(dist.group.WORLD)
        report("endpoints_ready")
        if args.endpoint_only:
            dist.barrier()
            print(
                "OCS_PHASE_RUNNER_ENDPOINT_PASS "
                + json.dumps({"rank": rank, "world_size": world_size}, sort_keys=True),
                flush=True,
            )
            return
        runtime = OCSRuntime()
        runner = OcsPhaseRunner(runtime=runtime)
        prepared = runner.prepare(
            build_two_phase_graph(rank, world_size, args.elements),
            operation_name="ocs_phase_runner_smoke_rank{}".format(rank),
        )
        report("phases_registered")
        if args.register_only:
            dist.barrier()
            print(
                "OCS_PHASE_RUNNER_REGISTER_PASS "
                + json.dumps({"rank": rank, "world_size": world_size}, sort_keys=True),
                flush=True,
            )
            prepared.close()
            return
        if args.barrier_only:
            barrier_plan = prepared.barriers_after_phase[0]
            if barrier_plan is None:
                raise RuntimeError("two-phase smoke graph is missing its OCS barrier")
            release = runtime.barrier_switch(barrier_plan)
            if release["status"] != "OK":
                raise RuntimeError("unexpected OCS release: {}".format(release))
            dist.barrier()
            print(
                "OCS_PHASE_RUNNER_BARRIER_PASS "
                + json.dumps(
                    {
                        "barrier_latency_us": release["latency_us"],
                        "rank": rank,
                        "world_size": world_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            prepared.close()
            return
        input_tensor = torch.full(
            (args.elements,), float(rank + 1), dtype=torch.float32, device="cuda")
        output_tensor = torch.empty_like(input_tensor)

        if args.first_phase_only:
            first_phase_name = prepared.operation_names[0]
            if first_phase_name is None:
                raise RuntimeError("two-phase smoke graph is missing its first data phase")
            result = runner._engine().execute_operation(
                first_phase_name, input_tensor, output_tensor)
            torch.cuda.synchronize()
            if not torch.equal(result, input_tensor):
                raise RuntimeError("first phase output does not match this rank's input")
            dist.barrier()
            print(
                "OCS_PHASE_RUNNER_FIRST_PHASE_PASS "
                + json.dumps({"rank": rank, "world_size": world_size}, sort_keys=True),
                flush=True,
            )
            prepared.close()
            return

        started_ns = time.perf_counter_ns()
        result = runner.execute(prepared, input_tensor, output_tensor=output_tensor)
        torch.cuda.synchronize()
        report("phases_executed")
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
        if dist.is_initialized():
            report("process_group_destroying")
            dist.destroy_process_group()
            report("process_group_destroyed")


if __name__ == "__main__":
    main()
