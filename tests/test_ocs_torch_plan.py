"""Tests for torch-distributed collective plans with OCS barriers."""

import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import (
    OCSRuntime,
    OcsTorchPlanRunner,
    TorchCollectivePhase,
    TorchCollectivePlan,
    build_torch_allreduce_alltoall_plan,
)
import pccl.ocs.torch_plan as torch_plan_module


class RecordingRuntime:
    def __init__(self, events):
        self.events = events
        self.plans = []

    def barrier_switch(self, plan, group=None, timeout=None):
        self.events.append(("barrier", plan.barrier_id, plan.epoch_id))
        self.plans.append(plan)
        return {"status": "OK", "link_state": "LINK_ALIGNED"}


def test_torch_collective_plan_requires_phase_boundaries():
    plan = build_torch_allreduce_alltoall_plan(world_size=2)

    with pytest.raises(ValueError, match="non-final"):
        TorchCollectivePlan(phases=(
            TorchCollectivePhase("all_reduce"),
            plan.phases[1],
            plan.phases[2],
        ))


def test_torch_collective_plan_supports_final_epoch_boundary():
    plan = build_torch_allreduce_alltoall_plan(
        world_size=2,
        first_barrier_id=20,
        first_epoch_id=30,
        include_final_barrier=True,
    )

    assert [phase.barrier_after.barrier_id for phase in plan.phases] == [20, 21, 22]
    assert [phase.barrier_after.epoch_id for phase in plan.phases] == [30, 31, 32]
    assert [phase.barrier_after.next_epoch_id for phase in plan.phases] == [31, 32, 33]


def test_torch_collective_plan_orders_collectives_and_barriers(monkeypatch):
    events = []
    runtime = RecordingRuntime(events)
    runner = OcsTorchPlanRunner(runtime=runtime)

    def fake_all_reduce(tensor, group=None):
        events.append(("all_reduce", float(tensor.item())))
        tensor.add_(1)

    def fake_all_to_all(output, input, group=None):
        events.append(("all_to_all_single", float(input.item())))
        output.copy_(input)
        output.add_(2)

    monkeypatch.setattr(torch_plan_module.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch_plan_module.dist, "all_to_all_single", fake_all_to_all)

    output = torch.zeros(1)
    result = runner.execute(
        build_torch_allreduce_alltoall_plan(
            world_size=2,
            first_barrier_id=31,
            first_epoch_id=9,
            include_final_barrier=True,
        ),
        torch.tensor([1.0]),
        output_tensor=output,
    )

    assert result is output
    assert result.item() == 5.0
    assert events == [
        ("all_reduce", 1.0),
        ("barrier", 31, 9),
        ("all_to_all_single", 2.0),
        ("barrier", 32, 10),
        ("all_reduce", 4.0),
        ("barrier", 33, 11),
    ]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _gloo_worker(rank, world_size, port, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        runtime = OCSRuntime()
        runner = OcsTorchPlanRunner(runtime=runtime)
        for iteration in range(3):
            result = runner.execute(
                build_torch_allreduce_alltoall_plan(
                    world_size=world_size,
                    first_barrier_id=iteration * 3,
                    first_epoch_id=iteration * 3,
                    include_final_barrier=True,
                ),
                torch.full((4,), float(rank + 1)),
            )
            assert torch.equal(result, torch.full_like(result, 6.0))
        queue.put((
            rank,
            [release["barrier_id"] for release in runtime.history],
            [release["link_state"] for release in runtime.history],
        ))
    except Exception as exc:  # pragma: no cover - reported to parent process
        queue.put((rank, type(exc).__name__, str(exc)))
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available()
    or not dist.is_gloo_available()
    or not hasattr(dist, "all_gather_object"),
    reason="requires torch.distributed gloo with object collectives",
)
def test_torch_collective_plan_cpu_gloo_2rank_multiepoch():
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_gloo_worker, args=(rank, world_size, port, queue))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)

    for proc in procs:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)

    assert all(proc.exitcode == 0 for proc in procs)
    results = [queue.get(timeout=5) for _ in range(world_size)]
    assert sorted(results) == [
        (0, list(range(9)), ["LINK_ALIGNED"] * 9),
        (1, list(range(9)), ["LINK_ALIGNED"] * 9),
    ]
