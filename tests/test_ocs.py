"""Tests for the OCS-aware runtime skeleton."""

from dataclasses import replace
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
    OCSPlan,
    OCSPlanController,
    OCSRuntime,
    StaticPlanController,
    SwitchConnector,
    TorchDistributedSwitchConnector,
    ocs_all_reduce,
)
from pccl.ocs import OCSBarrierTimeout, OCSPlanMismatchError
import pccl.ocs.runtime as runtime_module


class FakeConnector:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def exchange_ready(self, ready_record, group=None, timeout=None):
        self.calls.append((ready_record, group, timeout))
        return self.records if self.records is not None else [ready_record]


class FakeRuntime:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def next_plan(self, event_key, group=None):
        self.calls.append(("next_plan", event_key))
        return self.plan

    def barrier_switch(self, plan, group=None, timeout=None):
        self.calls.append(("barrier_switch", plan.barrier_id))
        return {"status": "OK"}


def test_controller_and_connector_protocols_are_structural():
    assert isinstance(StaticPlanController(), OCSPlanController)
    assert isinstance(TorchDistributedSwitchConnector(), SwitchConnector)
    assert isinstance(FakeConnector(None), SwitchConnector)


def test_ocs_plan_defaults_and_bitmap():
    plan = OCSPlan(participant_ranks=(0, 2, 3), payload=b"abc")

    assert plan.job_id == "default"
    assert plan.algorithm == "torch_native"
    assert plan.backend == "torch"
    assert plan.route_mode == "STATIC_PLAN"
    assert plan.participant_bitmap == 0b1101
    assert plan.payload == b"abc"

    ready = plan.ready_record(src_rank=2, world_size=4, arrive_seq=7, arrival_time_us=123)
    assert ready["msg_type"] == "OCS_BARRIER_READY"
    assert ready["src_rank"] == 2
    assert ready["arrive_seq"] == 7
    assert ready["participant_bitmap"] == 0b1101
    assert ready["payload_len"] == 3


def test_static_plan_controller_returns_default_plans_by_event():
    controller = StaticPlanController()

    p0 = controller.next_plan("all_reduce", rank=0, world_size=4)
    p1 = controller.next_plan("all_reduce", rank=0, world_size=4)

    assert p0.barrier_id == 0
    assert p0.epoch_id == 0
    assert p0.next_epoch_id == 1
    assert p0.participant_ranks == (0, 1, 2, 3)
    assert p1.barrier_id == 1
    assert p1.epoch_id == 1


def test_static_plan_controller_uses_precomputed_event_plans():
    plan = OCSPlan(
        job_id="job-a",
        barrier_id=11,
        epoch_id=4,
        next_epoch_id=5,
        topology_id=8,
        route_plan_id=9,
        algorithm="ring",
    )
    controller = StaticPlanController({"train_step": [plan]})

    selected = controller.next_plan("train_step", rank=0, world_size=2)

    assert selected.job_id == "job-a"
    assert selected.barrier_id == 11
    assert selected.participant_ranks == (0, 1)
    assert selected.algorithm == "ring"


def test_barrier_switch_releases_when_ready_records_match():
    plan = OCSPlan(
        participant_ranks=(0, 1),
        barrier_id=3,
        epoch_id=3,
        next_epoch_id=4,
        topology_id=7,
        route_plan_id=10,
        algorithm="rhd",
    )
    records = [
        plan.ready_record(src_rank=0, world_size=2),
        plan.ready_record(src_rank=1, world_size=2),
    ]
    runtime = OCSRuntime(connector=FakeConnector(records))

    release = runtime.barrier_switch(plan)

    assert release["msg_type"] == "OCS_BARRIER_RELEASE"
    assert release["status"] == "OK"
    assert release["barrier_id"] == 3
    assert release["topology_id"] == 7
    assert release["algorithm"] == "rhd"
    assert len(runtime.history) == 1


def test_barrier_switch_rejects_inconsistent_plans():
    plan = OCSPlan(participant_ranks=(0, 1), topology_id=7)
    different = OCSPlan(participant_ranks=(0, 1), topology_id=8)
    records = [
        plan.ready_record(src_rank=0, world_size=2),
        different.ready_record(src_rank=1, world_size=2),
    ]
    runtime = OCSRuntime(connector=FakeConnector(records))

    with pytest.raises(OCSPlanMismatchError, match="topology_id"):
        runtime.barrier_switch(plan)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "different-job"),
        ("next_epoch_id", 2),
        ("route_mode", "USER_PLAN"),
        ("backend", "pccl"),
    ],
)
def test_barrier_switch_rejects_other_plan_identity_mismatches(field, value):
    plan = OCSPlan(
        job_id="job-a",
        participant_ranks=(0, 1),
        barrier_id=3,
        epoch_id=0,
        next_epoch_id=1,
        route_mode="STATIC_PLAN",
        backend="torch",
    )
    different = replace(plan, **{field: value})
    records = [
        plan.ready_record(src_rank=0, world_size=2),
        different.ready_record(src_rank=1, world_size=2),
    ]
    runtime = OCSRuntime(connector=FakeConnector(records))

    with pytest.raises(OCSPlanMismatchError, match=field):
        runtime.barrier_switch(plan)


def test_barrier_switch_rejects_duplicate_ready_rank():
    plan = OCSPlan(participant_ranks=(0, 1), barrier_id=6)
    records = [
        plan.ready_record(src_rank=0, world_size=2),
        plan.ready_record(src_rank=0, world_size=2),
        plan.ready_record(src_rank=1, world_size=2),
    ]
    runtime = OCSRuntime(connector=FakeConnector(records))

    with pytest.raises(OCSPlanMismatchError, match="duplicate READY"):
        runtime.barrier_switch(plan)


def test_barrier_switch_reports_missing_rank():
    plan = OCSPlan(participant_ranks=(0, 1), barrier_id=5)
    records = [plan.ready_record(src_rank=0, world_size=2)]
    runtime = OCSRuntime(connector=FakeConnector(records))

    with pytest.raises(OCSBarrierTimeout, match="missing READY"):
        runtime.barrier_switch(plan)


def test_barrier_switch_rejects_non_static_route_mode():
    plan = OCSPlan(participant_ranks=(0,), route_mode="ID_ROUTE")
    runtime = OCSRuntime(connector=FakeConnector(None))

    with pytest.raises(NotImplementedError, match="STATIC_PLAN"):
        runtime.barrier_switch(plan)


def test_ocs_all_reduce_calls_barrier_before_collective(monkeypatch):
    calls = []
    plan = OCSPlan(participant_ranks=(0,))
    fake_runtime = FakeRuntime(plan)

    def fake_all_reduce(tensor, op=None, group=None):
        calls.append(("all_reduce", tensor.item()))
        tensor.add_(1)

    monkeypatch.setattr(runtime_module.dist, "all_reduce", fake_all_reduce)
    tensor = torch.tensor([2.0])

    result = ocs_all_reduce(tensor, runtime=fake_runtime)

    assert result is tensor
    assert tensor.item() == 3.0
    assert fake_runtime.calls == [("next_plan", "all_reduce"), ("barrier_switch", 0)]
    assert calls == [("all_reduce", 2.0)]


def test_ocs_all_reduce_rejects_async_op():
    fake_runtime = FakeRuntime(OCSPlan(participant_ranks=(0,)))

    with pytest.raises(NotImplementedError, match="async_op"):
        ocs_all_reduce(torch.tensor([1.0]), runtime=fake_runtime, async_op=True)

    assert fake_runtime.calls == []


def test_ocs_all_reduce_rejects_pccl_backend():
    fake_runtime = FakeRuntime(OCSPlan(participant_ranks=(0,), backend="pccl"))

    with pytest.raises(NotImplementedError, match="torch backend"):
        ocs_all_reduce(torch.tensor([1.0]), runtime=fake_runtime)

    assert fake_runtime.calls == [("next_plan", "all_reduce")]


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
        runtime = OCSRuntime(controller=StaticPlanController())
        tensor = torch.tensor([float(rank + 1)])
        ocs_all_reduce(tensor, runtime=runtime)
        queue.put((rank, float(tensor.item()), len(runtime.history)))
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
    reason="requires torch.distributed gloo with all_gather_object",
)
def test_ocs_all_reduce_cpu_gloo_2rank_smoke():
    world_size = 2
    port = _free_port()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
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
    assert sorted(results) == [(0, 3.0, 1), (1, 3.0, 1)]
