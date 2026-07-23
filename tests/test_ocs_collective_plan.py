"""Tests for fixed multi-collective OCS execution plans."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import (
    OCSCollectivePhase,
    OCSCollectivePlan,
    OcsCollectivePlanRunner,
    build_ring_allreduce_alltoall_plan,
)
from pccl.ocs import collective_plan as collective_plan_module


class RecordingEngine:
    def __init__(self, events, register_success=True):
        self.events = events
        self.register_success = register_success
        self.registered = {}

    def register_operation(self, name, filename):
        self.events.append(("register", name))
        self.registered[name] = json.loads(Path(filename).read_text(encoding="utf-8"))
        return self.register_success

    def execute_operation(self, name, input_tensor, output_tensor):
        self.events.append(("execute", name, float(input_tensor.item())))
        output_tensor.copy_(input_tensor)
        if name.endswith("allreduce_0"):
            output_tensor.add_(1)
        elif name.endswith("alltoall_1"):
            output_tensor.add_(2)
        else:
            output_tensor.add_(3)
        return output_tensor

    def reset_signals(self, name):
        self.events.append(("reset_signals", name))


class RecordingRuntime:
    def __init__(self, events):
        self.events = events
        self.plans = []

    def barrier_switch(self, plan, group=None, timeout=None):
        self.events.append(("barrier", plan.barrier_id, plan.topology_id))
        self.plans.append(plan)
        return {"status": "OK", "link_state": "LINK_ALIGNED"}


def _build_plan(**kwargs):
    params = dict(
        rank=0,
        world_size=2,
        tensor_size=8,
        executor="sm",
        first_barrier_id=41,
        first_epoch_id=7,
    )
    params.update(kwargs)
    return build_ring_allreduce_alltoall_plan(**params)


def test_collective_plan_requires_barrier_between_phases():
    plan = _build_plan()

    with pytest.raises(ValueError, match="non-final"):
        OCSCollectivePlan(phases=(
            OCSCollectivePhase("first", plan.phases[0].graph),
            plan.phases[1],
            plan.phases[2],
        ))


def test_collective_plan_supports_final_epoch_boundary():
    plan = _build_plan(include_final_barrier=True)

    assert [phase.barrier_after.barrier_id for phase in plan.phases] == [41, 42, 43]
    assert [phase.barrier_after.epoch_id for phase in plan.phases] == [7, 8, 9]
    assert [phase.barrier_after.next_epoch_id for phase in plan.phases] == [8, 9, 10]


def test_collective_plan_materializes_per_phase_collective_types(tmp_path):
    events = []
    engine = RecordingEngine(events)
    runner = OcsCollectivePlanRunner(engine=engine, json_dir=str(tmp_path))

    prepared = runner.prepare(_build_plan(), operation_name="fixed")

    assert prepared.collective_types == ("allreduce", "alltoall", "allreduce")
    assert prepared.operation_names == (
        "fixed_0_allreduce_0",
        "fixed_1_alltoall_1",
        "fixed_2_allreduce_2",
    )
    assert all(path.exists() for path in prepared.phase_files)
    assert [plan.barrier_id if plan else None for plan in prepared.barriers_after_phase] == [41, 42, None]
    assert engine.registered == {}
    specs = [json.loads(path.read_text(encoding="utf-8")) for path in prepared.phase_files]
    assert [spec["collective_type"] for spec in specs] == [
        "allreduce",
        "alltoall",
        "allreduce",
    ]
    assert all(spec["version"] == 2 for spec in specs)


def test_collective_plan_executes_collective_barrier_collective_order(tmp_path):
    events = []
    engine = RecordingEngine(events)
    runtime = RecordingRuntime(events)
    runner = OcsCollectivePlanRunner(engine=engine, runtime=runtime, json_dir=str(tmp_path))
    prepared = runner.prepare(_build_plan(), operation_name="ordered")

    output = torch.zeros(1)
    result = runner.execute(prepared, torch.tensor([1.0]), output_tensor=output)

    assert result is output
    assert result.item() == 7.0
    assert events == [
        ("register", "ordered_0_allreduce_0"),
        ("execute", "ordered_0_allreduce_0", 1.0),
        ("barrier", 41, 1),
        ("reset_signals", "ordered_0_allreduce_0"),
        ("register", "ordered_1_alltoall_1"),
        ("execute", "ordered_1_alltoall_1", 2.0),
        ("barrier", 42, 2),
        ("reset_signals", "ordered_1_alltoall_1"),
        ("register", "ordered_2_allreduce_2"),
        ("execute", "ordered_2_allreduce_2", 4.0),
    ]
    assert [plan.epoch_id for plan in runtime.plans] == [7, 8]


def test_collective_plan_fences_registration_before_execution(tmp_path, monkeypatch):
    events = []
    engine = RecordingEngine(events)
    runtime = RecordingRuntime(events)
    runner = OcsCollectivePlanRunner(engine=engine, runtime=runtime, json_dir=str(tmp_path))
    prepared = runner.prepare(_build_plan(), operation_name="fenced")

    monkeypatch.setattr(collective_plan_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        collective_plan_module.dist,
        "barrier",
        lambda group=None: events.append(("registration_fence", group)),
    )

    runner.execute(prepared, torch.tensor([1.0]))

    assert events[:3] == [
        ("register", "fenced_0_allreduce_0"),
        ("registration_fence", None),
        ("execute", "fenced_0_allreduce_0", 1.0),
    ]


def test_collective_plan_repeats_three_full_epochs_without_state_leak(tmp_path):
    events = []
    engine = RecordingEngine(events)
    runtime = RecordingRuntime(events)
    runner = OcsCollectivePlanRunner(engine=engine, runtime=runtime, json_dir=str(tmp_path))

    for iteration in range(3):
        first = iteration * 3
        plan = _build_plan(
            first_barrier_id=first,
            first_epoch_id=first,
            include_final_barrier=True,
        )
        prepared = runner.prepare(plan, operation_name=f"round_{iteration}")
        runner.execute(prepared, torch.tensor([1.0]))

    assert [plan.barrier_id for plan in runtime.plans] == list(range(9))
    assert [plan.epoch_id for plan in runtime.plans] == list(range(9))
    assert [plan.next_epoch_id for plan in runtime.plans] == list(range(1, 10))
    assert len([event for event in events if event[0] == "reset_signals"]) == 9
