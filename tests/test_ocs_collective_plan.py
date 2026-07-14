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


def _build_plan():
    return build_ring_allreduce_alltoall_plan(
        rank=0,
        world_size=2,
        tensor_size=8,
        executor="sm",
        first_barrier_id=41,
        first_epoch_id=7,
    )


def test_collective_plan_requires_barrier_between_phases():
    plan = _build_plan()

    with pytest.raises(ValueError, match="non-final"):
        OCSCollectivePlan(phases=(
            OCSCollectivePhase("first", plan.phases[0].graph),
            plan.phases[1],
            plan.phases[2],
        ))


def test_collective_plan_registers_per_phase_collective_types(tmp_path):
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
    assert [plan.barrier_id if plan else None for plan in prepared.barriers_after_phase] == [41, 42, None]
    assert [spec["collective_type"] for spec in engine.registered.values()] == [
        "allreduce",
        "alltoall",
        "allreduce",
    ]
    assert all(spec["version"] == 2 for spec in engine.registered.values())


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
        ("register", "ordered_1_alltoall_1"),
        ("register", "ordered_2_allreduce_2"),
        ("execute", "ordered_0_allreduce_0", 1.0),
        ("barrier", 41, 1),
        ("reset_signals", "ordered_0_allreduce_0"),
        ("execute", "ordered_1_alltoall_1", 2.0),
        ("barrier", 42, 2),
        ("reset_signals", "ordered_1_alltoall_1"),
        ("execute", "ordered_2_allreduce_2", 4.0),
    ]
    assert [plan.epoch_id for plan in runtime.plans] == [7, 8]
