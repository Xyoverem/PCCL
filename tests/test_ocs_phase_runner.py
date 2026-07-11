"""Tests for host orchestration of phased OCS IR graphs."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import DeviceType, OcsPhaseRunner, Stream, build_graph


class RecordingEngine:
    def __init__(self, events, register_success=True):
        self.events = events
        self.register_success = register_success
        self.registered = {}

    def register_operation(self, name, filename):
        spec = json.loads(Path(filename).read_text(encoding="utf-8"))
        self.events.append(("register", name))
        self.registered[name] = spec
        return self.register_success

    def execute_operation(self, name, input_tensor, output_tensor):
        self.events.append(("execute", name, float(input_tensor.item())))
        output_tensor.copy_(input_tensor)
        output_tensor.add_(1 if name.endswith("phase_0") else 2)
        return output_tensor


class RecordingRuntime:
    def __init__(self, events):
        self.events = events
        self.plans = []

    def barrier_switch(self, plan, group=None, timeout=None):
        self.events.append(("barrier", plan.barrier_id, plan.algorithm))
        self.plans.append(plan)
        return {"status": "OK"}


def _build_two_phase_graph():
    def build(op):
        op.tensor(dtype="float32", shape=(1,))
        with Stream("phase0"):
            op.sm_copy(source_rank=1, src_offset=0, dst_offset=0, size=1)
        op.ocs_barrier(
            barrier_id=12,
            epoch_id=4,
            next_epoch_id=5,
            participant_ranks=(0, 1),
            topology_id=9,
            route_plan_id=10,
            algorithm="ring",
        )
        with Stream("phase1"):
            op.sm_copy(source_rank=1, src_offset=0, dst_offset=0, size=1)

    return build_graph("phase_runner_graph", build, device=DeviceType.CUDA)


def test_phase_runner_registers_json_v2_per_data_phase(tmp_path):
    events = []
    engine = RecordingEngine(events)
    runtime = RecordingRuntime(events)
    runner = OcsPhaseRunner(engine=engine, runtime=runtime, json_dir=str(tmp_path))

    prepared = runner.prepare(_build_two_phase_graph(), operation_name="two_phase")

    assert prepared.operation_names == ("two_phase_phase_0", "two_phase_phase_1")
    assert prepared.barriers_after_phase[0].barrier_id == 12
    assert prepared.barriers_after_phase[1] is None
    assert list(engine.registered) == ["two_phase_phase_0", "two_phase_phase_1"]
    for spec in engine.registered.values():
        assert spec["version"] == 2
        assert "operations" in spec
        assert all(op["primitive"] != "ocs.barrier" for op in spec["operations"])


def test_phase_runner_executes_data_then_barrier_then_next_phase(tmp_path):
    events = []
    engine = RecordingEngine(events)
    runtime = RecordingRuntime(events)
    runner = OcsPhaseRunner(engine=engine, runtime=runtime, json_dir=str(tmp_path))
    prepared = runner.prepare(_build_two_phase_graph(), operation_name="ordered")

    output = torch.zeros(1)
    result = runner.execute(prepared, torch.tensor([1.0]), output_tensor=output)

    assert result is output
    assert result.item() == 4.0
    assert events == [
        ("register", "ordered_phase_0"),
        ("register", "ordered_phase_1"),
        ("execute", "ordered_phase_0", 1.0),
        ("barrier", 12, "ring"),
        ("execute", "ordered_phase_1", 2.0),
    ]
    assert runtime.plans[0].backend == "pccl"


def test_phase_runner_rejects_async_execution(tmp_path):
    runner = OcsPhaseRunner(
        engine=RecordingEngine([]),
        runtime=RecordingRuntime([]),
        json_dir=str(tmp_path),
    )
    prepared = runner.prepare(_build_two_phase_graph())

    with pytest.raises(NotImplementedError, match="blocking"):
        runner.execute(prepared, torch.tensor([1.0]), async_op=True)


def test_phase_runner_requires_ocs_barrier_graph(tmp_path):
    def build(op):
        op.tensor(dtype="float32", shape=(1,))
        op.sm_copy(source_rank=1, src_offset=0, dst_offset=0, size=1)

    runner = OcsPhaseRunner(
        engine=RecordingEngine([]),
        runtime=RecordingRuntime([]),
        json_dir=str(tmp_path),
    )

    with pytest.raises(ValueError, match="OCS barrier"):
        runner.prepare(build_graph("plain", build, device=DeviceType.CUDA))
