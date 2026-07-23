"""Execution Plan v1 parsing, validation, and backend compilation tests."""

import json
from pathlib import Path

import pytest
import torch

from pccl import (
    ExecutionPlanCompiler,
    OCSControlMessage,
    OCSExecutionPlan,
    OCSExecutionPlanError,
    OcsCollectivePlanRunner,
    build_ready,
)


ROOT = Path(__file__).parent.parent
EXAMPLE_PATH = ROOT / "examples" / "ocs_execution_plan_v1.json"


def _example_dict():
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def _torch_plan_dict():
    data = _example_dict()
    for phase in data["phases"]:
        phase["backend"] = "torch"
        phase["algorithm_type"] = "torch_native"
        phase["artifact_id"] = None
        phase["graph_digest"] = None
    return data


class _RecordingEngine:
    def __init__(self, events):
        self.events = events

    def register_operation(self, name, filename):
        manifest = json.loads(Path(filename).read_text(encoding="utf-8"))
        self.events.append(("register", name, manifest["collective_type"]))
        return True

    def execute_operation(self, name, input_tensor, output_tensor):
        self.events.append(("execute", name))
        output_tensor.copy_(input_tensor)
        return output_tensor

    def reset_signals(self, name):
        self.events.append(("reset", name))


class _RecordingRuntime:
    def __init__(self, events):
        self.events = events
        self.plans = []

    def barrier_switch(self, plan, group=None, timeout=None):
        self.events.append(("barrier", plan.barrier_id))
        self.plans.append(plan)
        return {"status": "OK", "link_state": "LINK_ALIGNED"}


def test_execution_plan_loads_example_and_has_stable_canonical_digest():
    plan = OCSExecutionPlan.load(EXAMPLE_PATH)
    reordered = OCSExecutionPlan.from_json(json.dumps(plan.to_dict(), sort_keys=True, indent=4))

    assert plan.schema_version == "ocs-pccl.execution-plan.v1"
    assert plan.rank_list == (0, 1, 2, 3)
    assert plan.digest == reordered.digest
    assert plan.digest.startswith("sha256:")
    assert len(plan.digest) == len("sha256:") + 64


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data.update(participant_bitmap="0x3"), "does not match rank_list"),
        (lambda data: data["phases"][1].update(phase_id=4), "contiguous from zero"),
        (
            lambda data: data["phases"][0]["barrier_after"].update(next_epoch=999),
            "must equal the next phase epoch",
        ),
        (
            lambda data: data["phases"][0]["barrier_after"]["route_plan"].update(
                source_topology_id=999
            ),
            "must equal current phase topology_id",
        ),
        (
            lambda data: data["phases"][1]["barrier_after"].update(barrier_id=4200),
            "unique and strictly increasing",
        ),
        (lambda data: data.update(unknown_field=True), "unknown fields"),
    ],
)
def test_execution_plan_rejects_invalid_cross_field_contracts(mutate, message):
    data = _example_dict()
    mutate(data)

    with pytest.raises(OCSExecutionPlanError, match=message):
        OCSExecutionPlan.from_dict(data)


def test_barrier_projection_targets_the_next_phase_and_commits_rich_payload():
    plan = OCSExecutionPlan.load(EXAMPLE_PATH)

    barriers = [plan.barrier_plan(index) for index in range(3)]

    assert [(item.barrier_id, item.epoch_id, item.next_epoch_id) for item in barriers] == [
        (4200, 100, 101),
        (4201, 101, 102),
        (4202, 102, 103),
    ]
    assert [item.topology_id for item in barriers] == [20, 30, 10]
    assert [item.algorithm for item in barriers] == ["direct", "tree", "ring"]
    payload = json.loads(barriers[0].payload.decode("utf-8"))
    assert payload["plan_id"] == plan.plan_id
    assert payload["phase_id"] == 0
    assert payload["next_phase_id"] == 1
    assert payload["route_plan"]["route_plan_id"] == "route-4200"


def test_execution_plan_compiles_example_into_three_pccl_artifacts(tmp_path):
    plan = OCSExecutionPlan.load(EXAMPLE_PATH)
    compiled = ExecutionPlanCompiler().compile(
        plan,
        rank=0,
        tensor_size=16,
        dtype="float32",
        executor="sm",
    )
    prepared = OcsCollectivePlanRunner(json_dir=str(tmp_path)).prepare(
        compiled, operation_name="schema_v1"
    )

    assert [phase.graph.collective_type for phase in compiled.phases] == [
        "allreduce",
        "alltoall",
        "allreduce",
    ]
    assert prepared.collective_types == ("allreduce", "alltoall", "allreduce")
    assert [barrier.barrier_id for barrier in prepared.barriers_after_phase] == [
        4200,
        4201,
        4202,
    ]
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in prepared.phase_files]
    assert all(manifest["version"] == 2 for manifest in manifests)
    assert all(manifest["operations"] for manifest in manifests)


def test_compiled_execution_plan_runs_three_rounds_without_barrier_state_leak(tmp_path):
    events = []
    engine = _RecordingEngine(events)
    runtime = _RecordingRuntime(events)
    runner = OcsCollectivePlanRunner(engine=engine, runtime=runtime, json_dir=str(tmp_path))
    compiler = ExecutionPlanCompiler()

    for round_index in range(3):
        data = _example_dict()
        data["plan_id"] = "round-{}".format(round_index)
        first = round_index * 3
        for phase_index, phase in enumerate(data["phases"]):
            phase["epoch"] = first + phase_index
            barrier = phase["barrier_after"]
            barrier["barrier_id"] = first + phase_index
            barrier["next_epoch"] = first + phase_index + 1
        plan = OCSExecutionPlan.from_dict(data)
        compiled = compiler.compile(
            plan,
            rank=0,
            tensor_size=16,
            dtype="float32",
            executor="sm",
        )
        prepared = runner.prepare(compiled, operation_name="round_{}".format(round_index))
        result = runner.execute(prepared, torch.ones(16))
        assert torch.equal(result, torch.ones(16))

    assert [plan.barrier_id for plan in runtime.plans] == list(range(9))
    assert [plan.epoch_id for plan in runtime.plans] == list(range(9))
    assert [plan.next_epoch_id for plan in runtime.plans] == list(range(1, 10))
    assert len([event for event in events if event[0] == "execute"]) == 9
    assert len([event for event in events if event[0] == "reset"]) == 9


def test_execution_plan_compiler_remaps_sparse_group_ranks():
    data = _example_dict()
    data["rank_list"] = [2, 5]
    data["participant_bitmap"] = "0x24"
    plan = OCSExecutionPlan.from_dict(data)

    compiled = ExecutionPlanCompiler().compile(
        plan,
        rank=2,
        tensor_size=8,
        dtype="float32",
        executor="sm",
    )

    peer_ranks = set()
    for phase in compiled.phases:
        for node in phase.graph.nodes.values():
            for field in ("source_rank", "target_rank"):
                value = getattr(node, field, None)
                if isinstance(value, int) and value >= 0:
                    peer_ranks.add(value)
    assert peer_ranks
    assert peer_ranks.issubset({2, 5})
    assert compiled.phases[0].barrier_after.participant_ranks == (2, 5)


def test_execution_plan_compiler_commits_the_resolved_auto_algorithm():
    data = _example_dict()
    data["phases"][1]["algorithm_type"] = "auto"
    plan = OCSExecutionPlan.from_dict(data)

    compiled = ExecutionPlanCompiler().compile(
        plan,
        rank=0,
        tensor_size=16,
        dtype="float32",
        executor="sm",
    )

    # Non-allreduce auto lowering currently chooses the ring template.
    assert compiled.phases[0].barrier_after.algorithm == "ring"


def test_execution_plan_compiles_torch_backend_sequence():
    plan = OCSExecutionPlan.from_dict(_torch_plan_dict())

    compiled = ExecutionPlanCompiler().compile(plan, rank=0)

    assert [phase.collective for phase in compiled.phases] == [
        "all_reduce",
        "all_to_all_single",
        "all_reduce",
    ]
    assert [phase.barrier_after.barrier_id for phase in compiled.phases] == [
        4200,
        4201,
        4202,
    ]


def test_execution_plan_compiler_rejects_collective_without_lowering():
    data = _example_dict()
    data["phases"][0]["op_type"] = "broadcast"
    plan = OCSExecutionPlan.from_dict(data)

    with pytest.raises(OCSExecutionPlanError, match="does not support op_type 'broadcast'"):
        ExecutionPlanCompiler().compile(
            plan,
            rank=0,
            tensor_size=16,
            dtype="float32",
            executor="sm",
        )


def test_execution_plan_compiler_verifies_bound_graph_digest():
    data = _example_dict()
    data["phases"][0]["graph_digest"] = "sha256:" + "0" * 64
    plan = OCSExecutionPlan.from_dict(data)

    with pytest.raises(OCSExecutionPlanError, match="graph_digest mismatch"):
        ExecutionPlanCompiler().compile(
            plan,
            rank=0,
            tensor_size=16,
            dtype="float32",
            executor="sm",
        )


def test_direct_algorithm_round_trips_through_wire_v1():
    plan = OCSExecutionPlan.load(EXAMPLE_PATH).barrier_plan(0)

    decoded = OCSControlMessage.decode(build_ready(plan, src_rank=0).encode())

    assert decoded.algorithm == "direct"
    assert decoded.route_plan_id == plan.route_plan_id
    assert decoded.matches_plan(plan)
