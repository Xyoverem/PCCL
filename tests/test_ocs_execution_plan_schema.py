"""Contract tests for the versioned OCS execution-plan schema and example."""

import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
SCHEMA_PATH = ROOT / "schemas" / "ocs_execution_plan_v1.schema.json"
EXAMPLE_PATH = ROOT / "examples" / "ocs_execution_plan_v1.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_execution_plan_schema_freezes_required_controller_fields():
    schema = _load(SCHEMA_PATH)
    phase_required = set(schema["$defs"]["phase"]["required"])
    barrier_required = set(schema["$defs"]["barrier"]["required"])

    assert schema["properties"]["schema_version"]["const"] == ("ocs-pccl.execution-plan.v1")
    assert set(schema["required"]) == {
        "schema_version",
        "job_id",
        "plan_id",
        "group_id",
        "rank_list",
        "participant_bitmap",
        "phases",
    }
    assert {
        "phase_id",
        "epoch",
        "op_type",
        "algorithm_type",
        "topology_id",
    }.issubset(phase_required)
    assert {
        "barrier_id",
        "next_epoch",
        "switch_action",
        "route_plan",
    }.issubset(barrier_required)


def test_execution_plan_example_is_a_three_phase_closed_epoch_cycle():
    plan = _load(EXAMPLE_PATH)
    phases = plan["phases"]
    ranks = plan["rank_list"]

    bitmap = sum(1 << rank for rank in ranks)
    assert int(plan["participant_bitmap"], 16) == bitmap
    assert [phase["phase_id"] for phase in phases] == [0, 1, 2]
    assert [phase["op_type"] for phase in phases] == [
        "allreduce",
        "alltoall",
        "allreduce",
    ]
    assert [phase["epoch"] for phase in phases] == [100, 101, 102]

    barriers = [phase["barrier_after"] for phase in phases]
    assert [barrier["barrier_id"] for barrier in barriers] == [4200, 4201, 4202]
    assert [barrier["next_epoch"] for barrier in barriers] == [101, 102, 103]
    assert [barrier["next_phase_id"] for barrier in barriers] == [1, 2, 0]

    for index, phase in enumerate(phases):
        route = phase["barrier_after"]["route_plan"]
        assert route["source_topology_id"] == phase["topology_id"]
        if index + 1 < len(phases):
            assert route["target_topology_id"] == phases[index + 1]["topology_id"]

    assert barriers[-1]["route_plan"]["target_topology_id"] == phases[0]["topology_id"]
