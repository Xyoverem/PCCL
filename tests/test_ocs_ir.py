"""Tests for the OCS barrier IR and phased manifest code generation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import (
    DeviceType,
    OcsBarrierNode,
    PrimitiveOpType,
    Stream,
    build_graph,
    compile_to_json_string,
)
from pccl.dsl.compiler import Compiler


def _build_graph_with_barrier():
    def build(op):
        op.tensor(dtype="float32", shape=(4096,))
        with Stream("left"):
            op.sm_copy(source_rank=1, src_offset=0, dst_offset=0, size=2048)
        with Stream("right"):
            op.sm_copy(source_rank=2, src_offset=2048, dst_offset=2048, size=2048)

        op.ocs_barrier(
            barrier_id=7,
            epoch_id=3,
            next_epoch_id=4,
            participant_ranks=(0, 1, 2, 3),
            topology_id=11,
            route_plan_id=19,
            group_id=5,
            algorithm="rhd",
            payload=b"next-topology",
            timeout_ms=200,
        )

        with Stream("left"):
            op.sm_copy(source_rank=3, src_offset=0, dst_offset=0, size=4096)

    return build_graph("ocs_phase_graph", build, device=DeviceType.CUDA)


def test_ocs_barrier_node_has_control_semantics():
    node = OcsBarrierNode(
        barrier_id=2,
        epoch_id=8,
        next_epoch_id=9,
        participant_ranks=(0, 2, 3),
        topology_id=4,
        route_plan_id=5,
        algorithm="ring",
        payload=b"route",
    )

    assert node.op_type == PrimitiveOpType.OCS_BARRIER
    assert node.device == DeviceType.CPU
    assert node.participant_bitmap == 0b1101
    assert node.to_params()["payload_hex"] == b"route".hex()
    assert node.to_ocs_plan().backend == "pccl"
    assert node.validate() is True


def test_ocs_barrier_requires_complete_participant_scope():
    node = OcsBarrierNode(
        barrier_id=0,
        epoch_id=0,
        next_epoch_id=1,
        participant_ranks=(),
        topology_id=0,
        route_plan_id=0,
    )

    with pytest.raises(ValueError, match="participant_ranks"):
        node.validate()


def test_dsl_barrier_joins_prior_streams_and_blocks_following_phase():
    graph = _build_graph_with_barrier()
    barriers = graph.get_nodes_by_type(PrimitiveOpType.OCS_BARRIER)
    assert len(barriers) == 1
    barrier = barriers[0]

    pre_nodes = [
        node for node in graph.nodes.values()
        if node.op_type == PrimitiveOpType.SM_COPY and node.op_id != graph.exit_points[0]
    ]
    assert {node.op_id for node in pre_nodes}.issubset(set(barrier.dependencies))

    post_nodes = [
        node for node in graph.nodes.values()
        if node.op_type == PrimitiveOpType.SM_COPY and node.op_id not in barrier.dependencies
    ]
    assert len(post_nodes) == 1
    assert barrier.op_id in post_nodes[0].dependencies
    assert graph.validate() is True


def test_ocs_graph_splits_into_data_phases():
    graph = _build_graph_with_barrier()
    phases = graph.split_ocs_phases()

    assert len(phases) == 2
    assert len(phases[0].nodes) == 2
    assert phases[0].barrier is not None
    assert phases[0].barrier.barrier_id == 7
    assert len(phases[1].nodes) == 1
    assert phases[1].barrier is None


def test_ocs_graph_generates_phased_manifest():
    manifest = json.loads(compile_to_json_string(_build_graph_with_barrier()))

    assert manifest["version"] == 3
    assert manifest["execution_model"] == "phased_ocs"
    assert "operations" not in manifest
    assert len(manifest["phases"]) == 2
    assert len(manifest["phases"][0]["operations"]) == 2
    assert len(manifest["phases"][1]["operations"]) == 1
    assert manifest["control_operations"] == [{
        "index": 0,
        "primitive": "ocs.barrier",
        "executor": "host",
        "after_phase": 0,
        "before_phase": 1,
        "params": {
            "group_id": 5,
            "barrier_id": 7,
            "epoch_id": 3,
            "next_epoch_id": 4,
            "participant_ranks": [0, 1, 2, 3],
            "participant_bitmap": 15,
            "topology_id": 11,
            "route_mode": "STATIC_PLAN",
            "route_plan_id": 19,
            "algorithm": "rhd",
            "backend": "pccl",
            "payload_hex": b"next-topology".hex(),
            "timeout_ms": 200,
        },
    }]


def test_graph_rejects_barrier_that_is_not_a_global_phase_cut():
    graph = _build_graph_with_barrier()
    barrier = graph.get_nodes_by_type(PrimitiveOpType.OCS_BARRIER)[0]
    barrier.dependencies.pop()

    with pytest.raises(ValueError, match="must depend on every node"):
        graph.validate()


def test_superopt_is_rejected_for_ocs_barrier_graphs():
    compiler = Compiler(enable_superopt=True)

    with pytest.raises(ValueError, match="Superoptimization across OCS barriers"):
        compiler.compile(_build_graph_with_barrier())
