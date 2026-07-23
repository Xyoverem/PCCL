"""Tests for MSCCL-style Algorithm IR generation and PCCL lowering."""

from collections import Counter
import json
from pathlib import Path

import pytest

from pccl import (
    AlgorithmBuffer,
    AlgorithmIRBuilder,
    AlgorithmIRError,
    AlgorithmIRLowerer,
    Compiler,
    ExecutionPlanCompiler,
    OCSExecutionPlan,
    OCSExecutionPlanError,
)
from pccl.dsl.algorithms import (
    AlgorithmIRCollectives,
    RingAllreduce,
    build_direct_alltoall_ir,
    build_ring_allreduce_ir,
)
from pccl.dsl.codegen import RuntimeGraphGenerator
from pccl.dsl.nodes import PrimitiveOpType


ROOT = Path(__file__).parent.parent
EXAMPLE_PATH = ROOT / "examples" / "ocs_execution_plan_v1.json"


def _simulate(algorithm, initial):
    state = dict(initial)
    for step in algorithm.steps:
        snapshot = dict(state)
        writes = {}
        for transfer in step.transfers:
            for offset in range(transfer.src.count):
                src = (
                    transfer.src.rank,
                    transfer.src.buffer.value,
                    transfer.src.index + offset,
                )
                dst = (
                    transfer.dst.rank,
                    transfer.dst.buffer.value,
                    transfer.dst.index + offset,
                )
                if transfer.primitive.value == "copy":
                    writes[dst] = snapshot[src]
                else:
                    writes[dst] = snapshot[dst] + snapshot[src]
        state.update(writes)
    return state


def _primitive_counter(graph):
    return Counter(node.op_type for node in graph.nodes.values())


def test_ring_algorithm_ir_has_stable_digest_and_inferred_dependencies():
    first = build_ring_allreduce_ir(4)
    second = build_ring_allreduce_ir(4)

    assert first.digest == second.digest
    assert len(first.steps) == 7
    assert sum(len(step.transfers) for step in first.steps) == 24
    assert any(transfer.dependencies for step in first.steps for transfer in step.transfers)
    for step in first.steps:
        for transfer in step.transfers:
            assert all(
                int(dependency.split("_")[0][4:]) < step.index
                for dependency in transfer.dependencies
            )


def test_algorithm_ir_rejects_parallel_writes_to_the_same_chunk():
    builder = AlgorithmIRBuilder(
        name="invalid",
        collective_type="custom",
        world_size=2,
        chunks_per_rank=2,
    )
    step = builder.step("conflict")
    destination = builder.chunk(1, 0, AlgorithmBuffer.OUTPUT)
    step.copy(builder.chunk(0, 0), destination)
    step.copy(builder.chunk(0, 1), destination)

    with pytest.raises(AlgorithmIRError, match="writes chunk .* more than once"):
        builder.build()


def test_algorithm_ir_rejects_parallel_read_write_hazard():
    builder = AlgorithmIRBuilder(
        name="invalid",
        collective_type="custom",
        world_size=2,
        chunks_per_rank=2,
    )
    step = builder.step("conflict")
    shared = builder.chunk(0, 0)
    step.copy(shared, builder.chunk(1, 0))
    step.copy(builder.chunk(1, 1), shared)

    with pytest.raises(AlgorithmIRError, match="reads and writes chunk .* in parallel"):
        builder.build()


def test_algorithm_ir_rejects_non_integer_world_size():
    builder = AlgorithmIRBuilder(
        name="invalid",
        collective_type="custom",
        world_size=2.5,
        chunks_per_rank=2,
    )
    builder.step("sync").sync(0, 1)

    with pytest.raises(AlgorithmIRError, match="world_size"):
        builder.build()


def test_algorithm_ir_lowerer_rejects_boolean_signal_base():
    with pytest.raises(AlgorithmIRError, match="signal_base"):
        AlgorithmIRLowerer(signal_base=True)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_generated_ring_allreduce_schedule_is_semantically_correct(world_size):
    algorithm = build_ring_allreduce_ir(world_size)
    initial = {
        (rank, "input", chunk): rank + 1
        for rank in range(world_size)
        for chunk in range(world_size)
    }

    result = _simulate(algorithm, initial)

    expected = sum(range(1, world_size + 1))
    assert all(
        result[(rank, "input", chunk)] == expected
        for rank in range(world_size)
        for chunk in range(world_size)
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_generated_direct_alltoall_schedule_is_semantically_correct(world_size):
    algorithm = build_direct_alltoall_ir(world_size)
    initial = {
        (rank, "input", chunk): rank * 100 + chunk
        for rank in range(world_size)
        for chunk in range(world_size)
    }
    initial.update(
        {(rank, "scratch", chunk): -1 for rank in range(world_size) for chunk in range(world_size)}
    )

    result = _simulate(algorithm, initial)

    for dst_rank in range(world_size):
        for src_rank in range(world_size):
            if src_rank == dst_rank:
                # PCCL keeps the local contribution in the in-place input slot.
                assert result[(dst_rank, "input", dst_rank)] == dst_rank * 100 + dst_rank
            else:
                assert result[(dst_rank, "scratch", src_rank)] == src_rank * 100 + dst_rank


@pytest.mark.parametrize("world_size,tensor_size", [(2, 8), (4, 16), (8, 64)])
def test_generated_ring_lowers_to_same_primitive_shape_as_template(world_size, tensor_size):
    generated = AlgorithmIRCollectives().build_allreduce(
        rank=0,
        world_size=world_size,
        tensor_size=tensor_size,
        dtype="float32",
        executor="sm",
    )
    template = RingAllreduce().build_allreduce(
        rank=0,
        world_size=world_size,
        tensor_size=tensor_size,
        dtype="float32",
        executor="sm",
    )

    assert generated.size() == template.size()
    assert _primitive_counter(generated) == _primitive_counter(template)
    generated_data = [
        (
            node.op_type,
            node.source_rank,
            node.src_offset,
            node.dst_offset,
            getattr(node, "remote_offset", None),
        )
        for node in generated.topological_sort()
        if node.op_type in {PrimitiveOpType.SM_COPY, PrimitiveOpType.SM_REDUCE}
    ]
    template_data = [
        (
            node.op_type,
            node.source_rank,
            node.src_offset,
            node.dst_offset,
            getattr(node, "remote_offset", None),
        )
        for node in template.topological_sort()
        if node.op_type in {PrimitiveOpType.SM_COPY, PrimitiveOpType.SM_REDUCE}
    ]
    assert generated_data == template_data


def test_generated_alltoall_lowers_to_template_shape_and_uses_sm_for_scratch():
    generated = AlgorithmIRCollectives().build_alltoall(
        rank=0,
        world_size=4,
        tensor_size=16,
        dtype="float32",
        executor="tma",
    )
    template = RingAllreduce().build_alltoall(
        rank=0,
        world_size=4,
        tensor_size=16,
        dtype="float32",
        executor="tma",
    )

    assert generated.size() == template.size()
    assert _primitive_counter(generated) == _primitive_counter(template)
    copies = generated.get_nodes_by_type(PrimitiveOpType.SM_COPY)
    assert len(copies) == 3
    assert not generated.get_nodes_by_type(PrimitiveOpType.TMA_COPY)
    assert [node.dst_offset for node in copies] == [20, 24, 28]


def test_algorithm_ir_lowering_compiles_to_runtime_json_v2():
    algorithm = build_ring_allreduce_ir(4)
    graph = AlgorithmIRLowerer().lower(
        algorithm,
        rank=0,
        tensor_size=16,
        dtype="float32",
        executor="tma",
    )

    manifest = RuntimeGraphGenerator().generate(Compiler().compile(graph))

    assert manifest["version"] == 2
    assert manifest["collective_type"] == "allreduce"
    assert manifest["operations"]
    assert {operation["primitive"] for operation in manifest["operations"]}.issuperset(
        {
            "notify",
            "wait_notify",
            "tma.reduce",
            "tma.copy",
        }
    )


def test_execution_plan_compiler_generated_mode_builds_three_phases():
    data = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    data["phases"][2]["algorithm_type"] = "ring"
    plan = OCSExecutionPlan.from_dict(data)

    compiled = ExecutionPlanCompiler(algorithm_lowering="generated").compile(
        plan,
        rank=0,
        tensor_size=16,
        dtype="float32",
        executor="sm",
    )

    assert [phase.graph.collective_type for phase in compiled.phases] == [
        "allreduce",
        "alltoall",
        "allreduce",
    ]
    assert [phase.barrier_after.algorithm for phase in compiled.phases] == [
        "direct",
        "ring",
        "ring",
    ]
    assert all(phase.graph.size() > 0 for phase in compiled.phases)


def test_execution_plan_compiler_generated_mode_rejects_missing_lowering():
    plan = OCSExecutionPlan.load(EXAMPLE_PATH)

    with pytest.raises(OCSExecutionPlanError, match="does not support phase 2"):
        ExecutionPlanCompiler(algorithm_lowering="generated").compile(
            plan,
            rank=0,
            tensor_size=16,
            dtype="float32",
            executor="sm",
        )
