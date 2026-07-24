"""MSCCL XML compatibility boundary and Execution Plan integration tests."""

from pathlib import Path
from xml.etree import ElementTree

import pytest

from pccl import (
    Compiler,
    ExecutionPlanCompiler,
    MSCCLCompatibilityError,
    MSCCLStepType,
    MSCCLXMLAlgorithm,
    OCSExecutionPlan,
    OCSExecutionPlanError,
    PrimitiveOpType,
)
from pccl.dsl.codegen import RuntimeGraphGenerator


FIXTURES = Path(__file__).parent / "fixtures"
RING = FIXTURES / "msccl_ring_allreduce_2.xml"
FUSED_RING = FIXTURES / "msccl_ring_allreduce_2_fused.xml"
FUSED_RING_4 = FIXTURES / "msccl_ring_allreduce_4_fused.xml"
ALLTOALL = FIXTURES / "msccl_direct_alltoall_2.xml"


def _three_phase_plan():
    phase_specs = [
        (0, "allreduce", "ring", 10, "msccl:ring2", 11),
        (1, "alltoall", "direct", 11, "msccl:direct-a2a-2", 12),
        (2, "allreduce", "ring", 12, "msccl:ring2", 10),
    ]
    phases = []
    for phase_id, op_type, algorithm, topology, artifact, next_topology in phase_specs:
        next_phase = (phase_id + 1) % len(phase_specs)
        phases.append(
            {
                "phase_id": phase_id,
                "epoch": 100 + phase_id,
                "op_type": op_type,
                "algorithm_type": algorithm,
                "backend": "pccl",
                "topology_id": topology,
                "artifact_id": artifact,
                "graph_digest": None,
                "barrier_after": {
                    "barrier_id": 500 + phase_id,
                    "next_epoch": 101 + phase_id,
                    "next_phase_id": next_phase,
                    "switch_action": "APPLY_ROUTE",
                    "route_plan": {
                        "route_plan_id": "route-{}".format(phase_id),
                        "route_mode": "STATIC_PLAN",
                        "source_topology_id": topology,
                        "target_topology_id": next_topology,
                        "payload": {"cross_connects": [[2, 5]]},
                    },
                },
            }
        )
    return OCSExecutionPlan.from_dict(
        {
            "schema_version": "ocs-pccl.execution-plan.v1",
            "job_id": "msccl-compat-test",
            "plan_id": "three-phase-0",
            "group_id": 9,
            "rank_list": [2, 5],
            "participant_bitmap": "0x0000000000000024",
            "phases": phases,
        }
    )


def test_parses_official_msccl_xml_shape_and_has_canonical_digest():
    algorithm = MSCCLXMLAlgorithm.load(RING)
    reformatted = MSCCLXMLAlgorithm.from_xml(RING.read_text(encoding="utf-8").replace("><", ">\n<"))

    assert algorithm.name == "ring2"
    assert algorithm.protocol == "Simple"
    assert algorithm.num_channels == 1
    assert algorithm.chunks_per_loop == 2
    assert algorithm.num_gpus == 2
    assert algorithm.collective == "allreduce"
    assert algorithm.inplace is True
    assert algorithm.digest == reformatted.digest
    assert algorithm.digest.startswith("sha256:")
    assert algorithm.gpus[0].threadblocks[0].steps[0].op_type is MSCCLStepType.RECV_REDUCE_COPY


def test_ring_lowering_preserves_transfer_signals_and_cross_tb_dependency():
    algorithm = MSCCLXMLAlgorithm.load(RING)
    graphs = [
        algorithm.lower(rank, tensor_size=1024, dtype="float32", executor="tma")
        for rank in range(2)
    ]

    sends = set()
    receives = set()
    for rank, graph in enumerate(graphs):
        graph.validate()
        for node in graph.nodes.values():
            if node.op_type is PrimitiveOpType.NOTIFY:
                sends.add((rank, node.target_rank, node.signal_id))
            elif node.op_type is PrimitiveOpType.WAIT_NOTIFY:
                receives.add((node.source_rank, rank, node.signal_id))
    assert sends == receives
    assert all(graph.size() == 6 for graph in graphs)
    assert all(graph.topological_sort()[0].op_type is PrimitiveOpType.NOTIFY for graph in graphs)

    rank0 = graphs[0]
    notify_ids = {
        node.op_id for node in rank0.nodes.values() if node.op_type is PrimitiveOpType.NOTIFY
    }
    assert any(
        notify_ids.intersection(node.dependencies)
        for node in rank0.nodes.values()
        if node.op_type is PrimitiveOpType.WAIT_NOTIFY
    )


def test_lowered_msccl_graph_compiles_to_runtime_json_v2():
    graph = MSCCLXMLAlgorithm.load(RING).lower(
        rank=0,
        tensor_size=1024,
        dtype="float32",
        executor="tma",
    )
    manifest = RuntimeGraphGenerator().generate(Compiler().compile(graph))

    assert manifest["version"] == 2
    assert manifest["collective_type"] == "allreduce"
    assert {operation["primitive"] for operation in manifest["operations"]} == {
        "notify",
        "wait_notify",
        "tma.reduce",
        "tma.copy",
    }


def test_nop_retains_an_executable_dependency_position():
    xml = """
    <algo name="local_nop" proto="Simple" nchannels="1" nchunksperloop="1"
          ngpus="1" coll="custom" inplace="1">
      <gpu id="0" i_chunks="1" o_chunks="0" s_chunks="0">
        <tb id="0" send="-1" recv="-1" chan="0">
          <step s="0" type="cpy" srcbuf="i" srcoff="0" dstbuf="i" dstoff="0"
                cnt="1" depid="-1" deps="-1" hasdep="0"/>
          <step s="1" type="nop" depid="-1" deps="-1" hasdep="0"/>
        </tb>
      </gpu>
    </algo>
    """
    graph = MSCCLXMLAlgorithm.from_xml(xml).lower(
        rank=0, tensor_size=16, dtype="float32", executor="sm"
    )

    noops = graph.get_nodes_by_type(PrimitiveOpType.NOOP)
    assert len(noops) == 1
    assert len(noops[0].dependencies) == 1
    manifest = RuntimeGraphGenerator().generate(Compiler().compile(graph))
    assert "noop" in {operation["primitive"] for operation in manifest["operations"]}


def test_direct_alltoall_maps_msccl_output_buffer_to_pccl_scratch_layout():
    graph = MSCCLXMLAlgorithm.load(ALLTOALL).lower(
        rank=0,
        tensor_size=1024,
        dtype="float32",
        executor="tma",
    )

    remote_copies = graph.get_nodes_by_type(PrimitiveOpType.SM_COPY)
    local_copies = graph.get_nodes_by_type(PrimitiveOpType.TMA_COPY)
    assert len(remote_copies) == 1
    assert remote_copies[0].source_rank == 1
    assert remote_copies[0].src_offset == 0
    assert remote_copies[0].dst_offset == 1024 + 512
    assert len(local_copies) == 1
    assert local_copies[0].source_rank == 0
    assert local_copies[0].src_offset == local_copies[0].dst_offset == 0


def test_official_fused_ring_is_decomposed_with_matched_signals():
    algorithm = MSCCLXMLAlgorithm.load(FUSED_RING)
    graphs = [
        algorithm.lower(rank, tensor_size=1024, dtype="float32", executor="sm") for rank in range(2)
    ]

    assert algorithm.gpus[0].threadblocks[0].steps[1].op_type is (
        MSCCLStepType.RECV_REDUCE_COPY_SEND
    )
    sends = {
        (rank, node.target_rank, node.signal_id)
        for rank, graph in enumerate(graphs)
        for node in graph.nodes.values()
        if node.op_type is PrimitiveOpType.NOTIFY
    }
    receives = {
        (node.source_rank, rank, node.signal_id)
        for rank, graph in enumerate(graphs)
        for node in graph.nodes.values()
        if node.op_type is PrimitiveOpType.WAIT_NOTIFY
    }
    assert sends == receives
    assert all(graph.size() == 6 for graph in graphs)
    assert all(len(graph.get_nodes_by_type(PrimitiveOpType.SM_REDUCE)) == 1 for graph in graphs)


def test_official_four_rank_fused_ring_covers_all_pipeline_instructions():
    algorithm = MSCCLXMLAlgorithm.load(FUSED_RING_4)
    graphs = [
        algorithm.lower(rank, tensor_size=1024, dtype="float32", executor="sm") for rank in range(4)
    ]

    fused_types = {
        step.op_type
        for gpu in algorithm.gpus
        for threadblock in gpu.threadblocks
        for step in threadblock.steps
        if step.op_type
        in {
            MSCCLStepType.RECV_COPY_SEND,
            MSCCLStepType.RECV_REDUCE_SEND,
            MSCCLStepType.RECV_REDUCE_COPY_SEND,
        }
    }
    assert fused_types == {
        MSCCLStepType.RECV_COPY_SEND,
        MSCCLStepType.RECV_REDUCE_SEND,
        MSCCLStepType.RECV_REDUCE_COPY_SEND,
    }

    sends = {
        (rank, node.target_rank, node.signal_id)
        for rank, graph in enumerate(graphs)
        for node in graph.nodes.values()
        if node.op_type is PrimitiveOpType.NOTIFY
    }
    receives = {
        (node.source_rank, rank, node.signal_id)
        for rank, graph in enumerate(graphs)
        for node in graph.nodes.values()
        if node.op_type is PrimitiveOpType.WAIT_NOTIFY
    }
    assert sends == receives
    assert all(graph.size() == 18 for graph in graphs)
    assert all(len(graph.get_nodes_by_type(PrimitiveOpType.SM_REDUCE)) == 3 for graph in graphs)
    assert all(len(graph.get_nodes_by_type(PrimitiveOpType.SM_COPY)) == 3 for graph in graphs)


@pytest.mark.parametrize(
    "instruction, data_primitive",
    [
        ("rcs", PrimitiveOpType.SM_COPY),
        ("rrs", PrimitiveOpType.SM_REDUCE),
        ("rrcs", PrimitiveOpType.SM_REDUCE),
    ],
)
def test_fused_pipeline_instructions_lower_to_wait_data_notify(instruction, data_primitive):
    xml = """
    <algo name="fused_pipeline" proto="Simple" nchannels="1"
          nchunksperloop="1" ngpus="3" coll="custom" inplace="1">
      <gpu id="0" i_chunks="1" o_chunks="0" s_chunks="0">
        <tb id="0" send="1" recv="-1" chan="0">
          <step s="0" type="s" srcbuf="i" srcoff="0" dstbuf="i" dstoff="0"
                cnt="1" depid="-1" deps="-1" hasdep="0"/>
        </tb>
      </gpu>
      <gpu id="1" i_chunks="1" o_chunks="0" s_chunks="0">
        <tb id="0" send="2" recv="0" chan="0">
          <step s="0" type="{instruction}" srcbuf="i" srcoff="0"
                dstbuf="i" dstoff="0" cnt="1" depid="-1" deps="-1"
                hasdep="0"/>
        </tb>
      </gpu>
      <gpu id="2" i_chunks="1" o_chunks="0" s_chunks="0">
        <tb id="0" send="-1" recv="1" chan="0">
          <step s="0" type="r" srcbuf="i" srcoff="0" dstbuf="i" dstoff="0"
                cnt="1" depid="-1" deps="-1" hasdep="0"/>
        </tb>
      </gpu>
    </algo>
    """.format(
        instruction=instruction
    )
    algorithm = MSCCLXMLAlgorithm.from_xml(xml)
    middle = algorithm.lower(rank=1, tensor_size=16, dtype="float32", executor="sm")

    assert [node.op_type for node in middle.topological_sort()] == [
        PrimitiveOpType.WAIT_NOTIFY,
        data_primitive,
        PrimitiveOpType.NOTIFY,
    ]


def test_fused_instruction_requires_both_threadblock_peers():
    root = ElementTree.parse(FUSED_RING).getroot()
    threadblock = root.find("./gpu[@id='0']/tb")
    assert threadblock is not None
    threadblock.set("send", "-1")
    first_step = threadblock.find("./step[@s='0']")
    assert first_step is not None
    first_step.attrib = {
        "s": "0",
        "type": "nop",
        "depid": "-1",
        "deps": "-1",
        "hasdep": "0",
    }
    xml = ElementTree.tostring(root, encoding="unicode")

    with pytest.raises(MSCCLCompatibilityError, match="requires both"):
        MSCCLXMLAlgorithm.from_xml(xml)


def test_unmatched_send_receive_is_rejected():
    xml = RING.read_text(encoding="utf-8").replace(
        '<step s="1" type="s" srcbuf="i" srcoff="0" dstbuf="i" dstoff="0" '
        'cnt="1" depid="0" deps="0" hasdep="0"/>',
        '<step s="1" type="nop" depid="0" deps="0" hasdep="0"/>',
        1,
    )

    with pytest.raises(MSCCLCompatibilityError, match="unmatched MSCCL transfer"):
        MSCCLXMLAlgorithm.from_xml(xml)


def test_dependency_cycle_is_rejected():
    xml = RING.read_text(encoding="utf-8").replace(
        'type="rrc" srcbuf="i" srcoff="1" dstbuf="i" dstoff="1" cnt="1" ' 'depid="-1" deps="-1"',
        'type="rrc" srcbuf="i" srcoff="1" dstbuf="i" dstoff="1" cnt="1" ' 'depid="0" deps="1"',
        1,
    )

    with pytest.raises(MSCCLCompatibilityError, match="contain a cycle"):
        MSCCLXMLAlgorithm.from_xml(xml)


def test_execution_plan_compiler_imports_three_msccl_phases_and_remaps_group_ranks():
    artifacts = {
        "msccl:ring2": RING,
        "msccl:direct-a2a-2": ALLTOALL,
    }
    compiled = ExecutionPlanCompiler(
        algorithm_lowering="msccl",
        artifact_resolver=artifacts.__getitem__,
    ).compile(
        _three_phase_plan(),
        rank=2,
        tensor_size=1024,
        dtype="float32",
        executor="tma",
    )

    assert [phase.graph.collective_type for phase in compiled.phases] == [
        "allreduce",
        "alltoall",
        "allreduce",
    ]
    assert [phase.barrier_after.barrier_id for phase in compiled.phases] == [500, 501, 502]
    for phase in compiled.phases:
        peers = set()
        for node in phase.graph.nodes.values():
            if hasattr(node, "source_rank"):
                peers.add(node.source_rank)
            if hasattr(node, "target_rank"):
                peers.add(node.target_rank)
        assert peers.issubset({2, 5})


def test_msccl_execution_plan_requires_artifact_and_resolver():
    plan = _three_phase_plan()

    with pytest.raises(OCSExecutionPlanError, match="artifact_resolver"):
        ExecutionPlanCompiler(algorithm_lowering="msccl").compile(plan, rank=2, tensor_size=1024)

    data = plan.to_dict()
    data["phases"][0].pop("artifact_id")
    without_artifact = OCSExecutionPlan.from_dict(data)
    with pytest.raises(OCSExecutionPlanError, match="requires artifact_id"):
        ExecutionPlanCompiler(
            algorithm_lowering="msccl", artifact_resolver=lambda _artifact_id: RING
        ).compile(without_artifact, rank=2, tensor_size=1024)


def test_msccl_execution_plan_rejects_collective_and_world_size_mismatch():
    plan = _three_phase_plan()

    with pytest.raises(OCSExecutionPlanError, match="does not match phase op_type"):
        ExecutionPlanCompiler(
            algorithm_lowering="msccl",
            artifact_resolver=lambda _artifact_id: ALLTOALL,
        ).compile(plan, rank=2, tensor_size=1024)

    xml = RING.read_text(encoding="utf-8").replace('ngpus="2"', 'ngpus="3"', 1)
    with pytest.raises(OCSExecutionPlanError, match="contains 2 <gpu> nodes"):
        ExecutionPlanCompiler(
            algorithm_lowering="msccl",
            artifact_resolver=lambda _artifact_id: xml,
        ).compile(plan, rank=2, tensor_size=1024)


def test_msccl_alltoall_rejects_layout_not_supported_by_current_engine():
    xml = ALLTOALL.read_text(encoding="utf-8").replace(
        'type="cpy" srcbuf="i" srcoff="0" dstbuf="o" dstoff="0"',
        'type="cpy" srcbuf="i" srcoff="0" dstbuf="o" dstoff="1"',
        1,
    )
    algorithm = MSCCLXMLAlgorithm.from_xml(xml)

    with pytest.raises(MSCCLCompatibilityError, match="currently requires"):
        algorithm.lower(rank=0, tensor_size=1024)
