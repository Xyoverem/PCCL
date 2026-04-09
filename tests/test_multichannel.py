"""Tests for multi-channel superoptimizer support.

Covers:
  1. IR channel field persistence (serialization, codegen)
  2. Channel-aware cost model
  3. Auto-channelization pass
  4. Structural matching through sync ops
  5. Multi-channel enumerator
  6. Multi-channel rule discovery
  7. 16-GPU integration with channelization
"""

import pytest
from pccl.dsl.graph import PrimitiveIRGraph
from pccl.dsl.nodes import (
    PrimitiveOpType,
    SmCopyNode, TmaCopyNode, CeCopyNode,
    SmReduceNode, TmaReduceNode,
    MultimemReduceNode, MultimemStoreNode,
    RdmaWriteNode, RdmaReadNode,
    NotifyNode, WaitNotifyNode,
    TensorInfo, ExecutorType, VALID_DTYPES,
)
from pccl.dsl.superopt.rule import PatternNode, PatternEdge, RewriteRule
from pccl.dsl.superopt.enumerator import (
    enumerate_skeletons, LinkType, _channel_assignments,
)
from pccl.dsl.superopt.cost_model import (
    GpuProfile, H100_PROFILE, critical_path_cost, op_latency,
)
from pccl.dsl.superopt.pass_ import SuperoptPass
from pccl.dsl.superopt.verifier import (
    check_equivalence, signatures_compatible, concrete_simulation_match,
)
from pccl.dsl.superopt import discover_rules
from pccl.dsl.passes.channelize import ChannelizePass
from pccl.dsl.compiler import Compiler
from pccl.dsl.codegen import RuntimeGraphGenerator

from pccl.dsl.pipeline import Pipeline
from pccl.dsl.decorators import CommunicationOp, Stream
from pccl.dsl.codegen.mapping import _EXECUTOR_NAME_MAP


def count_ops_by_type(graph):
    counts = {}
    for node in graph.nodes.values():
        ot = node.op_type.value
        counts[ot] = counts.get(ot, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# 1. IR channel field
# ---------------------------------------------------------------------------

class TestIRChannel:
    def test_irnode_default_channel(self):
        n = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        assert n.channel == 0

    def test_irnode_custom_channel(self):
        n = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        n.channel = 2
        assert n.channel == 2

    def test_pattern_node_default_channel(self):
        pn = PatternNode(op_type=PrimitiveOpType.SM_COPY, rank=0)
        assert pn.channel == 0

    def test_pattern_node_custom_channel(self):
        pn = PatternNode(op_type=PrimitiveOpType.SM_COPY, rank=0, channel=3)
        assert pn.channel == 3

    def test_pattern_node_serialization(self):
        pn = PatternNode(op_type=PrimitiveOpType.SM_COPY, rank=0, channel=2)
        d = pn.to_dict()
        assert d["channel"] == 2
        restored = PatternNode.from_dict(d)
        assert restored.channel == 2
        assert restored == pn

    def test_rewrite_rule_serialization_with_channel(self):
        rule = RewriteRule(
            source_nodes=[PatternNode(PrimitiveOpType.SM_COPY, 0, channel=0)],
            source_edges=[],
            replacement_nodes=[
                PatternNode(PrimitiveOpType.SM_COPY, 0, channel=0),
                PatternNode(PrimitiveOpType.SM_COPY, 0, channel=1),
            ],
            replacement_edges=[],
            rule_id="test_ch",
        )
        d = rule.to_dict()
        restored = RewriteRule.from_dict(d)
        assert restored.replacement_nodes[0].channel == 0
        assert restored.replacement_nodes[1].channel == 1

    def test_codegen_includes_channel(self):
        g = PrimitiveIRGraph(graph_name="ch_test")
        n1 = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        n1.channel = 0
        n2 = SmCopyNode(source_rank=1, src_offset=4096, dst_offset=4096, size=4096)
        n2.channel = 1
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(n1.op_id, n2.op_id)

        gen = RuntimeGraphGenerator()
        result = gen.generate(g)
        ops = result["operations"]
        assert ops[0]["channel"] == 0
        assert ops[1]["channel"] == 1


# ---------------------------------------------------------------------------
# 2. Channel-aware cost model
# ---------------------------------------------------------------------------

class TestChannelCostModel:
    def test_single_channel_cost_unchanged(self):
        nodes = [
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=0, channel=0),
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=0, channel=0),
        ]
        edges = [PatternEdge(0, 1)]
        cost = critical_path_cost(nodes, edges, H100_PROFILE, 4096)
        single_op = op_latency(PrimitiveOpType.SM_COPY, H100_PROFILE, 4096)
        assert abs(cost - 2 * single_op) < 0.01

    def test_two_channel_parallel_cheaper(self):
        seq_nodes = [
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=0, channel=0),
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=1, channel=0),
        ]
        seq_edges = [PatternEdge(0, 1)]
        seq_cost = critical_path_cost(seq_nodes, seq_edges, H100_PROFILE, 4096)

        par_nodes = [
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=0, channel=0),
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=1, channel=1),
        ]
        par_edges = []
        par_cost = critical_path_cost(par_nodes, par_edges, H100_PROFILE, 4096)

        assert par_cost < seq_cost

    def test_channel_launch_overhead_applied(self):
        par_nodes = [
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=0, channel=0),
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=1, channel=1),
        ]
        par_edges = []
        cost = critical_path_cost(par_nodes, par_edges, H100_PROFILE, 4096)

        single_op = op_latency(PrimitiveOpType.SM_COPY, H100_PROFILE, 4096)
        assert cost > single_op

    def test_three_channels(self):
        nodes = [
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=i, channel=i)
            for i in range(3)
        ]
        edges = []
        cost = critical_path_cost(nodes, edges, H100_PROFILE, 4096)
        single_op = op_latency(PrimitiveOpType.SM_COPY, H100_PROFILE, 4096)
        overhead = H100_PROFILE.channel_launch_overhead_us * 2
        assert abs(cost - (single_op + overhead)) < 0.01

    def test_profile_serialization_with_channel_fields(self):
        p = GpuProfile(num_channels=4, channel_launch_overhead_us=0.3)
        d = p.to_dict()
        assert d["num_channels"] == 4
        assert d["channel_launch_overhead_us"] == 0.3
        restored = GpuProfile.from_dict(d)
        assert restored.num_channels == 4
        assert restored.channel_launch_overhead_us == 0.3


# ---------------------------------------------------------------------------
# 3. Auto-channelization pass
# ---------------------------------------------------------------------------

class TestChannelizePass:
    def _build_contiguous_copy_chain(self, n, chunk_size=4096):
        g = PrimitiveIRGraph(graph_name="contiguous_chain")
        nodes = []
        for i in range(n):
            node = SmCopyNode(
                source_rank=1,
                src_offset=i * chunk_size,
                dst_offset=i * chunk_size,
                size=chunk_size,
            )
            g.add_node(node)
            nodes.append(node)
        for i in range(n - 1):
            g.add_edge(nodes[i].op_id, nodes[i + 1].op_id)
        return g

    def test_channelizes_contiguous_chain(self):
        g = self._build_contiguous_copy_chain(4)
        assert all(n.channel == 0 for n in g.nodes.values())

        ChannelizePass(num_channels=3).run(g)

        channels_used = set(n.channel for n in g.nodes.values())
        assert len(channels_used) >= 2

    def test_does_not_channelize_small_data(self):
        g = PrimitiveIRGraph(graph_name="small_data")
        n1 = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=64)
        n2 = SmCopyNode(source_rank=1, src_offset=64, dst_offset=64, size=64)
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(n1.op_id, n2.op_id)

        ChannelizePass(num_channels=2, min_data_size=4096).run(g)
        assert all(n.channel == 0 for n in g.nodes.values())

    def test_channelizes_four_ops_into_two_channels(self):
        g = self._build_contiguous_copy_chain(6)
        ChannelizePass(num_channels=2).run(g)

        ch_counts = {}
        for n in g.nodes.values():
            ch_counts[n.channel] = ch_counts.get(n.channel, 0) + 1
        assert ch_counts[0] == 3
        assert ch_counts[1] == 3

    def test_graph_valid_after_channelization(self):
        g = self._build_contiguous_copy_chain(4)
        ChannelizePass(num_channels=2).run(g)
        g.validate()

    def test_non_contiguous_not_merged(self):
        g = PrimitiveIRGraph(graph_name="non_contiguous")
        n1 = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        n2 = SmCopyNode(source_rank=1, src_offset=8192, dst_offset=8192, size=4096)
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(n1.op_id, n2.op_id)

        ChannelizePass(num_channels=2).run(g)
        assert all(n.channel == 0 for n in g.nodes.values())


# ---------------------------------------------------------------------------
# 4. Structural matching through sync ops
# ---------------------------------------------------------------------------

class TestSyncOpAbstraction:
    def test_match_through_notify_wait(self):
        g = PrimitiveIRGraph(graph_name="sync_between")
        c1 = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        notify = NotifyNode(signal_id=0, target_rank=1)
        wait = WaitNotifyNode(signal_id=0, source_rank=1)
        c2 = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        g.add_node(c1)
        g.add_node(notify)
        g.add_node(wait)
        g.add_node(c2)
        g.add_edge(c1.op_id, notify.op_id)
        g.add_edge(notify.op_id, wait.op_id)
        g.add_edge(wait.op_id, c2.op_id)

        rules = discover_rules(k=2, timeout_per_pair=0.5,
                               link_type=LinkType.INTRA, progress=False)
        shrinking = [r for r in rules if len(r.replacement_nodes) < len(r.source_nodes)]

        initial_data_ops = sum(
            1 for n in g.nodes.values()
            if n.op_type in {PrimitiveOpType.SM_COPY, PrimitiveOpType.TMA_COPY}
        )
        assert initial_data_ops == 2

        p = SuperoptPass(rules=shrinking, data_size_hint=4096)
        p.run(g)

        final_data_ops = sum(
            1 for n in g.nodes.values()
            if n.op_type in {PrimitiveOpType.SM_COPY, PrimitiveOpType.TMA_COPY,
                             PrimitiveOpType.CE_COPY}
        )
        assert final_data_ops <= initial_data_ops

    def test_sync_ops_preserved_after_matching(self):
        g = PrimitiveIRGraph(graph_name="sync_preserved")
        c1 = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        notify = NotifyNode(signal_id=0, target_rank=1)
        c2 = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        g.add_node(c1)
        g.add_node(notify)
        g.add_node(c2)
        g.add_edge(c1.op_id, notify.op_id)
        g.add_edge(notify.op_id, c2.op_id)

        rules = discover_rules(k=2, timeout_per_pair=0.5,
                               link_type=LinkType.INTRA, progress=False)
        shrinking = [r for r in rules if len(r.replacement_nodes) < len(r.source_nodes)]
        SuperoptPass(rules=shrinking, data_size_hint=4096).run(g)

        notify_count = sum(
            1 for n in g.nodes.values()
            if n.op_type == PrimitiveOpType.NOTIFY
        )
        assert notify_count >= 1


# ---------------------------------------------------------------------------
# 5. Multi-channel enumerator
# ---------------------------------------------------------------------------

class TestMultiChannelEnumerator:
    def test_single_channel_backward_compatible(self):
        skels_old = list(enumerate_skeletons(2, LinkType.INTRA, max_channels=1))
        for nodes, edges in skels_old:
            assert all(n.channel == 0 for n in nodes)

    def test_multi_channel_generates_more_patterns(self):
        sc = list(enumerate_skeletons(2, LinkType.INTRA, max_channels=1))
        mc = list(enumerate_skeletons(2, LinkType.INTRA, max_channels=2))
        assert len(mc) > len(sc)

    def test_multi_channel_patterns_have_multiple_channels(self):
        mc = list(enumerate_skeletons(2, LinkType.INTRA, max_channels=2))
        multi_ch = [
            (n, e) for n, e in mc
            if len(set(nd.channel for nd in n)) > 1
        ]
        assert len(multi_ch) > 0

    def test_channel_assignments(self):
        assigns = _channel_assignments(3, 2)
        assert [0, 0, 0] in assigns
        multi = [a for a in assigns if max(a) > 0]
        assert len(multi) > 0
        for a in assigns:
            assert max(a) < 2

    def test_channel_assignments_single(self):
        assigns = _channel_assignments(3, 1)
        assert len(assigns) == 1
        assert assigns[0] == [0, 0, 0]


# ---------------------------------------------------------------------------
# 6. Multi-channel rule discovery
# ---------------------------------------------------------------------------

class TestMultiChannelDiscovery:
    def test_discover_with_max_channels_1_same_as_before(self):
        rules_1ch = discover_rules(
            k=2, timeout_per_pair=0.5,
            link_type=LinkType.INTRA, progress=False, max_channels=1,
        )
        rules_default = discover_rules(
            k=2, timeout_per_pair=0.5,
            link_type=LinkType.INTRA, progress=False,
        )
        assert len(rules_1ch) == len(rules_default)

    def test_discover_with_multi_channel_finds_more_rules(self):
        rules_1ch = discover_rules(
            k=2, timeout_per_pair=0.5,
            link_type=LinkType.INTRA, progress=False, max_channels=1,
        )
        rules_2ch = discover_rules(
            k=2, timeout_per_pair=1.0,
            link_type=LinkType.INTRA, progress=False, max_channels=2,
        )
        assert len(rules_2ch) >= len(rules_1ch)

    def test_channelization_rules_have_multi_channel_replacement(self):
        rules = discover_rules(
            k=2, timeout_per_pair=1.0,
            link_type=LinkType.INTRA, progress=False, max_channels=2,
        )
        ch_rules = [
            r for r in rules
            if r.rule_id.startswith("channelize_")
        ]
        for r in ch_rules:
            channels = set(n.channel for n in r.replacement_nodes)
            assert len(channels) > 1


# ---------------------------------------------------------------------------
# 7. Compiler integration with channelization
# ---------------------------------------------------------------------------

class TestCompilerChannelize:
    def _build_contiguous_copies(self, n=4, chunk_size=4096):
        g = PrimitiveIRGraph(graph_name="compiler_ch_test")
        nodes = []
        for i in range(n):
            node = SmCopyNode(
                source_rank=1,
                src_offset=i * chunk_size,
                dst_offset=i * chunk_size,
                size=chunk_size,
            )
            g.add_node(node)
            nodes.append(node)
        for i in range(n - 1):
            g.add_edge(nodes[i].op_id, nodes[i + 1].op_id)
        return g

    def test_compiler_channelize_flag(self):
        g = self._build_contiguous_copies(6)
        compiler = Compiler(enable_channelize=True, num_channels=2)
        compiled = compiler.compile(g)
        channels = set(n.channel for n in compiled.nodes.values())
        assert len(channels) == 2

    def test_compiler_without_channelize(self):
        g = self._build_contiguous_copies(4)
        compiler = Compiler(enable_channelize=False)
        compiled = compiler.compile(g)
        assert all(n.channel == 0 for n in compiled.nodes.values())

    def test_superopt_then_channelize(self):
        g = self._build_contiguous_copies(4)
        compiler = Compiler(
            enable_superopt=True,
            enable_channelize=True,
            num_channels=2,
        )
        compiled = compiler.compile(g)
        compiled.validate()


# ---------------------------------------------------------------------------
# 8. Verifier multi-channel
# ---------------------------------------------------------------------------

class TestVerifierMultiChannel:
    def test_same_pattern_different_channels_equivalent(self):
        src = [PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=0, channel=0)]
        src_e = []
        repl = [PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=0, channel=1)]
        repl_e = []
        assert concrete_simulation_match(src, src_e, repl, repl_e)

    def test_signatures_compatible_across_channels(self):
        src = [
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=0, channel=0),
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=1, channel=0),
        ]
        repl = [
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=0, channel=0),
            PatternNode(PrimitiveOpType.SM_COPY, 0, chunk_id=1, channel=1),
        ]
        assert signatures_compatible(src, repl)


# ---------------------------------------------------------------------------
# 9. Multimem (NVLS) DSL nodes
# ---------------------------------------------------------------------------

class TestMultimem:
    def test_multimem_reduce_node_defaults(self):
        n = MultimemReduceNode(source_rank=0, src_offset=0, dst_offset=0,
                               remote_offset=0, count=1024)
        assert n.op_type == PrimitiveOpType.MULTIMEM_REDUCE
        assert n.executor == ExecutorType.MULTIMEM
        assert n.reduce_op == "sum"

    def test_multimem_store_node_defaults(self):
        n = MultimemStoreNode(source_rank=0, src_offset=0, dst_offset=0, size=4096)
        assert n.op_type == PrimitiveOpType.MULTIMEM_STORE
        assert n.executor == ExecutorType.MULTIMEM

    def test_multimem_reduce_params(self):
        n = MultimemReduceNode(source_rank=1, src_offset=10, dst_offset=20,
                               remote_offset=30, count=500, reduce_op="sum")
        p = n.to_params()
        assert p["reduce_op"] == "sum"
        assert p["source_rank"] == 1
        assert p["count"] == 500

    def test_multimem_store_params(self):
        n = MultimemStoreNode(source_rank=2, src_offset=100, dst_offset=200, size=8192)
        p = n.to_params()
        assert p["source_rank"] == 2
        assert p["size"] == 8192

    def test_multimem_reduce_validation(self):
        n = MultimemReduceNode(source_rank=0, src_offset=0, dst_offset=0,
                               remote_offset=0, count=100)
        assert n.validate()
        with pytest.raises(ValueError):
            MultimemReduceNode(source_rank=-1, src_offset=0, dst_offset=0,
                               remote_offset=0, count=100).validate()

    def test_multimem_store_validation(self):
        n = MultimemStoreNode(source_rank=0, src_offset=0, dst_offset=0, size=100)
        assert n.validate()
        with pytest.raises(ValueError):
            MultimemStoreNode(source_rank=-1, src_offset=0, dst_offset=0, size=100).validate()

    def test_multimem_codegen_executor(self):
        g = PrimitiveIRGraph(graph_name="multimem_test")
        n1 = MultimemReduceNode(source_rank=0, src_offset=0, dst_offset=0,
                                remote_offset=0, count=1024)
        n2 = MultimemStoreNode(source_rank=0, src_offset=0, dst_offset=0, size=1024)
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(n1.op_id, n2.op_id)

        gen = RuntimeGraphGenerator()
        result = gen.generate(g)
        assert "cuda_multimem" in result["executors"]
        assert result["operations"][0]["primitive"] == "multimem.reduce"
        assert result["operations"][0]["executor"] == "cuda_multimem"
        assert result["operations"][1]["primitive"] == "multimem.store"

    def test_multimem_via_decorator(self):
        with CommunicationOp(name="mm_dec") as op:
            op.tensor(dtype="bfloat16", shape=(1024,))
            r = op.multimem_reduce(source_rank=0, src_offset=0, dst_offset=0,
                                   remote_offset=0, count=512)
            s = op.multimem_store(source_rank=0, src_offset=0, dst_offset=0,
                                  size=512)
            g = op.get_graph()

        assert r.op_type == PrimitiveOpType.MULTIMEM_REDUCE
        assert s.op_type == PrimitiveOpType.MULTIMEM_STORE
        assert len(g.nodes) == 2

    def test_executor_name_map(self):
        assert _EXECUTOR_NAME_MAP[ExecutorType.MULTIMEM] == "cuda_multimem"
        assert _EXECUTOR_NAME_MAP[ExecutorType.RDMA] == "cuda_rdma"


# ---------------------------------------------------------------------------
# 10. Pipeline construct
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_pipeline_depth_validation(self):
        Pipeline(depth=1)
        Pipeline(depth=4)
        with pytest.raises(ValueError):
            Pipeline(depth=0)
        with pytest.raises(ValueError):
            Pipeline(depth=-1)

    def test_pipeline_channel_assignment(self):
        """Verify stage() assigns channel = iteration % depth."""
        with CommunicationOp(name="pipe_ch") as op:
            op.tensor(dtype="bfloat16", shape=(4096,))
            pipe = Pipeline(depth=3)
            channels_seen = []
            with pipe.bind(op):
                for i in range(6):
                    with pipe.stage():
                        n = op.sm_copy(source_rank=0, src_offset=0,
                                       dst_offset=0, size=1024)
                        channels_seen.append(n.channel)
            g = op.get_graph()

        assert channels_seen == [0, 1, 2, 0, 1, 2]

    def test_pipeline_intra_channel_deps(self):
        """Same-channel ops should be chained (sequential within channel)."""
        with CommunicationOp(name="pipe_deps") as op:
            op.tensor(dtype="bfloat16", shape=(4096,))
            pipe = Pipeline(depth=2)
            nodes = []
            with pipe.bind(op):
                for i in range(4):
                    with pipe.stage():
                        n = op.sm_copy(source_rank=0, src_offset=0,
                                       dst_offset=0, size=1024)
                        nodes.append(n)
            g = op.get_graph()

        # nodes[0] ch0, nodes[1] ch1, nodes[2] ch0, nodes[3] ch1
        # nodes[2] should depend on nodes[0] (same ch0)
        assert nodes[0].op_id in nodes[2].dependencies
        # nodes[3] should depend on nodes[1] (same ch1)
        assert nodes[1].op_id in nodes[3].dependencies

    def test_pipeline_inter_channel_independent(self):
        """Different channel ops should NOT have deps on each other."""
        with CommunicationOp(name="pipe_indep") as op:
            op.tensor(dtype="bfloat16", shape=(4096,))
            pipe = Pipeline(depth=2)
            nodes = []
            with pipe.bind(op):
                for i in range(2):
                    with pipe.stage():
                        n = op.sm_copy(source_rank=0, src_offset=0,
                                       dst_offset=0, size=1024)
                        nodes.append(n)
            g = op.get_graph()

        # nodes[0] ch0, nodes[1] ch1 -> no deps between them
        assert nodes[1].op_id not in nodes[0].dependencies
        assert nodes[0].op_id not in nodes[1].dependencies

    def test_pipeline_codegen(self):
        """Verify codegen produces correct channels with pipeline."""
        with CommunicationOp(name="pipe_cg") as op:
            op.tensor(dtype="bfloat16", shape=(4096,))
            pipe = Pipeline(depth=2)
            with pipe.bind(op):
                for i in range(4):
                    with pipe.stage():
                        op.sm_copy(source_rank=0, src_offset=0,
                                   dst_offset=0, size=1024)
            g = op.get_graph()

        gen = RuntimeGraphGenerator()
        result = gen.generate(g)
        ch_list = [op_dict["channel"] for op_dict in result["operations"]]
        assert ch_list == [0, 1, 0, 1]

    def test_pipeline_depth_1_single_channel(self):
        """depth=1 should put everything on channel 0 (no pipelining)."""
        with CommunicationOp(name="pipe_d1") as op:
            op.tensor(dtype="bfloat16", shape=(4096,))
            pipe = Pipeline(depth=1)
            with pipe.bind(op):
                for i in range(3):
                    with pipe.stage():
                        op.sm_copy(source_rank=0, src_offset=0,
                                   dst_offset=0, size=1024)
            g = op.get_graph()

        assert all(n.channel == 0 for n in g.nodes.values())


# ---------------------------------------------------------------------------
# 11. NOOP handling
# ---------------------------------------------------------------------------

class TestNoop:
    def test_noop_executor_mapping(self):
        from pccl.dsl.nodes import infer_executor
        ex = infer_executor(PrimitiveOpType.NOOP)
        assert ex == ExecutorType.SM


# ---------------------------------------------------------------------------
# 12. Dtype validation
# ---------------------------------------------------------------------------

class TestDtypeValidation:
    def test_valid_dtypes(self):
        for dt in VALID_DTYPES:
            info = TensorInfo(dtype=dt, shape=(1024,))
            assert str(info.dtype) == dt

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported dtype"):
            TensorInfo(dtype="int32", shape=(1024,))

    def test_invalid_shape_type(self):
        with pytest.raises(ValueError, match="shape must be a tuple"):
            TensorInfo(dtype="float32", shape=[1024])
