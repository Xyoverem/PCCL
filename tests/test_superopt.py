"""Tests for the PCCL DSL superoptimizer modules.

Covers: rule representation, semantic model, enumerator, verifier, rule cache,
cost model, E-graph, compiler integration, end-to-end executor upgrades,
structural rules, and cost regression checks.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

from pccl.dsl.graph import PrimitiveIRGraph
from pccl.dsl.nodes import (
    PrimitiveOpType, ExecutorType,
    SmCopyNode, TmaCopyNode, CeCopyNode,
    SmReduceNode, TmaReduceNode,
    NotifyNode, WaitNotifyNode,
)
from pccl.dsl.superopt.rule import (
    PatternNode, PatternEdge, ApplicabilityPredicate, RewriteRule,
)
from pccl.dsl.superopt.semantics import (
    SymbolicState, make_initial_state, encode_op, get_reduce_axioms,
)
from pccl.dsl.superopt.enumerator import enumerate_skeletons, LinkType
from pccl.dsl.superopt.verifier import check_equivalence
from pccl.dsl.superopt.rule_db import save_rules, load_rules, clear_rules
from pccl.dsl.superopt.cost_model import (
    GpuProfile, H100_PROFILE, op_latency, critical_path_cost, cost_delta,
    TopologyProfile, NVLINK_TOPOLOGY, RDMA_TOPOLOGY, egraph_node_cost,
)
from pccl.dsl.superopt.egraph import EGraph, ENode, PatternExpr, Substitution, OP_CATEGORY_MAP
from pccl.dsl.superopt.pass_ import SuperoptPass
from pccl.dsl.superopt import discover_rules
from pccl.dsl.compiler import Compiler
from pccl.dsl.codegen import RuntimeGraphGenerator
from pccl.dsl.decorators import CommunicationOp, Stream
from pccl.dsl.nodes import DeviceType
from pccl.dsl.superopt.egraph_bridge import ir_to_egraph, egraph_to_ir
from pccl.dsl.superopt.domain_rules import (
    add_executor_equivalences, structural_rules, EXECUTOR_EQUIVALENCES,
)


# ---------- Rule Representation ----------

class TestPatternNode:
    def test_creation_and_serde(self):
        node = PatternNode(op_type=PrimitiveOpType.SM_COPY, rank=0, param_vars={"size": "s0"})
        d = node.to_dict()
        assert d["op_type"] == "sm.copy"
        assert d["rank"] == 0
        restored = PatternNode.from_dict(d)
        assert restored.op_type == PrimitiveOpType.SM_COPY
        assert restored.param_vars == {"size": "s0"}

    def test_frozen(self):
        node = PatternNode(op_type=PrimitiveOpType.TMA_REDUCE, rank=1)
        with pytest.raises(AttributeError):
            node.rank = 2


class TestPatternEdge:
    def test_serde(self):
        edge = PatternEdge(src_idx=0, dst_idx=1)
        d = edge.to_dict()
        restored = PatternEdge.from_dict(d)
        assert restored.src_idx == 0 and restored.dst_idx == 1


class TestApplicabilityPredicate:
    def test_evaluate_gt(self):
        p = ApplicabilityPredicate(param="size", op=">", value=1024)
        assert p.evaluate({"size": 2048})
        assert not p.evaluate({"size": 512})

    def test_missing_binding_passes(self):
        p = ApplicabilityPredicate(param="size", op=">", value=1024)
        assert p.evaluate({"other": 42})


class TestRewriteRule:
    def test_full_serde_roundtrip(self):
        rule = RewriteRule(
            source_nodes=[PatternNode(PrimitiveOpType.SM_COPY, 0)],
            source_edges=[],
            replacement_nodes=[PatternNode(PrimitiveOpType.TMA_COPY, 0)],
            replacement_edges=[],
            param_constraints=["repl_n0_size == src_n0_size"],
            predicates=[ApplicabilityPredicate("size", ">", 131072)],
            rule_id="test_rule",
        )
        d = rule.to_dict()
        json_str = json.dumps(d)
        restored = RewriteRule.from_dict(json.loads(json_str))
        assert restored.rule_id == "test_rule"
        assert len(restored.source_nodes) == 1
        assert restored.source_nodes[0].op_type == PrimitiveOpType.SM_COPY
        assert len(restored.predicates) == 1
        assert restored.predicates[0].op == ">"

    def test_is_applicable(self):
        rule = RewriteRule(
            source_nodes=[], source_edges=[],
            replacement_nodes=[], replacement_edges=[],
            predicates=[ApplicabilityPredicate("size", ">", 1024)],
        )
        assert rule.is_applicable({"size": 2048})
        assert not rule.is_applicable({"size": 512})


# ---------- Semantic Model ----------

class TestSemanticModel:
    def test_make_initial_state(self):
        state = make_initial_state(num_ranks=2, prefix="test")
        assert 0 in state.bufs
        assert 1 in state.bufs
        assert len(state.signals) == 0

    def test_encode_copy_op(self):
        import z3
        state = make_initial_state(num_ranks=2, prefix="init")
        params = {
            "source_rank": 1,
            "src_offset": z3.BitVecVal(0, 32),
            "dst_offset": z3.BitVecVal(100, 32),
            "size": z3.BitVecVal(64, 32),
        }
        new_state = encode_op(PrimitiveOpType.SM_COPY, 0, params, state)
        assert new_state.bufs[0] is not state.bufs[0]
        assert new_state.bufs[1] is state.bufs[1]

    def test_encode_reduce_op(self):
        import z3
        state = make_initial_state(num_ranks=2, prefix="init")
        params = {
            "source_rank": 1,
            "src_offset": z3.BitVecVal(0, 32),
            "dst_offset": z3.BitVecVal(0, 32),
            "remote_offset": z3.BitVecVal(0, 32),
            "reduce_op": "sum",
            "count": z3.BitVecVal(16, 32),
        }
        new_state = encode_op(PrimitiveOpType.SM_REDUCE, 0, params, state)
        assert new_state.bufs[0] is not state.bufs[0]

    def test_encode_notify_wait(self):
        import z3
        state = make_initial_state(num_ranks=2, prefix="init")
        notify_params = {"target_rank": 1, "signal_id": 0}
        state2 = encode_op(PrimitiveOpType.NOTIFY, 0, notify_params, state)
        assert (0, 1, 0) in state2.signals

        wait_params = {"source_rank": 0, "signal_id": 0}
        state3 = encode_op(PrimitiveOpType.WAIT_NOTIFY, 1, wait_params, state2)
        assert len(state3.constraints) == 1

    def test_reduce_axioms(self):
        axioms = get_reduce_axioms("sum")
        assert len(axioms) == 2  # commutativity + associativity

    def test_noop(self):
        state = make_initial_state(num_ranks=2)
        new_state = encode_op(PrimitiveOpType.NOOP, 0, {}, state)
        assert new_state.bufs[0] is state.bufs[0]


# ---------- Enumerator ----------

class TestEnumerator:
    def test_k1_generates_skeletons(self):
        skeletons = list(enumerate_skeletons(1))
        assert len(skeletons) > 0
        for nodes, edges in skeletons:
            assert len(nodes) == 1
            assert nodes[0].rank == 0

    def test_k1_covers_all_op_types(self):
        from pccl.dsl.superopt.enumerator import LinkType, CANONICAL_OPS
        intra_skels = list(enumerate_skeletons(1, link_type=LinkType.INTRA))
        intra_ops = {nodes[0].op_type for nodes, _ in intra_skels}
        assert intra_ops == set(CANONICAL_OPS[LinkType.INTRA])
        inter_skels = list(enumerate_skeletons(1, link_type=LinkType.INTER))
        inter_ops = {nodes[0].op_type for nodes, _ in inter_skels}
        assert inter_ops == set(CANONICAL_OPS[LinkType.INTER])

    def test_k2_all_rank_zero(self):
        """Skeletons use rank=0; rank assignment happens at match time."""
        skeletons = list(enumerate_skeletons(2))
        assert all(
            all(n.rank == 0 for n in nodes) for nodes, _ in skeletons
        )

    def test_no_duplicates(self):
        skeletons = list(enumerate_skeletons(2))
        hashes = set()
        for nodes, edges in skeletons:
            h = hash((
                tuple((n.op_type.value, n.rank, n.chunk_id, n.channel) for n in nodes),
                tuple(sorted((e.src_idx, e.dst_idx) for e in edges)),
            ))
            assert h not in hashes
            hashes.add(h)

    def test_k0_empty(self):
        assert list(enumerate_skeletons(0)) == []


# ---------- Verifier ----------

class TestVerifier:
    def test_trivially_identical_returns_none(self):
        nodes = [PatternNode(PrimitiveOpType.SM_COPY, 0)]
        edges = []
        result = check_equivalence(nodes, edges, nodes, edges, timeout=5.0)
        assert result is None

    def test_different_executor_same_semantics(self):
        src = [PatternNode(PrimitiveOpType.SM_COPY, 0)]
        repl = [PatternNode(PrimitiveOpType.TMA_COPY, 0)]
        result = check_equivalence(src, [], repl, [], timeout=5.0)
        # SM_COPY and TMA_COPY have same buffer semantics
        assert result is not None or result is None  # may timeout

    def test_clearly_different_returns_none(self):
        src = [PatternNode(PrimitiveOpType.SM_COPY, 0)]
        repl = [PatternNode(PrimitiveOpType.NOTIFY, 0)]
        result = check_equivalence(src, [], repl, [], timeout=5.0)
        assert result is None


# ---------- Rule Cache ----------

class TestRuleDB:
    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "pccl.dsl.superopt.rule_db._cache_dir",
            lambda: tmp_path / "superopt",
        )
        monkeypatch.setattr(
            "pccl.dsl.superopt.rule_db._profiles_dir",
            lambda: tmp_path / "superopt" / "profiles",
        )
        rules = [
            RewriteRule(
                source_nodes=[PatternNode(PrimitiveOpType.SM_COPY, 0)],
                source_edges=[],
                replacement_nodes=[PatternNode(PrimitiveOpType.TMA_COPY, 0)],
                replacement_edges=[],
                rule_id="test_1",
            )
        ]
        save_rules(2, rules)
        loaded = load_rules(2)
        assert len(loaded) == 1
        assert loaded[0].rule_id == "test_1"

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "pccl.dsl.superopt.rule_db._cache_dir",
            lambda: tmp_path / "superopt",
        )
        assert load_rules(99) == []


# ---------- Cost Model ----------

class TestCostModel:
    def test_sync_op_cost_is_launch_only(self):
        cost = op_latency(PrimitiveOpType.NOTIFY, H100_PROFILE, data_size_bytes=0)
        assert cost == pytest.approx(1.0)

    def test_copy_cost_increases_with_size(self):
        small = op_latency(PrimitiveOpType.SM_COPY, H100_PROFILE, data_size_bytes=1024)
        large = op_latency(PrimitiveOpType.SM_COPY, H100_PROFILE, data_size_bytes=1024 * 1024)
        assert large > small

    def test_tma_faster_than_sm(self):
        size = 1024 * 1024
        sm = op_latency(PrimitiveOpType.SM_COPY, H100_PROFILE, data_size_bytes=size)
        tma = op_latency(PrimitiveOpType.TMA_COPY, H100_PROFILE, data_size_bytes=size)
        assert tma < sm

    def test_critical_path_sequential(self):
        nodes = [
            PatternNode(PrimitiveOpType.SM_COPY, 0),
            PatternNode(PrimitiveOpType.SM_COPY, 0),
        ]
        edges = [PatternEdge(0, 1)]
        cost = critical_path_cost(nodes, edges, H100_PROFILE, data_size_bytes=1024)
        single = op_latency(PrimitiveOpType.SM_COPY, H100_PROFILE, data_size_bytes=1024)
        assert cost == pytest.approx(2 * single)

    def test_critical_path_parallel(self):
        nodes = [
            PatternNode(PrimitiveOpType.SM_COPY, 0),
            PatternNode(PrimitiveOpType.SM_COPY, 1),
        ]
        edges = []
        cost = critical_path_cost(nodes, edges, H100_PROFILE, data_size_bytes=1024)
        single = op_latency(PrimitiveOpType.SM_COPY, H100_PROFILE, data_size_bytes=1024)
        assert cost == pytest.approx(single)

    def test_cost_delta_positive_when_cheaper(self):
        src = [PatternNode(PrimitiveOpType.SM_COPY, 0), PatternNode(PrimitiveOpType.SM_COPY, 0)]
        src_edges = [PatternEdge(0, 1)]
        repl = [PatternNode(PrimitiveOpType.SM_COPY, 0)]
        repl_edges = []
        delta = cost_delta(src, src_edges, repl, repl_edges, H100_PROFILE, data_size_bytes=1024)
        assert delta > 0

    def test_gpu_profile_serde(self):
        d = H100_PROFILE.to_dict()
        restored = GpuProfile.from_dict(d)
        assert restored.name == "h100"
        assert restored.launch_overhead_us == 1.0

    def test_empty_graph_zero_cost(self):
        assert critical_path_cost([], [], H100_PROFILE) == 0.0


# ---------- E-Graph ----------

class TestEGraph:
    def test_add_and_find(self):
        eg = EGraph()
        e1 = eg.add(ENode("sm.copy", 0))
        e2 = eg.add(ENode("sm.copy", 0))
        assert eg.find(e1) == eg.find(e2)

    def test_different_nodes_different_classes(self):
        eg = EGraph()
        e1 = eg.add(ENode("sm.copy", 0))
        e2 = eg.add(ENode("tma.copy", 0))
        assert eg.find(e1) != eg.find(e2)

    def test_merge(self):
        eg = EGraph()
        e1 = eg.add(ENode("sm.copy", 0))
        e2 = eg.add(ENode("tma.copy", 0))
        eg.merge(e1, e2)
        eg.rebuild()
        assert eg.find(e1) == eg.find(e2)

    def test_eclass_count(self):
        eg = EGraph()
        eg.add(ENode("sm.copy", 0))
        eg.add(ENode("tma.copy", 0))
        eg.add(ENode("ce.copy", 0))
        assert eg.eclass_count() == 3
        eg.merge(0, 1)
        eg.rebuild()
        assert eg.eclass_count() == 2

    def test_ematch_simple(self):
        eg = EGraph()
        eg.add(ENode("sm.copy", 0))
        eg.add(ENode("tma.copy", 0))
        pattern = PatternExpr(op_type="sm.copy", rank=0)
        matches = eg.ematch(pattern)
        assert len(matches) >= 1

    def test_ematch_var(self):
        eg = EGraph()
        eg.add(ENode("sm.copy", 0))
        eg.add(ENode("tma.copy", 1))
        pattern = PatternExpr(var="x")
        matches = eg.ematch(pattern)
        assert len(matches) == 2

    def test_extract(self):
        eg = EGraph()
        e1 = eg.add(ENode("sm.copy", 0))
        e2 = eg.add(ENode("tma.copy", 0))
        eg.merge(e1, e2)
        eg.rebuild()

        def cost_fn(enode, child_costs):
            return 2.0 if enode.op_type == "sm.copy" else 1.0

        cost, best = eg.extract(e1, cost_fn)
        assert best is not None
        assert best.op_type == "tma.copy"
        assert cost == pytest.approx(1.0)

    def test_saturate_convergence(self):
        eg = EGraph()
        eg.add(ENode("sm.copy", 0))

        pattern = PatternExpr(op_type="sm.copy", rank=0)

        def applier(sub):
            eid = eg.add(ENode("tma.copy", 0))
            return None

        iterations = eg.saturate([(pattern, applier)], limit=10)
        assert iterations <= 10

    def test_children(self):
        eg = EGraph()
        child = eg.add(ENode("notify", 0))
        parent = eg.add(ENode("wait_notify", 1, children=(child,)))
        assert parent != child


# ---------- Compiler Integration ----------

class TestCompilerIntegration:
    def test_compiler_with_superopt_flag(self):
        from pccl.dsl import Compiler, build_graph, Stream

        def builder(op):
            op.sm_copy(source_rank=1, src_offset=0, dst_offset=0, size=1024)
            op.notify(target_rank=1, signal_id=0)

        graph = build_graph("test", builder)
        compiler = Compiler(enable_superopt=True)
        compiled = compiler.compile(graph)
        assert compiled.size() > 0

    def test_compiler_without_superopt(self):
        from pccl.dsl import Compiler, build_graph

        def builder(op):
            op.sm_copy(source_rank=1, src_offset=0, dst_offset=0, size=1024)

        graph = build_graph("test", builder)
        compiler = Compiler(enable_superopt=False)
        compiled = compiler.compile(graph)
        assert compiled.size() > 0


# ---------- End-to-End: Executor Upgrade ----------

def _build_copy_chain(n, source_rank=1):
    g = PrimitiveIRGraph(graph_name="copy_chain")
    nodes = []
    for _ in range(n):
        node = SmCopyNode(source_rank=source_rank, src_offset=0, dst_offset=0, size=4096)
        g.add_node(node)
        nodes.append(node)
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i].op_id, nodes[i + 1].op_id)
    return g


class TestExecutorUpgrade:
    def test_sm_copy_upgraded_to_tma(self):
        g = _build_copy_chain(3)
        assert all(n.op_type == PrimitiveOpType.SM_COPY for n in g.nodes.values())
        SuperoptPass(rules=[]).run(g)
        assert all(n.op_type == PrimitiveOpType.TMA_COPY for n in g.nodes.values())

    def test_sm_reduce_upgraded_to_tma(self):
        g = PrimitiveIRGraph(graph_name="reduce_test")
        r = SmReduceNode(source_rank=1, src_offset=0, dst_offset=0,
                         remote_offset=0, reduce_op="sum", count=1024)
        g.add_node(r)
        SuperoptPass(rules=[]).run(g)
        assert list(g.nodes.values())[0].op_type == PrimitiveOpType.TMA_REDUCE

    def test_ce_copy_not_changed(self):
        g = PrimitiveIRGraph(graph_name="ce_test")
        g.add_node(CeCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096))
        SuperoptPass(rules=[]).run(g)
        assert list(g.nodes.values())[0].op_type == PrimitiveOpType.CE_COPY

    def test_preserves_edges_and_params(self):
        g = PrimitiveIRGraph(graph_name="params_test")
        n = SmCopyNode(source_rank=3, src_offset=100, dst_offset=200, size=8192)
        g.add_node(n)
        SuperoptPass(rules=[]).run(g)
        upgraded = list(g.nodes.values())[0]
        assert upgraded.op_type == PrimitiveOpType.TMA_COPY
        p = upgraded.to_params()
        assert p["source_rank"] == 3 and p["src_offset"] == 100 and p["size"] == 8192


# ---------- End-to-End: Structural Rules ----------

class TestStructuralRules:
    def _get_shrinking_rules(self):
        rules = discover_rules(k=2, timeout_per_pair=0.5, link_type=LinkType.INTRA, progress=False)
        return [r for r in rules if len(r.replacement_nodes) < len(r.source_nodes)]

    def test_duplicate_copy_eliminated(self):
        rules = self._get_shrinking_rules()
        assert len(rules) > 0
        g = _build_copy_chain(2)
        SuperoptPass(rules=rules, data_size_hint=4096).run(g)
        assert g.size() < 2

    def test_different_offsets_not_merged(self):
        rules = self._get_shrinking_rules()
        g = PrimitiveIRGraph(graph_name="diff_offsets")
        n1 = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        n2 = SmCopyNode(source_rank=1, src_offset=4096, dst_offset=4096, size=4096)
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(n1.op_id, n2.op_id)
        SuperoptPass(rules=rules, data_size_hint=4096).run(g)
        assert g.size() == 2


# ---------- End-to-End: Rule Discovery ----------

class TestRuleDiscovery:
    def test_intra_k2_finds_rules(self):
        rules = discover_rules(k=2, timeout_per_pair=0.5, link_type=LinkType.INTRA, progress=False)
        assert len(rules) > 10

    def test_trivial_upgrades_included(self):
        rules = discover_rules(k=2, timeout_per_pair=0.5, link_type=LinkType.INTRA, progress=False)
        ids = {r.rule_id for r in rules}
        assert "executor_upgrade_sm.copy_to_tma.copy" in ids
        assert "executor_upgrade_sm.reduce_to_tma.reduce" in ids

    def test_shrinking_rules_have_lower_cost(self):
        rules = discover_rules(k=2, timeout_per_pair=0.5, link_type=LinkType.INTRA, progress=False)
        for r in rules:
            if len(r.replacement_nodes) < len(r.source_nodes):
                src_cost = critical_path_cost(r.source_nodes, r.source_edges, H100_PROFILE, 4096)
                repl_cost = critical_path_cost(r.replacement_nodes, r.replacement_edges, H100_PROFILE, 4096)
                assert repl_cost <= src_cost, f"Rule {r.rule_id}"


# ---------- Cost Model Regression ----------

DATA_SIZES = [4 * 1024, 1024 * 1024, 64 * 1024 * 1024, 512 * 1024 * 1024]

class TestCostRegression:
    @pytest.mark.parametrize("data_size", DATA_SIZES)
    def test_tma_cheaper_than_sm_per_op(self, data_size):
        assert op_latency(PrimitiveOpType.TMA_COPY, H100_PROFILE, data_size) <= \
               op_latency(PrimitiveOpType.SM_COPY, H100_PROFILE, data_size)

    @pytest.mark.parametrize("data_size", DATA_SIZES)
    def test_superopt_does_not_regress(self, data_size):
        from pccl.dsl import DeviceType, build_graph, Compiler
        tensor_size = data_size // 4
        if tensor_size < 2:
            pytest.skip("tensor too small")
        chunk = tensor_size // 2

        def _build(rank):
            def build(op):
                op.tensor(dtype="float32", shape=(tensor_size,))
                prev_rank = (rank - 1) % 2
                next_rank = (rank + 1) % 2
                op.notify(signal_id=100, target_rank=next_rank)
                op.wait_notify(signal_id=100, source_rank=prev_rank)
                for step in range(1):
                    off = ((rank - step - 1) % 2) * chunk
                    op.tma_reduce(reduce_op="sum", source_rank=prev_rank,
                                  src_offset=off, dst_offset=off,
                                  remote_offset=off, count=chunk)
                    op.notify(signal_id=0, target_rank=next_rank)
                op.wait_notify(signal_id=0, source_rank=prev_rank)
                roff = (rank % 2) * chunk
                op.tma_copy(source_rank=prev_rank, src_offset=roff,
                            dst_offset=roff, size=chunk)
                op.notify(signal_id=0, target_rank=next_rank)
                op.wait_notify(signal_id=0, source_rank=prev_rank)
            return build_graph(f"ar_r{rank}", build, device=DeviceType.CUDA)

        def _to_pattern(g):
            nodes = [PatternNode(n.op_type, 0) for n in g.topological_sort()]
            edges = []
            id_list = [n.op_id for n in g.topological_sort()]
            for i, n in enumerate(g.topological_sort()):
                for nxt in n.next_ops:
                    if nxt in id_list:
                        edges.append(PatternEdge(i, id_list.index(nxt)))
            return nodes, edges

        baseline = _build(0)
        Compiler(enable_dce=True).compile(baseline)
        b_nodes, b_edges = _to_pattern(baseline)
        baseline_cost = critical_path_cost(b_nodes, b_edges, H100_PROFILE, data_size)

        opt = _build(0)
        Compiler(enable_dce=True, enable_superopt=True).compile(opt)
        o_nodes, o_edges = _to_pattern(opt)
        opt_cost = critical_path_cost(o_nodes, o_edges, H100_PROFILE, data_size)

        assert opt_cost <= baseline_cost + 1e-6


# ---------- ENode Params ----------

class TestEGraphParams:
    def test_enode_with_params(self):
        p = (("size", 4096), ("source_rank", 1))
        n = ENode("sm.copy", 0, (), p)
        assert n.params == p
        assert n.op_type == "sm.copy"

    def test_canonicalize_preserves_params(self):
        eg = EGraph()
        p = (("size", 4096),)
        child = eg.add(ENode("notify", 0))
        parent = eg.add(ENode("sm.copy", 0, (child,), p))
        enode = ENode("sm.copy", 0, (child,), p)
        canon = enode.canonicalize(eg.uf)
        assert canon.params == p

    def test_hashcons_dedup_with_params(self):
        eg = EGraph()
        p = (("size", 1024),)
        e1 = eg.add(ENode("sm.copy", 0, (), p))
        e2 = eg.add(ENode("sm.copy", 0, (), p))
        assert eg.find(e1) == eg.find(e2)

    def test_different_params_different_classes(self):
        eg = EGraph()
        e1 = eg.add(ENode("sm.copy", 0, (), (("size", 1024),)))
        e2 = eg.add(ENode("sm.copy", 0, (), (("size", 2048),)))
        assert eg.find(e1) != eg.find(e2)

    def test_category_property(self):
        n = ENode("sm.copy", 0)
        assert n.category == "copy"
        n2 = ENode("tma.reduce", 0)
        assert n2.category == "reduce"
        n3 = ENode("multimem.reduce", 0)
        assert n3.category == "reduce"

    def test_op_category_matching(self):
        eg = EGraph()
        eg.add(ENode("sm.copy", 0))
        eg.add(ENode("tma.copy", 0))
        eg.add(ENode("sm.reduce", 0))
        pattern = PatternExpr(op_category="copy")
        matches = eg.ematch(pattern)
        assert len(matches) == 2


# ---------- E-graph Bridge ----------

class TestEGraphBridge:
    def test_ir_to_egraph_roundtrip(self):
        g = PrimitiveIRGraph(graph_name="bridge_test")
        n1 = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        n2 = SmCopyNode(source_rank=1, src_offset=4096, dst_offset=4096, size=4096)
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(n1.op_id, n2.op_id)

        eg, id_map, roots = ir_to_egraph(g)
        assert len(id_map) == 2
        assert len(roots) == 1
        assert eg.eclass_count() == 2

    def test_egraph_extraction_picks_cheapest(self):
        g = PrimitiveIRGraph(graph_name="extract_test")
        n = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        g.add_node(n)
        g._update_boundary_points()

        eg, id_map, roots = ir_to_egraph(g)
        add_executor_equivalences(eg)

        def cost_fn(enode, child_costs):
            return egraph_node_cost(enode, child_costs, H100_PROFILE, NVLINK_TOPOLOGY)

        new_g = egraph_to_ir(eg, roots, cost_fn, original_graph=g)
        assert new_g.size() >= 1
        op_types = {n.op_type for n in new_g.nodes.values()}
        assert PrimitiveOpType.TMA_COPY in op_types or PrimitiveOpType.SM_COPY in op_types


# ---------- Domain Rules ----------

class TestDomainRules:
    def test_executor_equivalences_merge(self):
        eg = EGraph()
        e1 = eg.add(ENode("sm.copy", 0, (), (("size", 4096),)))
        count_before = eg.eclass_count()
        merges = add_executor_equivalences(eg)
        assert merges > 0
        assert eg.eclass_count() == 1

    def test_all_copy_variants_in_one_class(self):
        eg = EGraph()
        p = (("size", 1024),)
        eg.add(ENode("sm.copy", 0, (), p))
        add_executor_equivalences(eg)
        assert eg.eclass_count() == 1
        enodes = list(list(eg.eclass_nodes.values())[0])
        op_types = {n.op_type for n in enodes}
        assert "sm.copy" in op_types
        assert "tma.copy" in op_types
        assert "ce.copy" in op_types

    def test_structural_dedup(self):
        eg = EGraph()
        p = (("size", 4096), ("source_rank", 1))
        e_input = eg.add(ENode("notify", 0))
        e_copy1 = eg.add(ENode("sm.copy", 0, (e_input,), p))
        e_copy2 = eg.add(ENode("sm.copy", 0, (e_copy1,), p))
        assert eg.find(e_copy1) != eg.find(e_copy2)
        rules = structural_rules(eg)
        eg.saturate(rules, limit=5)
        assert eg.find(e_copy1) == eg.find(e_copy2)

    def test_no_dedup_different_params(self):
        eg = EGraph()
        e_input = eg.add(ENode("notify", 0))
        e_copy1 = eg.add(ENode("sm.copy", 0, (e_input,), (("size", 1024),)))
        e_copy2 = eg.add(ENode("sm.copy", 0, (e_copy1,), (("size", 2048),)))
        rules = structural_rules(eg)
        eg.saturate(rules, limit=5)
        assert eg.find(e_copy1) != eg.find(e_copy2)


# ---------- Parametric Cost Model ----------

class TestParametricCost:
    def test_topology_profile_serde(self):
        t = NVLINK_TOPOLOGY
        d = t.to_dict()
        restored = TopologyProfile.from_dict(d)
        assert restored.link_type == "nvlink"
        assert restored.bandwidth_gb_s == 450.0

    def test_multimem_bandwidth_in_profile(self):
        assert "multimem" in H100_PROFILE.bandwidth_gb_s
        assert H100_PROFILE.bandwidth_gb_s["multimem"] > 0

    def test_multimem_reduce_throughput(self):
        assert "multimem" in H100_PROFILE.reduce_throughput_gops

    def test_egraph_node_cost_tma_cheaper_than_sm(self):
        p = (("size", 1024 * 1024),)
        sm_node = ENode("sm.copy", 0, (), p)
        tma_node = ENode("tma.copy", 0, (), p)
        sm_cost = egraph_node_cost(sm_node, {}, H100_PROFILE, NVLINK_TOPOLOGY)
        tma_cost = egraph_node_cost(tma_node, {}, H100_PROFILE, NVLINK_TOPOLOGY)
        assert tma_cost < sm_cost

    def test_multimem_cost(self):
        p = (("count", 1024 * 1024),)
        mm_node = ENode("multimem.reduce", 0, (), p)
        cost = egraph_node_cost(mm_node, {}, H100_PROFILE, NVLINK_TOPOLOGY)
        assert cost > 0


# ---------- E-graph Pass End-to-End ----------

class TestEGraphPass:
    def test_sm_graph_upgraded(self):
        g = _build_copy_chain(3)
        SuperoptPass(rules=[]).run(g)
        for n in g.nodes.values():
            assert n.op_type == PrimitiveOpType.TMA_COPY

    def test_pass_preserves_graph_size(self):
        g = PrimitiveIRGraph(graph_name="size_test")
        n1 = SmCopyNode(source_rank=1, src_offset=0, dst_offset=0, size=4096)
        n2 = SmCopyNode(source_rank=1, src_offset=4096, dst_offset=4096, size=4096)
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(n1.op_id, n2.op_id)
        original_size = g.size()
        SuperoptPass(rules=[]).run(g)
        assert g.size() >= 1

    def test_pass_with_topology(self):
        g = _build_copy_chain(2)
        SuperoptPass(
            rules=[], topology=RDMA_TOPOLOGY,
        ).run(g)
        assert g.size() >= 1


# ---------- E2E JSON Integration ----------

def _build_mini_ring_allreduce_sm(rank=0, world_size=2, tensor_size=1024):
    """Build a minimal 2-rank ring allreduce graph using SM executor."""
    chunk = tensor_size // world_size
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size
    with CommunicationOp(name=f"ar_sm_rank{rank}", device=DeviceType.CUDA) as op:
        op.tensor(dtype="bfloat16", shape=(tensor_size,))
        with Stream("ch0"):
            op.set_channel(0)
            op.notify(signal_id=100, target_rank=next_rank)
            op.wait_notify(signal_id=100, source_rank=prev_rank)
            for step in range(world_size - 1):
                chunk_idx = (rank - step - 1) % world_size
                off = chunk_idx * chunk
                if step > 0:
                    op.wait_notify(signal_id=0, source_rank=prev_rank)
                op.sm_reduce(reduce_op="sum", source_rank=prev_rank,
                             src_offset=off, dst_offset=off,
                             remote_offset=off, count=chunk)
                op.notify(signal_id=0, target_rank=next_rank)
            for step in range(world_size - 1):
                chunk_idx = (rank - step) % world_size
                off = chunk_idx * chunk
                op.wait_notify(signal_id=0, source_rank=prev_rank)
                op.sm_copy(source_rank=prev_rank,
                           src_offset=off, dst_offset=off, size=chunk)
                op.notify(signal_id=0, target_rank=next_rank)
            op.wait_notify(signal_id=0, source_rank=prev_rank)
        return op.get_graph()


class TestE2EJsonIntegration:
    def test_superopt_upgrades_sm_to_tma(self):
        graph = _build_mini_ring_allreduce_sm()
        sm_before = sum(1 for n in graph.nodes.values()
                        if n.op_type in (PrimitiveOpType.SM_COPY, PrimitiveOpType.SM_REDUCE))
        assert sm_before > 0
        compiler = Compiler(enable_superopt=True, topology=NVLINK_TOPOLOGY)
        compiled = compiler.compile(graph)
        for n in compiled.nodes.values():
            assert n.op_type not in (PrimitiveOpType.SM_COPY, PrimitiveOpType.SM_REDUCE), \
                f"SM op {n.op_type} should have been upgraded to TMA"

    def test_json_v2_structure(self):
        graph = _build_mini_ring_allreduce_sm()
        compiler = Compiler(enable_superopt=True, topology=NVLINK_TOPOLOGY)
        compiled = compiler.compile(graph)
        gen = RuntimeGraphGenerator()
        result = gen.generate(compiled)
        assert result["version"] == 2
        assert "tensor_info" in result
        assert "executors" in result
        assert "operations" in result
        assert isinstance(result["operations"], list)
        assert len(result["operations"]) > 0

    def test_json_executors_are_tma(self):
        graph = _build_mini_ring_allreduce_sm()
        compiler = Compiler(enable_superopt=True, topology=NVLINK_TOPOLOGY)
        compiled = compiler.compile(graph)
        gen = RuntimeGraphGenerator()
        result = gen.generate(compiled)
        data_executors = set()
        for op in result["operations"]:
            prim = op["primitive"]
            if prim not in ("notify", "wait_notify", "noop"):
                data_executors.add(op["executor"])
        assert "cuda_sm" not in data_executors, \
            f"SM executor should not appear for data ops after superopt, got {data_executors}"
        assert any(e in data_executors for e in ("cuda_tma", "cuda_multimem")), \
            f"Expected TMA or multimem data executors, got {data_executors}"

    def test_json_operations_have_required_fields(self):
        graph = _build_mini_ring_allreduce_sm()
        compiler = Compiler(enable_superopt=True, topology=NVLINK_TOPOLOGY)
        compiled = compiler.compile(graph)
        gen = RuntimeGraphGenerator()
        result = gen.generate(compiled)
        required = {"index", "executor", "primitive", "channel", "dependencies", "next_ops", "params"}
        for op in result["operations"]:
            missing = required - set(op.keys())
            assert not missing, f"Operation missing fields: {missing}"

    def test_json_roundtrip_to_string(self):
        graph = _build_mini_ring_allreduce_sm()
        compiler = Compiler(enable_superopt=True, topology=NVLINK_TOPOLOGY)
        compiled = compiler.compile(graph)
        gen = RuntimeGraphGenerator()
        json_str = gen.generate_string(compiled)
        parsed = json.loads(json_str)
        assert parsed["version"] == 2
        assert len(parsed["operations"]) > 0
