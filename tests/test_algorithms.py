"""Tests for PCCL DSL algorithm templates and selector."""

import pytest
import json

from pccl.dsl.algorithms import (
    RingAllreduce, RecursiveHalvingDoubling, TreeAllreduce,
    ALGORITHMS, select_algorithm, select_algorithm_cost_based,
)
from pccl.dsl.algorithms.base import CollectiveAlgorithm
from pccl.dsl.nodes import PrimitiveOpType
from pccl.dsl.compiler import Compiler
from pccl.dsl.codegen import RuntimeGraphGenerator
from pccl.dsl.superopt import NVLINK_TOPOLOGY


# ---------- Ring Allreduce ----------

class TestRingAllreduce:
    def test_build_2gpu(self):
        alg = RingAllreduce()
        g = alg.build_allreduce(rank=0, world_size=2, tensor_size=2048)
        assert g.size() > 0
        assert alg.name == "ring"
        assert alg.bandwidth_optimal is True

    def test_build_4gpu(self):
        g = RingAllreduce().build_allreduce(rank=0, world_size=4, tensor_size=4096)
        assert g.size() > 0

    def test_build_8gpu(self):
        g = RingAllreduce().build_allreduce(rank=0, world_size=8, tensor_size=8192)
        assert g.size() == 44

    def test_step_count(self):
        assert RingAllreduce().step_count == "O(2(N-1))"

    def test_data_ops_scale_with_world_size(self):
        g2 = RingAllreduce().build_allreduce(rank=0, world_size=2, tensor_size=2048)
        g4 = RingAllreduce().build_allreduce(rank=0, world_size=4, tensor_size=4096)
        g8 = RingAllreduce().build_allreduce(rank=0, world_size=8, tensor_size=8192)
        assert g4.size() > g2.size()
        assert g8.size() > g4.size()

    def test_multichannel(self):
        g1 = RingAllreduce().build_allreduce(rank=0, world_size=2, tensor_size=2048, num_channels=1)
        g2 = RingAllreduce().build_allreduce(rank=0, world_size=2, tensor_size=2048, num_channels=2)
        assert g2.size() > g1.size()
        channels = set(n.channel for n in g2.nodes.values())
        assert channels == {0, 1}

    def test_sm_executor(self):
        g = RingAllreduce().build_allreduce(rank=0, world_size=2, tensor_size=2048, executor="sm")
        has_sm = any(n.op_type in (PrimitiveOpType.SM_COPY, PrimitiveOpType.SM_REDUCE)
                     for n in g.nodes.values())
        assert has_sm


# ---------- Recursive Halving-Doubling ----------

class TestRecursiveHD:
    def test_build_2gpu(self):
        alg = RecursiveHalvingDoubling()
        g = alg.build_allreduce(rank=0, world_size=2, tensor_size=2048)
        assert g.size() > 0
        assert alg.name == "rhd"

    def test_build_4gpu(self):
        g = RecursiveHalvingDoubling().build_allreduce(rank=0, world_size=4, tensor_size=4096)
        assert g.size() > 0

    def test_build_8gpu(self):
        g = RecursiveHalvingDoubling().build_allreduce(rank=0, world_size=8, tensor_size=8192)
        assert g.size() > 0

    def test_fewer_nodes_than_ring_at_8gpu(self):
        ring_g = RingAllreduce().build_allreduce(rank=0, world_size=8, tensor_size=8192)
        rhd_g = RecursiveHalvingDoubling().build_allreduce(rank=0, world_size=8, tensor_size=8192)
        assert rhd_g.size() < ring_g.size()

    def test_step_count(self):
        assert RecursiveHalvingDoubling().step_count == "O(2 log N)"

    def test_requires_power_of_2(self):
        with pytest.raises(ValueError, match="power-of-2"):
            RecursiveHalvingDoubling().build_allreduce(rank=0, world_size=3, tensor_size=3072)

    def test_bandwidth_optimal(self):
        assert RecursiveHalvingDoubling().bandwidth_optimal is True

    def test_has_correct_partner_pattern(self):
        g = RecursiveHalvingDoubling().build_allreduce(rank=0, world_size=4, tensor_size=4096)
        notify_targets = set()
        for n in g.nodes.values():
            if n.op_type == PrimitiveOpType.NOTIFY:
                notify_targets.add(n.target_rank)
        # rank 0 with N=4: partners at distance 1 (rank 1), 2 (rank 2)
        assert 1 in notify_targets


# ---------- Tree Allreduce ----------

class TestTreeAllreduce:
    def test_build_2gpu(self):
        alg = TreeAllreduce()
        g = alg.build_allreduce(rank=0, world_size=2, tensor_size=2048)
        assert g.size() > 0
        assert alg.name == "tree"

    def test_build_4gpu(self):
        g = TreeAllreduce().build_allreduce(rank=0, world_size=4, tensor_size=4096)
        assert g.size() > 0

    def test_build_8gpu(self):
        g = TreeAllreduce().build_allreduce(rank=0, world_size=8, tensor_size=8192)
        assert g.size() > 0

    def test_not_bandwidth_optimal(self):
        assert TreeAllreduce().bandwidth_optimal is False

    def test_requires_power_of_2(self):
        with pytest.raises(ValueError, match="power-of-2"):
            TreeAllreduce().build_allreduce(rank=0, world_size=6, tensor_size=6144)

    def test_step_count(self):
        assert TreeAllreduce().step_count == "O(2 log N)"

    def test_root_handles_all_data(self):
        g = TreeAllreduce().build_allreduce(rank=0, world_size=4, tensor_size=4096)
        reduce_ops = [n for n in g.nodes.values() if n.op_type == PrimitiveOpType.TMA_REDUCE]
        for r_op in reduce_ops:
            assert r_op.count == 4096


# ---------- Algorithm Selector ----------

class TestSelector:
    def test_large_message_picks_ring(self):
        alg = select_algorithm(8, 512 * 1024 * 1024)
        assert alg.name == "ring"

    def test_small_message_high_gpu_picks_rhd(self):
        alg = select_algorithm(8, 64 * 1024)
        assert alg.name == "rhd"

    def test_2gpu_always_ring(self):
        alg = select_algorithm(2, 64 * 1024)
        assert alg.name == "ring"

    def test_non_power_of_2_picks_ring(self):
        alg = select_algorithm(3, 64 * 1024)
        assert alg.name == "ring"

    def test_medium_message_4gpu_picks_rhd(self):
        alg = select_algorithm(4, 256 * 1024)
        assert alg.name == "rhd"


# ---------- Algorithm + Compile Integration ----------

class TestAlgorithmCompile:
    def _compile_and_check(self, alg, world_size):
        g = alg.build_allreduce(rank=0, world_size=world_size, tensor_size=1024 * world_size)
        compiler = Compiler(enable_superopt=True, topology=NVLINK_TOPOLOGY)
        compiled = compiler.compile(g)
        gen = RuntimeGraphGenerator()
        result = gen.generate(compiled)
        assert result["version"] == 2
        assert len(result["operations"]) > 0
        return result

    def test_ring_compiles(self):
        self._compile_and_check(RingAllreduce(), 2)

    def test_rhd_compiles(self):
        self._compile_and_check(RecursiveHalvingDoubling(), 2)

    def test_tree_compiles(self):
        self._compile_and_check(TreeAllreduce(), 2)

    def test_ring_8gpu_compiles(self):
        self._compile_and_check(RingAllreduce(), 8)

    def test_rhd_8gpu_compiles(self):
        self._compile_and_check(RecursiveHalvingDoubling(), 8)

    def test_tree_4gpu_compiles(self):
        self._compile_and_check(TreeAllreduce(), 4)

    def test_auto_selected_compiles(self):
        alg = select_algorithm(4, 64 * 1024)
        self._compile_and_check(alg, 4)
