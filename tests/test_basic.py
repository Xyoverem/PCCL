"""
Unit tests for PCCL v2 DSL (Simplified): node types, graph building, JSON v2 generation.

Run with: python -m pytest tests/test_basic.py -v
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import (
    DeviceType,
    ExecutorType,
    PrimitiveOpType,
    TensorInfo,
    ReduceOp,
    IRNode,
    SmReduceNode,
    SmCopyNode,
    TmaCopyNode,
    TmaReduceNode,
    CeCopyNode,
    RdmaWriteNode,
    RdmaReadNode,
    NotifyNode,
    WaitNotifyNode,
    PrimitiveIRGraph,
    build_graph,
    Stream,
    compile_to_json_string,
)


# ---------------------------------------------------------------------------
# Node creation and executor inference
# ---------------------------------------------------------------------------

class TestNodeTypes:
    def test_sm_reduce_node(self):
        node = SmReduceNode(
            device=DeviceType.CUDA,
            reduce_op="sum",
            source_rank=1,
            src_offset=0,
            dst_offset=0,
            remote_offset=0,
            count=1024,
        )
        assert node.op_type == PrimitiveOpType.SM_REDUCE
        assert node.executor == ExecutorType.SM
        assert node.reduce_op == "sum"
        assert node.source_rank == 1
        assert node.count == 1024
        assert node.op_id != ""

    def test_sm_copy_node(self):
        node = SmCopyNode(
            device=DeviceType.CUDA,
            source_rank=0,
            src_offset=100,
            dst_offset=200,
            size=4096,
        )
        assert node.op_type == PrimitiveOpType.SM_COPY
        assert node.executor == ExecutorType.SM
        assert node.size == 4096

    def test_tma_copy_node(self):
        node = TmaCopyNode(
            device=DeviceType.CUDA,
            source_rank=1,
            src_offset=0,
            dst_offset=0,
            size=8192,
        )
        assert node.op_type == PrimitiveOpType.TMA_COPY
        assert node.executor == ExecutorType.TMA

    def test_tma_reduce_node(self):
        node = TmaReduceNode(
            device=DeviceType.CUDA,
            reduce_op="sum",
            source_rank=1,
            src_offset=0,
            dst_offset=0,
            remote_offset=0,
            count=2048,
        )
        assert node.op_type == PrimitiveOpType.TMA_REDUCE
        assert node.executor == ExecutorType.TMA

    def test_ce_copy_node(self):
        node = CeCopyNode(
            device=DeviceType.CUDA,
            source_rank=0,
            src_offset=0,
            dst_offset=0,
            size=1024,
        )
        assert node.op_type == PrimitiveOpType.CE_COPY
        assert node.executor == ExecutorType.CE

    def test_rdma_write_node(self):
        node = RdmaWriteNode(
            device=DeviceType.RDMA,
            target_rank=1,
            src_offset=0,
            dst_offset=0,
            size=4096,
        )
        assert node.op_type == PrimitiveOpType.RDMA_WRITE
        assert node.executor == ExecutorType.RDMA

    def test_rdma_read_node(self):
        node = RdmaReadNode(
            device=DeviceType.RDMA,
            source_rank=1,
            src_offset=0,
            dst_offset=0,
            size=4096,
        )
        assert node.op_type == PrimitiveOpType.RDMA_READ
        assert node.executor == ExecutorType.RDMA

    def test_notify_node_cuda(self):
        node = NotifyNode(
            device=DeviceType.CUDA,
            signal_id=0,
            target_rank=1,
        )
        assert node.op_type == PrimitiveOpType.NOTIFY
        assert node.executor == ExecutorType.SM

    def test_wait_notify_node_cuda(self):
        node = WaitNotifyNode(
            device=DeviceType.CUDA,
            signal_id=0,
            source_rank=1,
        )
        assert node.op_type == PrimitiveOpType.WAIT_NOTIFY
        assert node.executor == ExecutorType.SM


class TestNodeValidation:
    def test_sm_reduce_invalid_reduce_op(self):
        node = SmReduceNode(reduce_op="invalid", source_rank=0, count=10)
        with pytest.raises(ValueError, match="invalid reduce_op"):
            node.validate()

    def test_sm_reduce_negative_source_rank(self):
        node = SmReduceNode(reduce_op="sum", source_rank=-1, count=10)
        with pytest.raises(ValueError, match="source_rank must be non-negative"):
            node.validate()

    def test_sm_copy_negative_source_rank(self):
        node = SmCopyNode(source_rank=-1, size=10)
        with pytest.raises(ValueError, match="source_rank must be non-negative"):
            node.validate()

    def test_notify_negative_target_rank(self):
        node = NotifyNode(signal_id=0, target_rank=-1)
        with pytest.raises(ValueError, match="target_rank must be non-negative"):
            node.validate()

    def test_wait_notify_negative_source_rank(self):
        node = WaitNotifyNode(signal_id=0, source_rank=-1)
        with pytest.raises(ValueError, match="source_rank must be non-negative"):
            node.validate()

    def test_valid_nodes_pass(self):
        node = SmReduceNode(reduce_op="sum", source_rank=0, count=100)
        assert node.validate() is True

        node2 = TmaCopyNode(source_rank=1, size=4096)
        assert node2.validate() is True


class TestTensorInfo:
    def test_tensor_info_creation(self):
        info = TensorInfo(dtype="float32", shape=(1024,))
        assert info.dtype == "float32"
        assert info.shape == (1024,)

    def test_tensor_info_numel(self):
        info = TensorInfo(dtype="float32", shape=(16, 64))
        assert info.numel() == 1024

    def test_tensor_info_requires_tuple(self):
        with pytest.raises(ValueError, match="shape must be a tuple"):
            TensorInfo(dtype="float32", shape=[1024])


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

class TestGraphBuilding:
    def test_empty_graph(self):
        graph = PrimitiveIRGraph(graph_name="test")
        assert graph.size() == 0
        assert graph.is_empty()

    def test_add_single_node(self):
        graph = PrimitiveIRGraph(graph_name="test")
        node = SmReduceNode(
            device=DeviceType.CUDA,
            reduce_op="sum",
            source_rank=1,
            count=1024,
        )
        graph.add_node(node)
        assert graph.size() == 1
        assert graph.get_node(node.op_id) is node

    def test_duplicate_node_raises(self):
        graph = PrimitiveIRGraph(graph_name="test")
        node = SmReduceNode(op_id="test_node", reduce_op="sum", source_rank=0, count=10)
        graph.add_node(node)
        node2 = SmCopyNode(op_id="test_node", source_rank=0, size=10)
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(node2)

    def test_add_edge(self):
        graph = PrimitiveIRGraph(graph_name="test")
        n1 = SmReduceNode(reduce_op="sum", source_rank=0, count=10)
        n2 = NotifyNode(signal_id=0, target_rank=1, device=DeviceType.CUDA)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge(n1.op_id, n2.op_id)

        assert n1.op_id in n2.dependencies
        assert n2.op_id in n1.next_ops

    def test_cycle_detection(self):
        graph = PrimitiveIRGraph(graph_name="test")
        n1 = SmReduceNode(op_id="n1", reduce_op="sum", source_rank=0, count=10)
        n2 = SmCopyNode(op_id="n2", source_rank=0, size=10)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge(n1.op_id, n2.op_id)
        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge(n2.op_id, n1.op_id)

    def test_topological_sort(self):
        graph = PrimitiveIRGraph(graph_name="test")
        n1 = SmReduceNode(op_id="a", reduce_op="sum", source_rank=0, count=10)
        n2 = NotifyNode(op_id="b", signal_id=0, target_rank=1, device=DeviceType.CUDA)
        n3 = WaitNotifyNode(op_id="c", signal_id=0, source_rank=1, device=DeviceType.CUDA)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        topo = graph.topological_sort()
        ids = [n.op_id for n in topo]
        assert ids == ["a", "b", "c"]

    def test_validate_empty_graph_raises(self):
        graph = PrimitiveIRGraph(graph_name="test")
        with pytest.raises(ValueError, match="empty"):
            graph.validate()

    def test_validate_valid_graph(self):
        graph = PrimitiveIRGraph(graph_name="test")
        n1 = SmReduceNode(reduce_op="sum", source_rank=0, count=10)
        graph.add_node(n1)
        assert graph.validate() is True


# ---------------------------------------------------------------------------
# DSL build_graph API
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_simple_build(self):
        def build(op):
            op.tensor(dtype="float32", shape=(1024,))
            op.sm_reduce(
                source_rank=1,
                src_offset=0,
                dst_offset=0,
                remote_offset=0,
                count=1024,
            )

        graph = build_graph("test_simple", build, device=DeviceType.CUDA)
        assert graph.size() == 1
        nodes = list(graph.nodes.values())
        assert nodes[0].op_type == PrimitiveOpType.SM_REDUCE

    def test_auto_dependency_sequential(self):
        def build(op):
            op.tensor(dtype="float32", shape=(1024,))
            op.sm_reduce(
                source_rank=1,
                src_offset=0, dst_offset=0, remote_offset=0, count=512,
            )
            op.notify(signal_id=0, target_rank=1)

        graph = build_graph("test_seq", build, device=DeviceType.CUDA)
        assert graph.size() == 2
        nodes = list(graph.nodes.values())
        reduce_node = nodes[0]
        notify_node = nodes[1]
        assert reduce_node.op_id in notify_node.dependencies

    def test_stream_isolation(self):
        def build(op):
            op.tensor(dtype="float32", shape=(1024,))
            with Stream("s0"):
                op.sm_reduce(
                    source_rank=1,
                    src_offset=0, dst_offset=0, remote_offset=0, count=512,
                )
                op.notify(signal_id=0, target_rank=1)
            with Stream("s1"):
                op.sm_copy(
                    source_rank=1,
                    src_offset=0, dst_offset=512, size=2048,
                )
                op.notify(signal_id=1, target_rank=1)

        graph = build_graph("test_streams", build, device=DeviceType.CUDA)
        assert graph.size() == 4

        nodes = list(graph.nodes.values())
        reduce_node = nodes[0]
        notify0 = nodes[1]
        copy_node = nodes[2]
        notify1 = nodes[3]

        assert reduce_node.op_id in notify0.dependencies
        assert copy_node.op_id in notify1.dependencies
        # Cross-stream: no dependency between s0 and s1
        assert copy_node.op_id not in notify0.dependencies
        assert reduce_node.op_id not in notify1.dependencies

    def test_mixed_executors(self):
        def build(op):
            op.tensor(dtype="float32", shape=(4096,))
            op.sm_reduce(
                source_rank=1,
                src_offset=0, dst_offset=0, remote_offset=0, count=2048,
            )
            op.tma_copy(
                source_rank=1,
                src_offset=0, dst_offset=0, size=8192,
            )

        graph = build_graph("test_mixed", build, device=DeviceType.CUDA)
        nodes = list(graph.nodes.values())
        assert nodes[0].executor == ExecutorType.SM
        assert nodes[1].executor == ExecutorType.TMA

    def test_all_op_types(self):
        """Test all remaining op types (CPU nodes and barrier removed)."""
        def build(op):
            op.tensor(dtype="float32", shape=(1024,))
            op.sm_reduce(source_rank=0, src_offset=0, dst_offset=0, remote_offset=0, count=256)
            op.sm_copy(source_rank=0, src_offset=0, dst_offset=0, size=1024)
            op.tma_copy(source_rank=0, src_offset=0, dst_offset=0, size=1024)
            op.tma_reduce(source_rank=0, src_offset=0, dst_offset=0, remote_offset=0, count=256)
            op.ce_copy(source_rank=0, src_offset=0, dst_offset=0, size=1024)
            op.rdma_write(target_rank=1, src_offset=0, dst_offset=0, size=1024)
            op.rdma_read(source_rank=1, src_offset=0, dst_offset=0, size=1024)
            op.notify(signal_id=0, target_rank=1)
            op.wait_notify(signal_id=0, source_rank=1)

        graph = build_graph("test_all_ops", build, device=DeviceType.CUDA)
        assert graph.size() == 9


# ---------------------------------------------------------------------------
# Node to_params()
# ---------------------------------------------------------------------------

class TestNodeParams:
    def test_sm_reduce_params(self):
        node = SmReduceNode(reduce_op="sum", source_rank=1, src_offset=0,
                            dst_offset=100, remote_offset=200, count=512)
        params = node.to_params()
        assert params["reduce_op"] == "sum"
        assert params["source_rank"] == 1
        assert params["count"] == 512
        assert params["dst_offset"] == 100
        assert params["remote_offset"] == 200

    def test_sm_copy_params(self):
        node = SmCopyNode(source_rank=2, src_offset=0, dst_offset=100, size=4096)
        params = node.to_params()
        assert params["source_rank"] == 2
        assert params["size"] == 4096

    def test_notify_params(self):
        node = NotifyNode(signal_id=3, target_rank=1)
        params = node.to_params()
        assert params["signal_id"] == 3
        assert params["target_rank"] == 1

    def test_wait_notify_params(self):
        node = WaitNotifyNode(signal_id=2, source_rank=0)
        params = node.to_params()
        assert params["signal_id"] == 2
        assert params["source_rank"] == 0

    def test_base_node_params_empty(self):
        node = IRNode()
        assert node.to_params() == {}


# ---------------------------------------------------------------------------
# JSON v2 generation
# ---------------------------------------------------------------------------

class TestJsonV2Generation:
    def _build_simple_graph(self):
        def build(op):
            op.tensor(dtype="float32", shape=(8192,))
            op.sm_reduce(
                source_rank=1,
                src_offset=0, dst_offset=0, remote_offset=0,
                count=4096,
            )
            op.notify(signal_id=0, target_rank=1)

        return build_graph("test_json", build, device=DeviceType.CUDA)

    def test_json_has_version_2(self):
        graph = self._build_simple_graph()
        json_str = compile_to_json_string(graph)
        data = json.loads(json_str)
        assert data["version"] == 2

    def test_json_has_tensor_info(self):
        graph = self._build_simple_graph()
        json_str = compile_to_json_string(graph)
        data = json.loads(json_str)
        assert "tensor_info" in data
        assert data["tensor_info"]["dtype"] == "float32"
        assert data["tensor_info"]["shape"] == [8192]

    def test_json_has_executors_list(self):
        graph = self._build_simple_graph()
        json_str = compile_to_json_string(graph)
        data = json.loads(json_str)
        assert "executors" in data
        assert isinstance(data["executors"], list)
        assert len(data["executors"]) > 0

    def test_json_operations_have_executor(self):
        graph = self._build_simple_graph()
        json_str = compile_to_json_string(graph)
        data = json.loads(json_str)
        for op in data["operations"]:
            assert "executor" in op
            assert "primitive" in op
            assert "index" in op
            assert "dependencies" in op
            assert "next_ops" in op
            assert "params" in op

    def test_json_sm_reduce_primitive_name(self):
        graph = self._build_simple_graph()
        json_str = compile_to_json_string(graph)
        data = json.loads(json_str)
        reduce_ops = [op for op in data["operations"] if "reduce" in op["primitive"]]
        assert len(reduce_ops) >= 1
        assert reduce_ops[0]["primitive"] == "sm.reduce"

    def test_json_mixed_executors(self):
        def build(op):
            op.tensor(dtype="float32", shape=(4096,))
            with Stream("s0"):
                op.sm_reduce(
                    source_rank=1,
                    src_offset=0, dst_offset=0, remote_offset=0, count=2048,
                )
            with Stream("s1"):
                op.tma_copy(
                    source_rank=1,
                    src_offset=0, dst_offset=0, size=8192,
                )

        graph = build_graph("test_mixed_json", build, device=DeviceType.CUDA)
        json_str = compile_to_json_string(graph)
        data = json.loads(json_str)

        executors = data["executors"]
        assert "cuda_sm" in executors
        assert "cuda_tma" in executors

    def test_json_dependency_indices(self):
        graph = self._build_simple_graph()
        json_str = compile_to_json_string(graph)
        data = json.loads(json_str)
        ops = data["operations"]
        # First op should have no dependencies
        assert ops[0]["dependencies"] == []
        # Second op should depend on first
        assert 0 in ops[1]["dependencies"]

    def test_json_roundtrip_ring_allreduce(self):
        """Test a realistic 2-GPU ring allreduce produces valid JSON."""
        world_size = 2
        tensor_size = 16384
        chunk_size = tensor_size // world_size
        element_size = 4

        for rank in range(world_size):
            def build(op, r=rank):
                op.tensor(dtype="float32", shape=(tensor_size,))
                prev_rank = (r - 1) % world_size
                next_rank = (r + 1) % world_size

                for step in range(world_size - 1):
                    offset = ((r - step - 1) % world_size) * chunk_size
                    with Stream(f"rs_{step}"):
                        if step > 0:
                            op.wait_notify(signal_id=step - 1, source_rank=prev_rank)
                        op.sm_reduce(
                            reduce_op="sum",
                            source_rank=prev_rank,
                            src_offset=offset * element_size,
                            dst_offset=offset * element_size,
                            remote_offset=offset * element_size,
                            count=chunk_size,
                        )
                        op.notify(signal_id=step, target_rank=next_rank)

                for step in range(world_size - 1):
                    offset = ((r - step) % world_size) * chunk_size
                    with Stream(f"ag_{step}"):
                        op.wait_notify(
                            signal_id=world_size - 1 + step,
                            source_rank=prev_rank,
                        )
                        op.tma_copy(
                            source_rank=prev_rank,
                            src_offset=offset * element_size,
                            dst_offset=offset * element_size,
                            size=chunk_size * element_size,
                        )
                        op.notify(
                            signal_id=world_size - 1 + step,
                            target_rank=next_rank,
                        )

            graph = build_graph(f"ring_ar_rank{rank}", build, device=DeviceType.CUDA)
            json_str = compile_to_json_string(graph)
            data = json.loads(json_str)

            assert data["version"] == 2
            assert len(data["operations"]) > 0
            assert "cuda_sm" in data["executors"]
            assert "cuda_tma" in data["executors"]

            # Verify all op indices are contiguous
            indices = [op["index"] for op in data["operations"]]
            assert indices == list(range(len(indices)))

            # Verify dependency indices reference valid ops
            for op in data["operations"]:
                for dep_idx in op["dependencies"]:
                    assert 0 <= dep_idx < len(data["operations"])
                for next_idx in op["next_ops"]:
                    assert 0 <= next_idx < len(data["operations"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
