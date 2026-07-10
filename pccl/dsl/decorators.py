"""Python DSL API for PCCL - Executor-Aware Architecture"""

from typing import Optional, List, Any, Callable, Dict
from dataclasses import dataclass, field

from .nodes import (
    DeviceType,
    TensorInfo,
    IRNodeVariant,
    SmReduceNode,
    SmCopyNode,
    TmaCopyNode,
    TmaReduceNode,
    MultimemReduceNode,
    MultimemStoreNode,
    CeCopyNode,
    RdmaWriteNode,
    RdmaReadNode,
    NotifyNode,
    WaitNotifyNode,
    OcsBarrierNode,
)
from .graph import PrimitiveIRGraph


@dataclass
class _StreamState:
    name: str
    last_node_id: Optional[str] = None
    pending_dependencies: List[str] = field(default_factory=list)

    def take_pending_dependencies(self) -> List[str]:
        deps = self.pending_dependencies.copy()
        self.pending_dependencies.clear()
        return deps


class Stream:
    _active_comm_op: Optional['CommunicationOp'] = None

    def __init__(self, name: str):
        self.name = name
        self._prev_stream_name: Optional[str] = None

    def __enter__(self) -> 'Stream':
        if not Stream._active_comm_op:
            raise RuntimeError("Stream must be used within a CommunicationOp context")
        self._comm_op = Stream._active_comm_op
        self._prev_stream_name = self._comm_op._active_stream_name
        self._comm_op._get_or_create_stream(self.name)
        self._comm_op._active_stream_name = self.name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._comm_op:
            self._comm_op._active_stream_name = self._prev_stream_name
        return False


class CommunicationOp:
    def __init__(
        self,
        name: str,
        device: DeviceType = DeviceType.CUDA,
        auto_dependency: bool = True,
    ):
        self.name = name
        self.default_device = device
        self.auto_dependency = auto_dependency

        self.graph = PrimitiveIRGraph(graph_name=name)
        self.current_tensor_info: Optional[TensorInfo] = None
        self.last_node_id: Optional[str] = None
        self._nodes_in_order: List[str] = []
        self._current_channel: int = 0

        self._streams: Dict[str, '_StreamState'] = {}
        self._active_stream_name: Optional[str] = None
        self._pending_dependencies: List[str] = []

    @property
    def active_stream(self) -> Optional['_StreamState']:
        if self._active_stream_name:
            return self._streams.get(self._active_stream_name)
        return None

    def _get_or_create_stream(self, name: str) -> '_StreamState':
        if name not in self._streams:
            self._streams[name] = _StreamState(name)
        return self._streams[name]

    def __enter__(self) -> 'CommunicationOp':
        Stream._active_comm_op = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Stream._active_comm_op = None
        if exc_type is None:
            self._finalize_dependencies()
        return False

    def tensor(self, dtype: Any, shape: tuple) -> 'CommunicationOp':
        self.current_tensor_info = TensorInfo(dtype=dtype, shape=shape)
        return self

    def set_channel(self, channel: int) -> 'CommunicationOp':
        self._current_channel = channel
        return self

    def _add_node(self, node: IRNodeVariant) -> IRNodeVariant:
        if self.current_tensor_info:
            node.tensor_info = self.current_tensor_info

        node.channel = self._current_channel

        if self.auto_dependency:
            stream = self.active_stream
            if stream:
                pending = stream.take_pending_dependencies()
                for dep_id in pending:
                    node.add_dependency(dep_id)
                if stream.last_node_id:
                    node.add_dependency(stream.last_node_id)
            else:
                if self.last_node_id:
                    node.add_dependency(self.last_node_id)

        self.graph.add_node(node)

        stream = self.active_stream
        if stream:
            stream.last_node_id = node.op_id
        else:
            self._track_node(node.op_id)
        return node

    # --- SM Operations ---
    def sm_reduce(self, source_rank: int, src_offset: int, dst_offset: int,
                  remote_offset: int, count: int, reduce_op: str = "sum") -> SmReduceNode:
        node = SmReduceNode(reduce_op=reduce_op, source_rank=source_rank,
                            src_offset=src_offset, dst_offset=dst_offset,
                            remote_offset=remote_offset, count=count, device=self.default_device)
        return self._add_node(node)

    def sm_copy(self, source_rank: int, src_offset: int, dst_offset: int,
                size: int) -> SmCopyNode:
        node = SmCopyNode(source_rank=source_rank, src_offset=src_offset,
                          dst_offset=dst_offset, size=size, device=self.default_device)
        return self._add_node(node)

    # --- TMA Operations ---
    def tma_copy(self, source_rank: int, src_offset: int, dst_offset: int,
                 size: int) -> TmaCopyNode:
        node = TmaCopyNode(source_rank=source_rank, src_offset=src_offset,
                           dst_offset=dst_offset, size=size, device=self.default_device)
        return self._add_node(node)

    def tma_reduce(self, source_rank: int, src_offset: int, dst_offset: int,
                   remote_offset: int, count: int, reduce_op: str = "sum") -> TmaReduceNode:
        node = TmaReduceNode(reduce_op=reduce_op, source_rank=source_rank,
                             src_offset=src_offset, dst_offset=dst_offset,
                             remote_offset=remote_offset, count=count, device=self.default_device)
        return self._add_node(node)

    # --- Copy Engine Operations ---
    def ce_copy(self, source_rank: int, src_offset: int, dst_offset: int,
                size: int) -> CeCopyNode:
        node = CeCopyNode(source_rank=source_rank, src_offset=src_offset,
                          dst_offset=dst_offset, size=size, device=self.default_device)
        return self._add_node(node)

    # --- Multimem (NVLS) Operations ---
    def multimem_reduce(self, source_rank: int, src_offset: int, dst_offset: int,
                        remote_offset: int, count: int,
                        reduce_op: str = "sum") -> MultimemReduceNode:
        node = MultimemReduceNode(reduce_op=reduce_op, source_rank=source_rank,
                                  src_offset=src_offset, dst_offset=dst_offset,
                                  remote_offset=remote_offset, count=count,
                                  device=self.default_device)
        return self._add_node(node)

    def multimem_store(self, source_rank: int, src_offset: int, dst_offset: int,
                       size: int) -> MultimemStoreNode:
        node = MultimemStoreNode(source_rank=source_rank, src_offset=src_offset,
                                 dst_offset=dst_offset, size=size,
                                 device=self.default_device)
        return self._add_node(node)

    # --- RDMA Operations ---
    def rdma_write(self, target_rank: int, src_offset: int, dst_offset: int,
                   size: int) -> RdmaWriteNode:
        node = RdmaWriteNode(target_rank=target_rank, src_offset=src_offset,
                             dst_offset=dst_offset, size=size, device=self.default_device)
        return self._add_node(node)

    def rdma_read(self, source_rank: int, src_offset: int, dst_offset: int,
                  size: int) -> RdmaReadNode:
        node = RdmaReadNode(source_rank=source_rank, src_offset=src_offset,
                            dst_offset=dst_offset, size=size, device=self.default_device)
        return self._add_node(node)

    # --- Synchronization ---
    def notify(self, signal_id: int, target_rank: int) -> NotifyNode:
        node = NotifyNode(signal_id=signal_id, target_rank=target_rank, device=self.default_device)
        return self._add_node(node)

    def wait_notify(self, signal_id: int, source_rank: int) -> WaitNotifyNode:
        node = WaitNotifyNode(signal_id=signal_id, source_rank=source_rank, device=self.default_device)
        return self._add_node(node)

    def ocs_barrier(
        self,
        barrier_id: int,
        epoch_id: int,
        next_epoch_id: int,
        participant_ranks: tuple,
        topology_id: int,
        route_plan_id: int,
        group_id: int = 0,
        route_mode: str = "STATIC_PLAN",
        algorithm: str = "auto",
        backend: str = "pccl",
        payload: bytes = b"",
        timeout_ms: int = 0,
    ) -> OcsBarrierNode:
        """Insert a graph-wide OCS reconfiguration boundary.

        The barrier joins every node built so far and becomes the dependency
        root for every later stream. This is intentionally stronger than a
        point-to-point notify/wait pair: it creates a safe host-control phase
        cut for topology commit and release.
        """
        node = OcsBarrierNode(
            group_id=group_id,
            barrier_id=barrier_id,
            epoch_id=epoch_id,
            next_epoch_id=next_epoch_id,
            participant_ranks=participant_ranks,
            topology_id=topology_id,
            route_mode=route_mode,
            route_plan_id=route_plan_id,
            algorithm=algorithm,
            backend=backend,
            payload=payload,
            timeout_ms=timeout_ms,
            device=DeviceType.CPU,
        )

        # A topology switch is valid only after every previously emitted
        # operation has completed. Depend on the whole existing graph rather
        # than the active stream alone, then reset all stream frontiers.
        for previous_id in self.graph.nodes:
            node.add_dependency(previous_id)
        self.graph.add_node(node)
        for dep_id in node.dependencies:
            self.graph.get_node(dep_id).add_next_op(node.op_id)

        self.last_node_id = node.op_id
        self._nodes_in_order.append(node.op_id)
        self._pending_dependencies.clear()
        for stream_state in self._streams.values():
            stream_state.last_node_id = node.op_id
            stream_state.pending_dependencies.clear()
        return node

    # --- Stream synchronization ---
    def wait(self, stream_name: str) -> None:
        target_stream = self._streams.get(stream_name)
        if not target_stream:
            raise ValueError(f"Stream '{stream_name}' not found")
        if not target_stream.last_node_id:
            raise ValueError(f"Stream '{stream_name}' has no operations to wait for")
        current_stream = self.active_stream
        if current_stream:
            current_stream.pending_dependencies.append(target_stream.last_node_id)
        else:
            self._pending_dependencies.append(target_stream.last_node_id)

    def wait_for(self, node: IRNodeVariant) -> None:
        if not node or not node.op_id:
            raise ValueError("Invalid node reference")
        current_stream = self.active_stream
        if current_stream:
            current_stream.pending_dependencies.append(node.op_id)
        else:
            self._pending_dependencies.append(node.op_id)

    # --- Internal ---
    def _track_node(self, node_id: str) -> None:
        self.last_node_id = node_id
        self._nodes_in_order.append(node_id)

    def _finalize_dependencies(self) -> None:
        if self.graph.is_empty():
            return

        for stream_state in self._streams.values():
            pending = stream_state.take_pending_dependencies()
            if pending and stream_state.last_node_id:
                last_node = self.graph.get_node(stream_state.last_node_id)
                for dep_id in pending:
                    last_node.add_dependency(dep_id)

        if self._pending_dependencies:
            if self.last_node_id:
                last_node = self.graph.get_node(self.last_node_id)
                for dep_id in self._pending_dependencies:
                    last_node.add_dependency(dep_id)
            self._pending_dependencies.clear()

        for node_id, node in self.graph.nodes.items():
            for dep_id in node.dependencies:
                if dep_id in self.graph.nodes:
                    dep_node = self.graph.nodes[dep_id]
                    if node_id not in dep_node.next_ops:
                        dep_node.add_next_op(node_id)

        try:
            self.graph.validate()
        except ValueError as e:
            raise ValueError(f"Graph validation failed: {e}")

    def get_graph(self) -> PrimitiveIRGraph:
        return self.graph


def build_graph(
    name: str,
    builder_func: Callable[['CommunicationOp'], None],
    device: DeviceType = DeviceType.CUDA,
) -> PrimitiveIRGraph:
    with CommunicationOp(name=name, device=device) as op:
        builder_func(op)
        return op.get_graph()
