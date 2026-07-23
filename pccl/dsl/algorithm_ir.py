"""Hardware-independent collective algorithm IR and PCCL primitive lowering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
from typing import Dict, List, Set, Tuple

from .decorators import CommunicationOp, Stream
from .graph import PrimitiveIRGraph
from .nodes import DeviceType, IRNodeVariant


class AlgorithmIRError(ValueError):
    """Raised when a collective algorithm description is inconsistent."""


class AlgorithmBuffer(str, Enum):
    """Logical buffers used before executor-specific memory layout is chosen."""

    INPUT = "input"
    OUTPUT = "output"
    SCRATCH = "scratch"


class AlgorithmPrimitive(str, Enum):
    """Data movement and reduction primitives understood by Algorithm IR."""

    COPY = "copy"
    REDUCE = "reduce"


@dataclass(frozen=True)
class ChunkRef:
    """A contiguous range of logical chunks in one rank-local buffer."""

    rank: int
    buffer: AlgorithmBuffer
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        try:
            buffer = AlgorithmBuffer(self.buffer)
        except ValueError as exc:
            raise AlgorithmIRError("unsupported algorithm buffer {!r}".format(self.buffer)) from exc
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise AlgorithmIRError("chunk rank must be a non-negative integer")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise AlgorithmIRError("chunk index must be a non-negative integer")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise AlgorithmIRError("chunk count must be a positive integer")
        object.__setattr__(self, "buffer", buffer)

    def keys(self) -> Tuple[Tuple[int, str, int], ...]:
        return tuple(
            (self.rank, self.buffer.value, index)
            for index in range(self.index, self.index + self.count)
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "buffer": self.buffer.value,
            "index": self.index,
            "count": self.count,
        }


@dataclass(frozen=True)
class AlgorithmTransfer:
    """A global send/receive relation executed as copy or reduce at dst.rank."""

    op_id: str
    primitive: AlgorithmPrimitive
    src: ChunkRef
    dst: ChunkRef
    dependencies: Tuple[str, ...] = ()
    requires_signal: bool = True
    channel: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.op_id, str) or not self.op_id:
            raise AlgorithmIRError("transfer op_id must not be empty")
        try:
            primitive = AlgorithmPrimitive(self.primitive)
        except ValueError as exc:
            raise AlgorithmIRError(
                "unsupported algorithm primitive {!r}".format(self.primitive)
            ) from exc
        if self.src.count != self.dst.count:
            raise AlgorithmIRError("transfer source and destination counts must match")
        if not isinstance(self.requires_signal, bool):
            raise AlgorithmIRError("transfer requires_signal must be a boolean")
        if isinstance(self.channel, bool) or not isinstance(self.channel, int) or self.channel < 0:
            raise AlgorithmIRError("transfer channel must be a non-negative integer")
        if any(
            not isinstance(dependency, str) or not dependency for dependency in self.dependencies
        ):
            raise AlgorithmIRError("transfer dependencies must contain non-empty operation IDs")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise AlgorithmIRError("transfer dependencies must not contain duplicates")
        object.__setattr__(self, "primitive", primitive)
        object.__setattr__(self, "dependencies", tuple(self.dependencies))

    def to_dict(self) -> Dict[str, object]:
        return {
            "op_id": self.op_id,
            "primitive": self.primitive.value,
            "src": self.src.to_dict(),
            "dst": self.dst.to_dict(),
            "dependencies": list(self.dependencies),
            "requires_signal": self.requires_signal,
            "channel": self.channel,
        }


@dataclass(frozen=True)
class AlgorithmSync:
    """One directed signal edge used to form an algorithm-level rendezvous."""

    src_rank: int
    dst_rank: int
    channel: int = 0

    def __post_init__(self) -> None:
        for name, value in (("src_rank", self.src_rank), ("dst_rank", self.dst_rank)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AlgorithmIRError("{} must be a non-negative integer".format(name))
        if self.src_rank == self.dst_rank:
            raise AlgorithmIRError("sync edge must connect two different ranks")
        if isinstance(self.channel, bool) or not isinstance(self.channel, int) or self.channel < 0:
            raise AlgorithmIRError("sync channel must be a non-negative integer")

    def to_dict(self) -> Dict[str, int]:
        return {
            "src_rank": self.src_rank,
            "dst_rank": self.dst_rank,
            "channel": self.channel,
        }


@dataclass(frozen=True)
class AlgorithmStep:
    """A globally ordered set of parallel transfers or synchronization edges."""

    index: int
    name: str
    transfers: Tuple[AlgorithmTransfer, ...] = ()
    sync_edges: Tuple[AlgorithmSync, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise AlgorithmIRError("step index must be a non-negative integer")
        if not isinstance(self.name, str) or not self.name:
            raise AlgorithmIRError("step name must not be empty")
        if self.transfers and self.sync_edges:
            raise AlgorithmIRError("one step cannot mix data transfers and sync edges")
        if not self.transfers and not self.sync_edges:
            raise AlgorithmIRError("step must contain transfers or sync edges")
        object.__setattr__(self, "transfers", tuple(self.transfers))
        object.__setattr__(self, "sync_edges", tuple(self.sync_edges))

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "transfers": [transfer.to_dict() for transfer in self.transfers],
            "sync_edges": [edge.to_dict() for edge in self.sync_edges],
        }


@dataclass(frozen=True)
class CollectiveAlgorithmIR:
    """A complete hardware-independent collective schedule for all ranks."""

    name: str
    collective_type: str
    world_size: int
    chunks_per_rank: int
    steps: Tuple[AlgorithmStep, ...]
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        self.validate()

    def validate(self) -> None:
        if self.version != 1:
            raise AlgorithmIRError("unsupported Algorithm IR version {}".format(self.version))
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.collective_type, str)
            or not self.collective_type
        ):
            raise AlgorithmIRError("algorithm name and collective_type must not be empty")
        if (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size < 2
        ):
            raise AlgorithmIRError("world_size must be at least two")
        if (
            isinstance(self.chunks_per_rank, bool)
            or not isinstance(self.chunks_per_rank, int)
            or self.chunks_per_rank <= 0
        ):
            raise AlgorithmIRError("chunks_per_rank must be positive")
        if not self.steps:
            raise AlgorithmIRError("algorithm must contain at least one step")
        if tuple(step.index for step in self.steps) != tuple(range(len(self.steps))):
            raise AlgorithmIRError("step indices must be contiguous from zero")

        op_step: Dict[str, int] = {}
        for step in self.steps:
            written_in_step: Set[Tuple[int, str, int]] = set()
            readers_in_step: Dict[Tuple[int, str, int], Set[str]] = {}
            signaled_targets: Set[Tuple[int, int]] = set()
            sync_targets: Set[Tuple[int, int]] = set()
            for transfer in step.transfers:
                for key in transfer.src.keys():
                    readers_in_step.setdefault(key, set()).add(transfer.op_id)
            for transfer in step.transfers:
                if transfer.op_id in op_step:
                    raise AlgorithmIRError("duplicate transfer op_id {!r}".format(transfer.op_id))
                self._validate_chunk(transfer.src)
                self._validate_chunk(transfer.dst)
                for key in transfer.dst.keys():
                    if key in written_in_step:
                        raise AlgorithmIRError(
                            "step {} writes chunk {} more than once".format(step.index, key)
                        )
                    parallel_readers = readers_in_step.get(key, set()).difference(
                        {
                            transfer.op_id,
                        }
                    )
                    if parallel_readers:
                        raise AlgorithmIRError(
                            "step {} reads and writes chunk {} in parallel".format(step.index, key)
                        )
                    written_in_step.add(key)
                if transfer.requires_signal and transfer.src.rank != transfer.dst.rank:
                    target = (transfer.dst.rank, transfer.channel)
                    if target in signaled_targets:
                        raise AlgorithmIRError(
                            (
                                "step {} has multiple signaled transfers into " "rank {} channel {}"
                            ).format(step.index, *target)
                        )
                    signaled_targets.add(target)
                for dependency in transfer.dependencies:
                    if dependency not in op_step or op_step[dependency] >= step.index:
                        raise AlgorithmIRError(
                            "transfer {!r} depends on unknown or non-prior op {!r}".format(
                                transfer.op_id, dependency
                            )
                        )
                op_step[transfer.op_id] = step.index

            for edge in step.sync_edges:
                self._validate_rank(edge.src_rank)
                self._validate_rank(edge.dst_rank)
                target = (edge.dst_rank, edge.channel)
                if target in sync_targets:
                    raise AlgorithmIRError(
                        "step {} has multiple sync edges into rank {} channel {}".format(
                            step.index, *target
                        )
                    )
                sync_targets.add(target)

    def _validate_rank(self, rank: int) -> None:
        if rank >= self.world_size:
            raise AlgorithmIRError("rank {} is outside world_size {}".format(rank, self.world_size))

    def _validate_chunk(self, chunk: ChunkRef) -> None:
        self._validate_rank(chunk.rank)
        if chunk.index + chunk.count > self.chunks_per_rank:
            raise AlgorithmIRError(
                "chunk [{}, {}) exceeds chunks_per_rank {}".format(
                    chunk.index, chunk.index + chunk.count, self.chunks_per_rank
                )
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "name": self.name,
            "collective_type": self.collective_type,
            "world_size": self.world_size,
            "chunks_per_rank": self.chunks_per_rank,
            "steps": [step.to_dict() for step in self.steps],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @property
    def digest(self) -> str:
        encoded = self.canonical_json().encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass
class _MutableStep:
    index: int
    name: str
    transfers: List[AlgorithmTransfer] = field(default_factory=list)
    sync_edges: List[AlgorithmSync] = field(default_factory=list)


class AlgorithmStepBuilder:
    """Fluent builder for one globally parallel Algorithm IR step."""

    def __init__(self, step: _MutableStep) -> None:
        self._step = step

    def copy(
        self,
        src: ChunkRef,
        dst: ChunkRef,
        requires_signal: bool = True,
        channel: int = 0,
    ) -> AlgorithmTransfer:
        return self._transfer(AlgorithmPrimitive.COPY, src, dst, requires_signal, channel)

    def reduce(
        self,
        src: ChunkRef,
        dst: ChunkRef,
        requires_signal: bool = True,
        channel: int = 0,
    ) -> AlgorithmTransfer:
        return self._transfer(AlgorithmPrimitive.REDUCE, src, dst, requires_signal, channel)

    def sync(self, src_rank: int, dst_rank: int, channel: int = 0) -> AlgorithmSync:
        edge = AlgorithmSync(src_rank=src_rank, dst_rank=dst_rank, channel=channel)
        self._step.sync_edges.append(edge)
        return edge

    def _transfer(
        self,
        primitive: AlgorithmPrimitive,
        src: ChunkRef,
        dst: ChunkRef,
        requires_signal: bool,
        channel: int,
    ) -> AlgorithmTransfer:
        op_id = "step{}_{}_{}".format(self._step.index, primitive.value, len(self._step.transfers))
        transfer = AlgorithmTransfer(
            op_id=op_id,
            primitive=primitive,
            src=src,
            dst=dst,
            requires_signal=requires_signal,
            channel=channel,
        )
        self._step.transfers.append(transfer)
        return transfer


class AlgorithmIRBuilder:
    """Build an Algorithm IR and infer chunk last-writer/reader dependencies."""

    def __init__(
        self,
        name: str,
        collective_type: str,
        world_size: int,
        chunks_per_rank: int,
    ) -> None:
        self.name = name
        self.collective_type = collective_type
        self.world_size = world_size
        self.chunks_per_rank = chunks_per_rank
        self._steps: List[_MutableStep] = []

    def chunk(
        self,
        rank: int,
        index: int,
        buffer: AlgorithmBuffer = AlgorithmBuffer.INPUT,
        count: int = 1,
    ) -> ChunkRef:
        return ChunkRef(rank=rank, buffer=buffer, index=index, count=count)

    def step(self, name: str) -> AlgorithmStepBuilder:
        step = _MutableStep(index=len(self._steps), name=name)
        self._steps.append(step)
        return AlgorithmStepBuilder(step)

    def build(self) -> CollectiveAlgorithmIR:
        steps = self._infer_dependencies()
        return CollectiveAlgorithmIR(
            name=self.name,
            collective_type=self.collective_type,
            world_size=self.world_size,
            chunks_per_rank=self.chunks_per_rank,
            steps=steps,
        )

    def _infer_dependencies(self) -> Tuple[AlgorithmStep, ...]:
        last_writer: Dict[Tuple[int, str, int], str] = {}
        last_readers: Dict[Tuple[int, str, int], Set[str]] = {}
        result: List[AlgorithmStep] = []

        for mutable in self._steps:
            transfers: List[AlgorithmTransfer] = []
            for transfer in mutable.transfers:
                dependencies: Set[str] = set()
                for key in transfer.src.keys():
                    writer = last_writer.get(key)
                    if writer is not None:
                        dependencies.add(writer)
                for key in transfer.dst.keys():
                    writer = last_writer.get(key)
                    if writer is not None:
                        dependencies.add(writer)
                    dependencies.update(last_readers.get(key, set()))
                transfers.append(replace(transfer, dependencies=tuple(sorted(dependencies))))

            # All transfers in one step are parallel and observe prior-step state.
            for transfer in transfers:
                for key in transfer.src.keys():
                    last_readers.setdefault(key, set()).add(transfer.op_id)
            for transfer in transfers:
                for key in transfer.dst.keys():
                    last_writer[key] = transfer.op_id
                    last_readers[key] = set()

            result.append(
                AlgorithmStep(
                    index=mutable.index,
                    name=mutable.name,
                    transfers=tuple(transfers),
                    sync_edges=tuple(mutable.sync_edges),
                )
            )
        return tuple(result)


class AlgorithmIRLowerer:
    """Lower global Algorithm IR into one rank-local PCCL PrimitiveIRGraph."""

    SIGNALS_PER_CHANNEL = 1024
    MAX_SIGNALS = 4096

    def __init__(self, signal_base: int = 0) -> None:
        if isinstance(signal_base, bool) or not isinstance(signal_base, int) or signal_base < 0:
            raise AlgorithmIRError("signal_base must be non-negative")
        self.signal_base = signal_base

    def lower(
        self,
        algorithm: CollectiveAlgorithmIR,
        rank: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
    ) -> PrimitiveIRGraph:
        if not isinstance(algorithm, CollectiveAlgorithmIR):
            raise TypeError("algorithm must be a CollectiveAlgorithmIR")
        algorithm.validate()
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= algorithm.world_size
        ):
            raise AlgorithmIRError(
                "rank {} is outside world_size {}".format(rank, algorithm.world_size)
            )
        if (
            isinstance(tensor_size, bool)
            or not isinstance(tensor_size, int)
            or tensor_size <= 0
            or tensor_size % algorithm.chunks_per_rank
        ):
            raise AlgorithmIRError("tensor_size must be divisible by chunks_per_rank")
        if executor not in {"sm", "tma"}:
            raise AlgorithmIRError("executor must be 'sm' or 'tma'")
        chunk_size = tensor_size // algorithm.chunks_per_rank
        completion_nodes: Dict[str, IRNodeVariant] = {}

        name = "{}_{}_rank{}".format(algorithm.name, executor, rank)
        with CommunicationOp(name=name, device=DeviceType.CUDA) as op:
            op.tensor(dtype=dtype, shape=(tensor_size,))
            for step in algorithm.steps:
                channels = self._step_channels(step)
                for channel in channels:
                    with Stream("ch{}".format(channel)):
                        op.set_channel(channel)
                        signal_id = self._signal_id(step.index, channel)
                        self._lower_sync_step(op, step, rank, channel, signal_id)
                        self._lower_transfer_step(
                            op,
                            step,
                            rank,
                            channel,
                            signal_id,
                            tensor_size,
                            chunk_size,
                            executor,
                            completion_nodes,
                        )
            graph = op.get_graph()
            graph.collective_type = algorithm.collective_type
            return graph

    def _signal_id(self, step_index: int, channel: int) -> int:
        if step_index >= self.SIGNALS_PER_CHANNEL:
            raise AlgorithmIRError("Algorithm IR has too many steps for one signal channel")
        signal_id = self.signal_base + channel * self.SIGNALS_PER_CHANNEL + step_index
        if signal_id >= self.MAX_SIGNALS:
            raise AlgorithmIRError("generated signal_id exceeds PCCL signal workspace")
        return signal_id

    @staticmethod
    def _step_channels(step: AlgorithmStep) -> Tuple[int, ...]:
        channels = {transfer.channel for transfer in step.transfers}
        channels.update(edge.channel for edge in step.sync_edges)
        return tuple(sorted(channels))

    @staticmethod
    def _lower_sync_step(
        op: CommunicationOp,
        step: AlgorithmStep,
        rank: int,
        channel: int,
        signal_id: int,
    ) -> None:
        for edge in step.sync_edges:
            if edge.channel == channel and edge.src_rank == rank:
                op.notify(signal_id=signal_id, target_rank=edge.dst_rank)
        for edge in step.sync_edges:
            if edge.channel == channel and edge.dst_rank == rank:
                op.wait_notify(signal_id=signal_id, source_rank=edge.src_rank)

    def _lower_transfer_step(
        self,
        op: CommunicationOp,
        step: AlgorithmStep,
        rank: int,
        channel: int,
        signal_id: int,
        tensor_size: int,
        chunk_size: int,
        executor: str,
        completion_nodes: Dict[str, IRNodeVariant],
    ) -> None:
        transfers = [transfer for transfer in step.transfers if transfer.channel == channel]
        for transfer in transfers:
            if transfer.requires_signal and transfer.src.rank == rank and transfer.dst.rank != rank:
                notify_node = op.notify(signal_id=signal_id, target_rank=transfer.dst.rank)
                self._attach_local_dependencies(
                    notify_node, transfer.dependencies, completion_nodes
                )

        for transfer in transfers:
            if transfer.dst.rank != rank:
                continue
            if transfer.requires_signal and transfer.src.rank != rank:
                wait = op.wait_notify(signal_id=signal_id, source_rank=transfer.src.rank)
                self._attach_local_dependencies(wait, transfer.dependencies, completion_nodes)

            src_offset = self._offset(transfer.src, tensor_size, chunk_size)
            dst_offset = self._offset(transfer.dst, tensor_size, chunk_size)
            count = transfer.src.count * chunk_size
            node = self._lower_data_op(
                op,
                transfer,
                src_offset=src_offset,
                dst_offset=dst_offset,
                count=count,
                tensor_size=tensor_size,
                executor=executor,
            )
            self._attach_local_dependencies(node, transfer.dependencies, completion_nodes)
            completion_nodes[transfer.op_id] = node

    @staticmethod
    def _attach_local_dependencies(
        node: IRNodeVariant,
        dependencies: Tuple[str, ...],
        completion_nodes: Dict[str, IRNodeVariant],
    ) -> None:
        for dependency in dependencies:
            previous = completion_nodes.get(dependency)
            if previous is not None:
                node.add_dependency(previous.op_id)

    @staticmethod
    def _offset(chunk: ChunkRef, tensor_size: int, chunk_size: int) -> int:
        base = tensor_size if chunk.buffer is AlgorithmBuffer.SCRATCH else 0
        return base + chunk.index * chunk_size

    @staticmethod
    def _lower_data_op(
        op: CommunicationOp,
        transfer: AlgorithmTransfer,
        src_offset: int,
        dst_offset: int,
        count: int,
        tensor_size: int,
        executor: str,
    ) -> IRNodeVariant:
        tma_compatible = (
            executor == "tma"
            and src_offset + count <= tensor_size
            and dst_offset + count <= tensor_size
        )
        if transfer.primitive is AlgorithmPrimitive.COPY:
            if tma_compatible:
                return op.tma_copy(
                    source_rank=transfer.src.rank,
                    src_offset=src_offset,
                    dst_offset=dst_offset,
                    size=count,
                )
            return op.sm_copy(
                source_rank=transfer.src.rank,
                src_offset=src_offset,
                dst_offset=dst_offset,
                size=count,
            )

        if tma_compatible:
            return op.tma_reduce(
                reduce_op="sum",
                source_rank=transfer.src.rank,
                src_offset=dst_offset,
                dst_offset=dst_offset,
                remote_offset=src_offset,
                count=count,
            )
        return op.sm_reduce(
            reduce_op="sum",
            source_rank=transfer.src.rank,
            src_offset=dst_offset,
            dst_offset=dst_offset,
            remote_offset=src_offset,
            count=count,
        )
