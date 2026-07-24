"""Import standard MSCCL XML schedules into PCCL Primitive IR graphs.

The XML artifact is the compatibility boundary for external schedule
generators such as AICCL and RLCCL. OCS topology and barrier information is
intentionally not embedded here; it belongs to the outer Execution Plan.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Optional, Tuple, Union
from xml.etree import ElementTree

from .decorators import CommunicationOp, Stream
from .graph import PrimitiveIRGraph
from .nodes import DeviceType, IRNode, IRNodeVariant


class MSCCLCompatibilityError(ValueError):
    """Raised when an MSCCL artifact cannot be represented by PCCL."""


class MSCCLBuffer(str, Enum):
    INPUT = "i"
    OUTPUT = "o"
    SCRATCH = "s"


class MSCCLStepType(str, Enum):
    SEND = "s"
    RECV = "r"
    RECV_COPY_SEND = "rcs"
    RECV_REDUCE_SEND = "rrs"
    RECV_REDUCE_COPY = "rrc"
    RECV_REDUCE_COPY_SEND = "rrcs"
    COPY = "cpy"
    REDUCE = "re"
    NOP = "nop"


_FUSED_STEP_TYPES = {
    MSCCLStepType.RECV_COPY_SEND,
    MSCCLStepType.RECV_REDUCE_SEND,
    MSCCLStepType.RECV_REDUCE_COPY_SEND,
}
_RECV_STEP_TYPES = {
    MSCCLStepType.RECV,
    MSCCLStepType.RECV_REDUCE_COPY,
    *_FUSED_STEP_TYPES,
}
_SEND_STEP_TYPES = {MSCCLStepType.SEND, *_FUSED_STEP_TYPES}
_RECV_REDUCE_STEP_TYPES = {
    MSCCLStepType.RECV_REDUCE_COPY,
    MSCCLStepType.RECV_REDUCE_SEND,
    MSCCLStepType.RECV_REDUCE_COPY_SEND,
}
_PROTOCOLS = {"Simple", "LL", "LL128"}
_MAX_SIGNALS = 4096


@dataclass(frozen=True, order=True)
class MSCCLStepKey:
    rank: int
    threadblock_id: int
    step_id: int


@dataclass(frozen=True)
class MSCCLXMLStep:
    key: MSCCLStepKey
    op_type: MSCCLStepType
    src_buffer: Optional[MSCCLBuffer]
    src_offset: int
    dst_buffer: Optional[MSCCLBuffer]
    dst_offset: int
    count: int
    dependency_threadblock: int
    dependency_step: int
    has_dependency: bool

    @property
    def dependency_key(self) -> Optional[MSCCLStepKey]:
        if self.dependency_threadblock < 0:
            return None
        return MSCCLStepKey(
            self.key.rank,
            self.dependency_threadblock,
            self.dependency_step,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "rank": self.key.rank,
            "threadblock_id": self.key.threadblock_id,
            "step_id": self.key.step_id,
            "type": self.op_type.value,
            "srcbuf": None if self.src_buffer is None else self.src_buffer.value,
            "srcoff": self.src_offset,
            "dstbuf": None if self.dst_buffer is None else self.dst_buffer.value,
            "dstoff": self.dst_offset,
            "cnt": self.count,
            "depid": self.dependency_threadblock,
            "deps": self.dependency_step,
            "hasdep": int(self.has_dependency),
        }


@dataclass(frozen=True)
class MSCCLXMLThreadblock:
    rank: int
    threadblock_id: int
    send_peer: int
    recv_peer: int
    channel: int
    steps: Tuple[MSCCLXMLStep, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.threadblock_id,
            "send": self.send_peer,
            "recv": self.recv_peer,
            "chan": self.channel,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class MSCCLXMLGpu:
    rank: int
    input_chunks: int
    output_chunks: int
    scratch_chunks: int
    threadblocks: Tuple[MSCCLXMLThreadblock, ...]

    def chunk_count(self, buffer: MSCCLBuffer) -> int:
        if buffer is MSCCLBuffer.INPUT:
            return self.input_chunks
        if buffer is MSCCLBuffer.OUTPUT:
            return self.output_chunks
        return self.scratch_chunks

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.rank,
            "i_chunks": self.input_chunks,
            "o_chunks": self.output_chunks,
            "s_chunks": self.scratch_chunks,
            "threadblocks": [tb.to_dict() for tb in self.threadblocks],
        }


MSCCLXMLSource = Union[str, bytes, PathLike]


@dataclass(frozen=True)
class MSCCLXMLAlgorithm:
    """Parsed standard MSCCL ``<algo>`` artifact."""

    name: str
    protocol: str
    num_channels: int
    chunks_per_loop: int
    num_gpus: int
    collective: str
    inplace: bool
    gpus: Tuple[MSCCLXMLGpu, ...]

    @classmethod
    def from_xml(cls, xml: Union[str, bytes]) -> "MSCCLXMLAlgorithm":
        try:
            root = ElementTree.fromstring(xml)
        except (ElementTree.ParseError, ValueError) as exc:
            raise MSCCLCompatibilityError("invalid MSCCL XML: {}".format(exc)) from exc
        if root.tag != "algo":
            raise MSCCLCompatibilityError("MSCCL XML root must be <algo>")

        name = _required_attr(root, "name", "algo")
        protocol = _required_attr(root, "proto", "algo")
        num_channels = _int_attr(root, "nchannels", "algo", minimum=1)
        chunks_per_loop = _int_attr(root, "nchunksperloop", "algo", minimum=1)
        num_gpus = _int_attr(root, "ngpus", "algo", minimum=1)
        collective = _required_attr(root, "coll", "algo")
        inplace_value = _int_attr(root, "inplace", "algo", minimum=0)
        if inplace_value not in {0, 1}:
            raise MSCCLCompatibilityError("algo.inplace must be 0 or 1")

        gpus: List[MSCCLXMLGpu] = []
        for gpu_node in root:
            if gpu_node.tag != "gpu":
                raise MSCCLCompatibilityError("unexpected <{}> inside <algo>".format(gpu_node.tag))
            rank = _int_attr(gpu_node, "id", "gpu", minimum=0)
            gpu_path = "gpu[{}]".format(rank)
            threadblocks: List[MSCCLXMLThreadblock] = []
            for tb_node in gpu_node:
                if tb_node.tag != "tb":
                    raise MSCCLCompatibilityError(
                        "unexpected <{}> inside {}".format(tb_node.tag, gpu_path)
                    )
                tb_id = _int_attr(tb_node, "id", gpu_path + ".tb", minimum=0)
                tb_path = "{}.tb[{}]".format(gpu_path, tb_id)
                steps: List[MSCCLXMLStep] = []
                for step_node in tb_node:
                    if step_node.tag != "step":
                        raise MSCCLCompatibilityError(
                            "unexpected <{}> inside {}".format(step_node.tag, tb_path)
                        )
                    step_id = _int_attr(step_node, "s", tb_path + ".step", minimum=0)
                    step_path = "{}.step[{}]".format(tb_path, step_id)
                    raw_type = _required_attr(step_node, "type", step_path)
                    try:
                        op_type = MSCCLStepType(raw_type)
                    except ValueError as exc:
                        raise MSCCLCompatibilityError(
                            "{} has unsupported MSCCL instruction {!r}".format(step_path, raw_type)
                        ) from exc
                    src_buffer = _optional_buffer(step_node.get("srcbuf"), step_path + ".srcbuf")
                    dst_buffer = _optional_buffer(step_node.get("dstbuf"), step_path + ".dstbuf")
                    src_offset = _optional_int_attr(step_node, "srcoff", -1, step_path)
                    dst_offset = _optional_int_attr(step_node, "dstoff", -1, step_path)
                    count = _optional_int_attr(step_node, "cnt", 0, step_path)
                    dep_tb = _optional_int_attr(step_node, "depid", -1, step_path)
                    dep_step = _optional_int_attr(step_node, "deps", -1, step_path)
                    hasdep = _optional_int_attr(step_node, "hasdep", 0, step_path)
                    if hasdep not in {0, 1}:
                        raise MSCCLCompatibilityError("{}.hasdep must be 0 or 1".format(step_path))
                    steps.append(
                        MSCCLXMLStep(
                            key=MSCCLStepKey(rank, tb_id, step_id),
                            op_type=op_type,
                            src_buffer=src_buffer,
                            src_offset=src_offset,
                            dst_buffer=dst_buffer,
                            dst_offset=dst_offset,
                            count=count,
                            dependency_threadblock=dep_tb,
                            dependency_step=dep_step,
                            has_dependency=bool(hasdep),
                        )
                    )
                threadblocks.append(
                    MSCCLXMLThreadblock(
                        rank=rank,
                        threadblock_id=tb_id,
                        send_peer=_int_attr(tb_node, "send", tb_path),
                        recv_peer=_int_attr(tb_node, "recv", tb_path),
                        channel=_int_attr(tb_node, "chan", tb_path, minimum=0),
                        steps=tuple(steps),
                    )
                )
            gpus.append(
                MSCCLXMLGpu(
                    rank=rank,
                    input_chunks=_int_attr(gpu_node, "i_chunks", gpu_path, minimum=0),
                    output_chunks=_int_attr(gpu_node, "o_chunks", gpu_path, minimum=0),
                    scratch_chunks=_int_attr(gpu_node, "s_chunks", gpu_path, minimum=0),
                    threadblocks=tuple(threadblocks),
                )
            )

        algorithm = cls(
            name=name,
            protocol=protocol,
            num_channels=num_channels,
            chunks_per_loop=chunks_per_loop,
            num_gpus=num_gpus,
            collective=collective,
            inplace=bool(inplace_value),
            gpus=tuple(sorted(gpus, key=lambda item: item.rank)),
        )
        algorithm.validate()
        return algorithm

    @classmethod
    def load(cls, path: Union[str, PathLike]) -> "MSCCLXMLAlgorithm":
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise MSCCLCompatibilityError(
                "cannot read MSCCL XML artifact {!r}: {}".format(str(path), exc)
            ) from exc
        return cls.from_xml(data)

    @classmethod
    def resolve(cls, source: Union["MSCCLXMLAlgorithm", MSCCLXMLSource]) -> "MSCCLXMLAlgorithm":
        if isinstance(source, cls):
            return source
        if isinstance(source, bytes):
            return cls.from_xml(source)
        if isinstance(source, PathLike):
            return cls.load(source)
        if isinstance(source, str):
            if source.lstrip().startswith("<"):
                return cls.from_xml(source)
            return cls.load(source)
        raise TypeError("MSCCL artifact must be XML text, bytes, a path, or a parsed artifact")

    def validate(self) -> None:
        if self.protocol not in _PROTOCOLS:
            raise MSCCLCompatibilityError(
                "unsupported MSCCL protocol {!r}; expected one of {}".format(
                    self.protocol, sorted(_PROTOCOLS)
                )
            )
        if len(self.gpus) != self.num_gpus:
            raise MSCCLCompatibilityError(
                "algo.ngpus={} but XML contains {} <gpu> nodes".format(
                    self.num_gpus, len(self.gpus)
                )
            )
        if sorted(gpu.rank for gpu in self.gpus) != list(range(self.num_gpus)):
            raise MSCCLCompatibilityError("MSCCL gpu ids must be exactly 0..ngpus-1")

        steps: Dict[MSCCLStepKey, MSCCLXMLStep] = {}
        threadblocks: Dict[Tuple[int, int], MSCCLXMLThreadblock] = {}
        for gpu in self.gpus:
            ids = [tb.threadblock_id for tb in gpu.threadblocks]
            if len(ids) != len(set(ids)):
                raise MSCCLCompatibilityError(
                    "gpu[{}] contains duplicate threadblock ids".format(gpu.rank)
                )
            for tb in gpu.threadblocks:
                threadblocks[(gpu.rank, tb.threadblock_id)] = tb
                self._validate_peer(
                    tb.send_peer, "gpu[{}].tb[{}].send".format(gpu.rank, tb.threadblock_id)
                )
                self._validate_peer(
                    tb.recv_peer, "gpu[{}].tb[{}].recv".format(gpu.rank, tb.threadblock_id)
                )
                if tb.send_peer == gpu.rank or tb.recv_peer == gpu.rank:
                    raise MSCCLCompatibilityError("MSCCL send/recv peers cannot be the local rank")
                if tb.channel >= self.num_channels:
                    raise MSCCLCompatibilityError(
                        "gpu[{}].tb[{}].chan={} exceeds nchannels={}".format(
                            gpu.rank, tb.threadblock_id, tb.channel, self.num_channels
                        )
                    )
                if [step.key.step_id for step in tb.steps] != list(range(len(tb.steps))):
                    raise MSCCLCompatibilityError(
                        "gpu[{}].tb[{}] step ids must be contiguous and ordered from 0".format(
                            gpu.rank, tb.threadblock_id
                        )
                    )
                for step in tb.steps:
                    if step.key in steps:
                        raise MSCCLCompatibilityError("duplicate MSCCL step {}".format(step.key))
                    steps[step.key] = step
                    self._validate_step(gpu, tb, step)

        for step in steps.values():
            dep = step.dependency_key
            if (step.dependency_threadblock < 0) != (step.dependency_step < 0):
                raise MSCCLCompatibilityError(
                    "{} must set depid and deps together".format(step.key)
                )
            if dep is not None and dep not in steps:
                raise MSCCLCompatibilityError("{} depends on missing step {}".format(step.key, dep))

        self._validate_dependency_dag(steps, threadblocks)
        self._pair_transfers()

    def _validate_peer(self, peer: int, path: str) -> None:
        if peer < -1 or peer >= self.num_gpus:
            raise MSCCLCompatibilityError(
                "{}={} is outside -1..{}".format(path, peer, self.num_gpus - 1)
            )

    def _validate_step(
        self,
        gpu: MSCCLXMLGpu,
        tb: MSCCLXMLThreadblock,
        step: MSCCLXMLStep,
    ) -> None:
        path = "gpu[{}].tb[{}].step[{}]".format(gpu.rank, tb.threadblock_id, step.key.step_id)
        if step.op_type is MSCCLStepType.NOP:
            return
        if step.count <= 0:
            raise MSCCLCompatibilityError("{}.cnt must be positive".format(path))
        if step.src_buffer is None or step.dst_buffer is None:
            raise MSCCLCompatibilityError("{} requires srcbuf and dstbuf".format(path))
        if step.src_offset < 0 or step.dst_offset < 0:
            raise MSCCLCompatibilityError("{} requires non-negative offsets".format(path))

        if step.op_type in _FUSED_STEP_TYPES:
            if tb.recv_peer < 0 or tb.send_peer < 0:
                raise MSCCLCompatibilityError(
                    "{} fused receive/send requires both tb.recv and tb.send peers".format(path)
                )
            self._validate_chunk_range(
                self.gpus[tb.recv_peer],
                step.src_buffer,
                step.src_offset,
                step.count,
                path + ".incoming_src",
            )
            for dst_gpu, suffix in (
                (gpu, ".local_dst"),
                (self.gpus[tb.send_peer], ".outgoing_dst"),
            ):
                self._validate_chunk_range(
                    dst_gpu,
                    step.dst_buffer,
                    step.dst_offset,
                    step.count,
                    path + suffix,
                )
            return
        if step.op_type is MSCCLStepType.SEND:
            if tb.send_peer < 0:
                raise MSCCLCompatibilityError("{} send has no tb.send peer".format(path))
            src_gpu, dst_gpu = gpu, self.gpus[tb.send_peer]
        elif step.op_type in _RECV_STEP_TYPES:
            if tb.recv_peer < 0:
                raise MSCCLCompatibilityError("{} receive has no tb.recv peer".format(path))
            src_gpu, dst_gpu = self.gpus[tb.recv_peer], gpu
        else:
            src_gpu = dst_gpu = gpu

        self._validate_chunk_range(
            src_gpu, step.src_buffer, step.src_offset, step.count, path + ".src"
        )
        self._validate_chunk_range(
            dst_gpu, step.dst_buffer, step.dst_offset, step.count, path + ".dst"
        )

    @staticmethod
    def _validate_chunk_range(
        gpu: MSCCLXMLGpu,
        buffer: MSCCLBuffer,
        offset: int,
        count: int,
        path: str,
    ) -> None:
        chunks = gpu.chunk_count(buffer)
        if offset + count > chunks:
            raise MSCCLCompatibilityError(
                "{} range [{}, {}) exceeds gpu[{}] {}-buffer chunks {}".format(
                    path, offset, offset + count, gpu.rank, buffer.value, chunks
                )
            )

    @staticmethod
    def _validate_dependency_dag(
        steps: Mapping[MSCCLStepKey, MSCCLXMLStep],
        threadblocks: Mapping[Tuple[int, int], MSCCLXMLThreadblock],
    ) -> None:
        dependencies: Dict[MSCCLStepKey, List[MSCCLStepKey]] = {key: [] for key in steps}
        for tb in threadblocks.values():
            for previous, current in zip(tb.steps, tb.steps[1:]):
                dependencies[current.key].append(previous.key)
        for step in steps.values():
            if step.dependency_key is not None:
                dependencies[step.key].append(step.dependency_key)

        visiting = set()
        visited = set()

        def visit(key: MSCCLStepKey) -> None:
            if key in visiting:
                raise MSCCLCompatibilityError("MSCCL step dependencies contain a cycle")
            if key in visited:
                return
            visiting.add(key)
            for dependency in dependencies[key]:
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in dependencies:
            visit(key)

    def _pair_transfers(self) -> Tuple[Tuple[MSCCLStepKey, MSCCLStepKey], ...]:
        sends: DefaultDict[Tuple[object, ...], List[MSCCLStepKey]] = defaultdict(list)
        receives: DefaultDict[Tuple[object, ...], List[MSCCLStepKey]] = defaultdict(list)
        for gpu in self.gpus:
            for tb in gpu.threadblocks:
                for step in tb.steps:
                    if step.op_type in _SEND_STEP_TYPES:
                        if step.op_type in _FUSED_STEP_TYPES:
                            transfer = self._transfer_key_fields(
                                gpu.rank,
                                tb.send_peer,
                                tb.channel,
                                step.dst_buffer,
                                step.dst_offset,
                                step.dst_buffer,
                                step.dst_offset,
                                step.count,
                            )
                        else:
                            transfer = self._transfer_key(gpu.rank, tb.send_peer, tb.channel, step)
                        sends[transfer].append(step.key)
                    if step.op_type in _RECV_STEP_TYPES:
                        transfer = self._transfer_key(tb.recv_peer, gpu.rank, tb.channel, step)
                        receives[transfer].append(step.key)

        pairs: List[Tuple[MSCCLStepKey, MSCCLStepKey]] = []
        for transfer in sorted(set(sends).union(receives)):
            send_keys = sorted(sends.get(transfer, ()))
            recv_keys = sorted(receives.get(transfer, ()))
            if len(send_keys) != len(recv_keys):
                raise MSCCLCompatibilityError(
                    "unmatched MSCCL transfer {}: {} send step(s), {} receive step(s)".format(
                        transfer, len(send_keys), len(recv_keys)
                    )
                )
            for send_key, recv_key in zip(send_keys, recv_keys):
                pairs.append((send_key, recv_key))
        return tuple(pairs)

    @staticmethod
    def _transfer_key(
        source_rank: int,
        target_rank: int,
        channel: int,
        step: MSCCLXMLStep,
    ) -> Tuple[object, ...]:
        return MSCCLXMLAlgorithm._transfer_key_fields(
            source_rank,
            target_rank,
            channel,
            step.src_buffer,
            step.src_offset,
            step.dst_buffer,
            step.dst_offset,
            step.count,
        )

    @staticmethod
    def _transfer_key_fields(
        source_rank: int,
        target_rank: int,
        channel: int,
        src_buffer: Optional[MSCCLBuffer],
        src_offset: int,
        dst_buffer: Optional[MSCCLBuffer],
        dst_offset: int,
        count: int,
    ) -> Tuple[object, ...]:
        return (
            source_rank,
            target_rank,
            channel,
            src_buffer.value if src_buffer else "",
            src_offset,
            dst_buffer.value if dst_buffer else "",
            dst_offset,
            count,
        )

    def matches_collective(self, op_type: str) -> bool:
        return _normalize_collective(self.collective) == _normalize_collective(op_type)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "proto": self.protocol,
            "nchannels": self.num_channels,
            "nchunksperloop": self.chunks_per_loop,
            "ngpus": self.num_gpus,
            "coll": self.collective,
            "inplace": int(self.inplace),
            "gpus": [gpu.to_dict() for gpu in self.gpus],
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def lower(
        self,
        rank: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        signal_base: int = 0,
    ) -> PrimitiveIRGraph:
        """Lower one GPU's MSCCL threadblocks to a rank-local PCCL DAG."""
        self.validate()
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < self.num_gpus:
            raise MSCCLCompatibilityError(
                "rank {} is outside MSCCL ngpus {}".format(rank, self.num_gpus)
            )
        if (
            isinstance(tensor_size, bool)
            or not isinstance(tensor_size, int)
            or tensor_size <= 0
            or tensor_size % self.chunks_per_loop
        ):
            raise MSCCLCompatibilityError(
                "tensor_size must be positive and divisible by nchunksperloop {}".format(
                    self.chunks_per_loop
                )
            )
        if executor not in {"sm", "tma"}:
            raise MSCCLCompatibilityError("executor must be 'sm' or 'tma'")
        if self.protocol != "Simple":
            raise MSCCLCompatibilityError(
                "PCCL lowering currently supports MSCCL proto='Simple' only; "
                "the artifact uses {!r}".format(self.protocol)
            )
        if isinstance(signal_base, bool) or not isinstance(signal_base, int) or signal_base < 0:
            raise MSCCLCompatibilityError("signal_base must be a non-negative integer")

        if _normalize_collective(self.collective) == "alltoall":
            self._validate_pccl_alltoall_layout()

        transfer_pairs = self._pair_transfers()
        send_signal_by_step: Dict[MSCCLStepKey, int] = {}
        recv_signal_by_step: Dict[MSCCLStepKey, int] = {}
        if signal_base + len(transfer_pairs) > _MAX_SIGNALS:
            raise MSCCLCompatibilityError("MSCCL schedule exceeds PCCL signal workspace")
        for offset, (send_key, recv_key) in enumerate(transfer_pairs):
            signal = signal_base + offset
            send_signal_by_step[send_key] = signal
            recv_signal_by_step[recv_key] = signal

        chunk_size = tensor_size // self.chunks_per_loop
        refs: Dict[MSCCLStepKey, Tuple[IRNode, IRNode]] = {}
        gpu = self.gpus[rank]
        name = "msccl_{}_{}_rank{}".format(self.name, executor, rank)
        with CommunicationOp(name=name, device=DeviceType.CUDA) as op:
            op.tensor(dtype=dtype, shape=(tensor_size,))
            # MSCCL threadblocks are concurrent, while the current PCCL JSON
            # executor uses insertion order to break ties between DAG roots.
            # Emit send-capable roots before receive-only roots so a blocking
            # wait cannot be selected before its independent rendezvous send.
            threadblocks = sorted(
                gpu.threadblocks,
                key=lambda item: (
                    item.send_peer < 0,
                    item.recv_peer >= 0,
                    item.threadblock_id,
                ),
            )
            for tb in threadblocks:
                with Stream("msccl_tb_{}".format(tb.threadblock_id)):
                    op.set_channel(tb.channel)
                    for step in tb.steps:
                        refs[step.key] = self._lower_step(
                            op,
                            tb,
                            step,
                            send_signal_by_step,
                            recv_signal_by_step,
                            rank,
                            tensor_size,
                            chunk_size,
                            executor,
                        )

            for tb in gpu.threadblocks:
                for step in tb.steps:
                    dependency = step.dependency_key
                    if dependency is not None:
                        first, _last = refs[step.key]
                        _dep_first, dependency_last = refs[dependency]
                        first.add_dependency(dependency_last.op_id)

            graph = op.get_graph()
            graph.collective_type = _normalize_collective(self.collective)
            return graph

    def _lower_step(
        self,
        op: CommunicationOp,
        tb: MSCCLXMLThreadblock,
        step: MSCCLXMLStep,
        send_signal_by_step: Mapping[MSCCLStepKey, int],
        recv_signal_by_step: Mapping[MSCCLStepKey, int],
        rank: int,
        tensor_size: int,
        chunk_size: int,
        executor: str,
    ) -> Tuple[IRNode, IRNode]:
        if step.op_type is MSCCLStepType.NOP:
            noop = op.noop()
            return noop, noop
        if step.op_type is MSCCLStepType.SEND:
            notify = op.notify(signal_id=send_signal_by_step[step.key], target_rank=tb.send_peer)
            return notify, notify
        if step.op_type in _RECV_STEP_TYPES:
            wait = op.wait_notify(signal_id=recv_signal_by_step[step.key], source_rank=tb.recv_peer)
            data = self._lower_data_step(
                op,
                step,
                source_rank=tb.recv_peer,
                rank=rank,
                tensor_size=tensor_size,
                chunk_size=chunk_size,
                executor=executor,
                reduce=step.op_type in _RECV_REDUCE_STEP_TYPES,
                remote_receive=True,
            )
            if step.op_type in _FUSED_STEP_TYPES:
                notify = op.notify(
                    signal_id=send_signal_by_step[step.key], target_rank=tb.send_peer
                )
                return wait, notify
            return wait, data

        data = self._lower_data_step(
            op,
            step,
            source_rank=rank,
            rank=rank,
            tensor_size=tensor_size,
            chunk_size=chunk_size,
            executor=executor,
            reduce=step.op_type is MSCCLStepType.REDUCE,
            remote_receive=False,
        )
        return data, data

    def _lower_data_step(
        self,
        op: CommunicationOp,
        step: MSCCLXMLStep,
        source_rank: int,
        rank: int,
        tensor_size: int,
        chunk_size: int,
        executor: str,
        reduce: bool,
        remote_receive: bool,
    ) -> IRNodeVariant:
        assert step.src_buffer is not None
        assert step.dst_buffer is not None
        src_offset = self._pccl_offset(
            step.src_buffer,
            step.src_offset,
            tensor_size,
            chunk_size,
            destination=False,
            remote_receive=remote_receive,
        )
        dst_offset = self._pccl_offset(
            step.dst_buffer,
            step.dst_offset,
            tensor_size,
            chunk_size,
            destination=True,
            remote_receive=remote_receive,
        )
        count = step.count * chunk_size
        tma_compatible = (
            executor == "tma"
            and src_offset + count <= tensor_size
            and dst_offset + count <= tensor_size
        )
        if not reduce:
            if tma_compatible:
                return op.tma_copy(
                    source_rank=source_rank,
                    src_offset=src_offset,
                    dst_offset=dst_offset,
                    size=count,
                )
            return op.sm_copy(
                source_rank=source_rank,
                src_offset=src_offset,
                dst_offset=dst_offset,
                size=count,
            )

        if tma_compatible:
            return op.tma_reduce(
                reduce_op="sum",
                source_rank=source_rank,
                src_offset=dst_offset,
                dst_offset=dst_offset,
                remote_offset=src_offset,
                count=count,
            )
        return op.sm_reduce(
            reduce_op="sum",
            source_rank=source_rank,
            src_offset=dst_offset,
            dst_offset=dst_offset,
            remote_offset=src_offset,
            count=count,
        )

    def _pccl_offset(
        self,
        buffer: MSCCLBuffer,
        index: int,
        tensor_size: int,
        chunk_size: int,
        destination: bool,
        remote_receive: bool,
    ) -> int:
        if buffer is MSCCLBuffer.SCRATCH:
            return tensor_size + index * chunk_size
        if (
            _normalize_collective(self.collective) == "alltoall"
            and buffer is MSCCLBuffer.OUTPUT
            and destination
            and remote_receive
        ):
            # PCCL's current alltoall engine assembles remote source slots from
            # tensor_size + source_rank * chunk_size.
            return tensor_size + index * chunk_size
        return index * chunk_size

    def _validate_pccl_alltoall_layout(self) -> None:
        if self.chunks_per_loop != self.num_gpus:
            raise MSCCLCompatibilityError(
                "PCCL currently supports MSCCL alltoall only when " "nchunksperloop == ngpus"
            )
        for gpu in self.gpus:
            for tb in gpu.threadblocks:
                for step in tb.steps:
                    if step.op_type is MSCCLStepType.NOP:
                        continue
                    if step.count != 1:
                        raise MSCCLCompatibilityError(
                            "PCCL MSCCL-alltoall compatibility currently requires cnt=1"
                        )
                    actual = (
                        step.src_buffer,
                        step.src_offset,
                        step.dst_buffer,
                        step.dst_offset,
                    )
                    if step.op_type is MSCCLStepType.SEND:
                        expected = (
                            MSCCLBuffer.INPUT,
                            tb.send_peer,
                            MSCCLBuffer.OUTPUT,
                            gpu.rank,
                        )
                    elif step.op_type is MSCCLStepType.RECV:
                        expected = (
                            MSCCLBuffer.INPUT,
                            gpu.rank,
                            MSCCLBuffer.OUTPUT,
                            tb.recv_peer,
                        )
                    elif step.op_type is MSCCLStepType.COPY:
                        expected = (
                            MSCCLBuffer.INPUT,
                            gpu.rank,
                            MSCCLBuffer.OUTPUT,
                            gpu.rank,
                        )
                    else:
                        raise MSCCLCompatibilityError(
                            "PCCL direct alltoall supports only s/r/cpy/nop MSCCL steps"
                        )
                    if actual != expected:
                        raise MSCCLCompatibilityError(
                            "MSCCL alltoall step {} uses layout {}; PCCL currently requires {}".format(
                                step.key, actual, expected
                            )
                        )


def _required_attr(node: ElementTree.Element, name: str, path: str) -> str:
    value = node.get(name)
    if value is None or not value:
        raise MSCCLCompatibilityError("{}.{} is required".format(path, name))
    return value


def _int_attr(
    node: ElementTree.Element,
    name: str,
    path: str,
    minimum: Optional[int] = None,
) -> int:
    value = _required_attr(node, name, path)
    try:
        result = int(value)
    except ValueError as exc:
        raise MSCCLCompatibilityError("{}.{} must be an integer".format(path, name)) from exc
    if minimum is not None and result < minimum:
        raise MSCCLCompatibilityError("{}.{} must be >= {}".format(path, name, minimum))
    return result


def _optional_int_attr(
    node: ElementTree.Element,
    name: str,
    default: int,
    path: str,
) -> int:
    value = node.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise MSCCLCompatibilityError("{}.{} must be an integer".format(path, name)) from exc


def _optional_buffer(value: Optional[str], path: str) -> Optional[MSCCLBuffer]:
    if value is None or value == "":
        return None
    try:
        return MSCCLBuffer(value)
    except ValueError as exc:
        raise MSCCLCompatibilityError(
            "{} must be one of i/o/s, got {!r}".format(path, value)
        ) from exc


def _normalize_collective(value: str) -> str:
    return value.lower().replace("_", "").replace("-", "")


__all__ = [
    "MSCCLBuffer",
    "MSCCLCompatibilityError",
    "MSCCLStepKey",
    "MSCCLStepType",
    "MSCCLXMLAlgorithm",
    "MSCCLXMLGpu",
    "MSCCLXMLStep",
    "MSCCLXMLThreadblock",
]
