"""Versioned OCS control-plane messages for real switch connectors.

The runtime owns barrier semantics.  This module only serializes a stable
control-plane envelope so a connector may use an FPGA mailbox, RPC, or RDMA
transport without changing the runtime API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import struct
import time
from typing import Dict, Optional, Tuple
import zlib

from .exceptions import OCSProtocolError
from .plan import OCSPlan


OCS_CONTROL_MAGIC = b"OCSB"
OCS_CONTROL_VERSION = 1
OCS_CONTROL_MAX_PAYLOAD_BYTES = 1 << 20

# Fixed 128-byte network-order header, followed by ``payload_len`` bytes and a
# four-byte CRC32 trailer.  Keeping the header fixed lets a hardware mailbox
# parse barrier identity and plan digest before reading an opaque route payload.
_HEADER = struct.Struct("!4sBBBBQIQQQiiQIBBHQQQ32sI")
_CRC = struct.Struct("!I")
_ACK_TARGET = struct.Struct("!BiQ")


class OCSControlMessageType(IntEnum):
    """Control-plane operations exchanged by ranks and the switch controller."""

    READY = 1
    RELEASE = 2
    ABORT = 3
    ACK = 4


class OCSControlStatus(IntEnum):
    """Outcome carried by RELEASE, ABORT, and ACK messages."""

    OK = 0
    LINK_NOT_READY = 1
    PLAN_MISMATCH = 2
    TIMEOUT = 3
    INTERNAL_ERROR = 4


_ROUTE_MODE_TO_ID = {
    "STATIC_PLAN": 1,
    "ID_ROUTE": 2,
    "SEGMENT_ROUTE": 3,
    "USER_PLAN": 4,
}
_ROUTE_MODE_FROM_ID = {value: key for key, value in _ROUTE_MODE_TO_ID.items()}
_ALGORITHM_TO_ID = {
    "ring": 1,
    "rhd": 2,
    "tree": 3,
    "auto": 4,
    "torch_native": 5,
}
_ALGORITHM_FROM_ID = {value: key for key, value in _ALGORITHM_TO_ID.items()}
_BACKEND_TO_ID = {"torch": 1, "pccl": 2}
_BACKEND_FROM_ID = {value: key for key, value in _BACKEND_TO_ID.items()}


def _time_us() -> int:
    return time.time_ns() // 1000


def _job_id_hash(job_id: str) -> int:
    digest = hashlib.blake2b(str(job_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big")


def plan_digest(plan: OCSPlan) -> bytes:
    """Return a SHA-256 commitment to every plan field, including payload."""
    canonical = {
        "algorithm": plan.algorithm,
        "backend": plan.backend,
        "barrier_id": plan.barrier_id,
        "epoch_id": plan.epoch_id,
        "group_id": plan.group_id,
        "job_id": plan.job_id,
        "next_epoch_id": plan.next_epoch_id,
        "participant_ranks": plan.participant_ranks,
        "payload_hex": plan.payload.hex(),
        "route_mode": plan.route_mode,
        "route_plan_id": plan.route_plan_id,
        "topology_id": plan.topology_id,
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).digest()


@dataclass(frozen=True)
class OCSControlMessageKey:
    """Idempotency key for a single logical control message."""

    job_id_hash: int
    group_id: int
    barrier_id: int
    epoch_id: int
    src_rank: int
    sequence: int
    message_type: OCSControlMessageType


@dataclass(frozen=True)
class OCSControlMessage:
    """A validated fixed-header OCS control-plane message."""

    message_type: OCSControlMessageType
    status: OCSControlStatus
    job_id_hash: int
    group_id: int
    barrier_id: int
    epoch_id: int
    next_epoch_id: int
    src_rank: int
    src_gpu_id: int
    participant_bitmap: int
    topology_id: int
    route_mode: str
    algorithm: str
    backend: str
    route_plan_id: int
    sequence: int
    send_time_us: int
    plan_digest: bytes
    payload: bytes = b""

    def __post_init__(self) -> None:
        if len(self.plan_digest) != hashlib.sha256().digest_size:
            raise ValueError("plan_digest must be a SHA-256 digest")
        if not isinstance(self.payload, bytes):
            object.__setattr__(self, "payload", bytes(self.payload))
        if len(self.payload) > OCS_CONTROL_MAX_PAYLOAD_BYTES:
            raise ValueError(
                "OCS control payload exceeds "
                f"{OCS_CONTROL_MAX_PAYLOAD_BYTES} bytes")

    @property
    def key(self) -> OCSControlMessageKey:
        return OCSControlMessageKey(
            job_id_hash=self.job_id_hash,
            group_id=self.group_id,
            barrier_id=self.barrier_id,
            epoch_id=self.epoch_id,
            src_rank=self.src_rank,
            sequence=self.sequence,
            message_type=self.message_type,
        )

    @classmethod
    def from_plan(
        cls,
        message_type: OCSControlMessageType,
        plan: OCSPlan,
        src_rank: int,
        src_gpu_id: Optional[int] = None,
        sequence: int = 0,
        status: OCSControlStatus = OCSControlStatus.OK,
        payload: Optional[bytes] = None,
        send_time_us: Optional[int] = None,
    ) -> "OCSControlMessage":
        """Build a message that commits to ``plan`` through its digest."""
        if payload is None:
            payload = plan.payload if message_type == OCSControlMessageType.READY else b""
        return cls(
            message_type=message_type,
            status=status,
            job_id_hash=_job_id_hash(plan.job_id),
            group_id=plan.group_id,
            barrier_id=plan.barrier_id,
            epoch_id=plan.epoch_id,
            next_epoch_id=plan.next_epoch_id,
            src_rank=int(src_rank),
            src_gpu_id=int(src_rank if src_gpu_id is None else src_gpu_id),
            participant_bitmap=plan.participant_bitmap,
            topology_id=plan.topology_id,
            route_mode=plan.route_mode,
            algorithm=plan.algorithm,
            backend=plan.backend,
            route_plan_id=plan.route_plan_id,
            sequence=int(sequence),
            send_time_us=_time_us() if send_time_us is None else int(send_time_us),
            plan_digest=plan_digest(plan),
            payload=bytes(payload),
        )

    def matches_plan(self, plan: OCSPlan) -> bool:
        """Check identity and digest before accepting a control-plane message."""
        return (
            self.job_id_hash == _job_id_hash(plan.job_id)
            and self.group_id == plan.group_id
            and self.barrier_id == plan.barrier_id
            and self.epoch_id == plan.epoch_id
            and self.next_epoch_id == plan.next_epoch_id
            and self.participant_bitmap == plan.participant_bitmap
            and self.topology_id == plan.topology_id
            and self.route_mode == plan.route_mode
            and self.algorithm == plan.algorithm
            and self.backend == plan.backend
            and self.route_plan_id == plan.route_plan_id
            and self.plan_digest == plan_digest(plan)
        )

    def encode(self) -> bytes:
        """Serialize to a fixed header, opaque payload, and CRC32 trailer."""
        try:
            header = _HEADER.pack(
                OCS_CONTROL_MAGIC,
                OCS_CONTROL_VERSION,
                int(self.message_type),
                int(self.status),
                _ROUTE_MODE_TO_ID[self.route_mode],
                self.job_id_hash,
                self.group_id,
                self.barrier_id,
                self.epoch_id,
                self.next_epoch_id,
                self.src_rank,
                self.src_gpu_id,
                self.participant_bitmap,
                self.topology_id,
                _ALGORITHM_TO_ID[self.algorithm],
                _BACKEND_TO_ID[self.backend],
                0,
                self.route_plan_id,
                self.sequence,
                self.send_time_us,
                self.plan_digest,
                len(self.payload),
            )
        except struct.error as exc:
            raise OCSProtocolError(f"OCS control header field is out of range: {exc}") from exc
        body = header + self.payload
        return body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)

    @classmethod
    def decode(cls, wire: bytes) -> "OCSControlMessage":
        """Verify and decode a wire message without trusting its payload length."""
        if not isinstance(wire, bytes):
            wire = bytes(wire)
        if len(wire) < _HEADER.size + _CRC.size:
            raise OCSProtocolError("OCS control message is shorter than its fixed header")

        header = wire[:_HEADER.size]
        payload_and_crc = wire[_HEADER.size:]
        fields = _HEADER.unpack(header)
        (
            magic,
            version,
            message_type,
            status,
            route_mode,
            job_id_hash,
            group_id,
            barrier_id,
            epoch_id,
            next_epoch_id,
            src_rank,
            src_gpu_id,
            participant_bitmap,
            topology_id,
            algorithm,
            backend,
            _reserved,
            route_plan_id,
            sequence,
            send_time_us,
            digest,
            payload_len,
        ) = fields
        if magic != OCS_CONTROL_MAGIC:
            raise OCSProtocolError("invalid OCS control message magic")
        if version != OCS_CONTROL_VERSION:
            raise OCSProtocolError(
                f"unsupported OCS control version {version}; expected {OCS_CONTROL_VERSION}")
        if payload_len > OCS_CONTROL_MAX_PAYLOAD_BYTES:
            raise OCSProtocolError("OCS control payload exceeds the protocol maximum")
        if len(payload_and_crc) != payload_len + _CRC.size:
            raise OCSProtocolError("OCS control payload length does not match the wire size")

        body = wire[:-_CRC.size]
        (expected_crc,) = _CRC.unpack(wire[-_CRC.size:])
        if zlib.crc32(body) & 0xFFFFFFFF != expected_crc:
            raise OCSProtocolError("OCS control CRC32 mismatch")

        try:
            return cls(
                message_type=OCSControlMessageType(message_type),
                status=OCSControlStatus(status),
                job_id_hash=job_id_hash,
                group_id=group_id,
                barrier_id=barrier_id,
                epoch_id=epoch_id,
                next_epoch_id=next_epoch_id,
                src_rank=src_rank,
                src_gpu_id=src_gpu_id,
                participant_bitmap=participant_bitmap,
                topology_id=topology_id,
                route_mode=_ROUTE_MODE_FROM_ID[route_mode],
                algorithm=_ALGORITHM_FROM_ID[algorithm],
                backend=_BACKEND_FROM_ID[backend],
                route_plan_id=route_plan_id,
                sequence=sequence,
                send_time_us=send_time_us,
                plan_digest=digest,
                payload=wire[_HEADER.size:-_CRC.size],
            )
        except (KeyError, ValueError) as exc:
            raise OCSProtocolError(f"invalid OCS control enum value: {exc}") from exc


def build_ready(
    plan: OCSPlan,
    src_rank: int,
    src_gpu_id: Optional[int] = None,
    sequence: int = 0,
) -> OCSControlMessage:
    """Create a READY message carrying the opaque plan payload."""
    return OCSControlMessage.from_plan(
        OCSControlMessageType.READY,
        plan,
        src_rank=src_rank,
        src_gpu_id=src_gpu_id,
        sequence=sequence,
    )


def build_release(
    plan: OCSPlan,
    controller_rank: int = -1,
    sequence: int = 0,
    status: OCSControlStatus = OCSControlStatus.OK,
) -> OCSControlMessage:
    """Create a RELEASE/ABORT outcome with the same plan commitment."""
    message_type = (
        OCSControlMessageType.RELEASE
        if status == OCSControlStatus.OK
        else OCSControlMessageType.ABORT
    )
    return OCSControlMessage.from_plan(
        message_type,
        plan,
        src_rank=controller_rank,
        src_gpu_id=-1,
        sequence=sequence,
        status=status,
        payload=b"",
    )


def build_ack(
    acknowledged: OCSControlMessage,
    src_rank: int,
    src_gpu_id: Optional[int] = None,
    sequence: int = 0,
    status: OCSControlStatus = OCSControlStatus.OK,
) -> OCSControlMessage:
    """Acknowledge a READY/RELEASE/ABORT by its idempotency key."""
    target = _ACK_TARGET.pack(
        int(acknowledged.message_type),
        acknowledged.src_rank,
        acknowledged.sequence,
    )
    return OCSControlMessage(
        message_type=OCSControlMessageType.ACK,
        status=status,
        job_id_hash=acknowledged.job_id_hash,
        group_id=acknowledged.group_id,
        barrier_id=acknowledged.barrier_id,
        epoch_id=acknowledged.epoch_id,
        next_epoch_id=acknowledged.next_epoch_id,
        src_rank=int(src_rank),
        src_gpu_id=int(src_rank if src_gpu_id is None else src_gpu_id),
        participant_bitmap=acknowledged.participant_bitmap,
        topology_id=acknowledged.topology_id,
        route_mode=acknowledged.route_mode,
        algorithm=acknowledged.algorithm,
        backend=acknowledged.backend,
        route_plan_id=acknowledged.route_plan_id,
        sequence=int(sequence),
        send_time_us=_time_us(),
        plan_digest=acknowledged.plan_digest,
        payload=target,
    )


def ack_target(message: OCSControlMessage) -> Tuple[OCSControlMessageType, int, int]:
    """Return ``(message_type, src_rank, sequence)`` acknowledged by an ACK."""
    if message.message_type != OCSControlMessageType.ACK:
        raise ValueError("ack_target requires an ACK message")
    if len(message.payload) != _ACK_TARGET.size:
        raise OCSProtocolError("ACK payload has an invalid target key")
    message_type, src_rank, sequence = _ACK_TARGET.unpack(message.payload)
    try:
        return OCSControlMessageType(message_type), src_rank, sequence
    except ValueError as exc:
        raise OCSProtocolError(f"ACK references an invalid message type: {message_type}") from exc
