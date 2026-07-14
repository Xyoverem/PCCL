"""Tests for the versioned OCS controller wire protocol."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import (
    OCSControlMessage,
    OCSControlMessageType,
    OCSControlStatus,
    OCSPlan,
    OCSProtocolError,
    ack_target,
    build_ack,
    build_ready,
    build_release,
    plan_digest,
)


def _plan():
    return OCSPlan(
        job_id="train-42",
        group_id=7,
        barrier_id=9,
        epoch_id=3,
        next_epoch_id=4,
        participant_ranks=(0, 2),
        topology_id=11,
        route_mode="STATIC_PLAN",
        route_plan_id=13,
        algorithm="ring",
        backend="pccl",
        payload=b"opaque-route-plan",
    )


def test_ready_wire_round_trip_commits_to_every_plan_field():
    plan = _plan()
    ready = build_ready(plan, src_rank=2, src_gpu_id=5, sequence=17)

    decoded = OCSControlMessage.decode(ready.encode())

    assert decoded == ready
    assert decoded.payload == plan.payload
    assert decoded.matches_plan(plan)
    assert decoded.key.src_rank == 2
    assert decoded.key.sequence == 17
    assert decoded.key.message_type == OCSControlMessageType.READY


def test_plan_digest_and_wire_validation_reject_different_plan():
    plan = _plan()
    ready = build_ready(plan, src_rank=0)
    changed = replace(plan, topology_id=12)

    assert plan_digest(plan) != plan_digest(changed)
    assert not ready.matches_plan(changed)


@pytest.mark.parametrize("mutator", [
    lambda wire: wire[:-1] + bytes([wire[-1] ^ 0x01]),
    lambda wire: b"BAD!" + wire[4:],
    lambda wire: wire[:-1],
])
def test_wire_decode_rejects_corruption(mutator):
    wire = build_ready(_plan(), src_rank=0).encode()

    with pytest.raises(OCSProtocolError):
        OCSControlMessage.decode(mutator(wire))


def test_release_and_ack_preserve_plan_commitment_and_target():
    plan = _plan()
    ready = build_ready(plan, src_rank=0, sequence=23)
    release = build_release(plan, controller_rank=-1, sequence=24)
    ack = build_ack(release, src_rank=2, sequence=25)

    assert release.message_type == OCSControlMessageType.RELEASE
    assert release.status == OCSControlStatus.OK
    assert release.payload == b""
    assert release.matches_plan(plan)
    assert OCSControlMessage.decode(release.encode()) == release
    assert ack_target(OCSControlMessage.decode(ack.encode())) == (
        OCSControlMessageType.RELEASE,
        -1,
        24,
    )
    assert ack.matches_plan(plan)
    assert ready.matches_plan(plan)


def test_non_ok_release_is_encoded_as_abort():
    abort = build_release(
        _plan(),
        status=OCSControlStatus.LINK_NOT_READY,
    )

    assert abort.message_type == OCSControlMessageType.ABORT
    assert abort.status == OCSControlStatus.LINK_NOT_READY


def test_ack_target_requires_an_ack_message():
    with pytest.raises(ValueError, match="ACK"):
        ack_target(build_ready(_plan(), src_rank=0))
