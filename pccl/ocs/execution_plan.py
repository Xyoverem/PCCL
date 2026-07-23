"""Versioned controller-side execution plans for OCS-PCCL."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, NoReturn, Optional, Tuple, Union, cast

from .exceptions import OCSExecutionPlanError
from .plan import OCSPlan, SUPPORTED_ALGORITHMS


OCS_EXECUTION_PLAN_VERSION = "ocs-pccl.execution-plan.v1"
_OP_TYPES = {
    "allreduce",
    "alltoall",
    "allgather",
    "reducescatter",
    "broadcast",
    "reduce",
    "custom",
}
_ALGORITHM_TYPES = {
    "ring",
    "rhd",
    "tree",
    "direct",
    "hierarchical",
    "auto",
    "torch_native",
    "custom",
}
_BACKENDS = {"pccl", "torch"}
_ROUTE_MODES = {"STATIC_PLAN", "ID_ROUTE", "SEGMENT_ROUTE", "USER_PLAN"}
_SWITCH_ACTIONS = {"KEEP", "APPLY_ROUTE"}
_EXTENSION_VALUE = re.compile(r"^x-[a-z0-9][a-z0-9._-]*$")
_GRAPH_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_BITMAP = re.compile(r"^0x[0-9a-f]+$")


def _fail(path: str, message: str) -> NoReturn:
    raise OCSExecutionPlanError("{}: {}".format(path, message))


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected an object")
    return cast(Mapping[str, Any], value)


def _required(mapping: Mapping[str, Any], field: str, path: str) -> Any:
    if field not in mapping:
        _fail(path, "missing required field {!r}".format(field))
    return mapping[field]


def _no_unknown(mapping: Mapping[str, Any], allowed: set, path: str) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        _fail(path, "unknown fields {}".format(unknown))


def _string(value: Any, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    if not allow_empty and not value:
        _fail(path, "must not be empty")
    return cast(str, value)


def _uint(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected a non-negative integer")
    if value < 0:
        _fail(path, "expected a non-negative integer")
    return cast(int, value)


def _optional_string(value: Any, path: str) -> Optional[str]:
    if value is None:
        return None
    return _string(value, path)


def _enum_or_extension(value: Any, allowed: set, path: str) -> str:
    result = _string(value, path)
    if result not in allowed and _EXTENSION_VALUE.fullmatch(result) is None:
        _fail(path, "unsupported value {!r}".format(result))
    return result


def _json_object(value: Any, path: str) -> Dict[str, Any]:
    mapping = _mapping(value, path)
    try:
        # A round trip both copies mutable input and rejects non-JSON payloads.
        return cast(
            Dict[str, Any],
            json.loads(json.dumps(mapping, sort_keys=True, separators=(",", ":"))),
        )
    except (TypeError, ValueError) as exc:
        _fail(path, "must contain only JSON values: {}".format(exc))


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _route_plan_wire_id(route_plan_id: str) -> int:
    """Project a string route identity into the uint64 wire-v1 field."""
    digest = hashlib.blake2b(route_plan_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big")


@dataclass(frozen=True)
class OCSRoutePlanSpec:
    route_plan_id: str
    route_mode: str
    source_topology_id: int
    target_topology_id: int
    payload: Dict[str, Any]

    @classmethod
    def from_dict(cls, value: Any, path: str) -> "OCSRoutePlanSpec":
        data = _mapping(value, path)
        _no_unknown(
            data,
            {
                "route_plan_id",
                "route_mode",
                "source_topology_id",
                "target_topology_id",
                "payload",
            },
            path,
        )
        route_mode = _string(_required(data, "route_mode", path), path + ".route_mode")
        if route_mode not in _ROUTE_MODES:
            _fail(path + ".route_mode", "unsupported value {!r}".format(route_mode))
        return cls(
            route_plan_id=_string(_required(data, "route_plan_id", path), path + ".route_plan_id"),
            route_mode=route_mode,
            source_topology_id=_uint(
                _required(data, "source_topology_id", path),
                path + ".source_topology_id",
            ),
            target_topology_id=_uint(
                _required(data, "target_topology_id", path),
                path + ".target_topology_id",
            ),
            payload=_json_object(_required(data, "payload", path), path + ".payload"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_plan_id": self.route_plan_id,
            "route_mode": self.route_mode,
            "source_topology_id": self.source_topology_id,
            "target_topology_id": self.target_topology_id,
            "payload": self.payload,
        }

    @property
    def wire_id(self) -> int:
        return _route_plan_wire_id(self.route_plan_id)


@dataclass(frozen=True)
class OCSBarrierTransition:
    barrier_id: int
    next_epoch: int
    next_phase_id: int
    switch_action: str
    route_plan: Optional[OCSRoutePlanSpec]

    @classmethod
    def from_dict(cls, value: Any, path: str) -> "OCSBarrierTransition":
        data = _mapping(value, path)
        _no_unknown(
            data,
            {
                "barrier_id",
                "next_epoch",
                "next_phase_id",
                "switch_action",
                "route_plan",
            },
            path,
        )
        switch_action = _string(_required(data, "switch_action", path), path + ".switch_action")
        if switch_action not in _SWITCH_ACTIONS:
            _fail(path + ".switch_action", "unsupported value {!r}".format(switch_action))
        route_value = _required(data, "route_plan", path)
        if switch_action == "KEEP":
            if route_value is not None:
                _fail(path + ".route_plan", "must be null when switch_action is KEEP")
            route_plan = None
        else:
            if route_value is None:
                _fail(path + ".route_plan", "is required when switch_action is APPLY_ROUTE")
            route_plan = OCSRoutePlanSpec.from_dict(route_value, path + ".route_plan")
        return cls(
            barrier_id=_uint(_required(data, "barrier_id", path), path + ".barrier_id"),
            next_epoch=_uint(_required(data, "next_epoch", path), path + ".next_epoch"),
            next_phase_id=_uint(_required(data, "next_phase_id", path), path + ".next_phase_id"),
            switch_action=switch_action,
            route_plan=route_plan,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "barrier_id": self.barrier_id,
            "next_epoch": self.next_epoch,
            "next_phase_id": self.next_phase_id,
            "switch_action": self.switch_action,
            "route_plan": None if self.route_plan is None else self.route_plan.to_dict(),
        }


@dataclass(frozen=True)
class OCSExecutionPhase:
    phase_id: int
    epoch: int
    op_type: str
    algorithm_type: str
    backend: str
    topology_id: int
    artifact_id: Optional[str]
    graph_digest: Optional[str]
    barrier_after: Optional[OCSBarrierTransition]

    @classmethod
    def from_dict(cls, value: Any, path: str) -> "OCSExecutionPhase":
        data = _mapping(value, path)
        _no_unknown(
            data,
            {
                "phase_id",
                "epoch",
                "op_type",
                "algorithm_type",
                "backend",
                "topology_id",
                "artifact_id",
                "graph_digest",
                "barrier_after",
            },
            path,
        )
        backend = _string(_required(data, "backend", path), path + ".backend")
        if backend not in _BACKENDS:
            _fail(path + ".backend", "unsupported value {!r}".format(backend))
        graph_digest = _optional_string(data.get("graph_digest"), path + ".graph_digest")
        if graph_digest is not None and _GRAPH_DIGEST.fullmatch(graph_digest) is None:
            _fail(path + ".graph_digest", "expected sha256:<64 lower-case hex digits>")
        barrier_value = _required(data, "barrier_after", path)
        return cls(
            phase_id=_uint(_required(data, "phase_id", path), path + ".phase_id"),
            epoch=_uint(_required(data, "epoch", path), path + ".epoch"),
            op_type=_enum_or_extension(
                _required(data, "op_type", path), _OP_TYPES, path + ".op_type"
            ),
            algorithm_type=_enum_or_extension(
                _required(data, "algorithm_type", path),
                _ALGORITHM_TYPES,
                path + ".algorithm_type",
            ),
            backend=backend,
            topology_id=_uint(_required(data, "topology_id", path), path + ".topology_id"),
            artifact_id=_optional_string(data.get("artifact_id"), path + ".artifact_id"),
            graph_digest=graph_digest,
            barrier_after=(
                None
                if barrier_value is None
                else OCSBarrierTransition.from_dict(barrier_value, path + ".barrier_after")
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "phase_id": self.phase_id,
            "epoch": self.epoch,
            "op_type": self.op_type,
            "algorithm_type": self.algorithm_type,
            "backend": self.backend,
            "topology_id": self.topology_id,
            "barrier_after": (None if self.barrier_after is None else self.barrier_after.to_dict()),
        }
        if self.artifact_id is not None:
            result["artifact_id"] = self.artifact_id
        if self.graph_digest is not None:
            result["graph_digest"] = self.graph_digest
        return result


@dataclass(frozen=True)
class OCSExecutionPlan:
    schema_version: str
    job_id: str
    plan_id: str
    group_id: int
    rank_list: Tuple[int, ...]
    participant_bitmap: str
    phases: Tuple[OCSExecutionPhase, ...]

    @classmethod
    def from_dict(cls, value: Any) -> "OCSExecutionPlan":
        path = "execution_plan"
        data = _mapping(value, path)
        _no_unknown(
            data,
            {
                "schema_version",
                "job_id",
                "plan_id",
                "group_id",
                "rank_list",
                "participant_bitmap",
                "phases",
            },
            path,
        )
        schema_version = _string(_required(data, "schema_version", path), path + ".schema_version")
        if schema_version != OCS_EXECUTION_PLAN_VERSION:
            _fail(
                path + ".schema_version",
                "unsupported version {!r}; expected {!r}".format(
                    schema_version, OCS_EXECUTION_PLAN_VERSION
                ),
            )

        ranks_value = _required(data, "rank_list", path)
        if not isinstance(ranks_value, list) or not ranks_value:
            _fail(path + ".rank_list", "expected a non-empty array")
        ranks = tuple(
            _uint(rank, "{}.rank_list[{}]".format(path, index))
            for index, rank in enumerate(ranks_value)
        )

        phases_value = _required(data, "phases", path)
        if not isinstance(phases_value, list) or not phases_value:
            _fail(path + ".phases", "expected a non-empty array")
        phases = tuple(
            OCSExecutionPhase.from_dict(phase, "{}.phases[{}]".format(path, index))
            for index, phase in enumerate(phases_value)
        )

        plan = cls(
            schema_version=schema_version,
            job_id=_string(_required(data, "job_id", path), path + ".job_id"),
            plan_id=_string(_required(data, "plan_id", path), path + ".plan_id"),
            group_id=_uint(_required(data, "group_id", path), path + ".group_id"),
            rank_list=ranks,
            participant_bitmap=_string(
                _required(data, "participant_bitmap", path),
                path + ".participant_bitmap",
            ),
            phases=phases,
        )
        plan.validate()
        return plan

    @classmethod
    def from_json(cls, value: Union[str, bytes, bytearray]) -> "OCSExecutionPlan":
        try:
            data = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise OCSExecutionPlanError("invalid Execution Plan JSON: {}".format(exc)) from exc
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OCSExecutionPlan":
        try:
            contents = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise OCSExecutionPlanError(
                "cannot read Execution Plan {!r}: {}".format(str(path), exc)
            ) from exc
        return cls.from_json(contents)

    def validate(self) -> None:
        if tuple(sorted(set(self.rank_list))) != self.rank_list:
            _fail("execution_plan.rank_list", "must be strictly increasing and unique")
        if self.rank_list[-1] >= 64:
            _fail("execution_plan.rank_list", "wire protocol v1 supports ranks 0..63")
        if _BITMAP.fullmatch(self.participant_bitmap) is None:
            _fail(
                "execution_plan.participant_bitmap",
                "expected a lower-case hexadecimal string beginning with 0x",
            )
        expected_bitmap = sum(1 << rank for rank in self.rank_list)
        if int(self.participant_bitmap, 16) != expected_bitmap:
            _fail(
                "execution_plan.participant_bitmap",
                "does not match rank_list; expected {}".format(hex(expected_bitmap)),
            )

        expected_phase_ids = tuple(range(len(self.phases)))
        actual_phase_ids = tuple(phase.phase_id for phase in self.phases)
        if actual_phase_ids != expected_phase_ids:
            _fail(
                "execution_plan.phases",
                "phase_id values must be contiguous from zero; got {}".format(actual_phase_ids),
            )

        barrier_ids = []
        for index, phase in enumerate(self.phases):
            barrier = phase.barrier_after
            path = "execution_plan.phases[{}].barrier_after".format(index)
            if index < len(self.phases) - 1 and barrier is None:
                _fail(path, "is required after every non-final phase")
            if barrier is None:
                continue
            if barrier.next_epoch <= phase.epoch:
                _fail(path + ".next_epoch", "must be greater than the current phase epoch")
            expected_next_phase = index + 1 if index < len(self.phases) - 1 else 0
            if barrier.next_phase_id != expected_next_phase:
                _fail(
                    path + ".next_phase_id",
                    "expected {} for a sequential phased plan".format(expected_next_phase),
                )
            target_phase = self.phases[barrier.next_phase_id]
            if index < len(self.phases) - 1 and barrier.next_epoch != target_phase.epoch:
                _fail(
                    path + ".next_epoch",
                    "must equal the next phase epoch {}".format(target_phase.epoch),
                )
            if barrier.switch_action == "KEEP":
                if phase.topology_id != target_phase.topology_id:
                    _fail(
                        path + ".switch_action",
                        "KEEP requires current and target phase topology_id to match",
                    )
            else:
                route = barrier.route_plan
                if route is None:
                    _fail(path + ".route_plan", "missing APPLY_ROUTE plan")
                if route.source_topology_id != phase.topology_id:
                    _fail(
                        path + ".route_plan.source_topology_id",
                        "must equal current phase topology_id {}".format(phase.topology_id),
                    )
                if route.target_topology_id != target_phase.topology_id:
                    _fail(
                        path + ".route_plan.target_topology_id",
                        "must equal target phase topology_id {}".format(target_phase.topology_id),
                    )
            barrier_ids.append(barrier.barrier_id)

        if barrier_ids != sorted(set(barrier_ids)):
            _fail(
                "execution_plan.phases",
                "barrier_id values must be unique and strictly increasing",
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "plan_id": self.plan_id,
            "group_id": self.group_id,
            "rank_list": list(self.rank_list),
            "participant_bitmap": self.participant_bitmap,
            "phases": [phase.to_dict() for phase in self.phases],
        }

    def canonical_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    def barrier_plan(
        self,
        phase_id: int,
        target_algorithm: Optional[str] = None,
    ) -> Optional[OCSPlan]:
        """Project one rich phase transition into the existing wire-v1 plan."""
        if phase_id < 0 or phase_id >= len(self.phases):
            raise IndexError("phase_id {} is outside this Execution Plan".format(phase_id))
        phase = self.phases[phase_id]
        barrier = phase.barrier_after
        if barrier is None:
            return None
        target_phase = self.phases[barrier.next_phase_id]
        algorithm = target_phase.algorithm_type if target_algorithm is None else target_algorithm
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise OCSExecutionPlanError(
                "phase {} algorithm {!r} cannot be represented by wire protocol v1".format(
                    target_phase.phase_id, algorithm
                )
            )

        route = barrier.route_plan
        transition_payload = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "phase_id": phase.phase_id,
            "next_phase_id": barrier.next_phase_id,
            "switch_action": barrier.switch_action,
            "route_plan": None if route is None else route.to_dict(),
        }
        return OCSPlan(
            job_id=self.job_id,
            group_id=self.group_id,
            barrier_id=barrier.barrier_id,
            epoch_id=phase.epoch,
            next_epoch_id=barrier.next_epoch,
            participant_ranks=self.rank_list,
            topology_id=target_phase.topology_id,
            route_mode="STATIC_PLAN" if route is None else route.route_mode,
            route_plan_id=0 if route is None else route.wire_id,
            algorithm=algorithm,
            backend=target_phase.backend,
            payload=_canonical_bytes(transition_payload),
        )
