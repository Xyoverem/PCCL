"""OCS-aware runtime interfaces for PCCL experiments."""

from .controller import OCSPlanController, StaticPlanController
from .collective_plan import (
    OCSCollectivePhase,
    OCSCollectivePlan,
    OcsCollectivePlanRunner,
    PreparedOcsCollectivePlan,
    build_ring_allreduce_alltoall_plan,
)
from .exceptions import (
    OCSError,
    OCSBarrierTimeout,
    OCSLinkNotReady,
    OCSPlanMismatchError,
    OCSProtocolError,
)
from .phase_runner import OcsPhaseRunner, PreparedOcsGraph
from .plan import OCSPlan
from .runtime import (
    OCSRuntime,
    OCSBarrierState,
    OCSLinkState,
    OCSSwitchResult,
    SwitchConnector,
    TorchDistributedSwitchConnector,
    ocs_all_reduce,
    ocs_barrier_switch,
)
from .torch_plan import (
    OcsTorchPlanRunner,
    TorchCollectivePhase,
    TorchCollectivePlan,
    build_torch_allreduce_alltoall_plan,
)
from .protocol import (
    OCS_CONTROL_MAGIC,
    OCS_CONTROL_MAX_PAYLOAD_BYTES,
    OCS_CONTROL_VERSION,
    OCSControlMessage,
    OCSControlMessageKey,
    OCSControlMessageType,
    OCSControlStatus,
    ack_target,
    build_ack,
    build_ready,
    build_release,
    plan_digest,
)

__all__ = [
    "OCSError",
    "OCSBarrierTimeout",
    "OCSLinkNotReady",
    "OCSProtocolError",
    "OCSPlan",
    "OCSPlanMismatchError",
    "OCSPlanController",
    "OCSCollectivePhase",
    "OCSCollectivePlan",
    "OCSRuntime",
    "OCSBarrierState",
    "OCSLinkState",
    "OCSSwitchResult",
    "OcsPhaseRunner",
    "OcsCollectivePlanRunner",
    "PreparedOcsCollectivePlan",
    "PreparedOcsGraph",
    "StaticPlanController",
    "SwitchConnector",
    "TorchDistributedSwitchConnector",
    "build_ring_allreduce_alltoall_plan",
    "OcsTorchPlanRunner",
    "TorchCollectivePhase",
    "TorchCollectivePlan",
    "build_torch_allreduce_alltoall_plan",
    "OCS_CONTROL_MAGIC",
    "OCS_CONTROL_MAX_PAYLOAD_BYTES",
    "OCS_CONTROL_VERSION",
    "OCSControlMessage",
    "OCSControlMessageKey",
    "OCSControlMessageType",
    "OCSControlStatus",
    "ack_target",
    "build_ack",
    "build_ready",
    "build_release",
    "ocs_all_reduce",
    "ocs_barrier_switch",
    "plan_digest",
]
