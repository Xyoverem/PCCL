"""OCS-aware runtime interfaces for PCCL experiments."""

from .controller import OCSPlanController, StaticPlanController
from .collective_plan import (
    OCSCollectivePhase,
    OCSCollectivePlan,
    OcsCollectivePlanRunner,
    PreparedOcsCollectivePlan,
    build_ring_allreduce_alltoall_plan,
)
from .exceptions import OCSError, OCSBarrierTimeout, OCSLinkNotReady, OCSPlanMismatchError
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

__all__ = [
    "OCSError",
    "OCSBarrierTimeout",
    "OCSLinkNotReady",
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
    "ocs_all_reduce",
    "ocs_barrier_switch",
]
