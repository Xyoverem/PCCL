"""Exceptions for the OCS-aware runtime."""


class OCSError(RuntimeError):
    """Base class for OCS runtime failures."""


class OCSPlanMismatchError(OCSError):
    """Raised when ranks arrive at the same barrier with different plans."""


class OCSBarrierTimeout(OCSError):
    """Raised when not all expected ranks arrive at a barrier."""
