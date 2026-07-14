"""Exceptions for the OCS-aware runtime."""


class OCSError(RuntimeError):
    """Base class for OCS runtime failures."""


class OCSPlanMismatchError(OCSError):
    """Raised when ranks arrive at the same barrier with different plans."""


class OCSBarrierTimeout(OCSError):
    """Raised when not all expected ranks arrive at a barrier."""


class OCSLinkNotReady(OCSError):
    """Raised when an OCS switch completes without a link-aligned indication."""
