"""Primitive Mapping for Code Generation - JSON v2 only."""

from typing import Dict
from ..nodes import ExecutorType, IRNodeVariant


_EXECUTOR_NAME_MAP: Dict[ExecutorType, str] = {
    ExecutorType.SM: "cuda_sm",
    ExecutorType.TMA: "cuda_tma",
    ExecutorType.CE: "cuda_ce",
    ExecutorType.HOST: "host",
    ExecutorType.RDMA: "cuda_rdma",
    ExecutorType.MULTIMEM: "cuda_multimem",
}


def get_primitive_name(node: IRNodeVariant) -> str:
    """Get runtime primitive name directly from op_type value (e.g. 'sm.reduce')."""
    return node.op_type.value


def get_executor_name(node: IRNodeVariant) -> str:
    """Get human-readable executor name for a node."""
    if node.executor:
        return _EXECUTOR_NAME_MAP.get(node.executor, "host")
    return "host"
