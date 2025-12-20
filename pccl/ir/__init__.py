"""
PCCL IR (Intermediate Representation) System

This module defines the IR structures for the three-layer architecture:
- Layer 1: Collective Primitives (AllReduce, Broadcast, etc.)
- Layer 2: Primitive IR (Write, Reduce, Copy, Signal, Wait)
- Layer 3: Hardware Primitives (CUDA multimem.reduce, RDMA verbs, etc.)
"""

from .json_serializer import IRSerializer
from .primitive_ir import *
from .cuda_primitives import *
from .rdma_primitives import *

__all__ = [
    'IRSerializer'
]