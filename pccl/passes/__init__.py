"""
PCCL Pass System

This module provides the infrastructure for defining, registering, and executing
IR transformation passes in the PCCL three-layer IR architecture.

Key Components:
- Pass: Base class for all transformation passes
- PassManager: Manages pass execution and dependencies
- PassRegistry: Registers and discovers passes
"""

from .base import Pass, PassResult, PassContext
from .registry import PassRegistry
from .manager import PassManager
from .pipeline import PassPipeline

# Import concrete pass implementations to ensure they register themselves
from .collective_to_primitive import *
from .primitive_to_hardware import *

__all__ = [
    'Pass',
    'PassResult',
    'PassContext',
    'PassRegistry',
    'PassManager',
    'PassPipeline'
]