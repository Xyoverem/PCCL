"""Compiler Passes for PCCL DSL - DependencyAnalysis + DCE + Channelize."""

from .dependency_analysis import DependencyAnalysisPass
from .dce import DeadCodeEliminationPass
from .channelize import ChannelizePass

__all__ = [
    "DependencyAnalysisPass",
    "DeadCodeEliminationPass",
    "ChannelizePass",
]
