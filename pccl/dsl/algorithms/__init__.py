"""PCCL DSL Algorithm Templates -- collective communication algorithms."""

from .base import CollectiveAlgorithm
from .ring import RingAllreduce
from .recursive_hd import RecursiveHalvingDoubling
from .tree import TreeAllreduce
from .generated import (
    AlgorithmIRCollectives,
    build_direct_alltoall_ir,
    build_ring_allreduce_ir,
)
from .selector import select_algorithm, select_algorithm_cost_based

ALGORITHMS = {
    "ring": RingAllreduce,
    "rhd": RecursiveHalvingDoubling,
    "tree": TreeAllreduce,
}

__all__ = [
    "CollectiveAlgorithm",
    "RingAllreduce",
    "RecursiveHalvingDoubling",
    "TreeAllreduce",
    "AlgorithmIRCollectives",
    "build_direct_alltoall_ir",
    "build_ring_allreduce_ir",
    "ALGORITHMS",
    "select_algorithm",
    "select_algorithm_cost_based",
]
