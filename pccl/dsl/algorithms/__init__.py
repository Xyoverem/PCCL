"""PCCL DSL Algorithm Templates -- collective communication algorithms."""

from .base import CollectiveAlgorithm
from .ring import RingAllreduce
from .recursive_hd import RecursiveHalvingDoubling
from .tree import TreeAllreduce
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
    "ALGORITHMS",
    "select_algorithm",
    "select_algorithm_cost_based",
]
