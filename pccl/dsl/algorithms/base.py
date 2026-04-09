"""Base class for collective algorithm templates."""

from abc import ABC, abstractmethod
from ..graph import PrimitiveIRGraph


class CollectiveAlgorithm(ABC):
    name: str = ""

    @abstractmethod
    def build_allreduce(
        self,
        rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> PrimitiveIRGraph:
        raise NotImplementedError

    def build_reduce_scatter(
        self,
        rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> PrimitiveIRGraph:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement reduce_scatter")

    def build_allgather(
        self,
        rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> PrimitiveIRGraph:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement allgather")

    def build_alltoall(
        self,
        rank: int,
        world_size: int,
        tensor_size: int,
        dtype: str = "bfloat16",
        executor: str = "tma",
        num_channels: int = 1,
    ) -> PrimitiveIRGraph:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement alltoall")

    @property
    def step_count(self) -> str:
        return "unknown"

    @property
    def bandwidth_optimal(self) -> bool:
        return False
