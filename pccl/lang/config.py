from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum, auto

class ReduceOp(Enum):
    SUM = auto()
    AVG = auto()
    MAX = auto()
    MIN = auto()

class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    ROCM = auto()

class InterconnectType(Enum):
    NVLINK = auto()
    PCIE = auto()
    RDMA = auto()
    ETHERNET = auto()
    INFINIBAND = auto()

class TopologyType(Enum):
    RING = auto()
    TREE = auto()
    MESH = auto()
    TORUS = auto()
    FAT_TREE = auto()
    HIERARCHICAL = auto()
    FULLY_CONNECTED = auto()

class AlgorithmType(Enum):
    RING = auto()
    TREE = auto()
    RABENSEIFNER = auto()
    DOUBLE_BINARY_TREE = auto()

@dataclass
class OperatorConfig:
    """Base class for declarative operator configurations"""
    name: str = ""
    device_type: DeviceType = DeviceType.CPU
    buffer_size: int = 128 * 1024 * 1024  # 128MB
    enable_overlap: bool = False
    pipeline_depth: int = 2
    topology: Optional['TopologyConfig'] = None

    def validate(self) -> bool:
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.pipeline_depth <= 0:
            raise ValueError("pipeline_depth must be positive")
        return True

@dataclass
class TopologyConfig:
    """Base class for network topology configurations"""
    topology_type: TopologyType = TopologyType.RING
    interconnect_type: InterconnectType = InterconnectType.PCIE
    bandwidth: float = 10.0  # GB/s
    latency: float = 1.0     # μs

    def validate(self) -> bool:
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if self.latency < 0:
            raise ValueError("latency cannot be negative")
        return True

@dataclass
class RingTopologyConfig(TopologyConfig):
    topology_type: TopologyType = TopologyType.RING

    def __post_init__(self):
        self.topology_type = TopologyType.RING

@dataclass
class TreeTopologyConfig(TopologyConfig):
    topology_type: TopologyType = TopologyType.TREE
    branching_factor: int = 2

    def __post_init__(self):
        self.topology_type = TopologyType.TREE

    def validate(self) -> bool:
        if self.branching_factor <= 1:
            raise ValueError("branching_factor must be > 1")
        return super().validate()

@dataclass
class HierarchicalTopologyConfig(TopologyConfig):
    topology_type: TopologyType = TopologyType.HIERARCHICAL
    intra_interconnect: InterconnectType = InterconnectType.NVLINK
    inter_interconnect: InterconnectType = InterconnectType.RDMA
    intra_bandwidth: float = 50.0
    inter_bandwidth: float = 10.0
    node_size: int = 4

    def __post_init__(self):
        self.topology_type = TopologyType.HIERARCHICAL

    def validate(self) -> bool:
        if self.intra_bandwidth <= 0 or self.inter_bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if self.node_size <= 1:
            raise ValueError("node_size must be > 1")
        return super().validate()

@dataclass
class AllreduceConfig(OperatorConfig):
    """Configuration for AllReduce collective operation"""
    reduce_op: ReduceOp = ReduceOp.SUM
    algorithm: AlgorithmType = AlgorithmType.RING
    participants: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = f"allreduce_{self.algorithm.name.lower()}"

    def validate(self) -> bool:
        if not self.participants:
            raise ValueError("participants list cannot be empty")
        if len(self.participants) != len(set(self.participants)):
            raise ValueError("participants must be unique")
        return super().validate()

@dataclass
class AllgatherConfig(OperatorConfig):
    """Configuration for AllGather collective operation"""
    participants: List[int] = field(default_factory=list)
    input_size: int = 0

    def __post_init__(self):
        if not self.name:
            self.name = "allgather"

    def validate(self) -> bool:
        if not self.participants:
            raise ValueError("participants list cannot be empty")
        if self.input_size < 0:
            raise ValueError("input_size cannot be negative")
        return super().validate()

@dataclass
class BroadcastConfig(OperatorConfig):
    """Configuration for Broadcast collective operation"""
    root_rank: int = 0
    participants: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = "broadcast"

    def validate(self) -> bool:
        if not self.participants:
            raise ValueError("participants list cannot be empty")
        if self.root_rank not in self.participants:
            raise ValueError("root_rank must be in participants")
        return super().validate()

@dataclass
class ReduceScatterConfig(OperatorConfig):
    """Configuration for Reduce-Scatter collective operation"""
    reduce_op: ReduceOp = ReduceOp.SUM
    participants: List[int] = field(default_factory=list)
    input_size: int = 0
    output_size: int = 0

    def ___post_init__(self):
        if not self.name:
            self.name = "reduce_scatter"

    def validate(self) -> bool:
        if not self.participants:
            raise ValueError("participants list cannot be empty")
        if self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        return super().validate()

@dataclass
class SendRecvConfig(OperatorConfig):
    """Configuration for Send/Recv point-to-point operations"""
    src_rank: int = 0
    dst_rank: int = 1
    tag: int = 0

    def __post_init__(self):
        if not self.name:
            self.name = "send_recv"

    def validate(self) -> bool:
        if self.src_rank == self.dst_rank:
            raise ValueError("src_rank and dst_rank must be different")
        if self.tag < 0:
            raise ValueError("tag cannot be negative")
        return super().validate()

@dataclass
class PipelineAllreduceConfig(AllreduceConfig):
    """Configuration for Pipeline AllReduce with compute-communication overlap"""
    compute_chunks: int = 4
    communication_chunks: int = 2

    def __post_init__(self):
        super().__post_init__()
        self.enable_overlap = True
        if not self.name:
            self.name = f"pipeline_allreduce_{self.algorithm.name.lower()}"

    def validate(self) -> bool:
        if self.compute_chunks <= 0 or self.communication_chunks <= 0:
            raise ValueError("compute_chunks and communication_chunks must be positive")
        return super().validate()

class ConfigBuilder:
    """Builder pattern for constructing operator configurations"""

    @staticmethod
    def allreduce(reduce_op: ReduceOp = ReduceOp.SUM,
                  algorithm: AlgorithmType = AlgorithmType.RING,
                  participants: Optional[List[int]] = None,
                  topology: Optional[TopologyConfig] = None,
                  buffer_size: int = 128 * 1024 * 1024,
                  enable_overlap: bool = False) -> AllreduceConfig:
        return AllreduceConfig(
            reduce_op=reduce_op,
            algorithm=algorithm,
            participants=participants or [],
            topology=topology,
            buffer_size=buffer_size,
            enable_overlap=enable_overlap
        )

    @staticmethod
    def ring_allreduce(reduce_op: ReduceOp = ReduceOp.SUM,
                       participants: Optional[List[int]] = None,
                       buffer_size: int = 128 * 1024 * 1024,
                       enable_overlap: bool = False) -> AllreduceConfig:
        return ConfigBuilder.allreduce(
            reduce_op=reduce_op,
            algorithm=AlgorithmType.RING,
            participants=participants,
            buffer_size=buffer_size,
            enable_overlap=enable_overlap,
            topology=RingTopologyConfig()
        )

    @staticmethod
    def tree_allreduce(reduce_op: ReduceOp = ReduceOp.SUM,
                       participants: Optional[List[int]] = None,
                       branching_factor: int = 2,
                       buffer_size: int = 128 * 1024 * 1024,
                       enable_overlap: bool = False) -> AllreduceConfig:
        return AllreduceConfig(
            reduce_op=reduce_op,
            algorithm=AlgorithmType.TREE,
            participants=participants or [],
            topology=TreeTopologyConfig(branching_factor=branching_factor),
            buffer_size=buffer_size,
            enable_overlap=enable_overlap
        )

    @staticmethod
    def hierarchical_allreduce(reduce_op: ReduceOp = ReduceOp.SUM,
                              participants: Optional[List[int]] = None,
                              node_size: int = 4,
                              intra_interconnect: InterconnectType = InterconnectType.NVLINK,
                              inter_interconnect: InterconnectType = InterconnectType.RDMA,
                              buffer_size: int = 128 * 1024 * 1024,
                              enable_overlap: bool = False) -> AllreduceConfig:
        return AllreduceConfig(
            reduce_op=reduce_op,
            algorithm=AlgorithmType.RABENSEIFNER,
            participants=participants or [],
            topology=HierarchicalTopologyConfig(
                intra_interconnect=intra_interconnect,
                inter_interconnect=inter_interconnect,
                node_size=node_size
            ),
            buffer_size=buffer_size,
            enable_overlap=enable_overlap
        )

    @staticmethod
    def pipeline_allreduce(reduce_op: ReduceOp = ReduceOp.SUM,
                          algorithm: AlgorithmType = AlgorithmType.RING,
                          participants: Optional[List[int]] = None,
                          compute_chunks: int = 4,
                          communication_chunks: int = 2,
                          buffer_size: int = 128 * 1024 * 1024) -> PipelineAllreduceConfig:
        return PipelineAllreduceConfig(
            reduce_op=reduce_op,
            algorithm=algorithm,
            participants=participants or [],
            compute_chunks=compute_chunks,
            communication_chunks=communication_chunks,
            buffer_size=buffer_size
        )

    @staticmethod
    def broadcast(root_rank: int = 0,
                  participants: Optional[List[int]] = None,
                  buffer_size: int = 128 * 1024 * 1024) -> BroadcastConfig:
        return BroadcastConfig(
            root_rank=root_rank,
            participants=participants or [root_rank],
            buffer_size=buffer_size
        )

    @staticmethod
    def allgather(participants: Optional[List[int]] = None,
                  input_size: int = 0,
                  buffer_size: int = 128 * 1024 * 1024) -> AllgatherConfig:
        return AllgatherConfig(
            participants=participants or [],
            input_size=input_size,
            buffer_size=buffer_size
        )

    @staticmethod
    def reduce_scatter(reduce_op: ReduceOp = ReduceOp.SUM,
                       participants: Optional[List[int]] = None,
                       input_size: int = 0,
                       output_size: int = 0,
                       buffer_size: int = 128 * 1024 * 1024) -> ReduceScatterConfig:
        return ReduceScatterConfig(
            reduce_op=reduce_op,
            participants=participants or [],
            input_size=input_size,
            output_size=output_size,
            buffer_size=buffer_size
        )