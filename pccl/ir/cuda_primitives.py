"""
CUDA Hardware Primitives

Defines CUDA-specific hardware primitives for Layer 3 of the three-layer IR architecture.
These primitives map to actual CUDA kernel operations and are executed by CUDA plugins.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .json_serializer import (
    IRValue, IROperation, IRGraph, IRType,
    DeviceType
)
from .primitive_ir import PrimitiveOperation


class CUDAPrimitiveType(Enum):
    """CUDA-specific primitive operation types"""
    MULTIMEM_REDUCE = "multimem_reduce"
    WARP_LEVEL_REDUCE = "warp_reduce"
    SHARED_MEMORY_COPY = "shared_copy"
    GLOBAL_MEMORY_SYNC = "global_sync"
    COOPERATIVE_LOAD = "cooperative_load"
    COOPERATIVE_STORE = "cooperative_store"
    ATOMIC_ADD = "atomic_add"
    ATOMIC_CAS = "atomic_cas"
    WARP_SYNC = "warp_sync"
    BLOCK_SYNC = "block_sync"
    SHUFFLE_SYNC = "shuffle_sync"
    VECTOR_LOAD = "vector_load"
    VECTOR_STORE = "vector_store"
    MEMORY_BARRIER = "memory_barrier"
    REGISTER_SPILL = "register_spill"
    REGISTER_FILL = "register_fill"


class CUDAMemorySpace(Enum):
    """CUDA memory spaces"""
    GLOBAL = "global"
    SHARED = "shared"
    CONSTANT = "constant"
    REGISTER = "register"
    LOCAL = "local"
    TEXTURE = "texture"


class CUDAThreadLevel(Enum):
    """CUDA thread hierarchy levels"""
    THREAD = "thread"
    WARP = "warp"
    BLOCK = "block"
    GRID = "grid"


@dataclass
class CUDALayoutInfo:
    """CUDA memory layout information"""
    stride: List[int]
    shape: List[int]
    dtype: str
    memory_space: CUDAMemorySpace
    alignment: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stride": self.stride,
            "shape": self.shape,
            "dtype": self.dtype,
            "memory_space": self.memory_space.value,
            "alignment": self.alignment
        }


@dataclass
class CUDAWarpInfo:
    """CUDA warp configuration"""
    warp_size: int = 32
    warp_id: int = 0
    lane_id: int = 0
    active_lanes: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warp_size": self.warp_size,
            "warp_id": self.warp_id,
            "lane_id": self.lane_id,
            "active_lanes": self.active_lanes
        }


class CUDAPrimitiveOperation(PrimitiveOperation):
    """Base class for CUDA hardware primitive operations"""

    def __init__(self,
                 id: str,
                 cuda_primitive_type: CUDAPrimitiveType,
                 inputs: List[str],
                 outputs: List[str],
                 attributes: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            id=id,
            op_type=cuda_primitive_type.value,
            inputs=inputs,
            outputs=outputs,
            attributes=attributes or {},
            metadata=metadata or {}
        )
        self.cuda_primitive_type = cuda_primitive_type


class CUDAMultiMemReduceOp(CUDAPrimitiveOperation):
    """CUDA multi-memory reduction operation"""

    def __init__(self,
                 id: str,
                 inputs: List[str],
                 output: str,
                 reduce_op: str = "sum",
                 dtype: str = "float32",
                 block_size: int = 256,
                 elements_per_thread: int = 4,
                 memory_space: CUDAMemorySpace = CUDAMemorySpace.GLOBAL):
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.MULTIMEM_REDUCE,
            inputs=inputs,
            outputs=[output],
            attributes={
                "reduce_op": reduce_op,
                "dtype": dtype,
                "block_size": block_size,
                "elements_per_thread": elements_per_thread,
                "memory_space": memory_space.value,
                "thread_level": CUDAThreadLevel.BLOCK.value
            }
        )


class CUDAWarpLevelReduceOp(CUDAPrimitiveOperation):
    """CUDA warp-level reduction operation"""

    def __init__(self,
                 id: str,
                 inputs: List[str],
                 output: str,
                 reduce_op: str = "sum",
                 dtype: str = "float32",
                 warp_info: Optional[CUDAWarpInfo] = None):
        warp_info = warp_info or CUDAWarpInfo()
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.WARP_LEVEL_REDUCE,
            inputs=inputs,
            outputs=[output],
            attributes={
                "reduce_op": reduce_op,
                "dtype": dtype,
                "warp_size": warp_info.warp_size,
                "warp_id": warp_info.warp_id,
                "thread_level": CUDAThreadLevel.WARP.value,
                "active_lanes": warp_info.active_lanes
            }
        )


class CUDASharedMemoryCopyOp(CUDAPrimitiveOperation):
    """CUDA shared memory copy operation"""

    def __init__(self,
                 id: str,
                 input: str,
                 output: str,
                 size: int,
                 dtype: str = "float32",
                 bank_conflict_avoidance: bool = True):
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.SHARED_MEMORY_COPY,
            inputs=[input],
            outputs=[output],
            attributes={
                "size": size,
                "dtype": dtype,
                "bank_conflict_avoidance": bank_conflict_avoidance,
                "thread_level": CUDAThreadLevel.BLOCK.value
            }
        )


class CUDAGlobalSyncOp(CUDAPrimitiveOperation):
    """CUDA global synchronization operation"""

    def __init__(self,
                 id: str,
                 sync_level: CUDAThreadLevel,
                 barrier_id: Optional[str] = None):
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.GLOBAL_SYNC,
            inputs=[],
            outputs=[],
            attributes={
                "sync_level": sync_level.value,
                "barrier_id": barrier_id or f"barrier_{sync_level.value}",
                "thread_level": sync_level.value
            }
        )


class CUDACooperativeLoadOp(CUDAPrimitiveOperation):
    """CUDA cooperative load operation"""

    def __init__(self,
                 id: str,
                 address: str,
                 output: str,
                 size: int,
                 dtype: str = "float32",
                 cache_hint: str = "ca",
                 is_volatile: bool = False):
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.COOPERATIVE_LOAD,
            inputs=[address],
            outputs=[output],
            attributes={
                "address": address,
                "size": size,
                "dtype": dtype,
                "cache_hint": cache_hint,
                "is_volatile": is_volatile,
                "thread_level": CUDAThreadLevel.THREAD.value
            }
        )


class CUDACooperativeStoreOp(CUDAPrimitiveOperation):
    """CUDA cooperative store operation"""

    def __init__(self,
                 id: str,
                 input: str,
                 address: str,
                 size: int,
                 dtype: str = "float32",
                 cache_hint: str = "wb",
                 is_streaming: bool = False):
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.COOPERATIVE_STORE,
            inputs=[input],
            outputs=[address],
            attributes={
                "input": input,
                "address": address,
                "size": size,
                "dtype": dtype,
                "cache_hint": cache_hint,
                "is_streaming": is_streaming,
                "thread_level": CUDAThreadLevel.THREAD.value
            }
        )


class CUDAAtomicAddOp(CUDAPrimitiveOperation):
    """CUDA atomic add operation"""

    def __init__(self,
                 id: str,
                 address: str,
                 value: str,
                 output: Optional[str] = None,
                 dtype: str = "float32",
                 memory_space: CUDAMemorySpace = CUDAMemorySpace.GLOBAL):
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.ATOMIC_ADD,
            inputs=[address, value],
            outputs=[output] if output else [],
            attributes={
                "address": address,
                "value": value,
                "dtype": dtype,
                "memory_space": memory_space.value,
                "thread_level": CUDAThreadLevel.THREAD.value
            }
        )


class CUDAWarpSyncOp(CUDAPrimitiveOperation):
    """CUDA warp synchronization operation"""

    def __init__(self,
                 id: str,
                 mask: int = 0xFFFFFFFF,
                 predicate: bool = True):
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.WARP_SYNC,
            inputs=[],
            outputs=[],
            attributes={
                "mask": mask,
                "predicate": predicate,
                "thread_level": CUDAThreadLevel.WARP.value
            }
        )


class CUDABlockSyncOp(CUDAPrimitiveOperation):
    """CUDA block synchronization operation"""

    def __init__(self,
                 id: str,
                 predicate: bool = True):
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.BLOCK_SYNC,
            inputs=[],
            outputs=[],
            attributes={
                "predicate": predicate,
                "thread_level": CUDAThreadLevel.BLOCK.value
            }
        )


class CUDAVectorLoadOp(CUDAPrimitiveOperation):
    """CUDA vectorized load operation"""

    def __init__(self,
                 id: str,
                 address: str,
                 outputs: List[str],
                 vector_size: int = 4,
                 dtype: str = "float32",
                 alignment: int = 16):
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.VECTOR_LOAD,
            inputs=[address],
            outputs=outputs,
            attributes={
                "address": address,
                "vector_size": vector_size,
                "dtype": dtype,
                "alignment": alignment,
                "thread_level": CUDAThreadLevel.THREAD.value
            }
        )


class CUDAVectorStoreOp(CUDAPrimitiveOperation):
    """CUDA vectorized store operation"""

    def __init__(self,
                 id: str,
                 inputs: List[str],
                 address: str,
                 vector_size: int = 4,
                 dtype: str = "float32",
                 alignment: int = 16):
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.VECTOR_STORE,
            inputs=inputs,
            outputs=[address],
            attributes={
                "inputs": inputs,
                "address": address,
                "vector_size": vector_size,
                "dtype": dtype,
                "alignment": alignment,
                "thread_level": CUDAThreadLevel.THREAD.value
            }
        )


class CUDAMemoryBarrierOp(CUDAPrimitiveOperation):
    """CUDA memory barrier operation"""

    def __init__(self,
                 id: str,
                 scope: str = "device",
                 ordering: str = "acquire_release"):
        super().__init__(
            id=id,
            cuda_primitive_type=CUDAPrimitiveType.MEMORY_BARRIER,
            inputs=[],
            outputs=[],
            attributes={
                "scope": scope,  # "thread", "block", "device", "system"
                "ordering": ordering,  # "acquire", "release", "acquire_release"
                "thread_level": CUDAThreadLevel.THREAD.value
            }
        )


class CUDAHardwarePrimitiveIRBuilder:
    """Builder for creating CUDA hardware primitive IR graphs"""

    def __init__(self, graph_id: str = "cuda_hardware_ir"):
        self.graph = IRGraph(ir_type=IRType.HARDWARE, values={}, operations={})
        self.graph.metadata["graph_id"] = graph_id
        self.graph.metadata["device_type"] = "cuda"
        self.value_counter = 0
        self.op_counter = 0

    def add_cuda_value(self,
                      dtype: str,
                      shape: List[int],
                      memory_space: CUDAMemorySpace = CUDAMemorySpace.GLOBAL,
                      layout: Optional[CUDALayoutInfo] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a CUDA-specific value to the graph"""
        value_id = f"cuda_value_{self.value_counter}"
        self.value_counter += 1

        value = IRValue(
            id=value_id,
            dtype=dtype,
            shape=shape,
            device_id=0,
            device_type=DeviceType.CUDA,
            metadata={
                "memory_space": memory_space.value,
                "layout": layout.to_dict() if layout else {},
                **(metadata or {})
            }
        )

        self.graph.add_value(value)
        return value_id

    def add_multimem_reduce(self,
                             inputs: List[str],
                             reduce_op: str = "sum",
                             block_size: int = 256,
                             elements_per_thread: int = 4) -> str:
        """Add multi-memory reduction operation"""
        op_id = f"multimem_reduce_{self.op_counter}"
        self.op_counter += 1

        output_value = self.add_cuda_value(
            dtype="float32",
            shape=[len(inputs) * 1024],  # Simplified shape inference
            memory_space=CUDAMemorySpace.GLOBAL
        )

        op = CUDAMultiMemReduceOp(
            id=op_id,
            inputs=inputs,
            output=output_value,
            reduce_op=reduce_op,
            block_size=block_size,
            elements_per_thread=elements_per_thread
        )

        self.graph.add_operation(op)
        return op_id

    def add_warp_reduce(self,
                       inputs: List[str],
                       reduce_op: str = "sum",
                       warp_info: Optional[CUDAWarpInfo] = None) -> str:
        """Add warp-level reduction operation"""
        op_id = f"warp_reduce_{self.op_counter}"
        self.op_counter += 1

        output_value = self.add_cuda_value(
            dtype="float32",
            shape=[1024],  # Simplified shape inference
            memory_space=CUDAMemorySpace.REGISTER
        )

        op = CUDAWarpLevelReduceOp(
            id=op_id,
            inputs=inputs,
            output=output_value,
            reduce_op=reduce_op,
            warp_info=warp_info
        )

        self.graph.add_operation(op)
        return op_id

    def add_shared_copy(self,
                       input_value: str,
                       size: int,
                       bank_conflict_avoidance: bool = True) -> str:
        """Add shared memory copy operation"""
        op_id = f"shared_copy_{self.op_counter}"
        self.op_counter += 1

        output_value = self.add_cuda_value(
            dtype="float32",
            shape=[size // 4],  # Assume 4-byte floats
            memory_space=CUDAMemorySpace.SHARED
        )

        op = CUDASharedMemoryCopyOp(
            id=op_id,
            input=input_value,
            output=output_value,
            size=size,
            bank_conflict_avoidance=bank_conflict_avoidance
        )

        self.graph.add_operation(op)
        return op_id

    def add_global_sync(self, sync_level: CUDAThreadLevel) -> str:
        """Add global synchronization operation"""
        op_id = f"global_sync_{self.op_counter}"
        self.op_counter += 1

        op = CUDAGlobalSyncOp(
            id=op_id,
            sync_level=sync_level
        )

        self.graph.add_operation(op)
        return op_id

    def add_atomic_add(self,
                       address: str,
                       value: str,
                       memory_space: CUDAMemorySpace = CUDAMemorySpace.GLOBAL) -> str:
        """Add atomic add operation"""
        op_id = f"atomic_add_{self.op_counter}"
        self.op_counter += 1

        output_value = None  # Atomic add typically doesn't return value

        op = CUDAAtomicAddOp(
            id=op_id,
            address=address,
            value=value,
            output=output_value,
            memory_space=memory_space
        )

        self.graph.add_operation(op)
        return op_id

    def add_vector_load(self,
                        address: str,
                        vector_size: int = 4,
                        alignment: int = 16) -> List[str]:
        """Add vectorized load operation"""
        op_id = f"vector_load_{self.op_counter}"
        self.op_counter += 1

        output_values = []
        for i in range(vector_size):
            output_val = self.add_cuda_value(
                dtype="float32",
                shape=[],
                memory_space=CUDAMemorySpace.REGISTER
            )
            output_values.append(output_val)

        op = CUDAVectorLoadOp(
            id=op_id,
            address=address,
            outputs=output_values,
            vector_size=vector_size,
            alignment=alignment
        )

        self.graph.add_operation(op)
        return output_values

    def get_graph(self) -> IRGraph:
        """Get the built hardware primitive IR graph"""
        return self.graph

    def get_value(self, value_id: str) -> Optional[IRValue]:
        """Get a CUDA value by ID"""
        return self.graph.get_value(value_id)

    def get_operation(self, op_id: str) -> Optional[IROperation]:
        """Get a CUDA operation by ID"""
        return self.graph.get_operation(op_id)


def create_cuda_multimem_allreduce_example() -> IRGraph:
    """Create an example CUDA multi-memory AllReduce hardware primitive pattern"""
    builder = CUDAHardwarePrimitiveIRBuilder("cuda_multimem_allreduce")

    # Input data values (simplified)
    input_values = []
    for i in range(4):
        input_val = builder.add_cuda_value(
            dtype="float32",
            shape=[1024],
            memory_space=CUDAMemorySpace.GLOBAL,
            metadata={"rank": i}
        )
        input_values.append(input_val)

    # Stage 1: Reduce in shared memory per block
    shared_reduced = []
    for i in range(4):
        shared_val = builder.add_shared_copy(input_values[i], 1024, True)
        shared_reduced.append(shared_val)

    # Stage 2: Multi-memory reduction across blocks
    final_result = builder.add_multimem_reduce(
        inputs=shared_reduced,
        reduce_op="sum",
        block_size=256,
        elements_per_thread=4
    )

    # Stage 3: Global synchronization
    builder.add_global_sync(CUDAThreadLevel.GRID)

    return builder.get_graph()


def create_cuda_warp_optimized_pattern() -> IRGraph:
    """Create a CUDA warp-optimized primitive pattern"""
    builder = CUDAHardwarePrimitiveIRBuilder("cuda_warp_optimized")

    # Input values
    input_vals = [
        builder.add_cuda_value("float32", [32], CUDAMemorySpace.REGISTER),
        builder.add_cuda_value("float32", [32], CUDAMemorySpace.REGISTER)
    ]

    # Warp-level reduction
    warp_result = builder.add_warp_reduce(input_vals, "sum")

    # Vector load/store for efficiency
    address_val = builder.add_cuda_value("uint64", [1], CUDAMemorySpace.GLOBAL)
    vector_vals = builder.add_vector_load(address_val, vector_size=4)

    output_address = builder.add_cuda_value("uint64", [1], CUDAMemorySpace.GLOBAL)
    builder.add_vector_store(vector_vals, output_address)

    # Atomic operations for thread safety
    builder.add_atomic_add(output_address, "atomic_value_1")

    # Memory barriers
    builder.add_global_sync(CUDAThreadLevel.BLOCK)
    builder.add_memory_barrier_op("device", "acquire_release")

    return builder.get_graph()