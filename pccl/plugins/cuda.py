import torch
import typing
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import pccl_native_cuda as _cuda_native
except ImportError:
    _cuda_native = None

class CudaDevice:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._native = None
        if _cuda_native and torch.cuda.is_available():
            try:
                self._native = _cuda_native.CudaDevice()
            except Exception as e:
                print(f"Failed to initialize CUDA device: {e}")

    @property
    def available(self) -> bool:
        if not torch.cuda.is_available():
            return False
        if self._native:
            return self._native.allocatorAvailable()
        return self.device_id < torch.cuda.device_count()

    @property
    def name(self) -> str:
        if self._native:
            return get_cuda_device_name(self.device_id)
        return torch.cuda.get_device_name(self.device_id) if torch.cuda.is_available() else "Unknown"

    @property
    def total_memory(self) -> int:
        if self._native:
            executor = _cuda_native.create_cuda_executor(self.device_id)
            executor.initialize()
            total = executor.getTotalMemory()
            executor.shutdown()
            return total
        elif torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
            return torch.cuda.get_device_properties(self.device_id).total_memory
        return 0

    @property
    def available_memory(self) -> int:
        if self._native:
            executor = _cuda_native.create_cuda_executor(self.device_id)
            executor.initialize()
            available = executor.getAvailableMemory()
            executor.shutdown()
            return available
        elif torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
            free, total = torch.cuda.mem_get_info()
            return free
        return 0

    def allocate(self, size: int) -> Optional[torch.Tensor]:
        if not self.available:
            return None

        try:
            tensor = torch.empty(size, dtype=torch.uint8, device=f"cuda:{self.device_id}")
            return tensor
        except Exception as e:
            print(f"Failed to allocate {size} bytes on CUDA device {self.device_id}: {e}")
            return None

    def deallocate(self, tensor: torch.Tensor):
        if tensor.is_cuda and tensor.get_device() == self.device_id:
            del tensor

class CudaExecutor:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._native = None
        self._stream = None

        if _cuda_native and torch.cuda.is_available():
            try:
                self._native = _cuda_native.create_cuda_executor(device_id)
                self._native.initialize()
                self._stream = self._native.getCurrentStream()
            except Exception as e:
                print(f"Failed to initialize CUDA executor: {e}")

    def __del__(self):
        if self._native:
            self._native.shutdown()

    def allocate(self, size: int) -> Optional[torch.Tensor]:
        if not self._native:
            return torch.empty(size, dtype=torch.uint8, device=f"cuda:{self.device_id}")

        ptr = self._native.allocate(size)
        if ptr:
            return torch.frombuffer(ptr, dtype=torch.uint8, count=size).to(f"cuda:{self.device_id}")
        return None

    def free(self, tensor: torch.Tensor):
        if self._native and tensor.is_cuda and tensor.get_device() == self.device_id:
            if tensor.is_contiguous():
                self._native.free(tensor.data_ptr())

    def copy(self, dst: torch.Tensor, src: torch.Tensor):
        if dst.shape != src.shape:
            raise ValueError("Source and destination tensors must have the same shape")

        if self._native:
            if dst.is_contiguous() and src.is_contiguous():
                size = src.numel() * src.element_size()
                self._native.copy(dst.data_ptr(), src.data_ptr(), size)
            else:
                dst.copy_(src)
        else:
            dst.copy_(src)

    def copy_from_host(self, dst: torch.Tensor, src: torch.Tensor):
        if self._native:
            if dst.is_contiguous() and src.is_contiguous():
                size = src.numel() * src.element_size()
                self._native.copyFromHost(dst.data_ptr(), src.data_ptr(), size)
            else:
                dst.copy_(src, non_blocking=True)
        else:
            dst.copy_(src, non_blocking=True)

    def copy_to_host(self, dst: torch.Tensor, src: torch.Tensor):
        if self._native:
            if dst.is_contiguous() and src.is_contiguous():
                size = src.numel() * src.element_size()
                self._native.copyToHost(dst.data_ptr(), src.data_ptr(), size)
            else:
                dst.copy_(src, non_blocking=True)
        else:
            dst.copy_(src, non_blocking=True)

    def synchronize(self):
        if self._native:
            self._native.synchronize()
        else:
            torch.cuda.synchronize(self.device_id)

    def get_device_id(self) -> int:
        return self.device_id

    def get_total_memory(self) -> int:
        if self._native:
            return self._native.getTotalMemory()
        elif torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
            return torch.cuda.get_device_properties(self.device_id).total_memory
        return 0

    def get_available_memory(self) -> int:
        if self._native:
            return self._native.getAvailableMemory()
        elif torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
            free, total = torch.cuda.mem_get_info()
            return free
        return 0

    def supports_p2p(self) -> bool:
        if self._native:
            return self._native.supportsP2P()
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            return torch.cuda.device_count() > 1
        return False

    def enable_p2p(self, other_device_id: int):
        if self._native:
            self._native.enableP2P(other_device_id)
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            try:
                torch.cuda.set_device(self.device_id)
                torch.cuda.set_device(other_device_id)
            except Exception as e:
                print(f"Failed to enable P2P: {e}")

class CudaStreamManager:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._native = None
        if _cuda_native:
            try:
                self._native = _cuda_native.CudaStreamManager()
                self._native.initialize(device_id)
            except Exception as e:
                print(f"Failed to initialize CUDA stream manager: {e}")

    def get_stream(self, stream_id: int = 0):
        if self._native:
            return self._native.getStream(stream_id)
        return torch.cuda.current_stream(self.device_id)

    def create_stream(self):
        if self._native:
            return self._native.createStream()
        return torch.cuda.Stream()

    def synchronize_stream(self, stream_id: int = 0):
        if self._native:
            self._native.synchronizeStream(stream_id)
        else:
            torch.cuda.current_stream(self.device_id).synchronize()

class CudaMemoryManager:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._native = None
        if _cuda_native:
            try:
                self._native = _cuda_native.CudaMemoryManager()
                self._native.initialize(device_id)
            except Exception as e:
                print(f"Failed to initialize CUDA memory manager: {e}")

    def allocate(self, size: int) -> Optional[torch.Tensor]:
        if self._native:
            ptr = self._native.allocate(size)
            if ptr:
                return torch.frombuffer(ptr, dtype=torch.uint8, count=size).to(f"cuda:{self.device_id}")
        return torch.empty(size, dtype=torch.uint8, device=f"cuda:{self.device_id}")

    def free(self, tensor: torch.Tensor):
        if self._native and tensor.is_cuda and tensor.get_device() == self.device_id:
            if tensor.is_contiguous():
                self._native.free(tensor.data_ptr())

    def get_allocated_bytes(self) -> int:
        if self._native:
            return self._native.getAllocatedBytes()
        return 0

    def get_allocation_count(self) -> int:
        if self._native:
            return self._native.getAllocationCount()
        return 0

def create_device(device_id: int = 0) -> CudaDevice:
    return CudaDevice(device_id)

def create_executor(device_id: int = 0) -> CudaExecutor:
    return CudaExecutor(device_id)

def create_memory_manager(device_id: int = 0) -> CudaMemoryManager:
    return CudaMemoryManager(device_id)

def create_stream_manager(device_id: int = 0) -> CudaStreamManager:
    return CudaStreamManager(device_id)

def get_device_count() -> int:
    if _cuda_native:
        return _cuda_native.get_cuda_device_count()
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def set_device(device_id: int) -> bool:
    if _cuda_native:
        return _cuda_native.set_cuda_device(device_id)
    elif torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return True
    return False

def get_device_name(device_id: int) -> str:
    if _cuda_native:
        return _cuda_native.get_cuda_device_name(device_id)
    elif torch.cuda.is_available():
        return torch.cuda.get_device_name(device_id)
    return "Unknown CUDA Device"

def get_device_properties(device_id: int) -> dict:
    if _cuda_native:
        return _cuda_native.get_cuda_device_properties(device_id)
    elif torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device_id)
        return {
            "name": props.name,
            "major": props.major,
            "minor": props.minor,
            "totalGlobalMem": props.total_memory,
            "sharedMemPerBlock": props.shared_memory_per_block,
            "maxThreadsPerBlock": props.max_threads_per_block,
            "maxGridSize": [props.max_grid_dim[0], props.max_grid_dim[1], props.max_grid_dim[2]],
            "maxThreadsDim": [props.max_threads_dim[0], props.max_threads_dim[1], props.max_threads_dim[2]],
            "warpSize": props.warp_size,
            "memoryClockRate": props.memory_clock_rate,
            "memoryBusWidth": props.memory_bus_width,
            "l2CacheSize": props.l2_cache_size,
            "maxThreadsPerMultiProcessor": props.max_threads_per_multi_processor,
            "multiProcessorCount": props.multi_processor_count,
            "concurrentKernels": props.concurrent_kernels,
            "integrated": props.integrated,
            "canMapHostMemory": props.can_map_host_memory,
            "computeMode": int(props.compute_mode)
        }
    return {}

def can_access_peer(device_id: int, peer_device_id: int) -> bool:
    if _cuda_native:
        return _cuda_native.cuda_can_access_peer(device_id, peer_device_id)
    elif torch.cuda.is_available() and torch.cuda.device_count() > max(device_id, peer_device_id):
        return torch.cuda.can_device_access_peer(device_id, peer_device_id)
    return False

def enable_peer_access(device_id: int, peer_device_id: int) -> bool:
    if _cuda_native:
        return _cuda_native.cuda_enable_peer_access(peer_device_id)
    elif torch.cuda.is_available() and torch.cuda.device_count() > max(device_id, peer_device_id):
        try:
            torch.cuda.set_device(device_id)
            torch.cuda.set_device(peer_device_id)
            return True
        except Exception:
            return False
    return False

def get_memory_info(device_id: int) -> Tuple[int, int]:
    if _cuda_native:
        return _cuda_native.cuda_get_memory_info(device_id)
    elif torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        free, total = torch.cuda.mem_get_info()
        return free, total
    return 0, 0

def is_available() -> bool:
    return get_device_count() > 0

def benchmark_copy(size: int, iterations: int = 100) -> float:
    if not is_available():
        return 0.0

    try:
        import time
        src = torch.randn(size, device="cuda")
        dst = torch.empty_like(src)

        executor = create_executor()
        start = time.time()
        for _ in range(iterations):
            executor.copy(dst, src)
        executor.synchronize()
        end = time.time()

        return (end - start) * 1000 / iterations
    except Exception as e:
        print(f"CUDA copy benchmark failed: {e}")
        return 0.0

def benchmark_allreduce(size: int, iterations: int = 10) -> float:
    if not is_available():
        return 0.0

    try:
        from ..lang.operator import allreduce

        allreduce_op = allreduce(
            algorithm="ring",
            participants=list(range(min(4, get_device_count()))),
            reduce_op="sum"
        )

        data = torch.randn(size, device="cuda")

        import time
        start = time.time()
        for _ in range(iterations):
            result = allreduce_op.execute(data)
        torch.cuda.synchronize()
        end = time.time()

        return (end - start) * 1000 / iterations
    except Exception as e:
        print(f"CUDA AllReduce benchmark failed: {e}")
        return 0.0

from ..lang.operator import allreduce, broadcast

__all__ = [
    "CudaDevice",
    "CudaExecutor",
    "CudaStreamManager",
    "CudaMemoryManager",
    "create_device",
    "create_executor",
    "create_memory_manager",
    "create_stream_manager",
    "get_device_count",
    "set_device",
    "get_device_name",
    "get_device_properties",
    "can_access_peer",
    "enable_peer_access",
    "get_memory_info",
    "is_available",
    "benchmark_copy",
    "benchmark_allreduce"
]