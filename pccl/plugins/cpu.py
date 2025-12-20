import torch
import typing
from typing import List, Optional, Tuple, Union
import numpy as np
import threading

try:
    import pccl_native_cpu as _cpu_native
except ImportError:
    _cpu_native = None

class CpuDevice:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._native = None
        if _cpu_native:
            try:
                self._native = _cpu_native.CpuDevice()
            except Exception as e:
                print(f"Failed to initialize CPU device: {e}")

    @property
    def available(self) -> bool:
        if self._native:
            return self._native.allocatorAvailable()
        return True

    def allocate(self, size: int) -> Optional[torch.Tensor]:
        if not self.available:
            return None

        try:
            tensor = torch.empty(size, dtype=torch.uint8)
            return tensor
        except Exception as e:
            print(f"Failed to allocate {size} bytes on CPU: {e}")
            return None

    def deallocate(self, tensor: torch.Tensor):
        pass

class CpuExecutor:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._native = None

        if _cpu_native:
            try:
                self._native = _cpu_native.create_cpu_executor(device_id)
                self._native.initialize()
            except Exception as e:
                print(f"Failed to initialize CPU executor: {e}")

    def __del__(self):
        if self._native:
            self._native.shutdown()

    def allocate(self, size: int) -> Optional[torch.Tensor]:
        if self._native:
            ptr = self._native.allocate(size)
            if ptr:
                return torch.from_numpy(np.frombuffer(ptr, dtype=np.uint8, count=size))
        return torch.empty(size, dtype=torch.uint8)

    def free(self, tensor: torch.Tensor):
        if self._native:
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

    def synchronize(self):
        if self._native:
            self._native.synchronize()

    def get_device_id(self) -> int:
        return self.device_id

    def get_total_memory(self) -> int:
        if self._native:
            return self._native.getTotalMemory()
        return 0

    def get_available_memory(self) -> int:
        if self._native:
            return self._native.getAvailableMemory()
        return 0

class CpuMemoryManager:
    def __init__(self):
        self._native = None
        if _cpu_native:
            try:
                self._native = _cpu_native.CpuMemoryManager()
            except Exception as e:
                print(f"Failed to initialize CPU memory manager: {e}")

    def allocate(self, size: int) -> Optional[torch.Tensor]:
        if self._native:
            ptr = self._native.allocate(size)
            if ptr:
                return torch.from_numpy(np.frombuffer(ptr, dtype=np.uint8, count=size))
        return torch.empty(size, dtype=torch.uint8)

    def free(self, tensor: torch.Tensor):
        if self._native and tensor.is_contiguous():
            self._native.free(tensor.data_ptr())

    def get_allocated_bytes(self) -> int:
        if self._native:
            return self._native.getAllocatedBytes()
        return 0

    def get_allocation_count(self) -> int:
        if self._native:
            return self._native.getAllocationCount()
        return 0

    def clear(self):
        if self._native:
            self._native.clear()

class CpuThreadPool:
    def __init__(self, num_threads: int = None):
        if num_threads is None:
            num_threads = get_cpu_core_count()

        self._native = None
        if _cpu_native:
            try:
                self._native = _cpu_native.CpuThreadPool(num_threads)
            except Exception as e:
                print(f"Failed to initialize CPU thread pool: {e}")
                self._pool = None
        else:
            import concurrent.futures
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    def submit(self, fn, *args, **kwargs):
        if self._native:
            import functools
            bound_fn = functools.partial(fn, *args, **kwargs)
            self._native.submit(bound_fn)
        elif hasattr(self, '_pool') and self._pool:
            return self._pool.submit(fn, *args, **kwargs)

    def get_thread_count(self) -> int:
        if self._native:
            return self._native.getThreadCount()
        elif hasattr(self, '_pool') and self._pool:
            return self._pool._max_workers
        return 1

    def get_queue_size(self) -> int:
        if self._native:
            return self._native.getQueueSize()
        return 0

class CpuKernelRegistry:
    def __init__(self):
        self._native = None
        if _cpu_native:
            try:
                self._native = _cpu_native.CpuKernelRegistry()
            except Exception as e:
                print(f"Failed to initialize CPU kernel registry: {e}")

    def register_kernel(self, name: str, kernel_func):
        if self._native:
            def cpu_kernel_wrapper(params):
                import ctypes
                kernel_args = params.args
                grid_idx = (params.grid_x, params.grid_y, params.grid_z)
                block_idx = (params.block_x, params.block_y, params.block_z)
                grid_dim = (params.grid_dim_x, params.grid_dim_y, params.grid_dim_z)
                block_dim = (params.block_dim_x, params.block_dim_y, params.block_dim_z)

                kernel_func(kernel_args, grid_idx, block_idx, grid_dim, block_dim)

            self._native.registerKernel(name, cpu_kernel_wrapper)
        else:
            self._kernels[name] = kernel_func

    def get_kernel(self, name: str):
        if self._native:
            return self._native.getKernel(name)
        else:
            return getattr(self, '_kernels', {}).get(name)

    def has_kernel(self, name: str) -> bool:
        if self._native:
            return self._native.hasKernel(name)
        else:
            return name in getattr(self, '_kernels', {})

    def get_kernel_names(self) -> List[str]:
        if self._native:
            return self._native.getKernelNames()
        else:
            return list(getattr(self, '_kernels', {}).keys())

    def clear(self):
        if self._native:
            self._native.clear()
        else:
            if hasattr(self, '_kernels'):
                self._kernels.clear()

def create_device(device_id: int = 0) -> CpuDevice:
    return CpuDevice(device_id)

def create_executor(device_id: int = 0) -> CpuExecutor:
    return CpuExecutor(device_id)

def create_memory_manager() -> CpuMemoryManager:
    return CpuMemoryManager()

def create_thread_pool(num_threads: int = None) -> CpuThreadPool:
    return CpuThreadPool(num_threads)

def create_kernel_registry() -> CpuKernelRegistry:
    return CpuKernelRegistry()

def get_cpu_core_count() -> int:
    if _cpu_native:
        return _cpu_native.get_cpu_core_count()
    return torch.get_num_threads()

def get_cpu_memory_info() -> Tuple[int, int]:
    if _cpu_native:
        return _cpu_native.get_cpu_memory_info()
    return (0, 0)

def is_available() -> bool:
    return True

def benchmark_copy(size: int, iterations: int = 100) -> float:
    import time
    src = torch.randn(size)
    dst = torch.empty_like(src)

    executor = create_executor()
    if not executor._native:
        start = time.time()
        for _ in range(iterations):
            dst.copy_(src)
        end = time.time()
        return (end - start) * 1000 / iterations

    start = time.time()
    for _ in range(iterations):
        executor.copy(dst, src)
    end = time.time()

    return (end - start) * 1000 / iterations

def benchmark_allreduce(size: int, iterations: int = 10) -> float:
    import time
    data = torch.randn(size)

    start = time.time()
    for _ in range(iterations):
        result = data * 2  # Simple CPU reduction
    end = time.time()

    return (end - start) * 1000 / iterations

from ..lang.operator import allreduce, broadcast

__all__ = [
    "CpuDevice",
    "CpuExecutor",
    "CpuMemoryManager",
    "CpuThreadPool",
    "CpuKernelRegistry",
    "create_device",
    "create_executor",
    "create_memory_manager",
    "create_thread_pool",
    "create_kernel_registry",
    "get_cpu_core_count",
    "get_cpu_memory_info",
    "is_available",
    "benchmark_copy",
    "benchmark_allreduce"
]