import torch
import typing
from typing import List, Optional, Tuple, Union
import numpy as np
import os

try:
    import pccl_native_rdma as _rdma_native
except ImportError:
    _rdma_native = None

class RdmaDevice:
    def __init__(self):
        self._native = None
        if _rdma_native:
            try:
                self._native = _rdma_native.RdmaDevice()
            except Exception as e:
                print(f"Failed to initialize RDMA device: {e}")

    @property
    def available(self) -> bool:
        if self._native:
            return self._native.remoteCommAvailable()
        return False

    def activate(self) -> str:
        if self._native:
            return self._native.activate()
        return ""

    def register_buffer(self, addr, size: int) -> str:
        if self._native:
            return self._native.registerBuffer(addr, size)
        return ""

    def connect(self, handle: str):
        if self._native:
            self._native.connect(handle)

    def disconnect(self, handle: str):
        if self._native:
            self._native.disconnect(handle)

class RdmaExecutor:
    def __init__(self):
        self._native = None
        if _rdma_native:
            try:
                self._native = _rdma_native.create_rdma_executor()
                self._native.initialize()
            except Exception as e:
                print(f"Failed to initialize RDMA executor: {e}")

    def __del__(self):
        if self._native:
            self._native.shutdown()

    def allocate(self, size: int) -> Optional[torch.Tensor]:
        if not self._native:
            return torch.empty(size, dtype=torch.uint8)

        ptr = self._native.allocate(size)
        if ptr:
            return torch.frombuffer(ptr, dtype=np.uint8, count=size)
        return None

    def free(self, tensor: torch.Tensor):
        if self._native and tensor.is_contiguous():
            self._native.free(tensor.data_ptr())

    def copy(self, dst: torch.Tensor, src: torch.Tensor):
        if dst.shape != src.shape:
            raise ValueError("Source and destination tensors must have the same shape")

        if self._native and dst.is_contiguous() and src.is_contiguous():
            size = src.numel() * src.element_size()
            self._native.copy(dst.data_ptr(), src.data_ptr(), size)
        else:
            dst.copy_(src)

    def synchronize(self):
        if self._native:
            self._native.synchronize()

    def get_rdma_device(self):
        return self._native.getRdmaDevice() if self._native else None

    def get_memory_manager(self):
        return self._native.getMemoryManager() if self._native else None

    def get_connection_manager(self):
        return self._native.getConnectionManager() if self._native else None

class RdmaMemoryManager:
    def __init__(self):
        self._native = None
        if _rdma_native:
            try:
                self._native = _rdma_native.RdmaMemoryManager()
                self._native.initialize(0)
            except Exception as e:
                print(f"Failed to initialize RDMA memory manager: {e}")

    def allocate(self, size: int) -> Optional[torch.Tensor]:
        if self._native:
            ptr = self._native.allocate(size)
            if ptr:
                return torch.frombuffer(ptr, dtype=np.uint8, count=size)
        return torch.empty(size, dtype=torch.uint8)

    def free(self, tensor: torch.Tensor):
        if self._native and tensor.is_contiguous():
            self._native.free(tensor.data_ptr())

    def register_buffer(self, tensor: torch.Tensor) -> str:
        if self._native and tensor.is_contiguous():
            return self._native.registerBuffer(tensor.data_ptr(), tensor.numel() * tensor.element_size())
        return ""

    def unregister_buffer(self, tensor: torch.Tensor) -> bool:
        if self._native and tensor.is_contiguous():
            return self._native.unregisterBuffer(tensor.data_ptr())
        return False

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

class RdmaConnectionManager:
    def __init__(self):
        self._native = None
        if _rdma_native:
            try:
                self._native = _rdma_native.RdmaConnectionManager()
                self._native.initialize(0)
            except Exception as e:
                print(f"Failed to initialize RDMA connection manager: {e}")

    def create_connection(self) -> str:
        if self._native:
            return self._native.createConnection()
        return ""

    def connect_to_peer(self, peer_handle: str):
        if self._native:
            self._native.connectToPeer(peer_handle)

    def disconnect_from_peer(self, peer_handle: str):
        if self._native:
            self._native.disconnectFromPeer(peer_handle)

    def is_connected(self, peer_handle: str) -> bool:
        if self._native:
            return self._native.isConnected(peer_handle)
        return False

    def get_connected_peers(self) -> List[str]:
        if self._native:
            return self._native.getConnectedPeers()
        return []

    def clear(self):
        if self._native:
            self._native.clear()

    def get_connection_count(self) -> int:
        if self._native:
            return self._native.getConnectionCount()
        return 0

def create_device() -> RdmaDevice:
    return RdmaDevice()

def create_executor() -> RdmaExecutor:
    return RdmaExecutor()

def create_memory_manager() -> RdmaMemoryManager:
    return RdmaMemoryManager()

def create_connection_manager() -> RdmaConnectionManager:
    return RdmaConnectionManager()

def is_available() -> bool:
    if _rdma_native:
        return _rdma_native.rdma_is_available()
    return False

def setup_environment() -> bool:
    if _rdma_native:
        return _rdma_native.rdma_setup_environment()

    os.environ.setdefault("PCCL_DISABLE_IB", "0")
    os.environ.setdefault("PCCL_IB_DEVICE", "mlx5_0")
    os.environ.setdefault("PCCL_IB_GID_INDEX", "0")
    os.environ.setdefault("PCCL_IB_PORT_NUM", "1")

    try:
        device = RdmaDevice()
        return device.available
    except Exception:
        return False

def get_device_list() -> List[str]:
    if _rdma_native:
        return _rdma_native.rdma_get_device_list()

    devices = []
    if setup_environment():
        device = RdmaDevice()
        if device.available:
            info = device.activate()
            devices.append(info)

    return devices

def benchmark_rdma_latency(size: int = 1024, iterations: int = 1000) -> float:
    if not is_available():
        return 0.0

    try:
        device = RdmaDevice()
        if not device.available:
            return 0.0

        conn_manager = create_connection_manager()
        mem_manager = create_memory_manager()

        buffer_size = size * 4  # 4 bytes per float
        send_buffer = mem_manager.allocate(buffer_size)
        recv_buffer = mem_manager.allocate(buffer_size)

        send_handle = mem_manager.register_buffer(send_buffer)
        recv_handle = mem_manager.register_buffer(recv_buffer)

        import time
        start = time.time()

        for _ in range(iterations):
            pass  # Simulate RDMA operation

        end = time.time()
        latency = (end - start) * 1e6 / iterations

        mem_manager.free(send_buffer)
        mem_manager.free(recv_buffer)

        return latency
    except Exception as e:
        print(f"RDMA latency benchmark failed: {e}")
        return 0.0

def benchmark_rdma_bandwidth(size: int = 1024*1024, iterations: int = 100) -> float:
    if not is_available():
        return 0.0

    try:
        device = RdmaDevice()
        if not device.available:
            return 0.0

        import time
        start = time.time()

        for _ in range(iterations):
            pass  # Simulate RDMA transfer

        end = time.time()
        bandwidth = (size * iterations) / ((end - start) * 1e9)  # GB/s

        return bandwidth
    except Exception as e:
        print(f"RDMA bandwidth benchmark failed: {e}")
        return 0.0

from ..lang.operator import allreduce, broadcast

__all__ = [
    "RdmaDevice",
    "RdmaExecutor",
    "RdmaMemoryManager",
    "RdmaConnectionManager",
    "create_device",
    "create_executor",
    "create_memory_manager",
    "create_connection_manager",
    "is_available",
    "setup_environment",
    "get_device_list",
    "benchmark_rdma_latency",
    "benchmark_rdma_bandwidth"
]