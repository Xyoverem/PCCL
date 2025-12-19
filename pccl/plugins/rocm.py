import torch
import typing
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import pccl_native_rocm as _rocm_native
except ImportError:
    _rocm_native = None

class ROCmDevice:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._native = None
        if _rocm_native:
            try:
                self._native = _rocm_native.ROCmDevice(device_id)
                self._native.initialize()
            except Exception as e:
                print(f"Failed to initialize ROCm device {device_id}: {e}")

    def __del__(self):
        if self._native:
            self._native.shutdown()

    @property
    def name(self) -> str:
        if self._native:
            return self._native.getDeviceName()
        return get_device_name(self.device_id)

    @property
    def total_memory(self) -> int:
        if self._native:
            return self._native.getTotalMemory()
        free_mem, total_mem = get_memory_info(self.device_id)
        return total_mem

    @property
    def available_memory(self) -> int:
        if self._native:
            return self._native.getAvailableMemory()
        free_mem, total_mem = get_memory_info(self.device_id)
        return free_mem

    @property
    def compute_units(self) -> int:
        if self._native:
            return self._native.getComputeUnits()
        return 64

    @property
    def gcn_arch(self) -> str:
        if self._native:
            return self._native.getGcnArch()
        return "unknown"

    @property
    def supports_p2p(self) -> bool:
        if self._native:
            return self._native.supportsP2P()
        return True

    def set_device(self):
        if _rocm_native:
            _rocm_native.set_rocm_device(self.device_id)
        else:
            torch.cuda.set_device(self.device_id)

    def synchronize(self):
        if self._native:
            self._native.synchronize()
        else:
            torch.cuda.synchronize(self.device_id)

class ROCmExecutor:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._native = None
        self._stream = None

        if _rocm_native:
            try:
                self._native = _rocm_native.create_rocm_executor(device_id)
                self._native.initialize()
                self._stream = self._native.getCurrentStream()
            except Exception as e:
                print(f"Failed to initialize ROCm executor: {e}")

    def __del__(self):
        if self._native:
            self._native.shutdown()

    def allocate(self, size: int) -> Optional[torch.Tensor]:
        if not torch.cuda.is_available():
            return None

        try:
            tensor = torch.empty(size, dtype=torch.uint8, device=f"cuda:{self.device_id}")
            return tensor
        except Exception as e:
            print(f"Failed to allocate {size} bytes: {e}")
            return None

    def copy(self, dst: torch.Tensor, src: torch.Tensor):
        if dst.device != src.device:
            print("Source and destination devices must match")
            return

        if self._native:
            size = src.numel() * src.element_size()
            self._native.copy(dst.data_ptr(), src.data_ptr(), size)
        else:
            dst.copy_(src)

    def copy_from_host(self, dst: torch.Tensor, src: Union[torch.Tensor, np.ndarray]):
        if isinstance(src, np.ndarray):
            src = torch.from_numpy(src)

        if dst.device != src.device and not src.is_cuda:
            if self._native:
                size = src.numel() * src.element_size()
                self._native.copyFromHost(dst.data_ptr(), src.data_ptr(), size)
            else:
                dst.copy_(src, non_blocking=True)

    def copy_to_host(self, dst: Union[torch.Tensor, np.ndarray], src: torch.Tensor):
        if isinstance(dst, np.ndarray):
            dst = torch.from_numpy(dst)

        if dst.device != src.device and dst.is_cpu:
            if self._native:
                size = src.numel() * src.element_size()
                self._native.copyToHost(dst.data_ptr(), src.data_ptr(), size)
            else:
                dst.copy_(src, non_blocking=True)

    def synchronize(self):
        if self._native:
            self._native.synchronize()
        else:
            torch.cuda.synchronize(self.device_id)

    def supports_p2p(self) -> bool:
        if self._native:
            return self._native.supportsP2P()
        return torch.cuda.device_count() > 1

    def enable_p2p(self, other_device_id: int):
        if self._native:
            self._native.enableP2P(other_device_id)
        else:
            if torch.cuda.device_count() > 1:
                try:
                    torch.cuda.set_device(self.device_id)
                    torch.cuda.set_device(other_device_id)
                except Exception as e:
                    print(f"Failed to enable P2P: {e}")

class ROCmMemoryPool:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._allocated_blocks = {}
        self._free_blocks = {}

    def allocate(self, size: int) -> Optional[torch.Tensor]:
        rounded_size = ((size + 511) // 512) * 512

        if rounded_size in self._free_blocks and self._free_blocks[rounded_size]:
            block = self._free_blocks[rounded_size].pop()
            self._allocated_blocks[block.data_ptr()] = block
            return block

        try:
            tensor = torch.empty(rounded_size, dtype=torch.uint8, device=f"cuda:{self.device_id}")
            self._allocated_blocks[tensor.data_ptr()] = tensor
            return tensor[:size]
        except Exception as e:
            print(f"Failed to allocate {size} bytes: {e}")
            return None

    def free(self, tensor: torch.Tensor):
        if tensor.data_ptr() in self._allocated_blocks:
            block = self._allocated_blocks.pop(tensor.data_ptr())
            size = block.numel()

            if size not in self._free_blocks:
                self._free_blocks[size] = []
            self._free_blocks[size].append(block)

    def clear(self):
        self._allocated_blocks.clear()
        self._free_blocks.clear()

class ROCmTopologyBuilder:
    @staticmethod
    def build_p2p_topology() -> Tuple[List[dict], List[dict]]:
        devices = []
        links = []

        device_count = get_device_count()
        for i in range(device_count):
            device_name = get_device_name(i)
            free_mem, total_mem = get_memory_info(i)

            device_info = {
                "device_id": i,
                "device_type": "ROCM",
                "device_name": device_name,
                "memory_bandwidth": 900.0,
                "compute_capability": 8.0,
                "memory_size": total_mem,
                "numa_node": i % 2
            }
            devices.append(device_info)

        for i in range(device_count):
            for j in range(i + 1, device_count):
                if can_access_peer(i, j):
                    bandwidth = estimate_p2p_bandwidth(i, j)
                    latency = estimate_p2p_latency(i, j)

                    link_info = {
                        "src_device": i,
                        "dst_device": j,
                        "interconnect_type": "XGMI" if i // 2 == j // 2 else "PCIe",
                        "bandwidth": bandwidth,
                        "latency": latency,
                        "bidirectional": True
                    }
                    links.append(link_info)

        return devices, links

    @staticmethod
    def is_p2p_supported(device1: int, device2: int) -> bool:
        return can_access_peer(device1, device2)

    @staticmethod
    def estimate_p2p_bandwidth(device1: int, device2: int) -> float:
        if device1 // 2 == device2 // 2:
            return 350.0
        else:
            return 50.0

    @staticmethod
    def estimate_p2p_latency(device1: int, device2: int) -> float:
        if device1 // 2 == device2 // 2:
            return 0.2
        else:
            return 2.0

def get_device_count() -> int:
    if _rocm_native:
        return _rocm_native.get_rocm_device_count()
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def set_device(device_id: int) -> bool:
    if _rocm_native:
        return _rocm_native.set_rocm_device(device_id)
    elif torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return True
    return False

def get_device_name(device_id: int) -> str:
    if _rocm_native:
        return _rocm_native.get_rocm_device_name(device_id)
    elif torch.cuda.is_available():
        return torch.cuda.get_device_name(device_id)
    return "Unknown ROCm Device"

def get_memory_info(device_id: int) -> Tuple[int, int]:
    if _rocm_native:
        free_mem, total_mem = _rocm_native.rocm_memory_info(device_id)
        return free_mem, total_mem
    elif torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return torch.cuda.mem_get_info()
    return 0, 0

def can_access_peer(device1: int, device2: int) -> bool:
    if _rocm_native:
        device_manager = _rocm_native.create_rocm_device_manager()
        device_manager.initialize()
        return device_manager.canAccessPeer(device1, device2)
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return torch.cuda.can_device_access_peer(device1, device2)
    return False

def enable_p2p_between_all_devices():
    device_count = get_device_count()
    if device_count <= 1:
        return

    if _rocm_native:
        device_manager = _rocm_native.create_rocm_device_manager()
        device_manager.initialize()
        device_manager.enableP2PBetweenAllDevices()
    elif torch.cuda.is_available():
        for i in range(device_count):
            for j in range(i + 1, device_count):
                if torch.cuda.can_device_access_peer(i, j):
                    try:
                        torch.cuda.set_device(i)
                        torch.cuda.set_device(j)
                    except Exception:
                        pass

def estimate_p2p_bandwidth(device1: int, device2: int) -> float:
    return ROCmTopologyBuilder.estimate_p2p_bandwidth(device1, device2)

def estimate_p2p_latency(device1: int, device2: int) -> float:
    return ROCmTopologyBuilder.estimate_p2p_latency(device1, device2)

def create_executor(device_id: int = 0) -> ROCmExecutor:
    return ROCmExecutor(device_id)

def create_device(device_id: int = 0) -> ROCmDevice:
    return ROCmDevice(device_id)

def create_memory_pool(device_id: int = 0) -> ROCmMemoryPool:
    return ROCmMemoryPool(device_id)

def is_available() -> bool:
    return get_device_count() > 0

def get_device_summary() -> str:
    if not is_available():
        return "No ROCm devices available"

    summary = "ROCm Devices:\n"
    device_count = get_device_count()
    summary += f"Device Count: {device_count}\n"

    for i in range(device_count):
        device_name = get_device_name(i)
        free_mem, total_mem = get_memory_info(i)
        summary += f"Device {i}: {device_name}\n"
        summary += f"  Memory: {total_mem / (1024**3):.2f} GB total, {free_mem / (1024**3):.2f} GB free\n"

    return summary

def benchmark_allreduce(size: int, algorithm: str = "ring", iterations: int = 10) -> float:
    if not is_available():
        return 0.0

    try:
        import time
        data = torch.randn(size, device="cuda")

        allreduce_op = allreduce(
            algorithm=algorithm,
            participants=[0, 1, 2, 3] if get_device_count() >= 4 else [0, 1],
            reduce_op="sum"
        )

        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(iterations):
            result = allreduce_op.execute(data)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / iterations * 1000
        return avg_time

    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 0.0

from ..lang.operator import allreduce, broadcast

__all__ = [
    "ROCmDevice",
    "ROCmExecutor",
    "ROCmMemoryPool",
    "ROCmTopologyBuilder",
    "get_device_count",
    "set_device",
    "get_device_name",
    "get_memory_info",
    "can_access_peer",
    "enable_p2p_between_all_devices",
    "estimate_p2p_bandwidth",
    "estimate_p2p_latency",
    "create_executor",
    "create_device",
    "create_memory_pool",
    "is_available",
    "get_device_summary",
    "benchmark_allreduce"
]