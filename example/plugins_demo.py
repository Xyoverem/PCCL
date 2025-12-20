"""
PCCL Plugins Demo
Demonstrates usage of all available plugins (CPU, CUDA, RDMA, ROCm)
"""

import torch
import pccl
from pccl.lang import communication, allreduce, broadcast
from pccl.plugins import *

def print_separator(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def demo_cpu_plugin():
    print_separator("CPU Plugin Demo")

    if not cpu_is_available():
        print("CPU plugin is not available")
        return

    print(f"CPU Core Count: {get_cpu_core_count()}")
    total_mem, available_mem = get_cpu_memory_info()
    print(f"CPU Memory: {available_mem / (1024**3):.2f} GB available / {total_mem / (1024**3):.2f} GB total")

    cpu_device = create_cpu_device()
    cpu_executor = create_cpu_executor()

    if cpu_device.available:
        print("CPU device is available")

        size = 1024 * 1024
        tensor = cpu_device.allocate(size)
        print(f"Allocated {size} bytes on CPU")

        memory_manager = create_cpu_memory_manager()
        allocated = memory_manager.get_allocated_bytes()
        print(f"Memory manager allocated: {allocated} bytes")

        if tensor is not None:
            cpu_device.deallocate(tensor)

        memory_manager.clear()

    thread_pool = create_cpu_thread_pool(4)
    print(f"Thread pool threads: {thread_pool.get_thread_count()}")

    copy_time = cpu_benchmark_copy(1024 * 1024, 10)
    print(f"CPU copy benchmark (1MB): {copy_time:.4f} ms")

    allreduce_time = cpu_benchmark_allreduce(1024 * 1024, 5)
    print(f"CPU AllReduce benchmark (1MB): {allreduce_time:.4f} ms")

def demo_cuda_plugin():
    print_separator("CUDA Plugin Demo")

    if not cuda_is_available():
        print("CUDA plugin is not available - CUDA not detected")
        return

    device_count = get_cuda_device_count()
    print(f"CUDA Devices Available: {device_count}")

    for device_id in range(device_count):
        print(f"\nDevice {device_id}:")
        print(f"  Name: {get_cuda_device_name(device_id)}")

        props = get_cuda_device_properties(device_id)
        if props:
            print(f"  Compute Capability: {props['major']}.{props['minor']}")
            print(f"  Memory: {props['totalGlobalMem'] / (1024**3):.2f} GB")
            print(f"  Multiprocessors: {props['multiProcessorCount']}")

        free_mem, total_mem = get_cuda_memory_info(device_id)
        print(f"  Memory Usage: {free_mem / (1024**3):.2f} GB free / {total_mem / (1024**3):.2f} GB total")

        if device_count > 1:
            for other_id in range(device_count):
                if other_id != device_id:
                    can_access = cuda_can_access_peer(device_id, other_id)
                    print(f"  Can access peer {other_id}: {can_access}")

                    if can_access:
                        cuda_enable_peer_access(device_id, other_id)
                        print(f"  Enabled P2P to device {other_id}")
    if device_count > 0:
        cuda_device = create_cuda_device(0)
        cuda_executor = create_cuda_executor(0)

        size = 1024 * 1024
        tensor = cuda_device.allocate(size)
        if tensor is not None:
            print(f"\nAllocated {size} bytes on CUDA device 0")
            print(f"Tensor shape: {tensor.shape}, device: {tensor.device}")
            cuda_device.deallocate(tensor)

        copy_time = cuda_benchmark_copy(1024 * 1024, 100)
        print(f"CUDA copy benchmark (1MB): {copy_time:.4f} ms")

        allreduce_time = cuda_benchmark_allreduce(1024 * 1024, 10)
        print(f"CUDA AllReduce benchmark (1MB): {allreduce_time:.4f} ms")

def demo_rocm_plugin():
    print_separator("ROCm Plugin Demo")

    if not rocm_is_available():
        print("ROCm plugin is not available - ROCm not detected")
        return

    device_count = get_rocm_device_count()
    print(f"ROCm Devices Available: {device_count}")

    for device_id in range(device_count):
        print(f"\nDevice {device_id}:")
        print(f"  Name: {get_rocm_device_name(device_id)}")

        free_mem, total_mem = get_rocm_memory_info(device_id)
        print(f"  Memory Usage: {free_mem / (1024**3):.2f} GB free / {total_mem / (1024**3):.2f} GB total")

        if device_count > 1:
            for other_id in range(device_count):
                if other_id != device_id:
                    can_access = rocm_can_access_peer(device_id, other_id)
                    print(f"  Can access peer {other_id}: {can_access}")
    if device_count > 0:
        rocm_device = create_rocm_device(0)
        rocm_executor = create_rocm_executor(0)

        size = 1024 * 1024
        tensor = rocm_device.allocate(size)
        if tensor is not None:
            print(f"\nAllocated {size} bytes on ROCm device 0")
            rocm_device.deallocate(tensor)

        allreduce_time = rocm_benchmark_allreduce(1024 * 1024, 10)
        print(f"ROCm AllReduce benchmark (1MB): {allreduce_time:.4f} ms")

def demo_rdma_plugin():
    print_separator("RDMA Plugin Demo")

    if not rdma_is_available():
        print("RDMA plugin is not available")
        print("To enable RDMA:")
        print("1. Install RDMA libraries (libibverbs, librdmacm)")
        print("2. Ensure Infiniband devices are available")
        print("3. Set environment variables:")
        print("   export PCCL_DISABLE_IB=0")
        print("   export PCCL_IB_DEVICE=mlx5_0")
        print("   export PCCL_IB_PORT_NUM=1")
        print("   export PCCL_IB_GID_INDEX=0")
        return

    print("RDMA plugin is available")

    if setup_environment():
        print("RDMA environment setup successful")

        device_list = get_device_list()
        print(f"Found {len(device_list)} RDMA devices")

        conn_manager = create_rdma_connection_manager()
        local_info = conn_manager.create_connection()
        if local_info:
            print("Created local RDMA connection info")
            print(f"Connection info length: {len(local_info)} bytes")

        mem_manager = create_rdma_memory_manager()
        buffer = mem_manager.allocate(1024 * 1024)
        if buffer is not None:
            print("Allocated 1MB RDMA buffer")

            buffer_handle = mem_manager.register_buffer(buffer)
            if buffer_handle:
                print("Registered RDMA buffer")

            mem_manager.free(buffer)
        latency = benchmark_rdma_latency(1024, 100)
        print(f"RDMA latency benchmark (1KB): {latency:.2f} μs")

        bandwidth = benchmark_rdma_bandwidth(1024 * 1024, 10)
        print(f"RDMA bandwidth benchmark (1MB): {bandwidth:.2f} GB/s")

def demo_communication_patterns():
    print_separator("Communication Patterns with Different Plugins")

    algorithms = ["ring", "tree", "rabenseifner"]
    size = 1024 * 1024

    for algo in algorithms:
        print(f"\n{algo.upper()} Algorithm:")

        try:
            allreduce_op = allreduce(
                algorithm=algo,
                participants=[0, 1, 2, 3],
                reduce_op="sum"
            )

            if torch.cuda.is_available() and cuda_is_available():
                data_cuda = torch.randn(size, device="cuda")
                result_cuda = allreduce_op.execute(data_cuda)
                print(f"  CUDA: {result_cuda.norm().item():.4f}")

            if rocm_is_available():
                print("  ROCm: Plugin available")

            if cpu_is_available():
                data_cpu = torch.randn(size)
                result_cpu = allreduce_op.execute(data_cpu)
                print(f"  CPU: {result_cpu.norm().item():.4f}")

        except Exception as e:
            print(f"  Error with {algo}: {e}")

def demo_declarative_patterns():
    print_separator("Declarative Communication Patterns")

    @communication
    class MultiDeviceTraining:
        cpu_allreduce = allreduce(
            reduce_op="sum",
            algorithm="ring",
            participants=[0, 1],
            enable_overlap=False
        )

        gpu_allreduce = allreduce(
            reduce_op="sum",
            algorithm="ring",
            participants=[0, 1, 2, 3],
            enable_overlap=True
        )

    print("Created declarative communication pattern")
    pattern = MultiDeviceTraining()

    if cpu_is_available():
        try:
            grads_cpu = [torch.randn(512, 512) for _ in range(2)]
            result_cpu = pattern.cpu_allreduce.execute(grads_cpu)
            print(f"CPU AllReduce result shape: {[g.shape for g in result_cpu]}")
        except Exception as e:
            print(f"CPU execution failed: {e}")
    if torch.cuda.is_available() and cuda_is_available():
        try:
            grads_gpu = [torch.randn(512, 512, device="cuda") for _ in range(4)]
            result_gpu = pattern.gpu_allreduce.execute(grads_gpu)
            print(f"GPU AllReduce result shape: {[g.shape for g in result_gpu]}")
        except Exception as e:
            print(f"GPU execution failed: {e}")

def benchmark_all_plugins():
    print_separator("Cross-Plugin Performance Comparison")

    sizes = [1024, 4096, 16384, 65536]

    print(f"{'Size':<8} {'CPU (ms)':<12} {'CUDA (ms)':<12} {'ROCm (ms)':<12} {'RDMA (μs)':<12}")
    print("-" * 60)

    for size in sizes:
        cpu_time = cpu_benchmark_allreduce(size, 10) if cpu_is_available() else 0.0
        cuda_time = cuda_benchmark_allreduce(size, 10) if cuda_is_available() else 0.0
        rocm_time = rocm_benchmark_allreduce(size, 10) if rocm_is_available() else 0.0
        rdma_time = benchmark_rdma_latency(size, 100) if rdma_is_available() else 0.0

        print(f"{size:<8} {cpu_time:<12.4f} {cuda_time:<12.4f} {rocm_time:<12.4f} {rdma_time:<12.2f}")

def main():
    print("PCCL Plugins Demo")
    print("Demonstrating CPU, CUDA, RDMA, and ROCm plugin functionality")

    demo_cpu_plugin()
    demo_cuda_plugin()
    demo_rocm_plugin()
    demo_rdma_plugin()

    demo_communication_patterns()
    demo_declarative_patterns()
    benchmark_all_plugins()

    print_separator("Demo Complete")
    print("PCCL plugin system successfully demonstrated!")
    print("\nPlugin Summary:")
    print(f"  CPU: {'✓ Available' if cpu_is_available() else '✗ Not Available'}")
    print(f"  CUDA: {'✓ Available' if cuda_is_available() else '✗ Not Available'}")
    print(f"  ROCm: {'✓ Available' if rocm_is_available() else '✗ Not Available'}")
    print(f"  RDMA: {'✓ Available' if rdma_is_available() else '✗ Not Available'}")

if __name__ == "__main__":
    main()