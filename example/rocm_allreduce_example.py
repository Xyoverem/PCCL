import torch
import pccl
import numpy as np
from pccl.lang import communication, allreduce, broadcast, ReduceOp
from pccl.plugins import rocm

def rocm_device_info():
    print("=== ROCm Device Information ===")
    try:
        device_count = rocm.get_device_count()
        print(f"Available ROCm devices: {device_count}")

        for i in range(device_count):
            device_name = rocm.get_device_name(i)
            free_mem, total_mem = rocm.get_memory_info(i)
            print(f"Device {i}: {device_name}")
            print(f"  Memory: {free_mem / (1024**3):.2f} GB free / {total_mem / (1024**3):.2f} GB total")
    except Exception as e:
        print(f"ROCm not available: {e}")

def simple_rocm_allreduce():
    print("\n=== Simple ROCm AllReduce ===")
    try:
        size = 1024 * 1024
        data = torch.randn(size, device="cuda")

        print(f"Input tensor: shape={data.shape}, device={data.device}")
        print(f"Input norm: {data.norm().item():.4f}")

        rocm_executor = rocm.create_executor()
        rocm_executor.initialize()

        allreduce_op = pccl.allreduce(
            algorithm="ring",
            participants=[0, 1, 2, 3],
            reduce_op="sum"
        )

        result = allreduce_op.execute(data)

        print(f"Output tensor: shape={result.shape}, device={result.device}")
        print(f"Output norm: {result.norm().item():.4f}")

        rocm_executor.shutdown()

    except Exception as e:
        print(f"Simple ROCm AllReduce failed: {e}")

@communication
class ROCmTrainingPattern:
    gradient_allreduce = allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=[0, 1, 2, 3, 4, 5, 6, 7],
        enable_overlap=True
    )

    weight_broadcast = broadcast(
        root_rank=0,
        participants=[0, 1, 2, 3, 4, 5, 6, 7]
    )

def rocm_distributed_training():
    print("\n=== ROCm Distributed Training Pattern ===")
    try:
        comm_pattern = ROCmTrainingPattern()

        gradients = []
        for i in range(8):
            grad = torch.randn(512, 512, device="cuda") * 0.1
            gradients.append(grad)

        print(f"Gradient shapes: {[g.shape for g in gradients]}")
        print(f"Initial gradient norms: {[g.norm().item():.4f for g in gradients[:3]]}...")

        allreduced_grads = comm_pattern.gradient_allreduce.execute(gradients)

        print(f"AllReduced gradient shapes: {[g.shape for g in allreduced_grads]}")
        print(f"Final gradient norms: {[g.norm().item():.4f for g in allreduced_grads[:3]]}...")

    except Exception as e:
        print(f"ROCm distributed training failed: {e}")

def rocm_memory_management():
    print("\n=== ROCm Memory Management ===")
    try:
        rocm_executor = rocm.create_executor()
        rocm_executor.initialize()

        sizes = [1024, 2048, 4096, 8192]
        allocations = []

        print("Allocating memory buffers:")
        for i, size in enumerate(sizes):
            tensor = torch.randn(size, size, device="cuda")
            allocations.append(tensor)
            free_mem, total_mem = rocm.get_memory_info(0)
            print(f"  Buffer {i+1}: {tensor.numel() * tensor.element_size() / (1024**2):.2f} MB")
            print(f"  Available memory: {free_mem / (1024**3):.2f} GB")

        print("\nPerforming memory copy test:")
        src = allocations[0]
        dst = torch.empty_like(src)

        rocm_executor.copy(dst, src, src.numel() * src.element_size())

        print(f"Copy test passed: {torch.allclose(src, dst)}")

        rocm_executor.shutdown()

    except Exception as e:
        print(f"ROCm memory management failed: {e}")

def rocm_stream_operations():
    print("\n=== ROCm Stream Operations ===")
    try:
        rocm_executor = rocm.create_executor()
        rocm_executor.initialize()

        stream_manager = rocm_executor.get_stream_manager()

        stream1 = stream_manager.createStream()
        stream2 = stream_manager.createStream()

        event1 = stream_manager.createEvent()
        event2 = stream_manager.createEvent()

        data1 = torch.randn(1024, 1024, device="cuda")
        data2 = torch.randn(1024, 1024, device="cuda")

        stream_manager.recordEvent(event1, stream1)
        stream_manager.waitForEvent(event1, stream2)
        stream_manager.recordEvent(event2, stream2)

        print("Stream operations completed successfully")
        print(f"Stream 1 busy: {stream_manager.isStreamBusy(0)}")
        print(f"Stream 2 busy: {stream_manager.isStreamBusy(1)}")

        stream_manager.synchronizeAllStreams()
        stream_manager.destroyStream(stream1)
        stream_manager.destroyStream(stream2)
        stream_manager.destroyEvent(event1)
        stream_manager.destroyEvent(event2)

        rocm_executor.shutdown()

    except Exception as e:
        print(f"ROCm stream operations failed: {e}")

def rocm_topology_awareness():
    print("\n=== ROCm Topology Awareness ===")
    try:
        from pccl.lang.topology import TopologyManager, TopologyConfig

        topology_manager = TopologyManager()
        topo_config = TopologyConfig()
        topo_config.enable_gpu_p2p = True
        topo_config.enable_nvlink_detection = True

        topology = topology_manager.build_topology(topo_config)

        print("Detected devices:")
        for device in topology.devices:
            print(f"  Device {device.device_id}: {device.device_name}")
            print(f"    Type: {device.device_type}")
            print(f"    Memory: {device.memory_size / (1024**3):.2f} GB")
            print(f"    Bandwidth: {device.memory_bandwidth:.2f} GB/s")

        print("\nInterconnects:")
        for link in topology.links:
            print(f"  Device {link.src_device} <-> Device {link.dst_device}")
            print(f"    Type: {link.interconnect_type}")
            print(f"    Bandwidth: {link.bandwidth:.2f} GB/s")
            print(f"    Latency: {link.latency:.4f} ms")

    except Exception as e:
        print(f"ROCm topology awareness failed: {e}")

def benchmark_rocm_allreduce():
    print("\n=== ROCm AllReduce Benchmark ===")
    try:
        algorithms = ["ring", "tree"]
        sizes = [1024, 4096, 16384, 65536]

        for algo in algorithms:
            print(f"\nAlgorithm: {algo}")
            for size in sizes:
                data = torch.randn(size, device="cuda")

                allreduce_op = pccl.allreduce(
                    algorithm=algo,
                    participants=[0, 1, 2, 3],
                    reduce_op="sum"
                )

                import time
                start_time = time.time()
                result = allreduce_op.execute(data)
                end_time = time.time()

                elapsed_ms = (end_time - start_time) * 1000
                bandwidth = (size * 4 * 2) / (elapsed_ms * 1e6)

                print(f"  Size {size:5d}: {elapsed_ms:8.4f} ms, {bandwidth:8.2f} GB/s")

    except Exception as e:
        print(f"ROCm benchmark failed: {e}")

if __name__ == "__main__":
    print("PCCL ROCm Plugin Examples")
    print("=" * 50)

    rocm_device_info()
    simple_rocm_allreduce()
    rocm_distributed_training()
    rocm_memory_management()
    rocm_stream_operations()
    rocm_topology_awareness()
    benchmark_rocm_allreduce()

    print("\nROCm examples completed!")