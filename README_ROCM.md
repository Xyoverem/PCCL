# PCCL ROCm Plugin

The PCCL ROCm plugin provides support for AMD GPU devices through the ROCm platform, enabling high-performance communication operations on AMD hardware.

## Features

- **Device Management**: Complete ROCm device discovery and management
- **Memory Management**: HIP-based memory allocation and management
- **Communication Primitives**: AllReduce, Reduce, Broadcast, and point-to-point operations
- **P2P Support**: Peer-to-peer communication between AMD GPUs
- **Topology Awareness**: Automatic detection of XGMI and PCIe interconnects
- **Stream Management**: HIP stream and event management
- **BLAS Integration**: rocBLAS integration for optimized compute operations

## Prerequisites

### ROCm Installation
- ROCm 5.0 or later
- HIP runtime libraries
- rocBLAS for BLAS operations
- rocrand for random number generation

### Environment Setup
```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH
export PATH=$ROCM_PATH/bin:$PATH
```

## Building with ROCm Support

The PCCL build system automatically detects ROCm and includes the plugin if available:

```bash
# Set ROCm installation path
export ROCM_PATH=/opt/rocm

# Build PCCL with ROCm support
python setup.py build_ext --inplace
```

## Usage

### Basic ROCm Device Operations

```python
import pccl
from pccl.plugins import rocm

# Check ROCm availability
if rocm.is_available():
    device_count = rocm.get_device_count()
    print(f"Found {device_count} ROCm devices")

    # Create ROCm executor
    executor = rocm.create_executor(device_id=0)
    executor.initialize()

    # Allocate memory
    size = 1024 * 1024
    tensor = torch.randn(size, device="cuda")

    # Perform operations
    executor.synchronize()
    executor.shutdown()
```

### ROCm AllReduce Operations

```python
from pccl.lang import allreduce

# Create AllReduce operation for ROCm
allreduce_op = allreduce(
    algorithm="ring",
    participants=[0, 1, 2, 3],
    reduce_op="sum"
)

# Execute on ROCm device
data = torch.randn(1024, device="cuda")
result = allreduce_op.execute(data)
```

### Memory Management

```python
# Use memory pool for better performance
memory_pool = rocm.create_memory_pool()

# Allocate from pool
buffer1 = memory_pool.allocate(1024 * 1024)
buffer2 = memory_pool.allocate(2048 * 2048)

# Use buffers...

# Return to pool
memory_pool.free(buffer1)
memory_pool.free(buffer2)
```

### Stream Operations

```python
executor = rocm.create_executor()
stream_manager = executor.get_stream_manager()

# Create multiple streams
stream1 = stream_manager.createStream()
stream2 = stream_manager.createStream()

# Create events for synchronization
event1 = stream_manager.createEvent()
event2 = stream_manager.createEvent()

# Record and wait for events
stream_manager.recordEvent(event1, stream1)
stream_manager.waitForEvent(event1, stream2)

# Clean up
stream_manager.synchronizeAllStreams()
```

### Topology Detection

```python
from pccl.lang.topology import TopologyManager, TopologyConfig

topology_manager = TopologyManager()
topo_config = TopologyConfig()
topo_config.enable_gpu_p2p = True

topology = topology_manager.build_topology(topo_config)

print("Detected ROCm topology:")
for device in topology.devices:
    if device.device_type == "ROCM":
        print(f"  Device {device.device_id}: {device.device_name}")

for link in topology.links:
    print(f"  Link: {link.src_device} <-> {link.dst_device} ({link.interconnect_type})")
```

## Performance Optimization

### Algorithm Selection
PCCL automatically selects the optimal algorithm based on:
- Number of participating devices
- Data size
- Network topology (XGMI vs PCIe)
- Memory bandwidth characteristics

```python
# Automatic algorithm selection
allreduce_op = allreduce(
    algorithm="auto",  # PCCL will choose optimal algorithm
    participants=[0, 1, 2, 3, 4, 5, 6, 7],
    enable_overlap=True
)
```

### Memory Optimization
- Use unified memory for频繁的数据传输
- Enable P2P for direct GPU-GPU communication
- Pool allocation to reduce allocation overhead

### P2P Configuration
```python
# Enable P2P between all devices
rocm.enable_p2p_between_all_devices()

# Check P2P availability
if rocm.can_access_peer(0, 1):
    print("P2P communication available between devices 0 and 1")
```

## Benchmarking

```python
def benchmark_rocm_allreduce():
    sizes = [1024, 4096, 16384, 65536]
    algorithms = ["ring", "tree"]

    for algo in algorithms:
        print(f"\nAlgorithm: {algo}")
        for size in sizes:
            avg_time = rocm.benchmark_allreduce(size, algorithm=algo)
            bandwidth = (size * 4 * 2) / (avg_time * 1e6)
            print(f"  Size {size:5d}: {avg_time:8.4f} ms, {bandwidth:8.2f} GB/s")

benchmark_rocm_allreduce()
```

## Troubleshooting

### Common Issues

1. **ROCm not detected**
   ```bash
   echo $ROCM_PATH
   ls -la $ROCM_PATH/lib/libhip.so
   ```

2. **Device access errors**
   ```bash
   # Check device permissions
   ls -la /dev/kfd
   sudo usermod -a -G video,render $USER
   ```

3. **Memory allocation failures**
   - Check available memory: `rocm.get_memory_info(device_id)`
   - Reduce tensor sizes or enable memory pooling

### Debug Mode

Enable debug logging for detailed error information:

```python
# Rebuild with debug mode
python setup.py build_ext --inplace --debug

# Enable runtime debug
import pccl
pccl.set_debug_level(1)
```

## Architecture

The ROCm plugin consists of several components:

- **ROCmDeviceManager**: Device discovery and management
- **ROCmExecutor**: Main execution engine
- **ROCmMemoryManager**: Memory allocation and P2P operations
- **ROCmStreamManager**: HIP stream and event management
- **ROCmKernelRegistry**: Kernel compilation and management
- **ROCmTopologyBuilder**: System topology detection

## Examples

See `example/rocm_allreduce_example.py` for comprehensive usage examples including:
- Device information display
- Simple AllReduce operations
- Distributed training patterns
- Memory management
- Stream operations
- Topology awareness
- Performance benchmarking

## Integration with Existing Code

The ROCm plugin is fully compatible with the existing PCCL Python DSL:

```python
# Declarative communication pattern works with ROCm
@communication
class ROcmTraining:
    gradient_allreduce = allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=[0, 1, 2, 3],
        enable_overlap=True
    )

# Usage is transparent - PCCL automatically uses ROCm when available
pattern = ROcmTraining()
result = pattern.gradient_allreduce.execute(gradients)
```