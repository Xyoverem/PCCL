# PCCL Plugin System

PCCL features a comprehensive plugin architecture that supports multiple hardware platforms and communication mechanisms. The plugin system provides a unified interface while allowing platform-specific optimizations.

## Supported Plugins

### 1. CPU Plugin
- **Purpose**: CPU-based communication and computation
- **Features**:
  - Multi-threading support with thread pool
  - Memory management with shared memory support
  - Kernel registry for custom CPU functions
  - Optimized copy and reduction operations

```python
from pccl.plugins.cpu import create_cpu_executor

executor = create_cpu_executor()
tensor = executor.allocate(1024 * 1024)
executor.synchronize()
```

### 2. CUDA Plugin
- **Purpose**: NVIDIA GPU acceleration
- **Features**:
  - CUDA streams and events management
  - GPU memory allocation and management
  - P2P communication between GPUs
  - cuBLAS integration
  - IPC memory sharing
  - Dynamic kernel compilation

```python
from pccl.plugins.cuda import create_cuda_executor

executor = create_cuda_executor(0)  # GPU 0
tensor = executor.allocate(1024 * 1024)
executor.enable_p2p(1)  # Enable P2P to GPU 1
```

### 3. ROCm Plugin
- **Purpose**: AMD GPU support
- **Features**:
  - HIP runtime integration
  - AMD GPU memory management
  - P2P communication between AMD GPUs
  - rocBLAS integration
  - Kernel compilation with HIPRTC
  - XGMI and PCIe topology awareness

```python
from pccl.plugins.rocm import create_rocm_executor

executor = create_rocm_executor(0)
tensor = executor.allocate(1024 * 1024)
```

### 4. RDMA Plugin
- **Purpose**: High-performance network communication
- **Features**:
  - Infiniband/RoCE support
  - Zero-copy data transfer
  - Remote memory access
  - Connection management
  - RDMA memory registration

```python
from pccl.plugins.rdma import create_rdma_executor

executor = create_rdma_executor()
conn_manager = executor.getConnectionManager()
local_info = conn_manager.create_connection()
```

## Installation

### Automatic Detection
The build system automatically detects available plugins:

```bash
# Build with all available plugins
python setup.py build_ext --inplace

# Install in development mode
pip install -e .
```

### Manual Plugin Selection
You can control which plugins to build:

```bash
# CPU plugin (always built)
# CUDA plugin (requires CUDA_HOME)
export CUDA_HOME=/usr/local/cuda

# ROCm plugin (requires ROCM_PATH)
export ROCM_PATH=/opt/rocm

# RDMA plugin (requires libibverbs)
sudo apt-get install libibverbs-dev librdmacm-dev
```

## Plugin Architecture

### Core Components

1. **Device Classes**: Hardware abstraction layer
   - Memory allocation/deallocation
   - Device properties and capabilities
   - IPC and P2P support

2. **Executor Classes**: Runtime execution engine
   - Memory management
   - Stream/synchronization handling
   - Kernel execution
   - Performance measurement

3. **Memory Managers**: Specialized memory handling
   - Platform-specific allocation strategies
   - Buffer registration for IPC/RDMA
   - Memory pool management

4. **Python Bindings**: Native Python interface
   - Direct access to C++ implementations
   - Tensor integration with PyTorch
   - Error handling and validation

### Plugin Detection and Loading

```python
import pccl.plugins as plugins

# Check plugin availability
print(f"CPU: {plugins.cpu_is_available()}")
print(f"CUDA: {plugins.cuda_is_available()}")
print(f"ROCm: {plugins.rocm_is_available()}")
print(f"RDMA: {plugins.rdma_is_available()}")

# Create executors for available plugins
if plugins.cuda_is_available():
    cuda_executor = plugins.create_cuda_executor()

if plugins.rocm_is_available():
    rocm_executor = plugins.create_rocm_executor()
```

## Usage Examples

### Cross-Platform Communication

```python
import pccl
from pccl.plugins import cuda_is_available, rocm_is_available
from pccl.lang import allreduce

def create_allreduce_op():
    # Choose the best available platform
    if cuda_is_available():
        device = "cuda"
        participants = list(range(min(4, cuda.get_device_count())))
    elif rocm_is_available():
        device = "cuda"  # ROCm uses CUDA interface in PyTorch
        participants = list(range(min(4, rocm.get_device_count())))
    else:
        device = "cpu"
        participants = [0, 1]

    return allreduce(
        algorithm="ring",
        participants=participants,
        reduce_op="sum"
    )

# Use with automatic platform selection
allreduce_op = create_allreduce_op()
data = torch.randn(1024, device=device)
result = allreduce_op.execute(data)
```

### Memory Management

```python
from pccl.plugins import create_cuda_memory_manager, create_rdma_memory_manager

# GPU memory management
if cuda_is_available():
    gpu_manager = create_cuda_memory_manager()
    gpu_buffer = gpu_manager.allocate(1024 * 1024)
    gpu_buffer_handle = gpu_manager.register_buffer(gpu_buffer)

    # Use buffer for communication...

    gpu_manager.free(gpu_buffer)

# RDMA memory management
if rdma_is_available():
    rdma_manager = create_rdma_memory_manager()
    rdma_buffer = rdma_manager.allocate(1024 * 1024)
    rdma_handle = rdma_manager.register_buffer(rdma_buffer)

    # Use for remote memory access...

    rdma_manager.free(rdma_buffer)
```

### Performance Benchmarking

```python
from pccl.plugins import (
    cpu_benchmark_allreduce,
    cuda_benchmark_allreduce,
    rocm_benchmark_allreduce,
    rdma_benchmark_latency
)

def benchmark_all_platforms():
    size = 1024 * 1024

    if cpu_is_available():
        cpu_time = cpu_benchmark_allreduce(size)
        print(f"CPU AllReduce: {cpu_time:.4f} ms")

    if cuda_is_available():
        cuda_time = cuda_benchmark_allreduce(size)
        print(f"CUDA AllReduce: {cuda_time:.4f} ms")

    if rocm_is_available():
        rocm_time = rocm_benchmark_allreduce(size)
        print(f"ROCm AllReduce: {rocm_time:.4f} ms")

    if rdma_is_available():
        rdma_time = rdma_benchmark_latency(size)
        print(f"RDMA Latency: {rdma_time:.2f} μs")
```

### P2P Communication Setup

```python
from pccl.plugins import create_cuda_executor, create_rocm_executor

def setup_p2p_communication():
    # CUDA P2P
    if cuda_is_available() and cuda.get_device_count() > 1:
        executor = create_cuda_executor(0)
        if executor.supports_p2p():
            executor.enable_p2p(1)
            print("CUDA P2P enabled between devices 0 and 1")

    # ROCm P2P
    if rocm_is_available() and rocm.get_device_count() > 1:
        executor = create_rocm_executor(0)
        if executor.supports_p2P():
            executor.enable_p2P(1)
            print("ROCm P2P enabled between devices 0 and 1")
```

## Integration with PCCL DSL

The plugins integrate seamlessly with the PCCL Python DSL:

```python
from pccl.lang import communication, allreduce
from pccl.plugins import cuda_is_available, rocm_is_available

@communication
class DistributedTraining:
    # Platform-aware AllReduce
    gradient_allreduce = allreduce(
        reduce_op="sum",
        algorithm="hierarchical",  # Auto-selects based on topology
        enable_overlap=True,
        participants=list(range(8)) if cuda_is_available() else [0, 1]
    )

# Usage is transparent
pattern = DistributedTraining()
result = pattern.gradient_allreduce.execute(gradients)
```

## Development and Extension

### Creating Custom Plugins

To add a new plugin:

1. **Create C++ implementation**:
   ```cpp
   // include/plugins/my_executor/device.h
   class MyDevice : public DeviceBase {
   public:
     bool allocatorAvailable() override;
     void* allocate(long nbytes) override;
     // ... other methods
   };
   ```

2. **Create Python bindings**:
   ```cpp
   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     py::class_<MyDevice>(m, "MyDevice")
       .def(py::init<>())
       .def("allocatorAvailable", &MyDevice::allocatorAvailable);
   }
   ```

3. **Create Python wrapper**:
   ```python
   # pccl/plugins/my.py
   class MyDevice:
       def __init__(self):
           self._native = _my_native.MyDevice() if _my_native else None
   ```

4. **Update setup.py**:
   ```python
   if my_library_available:
       extensions.append(
           CppExtension(name='pccl_native_my', ...)
       )
   ```

### Plugin Configuration

Most plugins support environment variable configuration:

```bash
# CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ROCm
export ROCM_VISIBLE_DEVICES=0,1

# RDMA
export PCCL_IB_DEVICE=mlx5_0
export PCCL_IB_PORT_NUM=1
export PCCL_IB_GID_INDEX=0
```

## Troubleshooting

### Common Issues

1. **Plugin Not Detected**:
   ```bash
   # Check installation
   python -c "import pccl.plugins; print(pccl.plugins.cuda_is_available())"
   ```

2. **Build Failures**:
   - Verify library dependencies
   - Check environment variables
   - Ensure correct compiler versions

3. **Runtime Errors**:
   - Check device availability
   - Verify permissions (especially for RDMA)
   - Check memory requirements

### Debug Mode

Enable debug output:

```python
import pccl.plugins as plugins

# Enable plugin-specific debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual plugins
if not plugins.cuda_is_available():
    print("CUDA plugin not available - checking CUDA installation")
```

## Performance Optimization

### Plugin Selection Guidelines

- **CPU**: Best for small data sizes or when GPU not available
- **CUDA**: Optimal for NVIDIA GPUs with large data transfers
- **ROCm**: Best for AMD GPU workloads
- **RDMA**: Lowest latency for inter-node communication

### Memory Optimization

```python
# Use memory pools for frequent allocations
from pccl.plugins import create_cuda_memory_manager

manager = create_cuda_memory_manager()
buffers = [manager.allocate(1024*1024) for _ in range(10)]

# Register buffers for IPC/RDMA if needed
handles = [manager.register_buffer(buf) for buf in buffers]

# Use buffers...

# Cleanup
for buf in buffers:
    manager.free(buf)
```

## Examples and Documentation

- `example/plugins_demo.py` - Comprehensive plugin demonstration
- `example/rocm_allreduce_example.py` - ROCm-specific usage
- `README_DSL.md` - Python DSL documentation
- `README_ROCM.md` - ROCm plugin guide

## Contributing

To contribute to the plugin system:

1. Follow the existing plugin patterns
2. Add comprehensive tests
3. Update documentation
4. Ensure cross-platform compatibility
5. Include performance benchmarks

The plugin system is designed to be extensible while maintaining a consistent API across all platforms.