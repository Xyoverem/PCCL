# PCCL Python DSL - Declarative Communication Operators

PCCL (Programmable Communication & Computation Library) provides a high-level Python DSL for defining and executing communication operations with automatic optimization.

## Features

- **Declarative Syntax**: Define communication patterns using Python decorators and configuration classes
- **Automatic Optimization**: Compile-time optimization of communication algorithms
- **Topology-Aware**: Automatic topology discovery and algorithm selection
- **Multiple Algorithms**: Ring, Tree, Rabenseifner, and Hierarchical AllReduce
- **Compute-Communication Overlap**: Pipeline execution for better performance
- **Performance Profiling**: Built-in benchmarking and profiling tools

## Quick Start

### Basic Usage

```python
import pccl
from pccl.lang import allreduce, communication

# Simple AllReduce
allreduce_op = allreduce(
    reduce_op="sum",
    algorithm="ring",
    participants=[0, 1, 2, 3],
    buffer_size=64 * 1024 * 1024
)

# Execute with numpy array
import numpy as np
data = np.random.randn(1024, 1024).astype(np.float32)
result = allreduce_op.execute(data, participants=[0, 1, 2, 3])
```

### Declarative Communication Patterns

```python
from pccl.lang import communication, broadcast, allgather

@communication
class DistributedTraining:
    """Communication pattern for distributed training"""

    gradient_allreduce = allreduce(
        reduce_op="sum",
        algorithm="hierarchical",
        participants=[0, 1, 2, 3, 4, 5, 6, 7],
        enable_overlap=True
    )

    param_broadcast = broadcast(
        root_rank=0,
        participants=[0, 1, 2, 3, 4, 5, 6, 7]
    )

# Compile and use
comm_pattern = DistributedTraining()
plan = comm_pattern.compile(participants=list(range(8)))
result = comm_pattern.execute(gradient_data)
```

## Configuration Options

### AllReduce Algorithms

```python
from pccl.lang import ConfigBuilder, AllreduceAlgorithm, ReduceOp

# Ring AllReduce
ring_config = ConfigBuilder.ring_allreduce(
    reduce_op=ReduceOp.SUM,
    participants=[0, 1, 2, 3],
    enable_overlap=True
)

# Tree AllReduce
tree_config = ConfigBuilder.tree_allreduce(
    reduce_op=ReduceOp.SUM,
    participants=[0, 1, 2, 3, 4, 5, 6, 7],
    branching_factor=2
)

# Hierarchical AllReduce
hier_config = ConfigBuilder.hierarchical_allreduce(
    reduce_op=ReduceOp.SUM,
    participants=[0, 1, 2, 3, 4, 5, 6, 7],
    node_size=4,
    intra_interconnect=pccl.lang.InterconnectType.NVLINK
)
```

### Reduction Operations

- `SUM`: Sum reduction (default)
- `AVG`: Average reduction
- `MAX`: Maximum reduction
- `MIN`: Minimum reduction

### Algorithms

- `RING`: Ring-based AllReduce (default for small groups)
- `TREE`: Tree-based AllReduce (good for medium groups)
- `RABENSEIFNER`: Combined reduce-scatter + allgather (good for large data)
- `DOUBLE_BINARY_TREE`: Dual binary tree (low latency)

## Topology Management

### Automatic Discovery

```python
from pccl.lang import discover_topology

# Auto-discover system topology
topology = discover_topology(num_devices=4)
print(f"Topology type: {topology['type']}")
print(f"Total bandwidth: {topology['metrics'].total_bandwidth} GB/s")
```

### Manual Configuration

```python
from pccl.lang import ring_topology, hierarchical_topology

# Ring topology
ring_topo = ring_topology(
    devices=[0, 1, 2, 3],
    bandwidth=25.0,
    latency=0.5
)

# Hierarchical topology
hier_topo = hierarchical_topology(
    node_groups=[[0, 1], [2, 3]],
    intra_bandwidth=50.0,  # NVLink
    inter_bandwidth=10.0   # RDMA
)
```

## Advanced Features

### Pipeline Execution with Overlap

```python
from pccl.lang import PipelineAllreduce

pipeline_allreduce = PipelineAllreduce(
    reduce_op="sum",
    algorithm="ring",
    participants=[0, 1, 2, 3],
    compute_chunks=4,
    communication_chunks=2
)

plan = pipeline_allreduce.compile()
```

### Composite Operators

```python
from pccl.lang import CompositeOperator, Allreduce, Broadcast

allreduce_op = Allreduce(reduce_op="sum", participants=[0, 1, 2, 3])
broadcast_op = Broadcast(root_rank=0, participants=[0, 1, 2, 3])

composite = CompositeOperator(
    operators=[allreduce_op, broadcast_op],
    name="training_composite"
)

result = composite.execute(data, participants=[0, 1, 2, 3])
```

### Performance Profiling

```python
from pccl.lang import profile

@profile
def benchmark_allreduce():
    allreduce_op = allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=[0, 1, 2, 3]
    )

    data = np.random.randn(1024*1024).astype(np.float32)
    return allreduce_op.execute(data)

# Run benchmark
benchmark_allreduce()

# Get profiler results
profiler = pccl.lang.get_profiler()
summary = profiler.get_profile_summary()
print(f"Average time: {summary['avg_time_ms']:.3f} ms")
```

### Asynchronous Execution

```python
import time

allreduce_op = allreduce(
    reduce_op="sum",
    algorithm="ring",
    participants=[0, 1, 2, 3]
)

# Start async execution
data = np.random.randn(1024*1024).astype(np.float32)
handle = allreduce_op.async_execute(data)

# Do other work while communication happens
compute_work()

# Wait for completion
result = handle.wait()
```

## Examples

### Distributed Training

```python
import pccl
from pccl.lang import communication, allreduce, broadcast

@communication
class DistributedTraining:
    gradient_allreduce = allreduce(
        reduce_op="sum",
        algorithm="hierarchical",
        participants=range(8),
        enable_overlap=True
    )

# In training loop
comm_pattern = DistributedTraining()
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()

    # AllReduce gradients
    grads = [p.grad for p in model.parameters()]
    allreduced_grads = comm_pattern.gradient_allreduce.execute(grads)

    optimizer.step()
```

### Collective Communication Patterns

```python
# Multiple collective operations
from pccl.lang import Allreduce, Allgather, ReduceScatter

allreduce_op = Allreduce(reduce_op="sum", participants=[0, 1, 2, 3])
allgather_op = Allgather(participants=[0, 1, 2, 3])
reduce_scatter_op = ReduceScatter(reduce_op="sum", participants=[0, 1, 2, 3])

# Execute in sequence
data = np.random.randn(1024, 1024).astype(np.float32)
result1 = allreduce_op.execute(data)
result2 = allgather_op.execute(data)
result3 = reduce_scatter_op.execute(data)
```

## Installation

### Build from Source

```bash
# Clone the repository
git clone https://github.com/your-org/pccl.git
cd pccl

# Build C++ extension
python setup.py build_ext --inplace

# Install in development mode
pip install -e .
```

### Dependencies

- Python 3.8+
- PyTorch (optional, for tensor integration)
- NumPy
- C++20 compatible compiler
- CUDA toolkit (optional, for GPU support)

## Running Examples

```bash
# Basic DSL demonstration
python example/python_dsl_demo.py

# AllReduce algorithms comparison
python example/collective/allreduce_example.py

# Distributed training example
python example/distributed_training.py
```

## Performance Tuning

### Buffer Size Optimization

```python
# Larger buffers for big data transfers
allreduce_op = allreduce(
    buffer_size=256 * 1024 * 1024,  # 256MB
    participants=range(8)
)

# Smaller buffers for latency-sensitive operations
broadcast_op = broadcast(
    buffer_size=4 * 1024 * 1024,  # 4MB
    root_rank=0
)
```

### Algorithm Selection Guidelines

- **Ring**: Best for small groups (< 8 participants) and uniform topologies
- **Tree**: Good for medium groups (8-32 participants) with balanced topology
- **Rabenseifner**: Best for large data transfers and hierarchical topologies
- **Hierarchical**: Optimal for multi-node GPU clusters with NVLink/RDMA

### Overlap Optimization

```python
# Enable compute-communication overlap
allreduce_op = allreduce(
    enable_overlap=True,
    pipeline_depth=4,  # Higher depth for better overlap
    participants=range(8)
)
```

## API Reference

### Core Classes

- `Allreduce`: AllReduce collective operation
- `Broadcast`: Broadcast collective operation
- `Allgather`: AllGather collective operation
- `PipelineAllreduce`: AllReduce with compute overlap
- `CompositeOperator`: Multiple operations combined

### Configuration

- `ConfigBuilder`: Factory for operation configurations
- `AllreduceConfig`: AllReduce configuration
- `TopologyConfig`: Topology configuration

### Topology

- `TopologyBuilder`: Build network topologies
- `TopologyDiscovery`: Automatic topology detection
- `TopologyOptimizer`: Performance optimization

### Execution

- `compile()`: Compile configuration to execution plan
- `execute()`: Synchronous execution
- `async_execute()`: Asynchronous execution

## Contributing

PCCL welcomes contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

PCCL is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.