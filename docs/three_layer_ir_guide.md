# PCCL Three-Layer IR Architecture Guide

## Overview

PCCL (Programmable Communication & Computation Library) implements a revolutionary three-layer IR architecture that provides unprecedented control and optimization capabilities for high-performance communication operations.

### Architecture Vision

The three-layer IR architecture serves different user levels:

1. **Layer 1: Collective Primitives** - High-level collective operations for beginners
2. **Layer 2: Primitive IR** - Five basic operations (Write/Reduce/Copy/Signal/Wait) for developers
3. **Layer 3: Hardware Primitives** - Hardware-specific operations for engineers

### User Model

- **初学者 (Beginners)**: Use high-level collective APIs directly
- **开发者 (Developers)**: Access and control primitive IR operations
- **工程师 (Engineers)**: Fine-tune hardware-specific primitives

## Quick Start

### Installation

```bash
# Build PCCL
python setup.py build_ext --inplace

# Install in development mode
pip install -e .
```

### Basic Usage Examples

#### Level 1: High-Level Collective APIs

```python
import pccl

# Simple AllReduce - perfect for beginners
result = pccl.allreduce(tensor, op='sum')

# Other collective operations
broadcast_result = pccl.broadcast(data, root=0)
allgather_result = pccl.allgather(local_data)
```

#### Level 2: Primitive IR Control

```python
from pccl.passes import PassManager, PassContext
from pccl.ir.json_serializer import IRGraph, IRType

# Create collective IR
collective_graph = create_allreduce_graph()

# Lower to primitive IR with control
manager = PassManager()
context = PassContext(
    target_device="cuda",
    optimization_level="performance"
)

primitive_graph = manager.execute_pass("collective_to_primitive", collective_graph, context).ir

# Access and analyze primitive operations
for op_id, operation in primitive_graph.operations.items():
    print(f"Operation: {operation.op_type}")
    print(f"  Inputs: {operation.inputs}")
    print(f"  Outputs: {operation.outputs}")
    print(f"  Attributes: {operation.attributes}")
```

#### Level 3: Hardware-Specific Optimization

```python
# Fine-tune hardware primitives
from pccl.passes.primitive_to_hardware import PrimitiveToHardwareLowering

hardware_context = PassContext(target_device="cuda")
hardware_lowering = PrimitiveToHardwareLowering()

# Control hardware-specific lowering
hardware_graph = hardware_lowering.execute(primitive_graph, hardware_context).ir

# Optimize for specific hardware features
if "multimem_reduce" in available_cuda_features:
    hardware_graph = apply_multimem_optimization(hardware_graph)
```

## Lowering System

### Pass Architecture

The PCCL lowering system uses a flexible pass-based architecture:

```python
from pccl.passes import Pass, PassResult, PassContext

class CustomLoweringPass(Pass):
    def __init__(self):
        super().__init__(
            name="custom_lowering",
            pass_type=PassType.LAYER1_TO_LAYER2
        )

    def declare_capabilities(self):
        return PassCapabilities(
            input_ir_types=["collective"],
            output_ir_types=["primitive"],
            supported_devices=["cuda", "cpu", "rdma"]
        )

    def execute(self, ir, context):
        # Custom lowering logic
        return PassResult(success=True, ir=lowered_ir)
```

### Predefined Pipelines

```python
from pccl.passes.pipeline import StandardPipelines

# Complete lowering pipeline
pipeline = StandardPipelines.full_lowering_pipeline("cuda")
result = pipeline.execute(collective_graph, context)

# Optimization-only pipeline
opt_pipeline = StandardPipelines.optimization_pipeline("aggressive")
optimized_graph = opt_pipeline.execute(primitive_graph, context).ir
```

## IR Types and Operations

### Layer 1: Collective Primitives

#### Supported Operations
- **AllReduce**: Reduce and distribute data across all ranks
- **Broadcast**: Distribute data from root to all ranks
- **AllGather**: Gather data from all ranks to all ranks
- **Reduce-Scatter**: Reduce and scatter results
- **Reduce**: Reduce data to root rank
- **All-to-All**: Exchange data between all ranks

#### Algorithm Support
- **Ring**: Standard ring-based algorithms
- **Tree**: Hierarchical tree algorithms
- **Rabenseifner**: Optimized for hierarchical networks
- **Direct**: Point-to-point for small groups

### Layer 2: Primitive Operations

#### Five Basic Operations

1. **Write**: Store data to memory
   ```python
   write_op = WriteOp(
       id="write_0",
       input_value="data",
       output_value="memory_location",
       address=0x1000,
       size=1024
   )
   ```

2. **Reduce**: Combine multiple values
   ```python
   reduce_op = ReduceOp(
       id="reduce_0",
       input_values=["data1", "data2"],
       output_value="reduced_data",
       reduce_op=ReduceOpType.SUM
   )
   ```

3. **Copy**: Transfer data between locations
   ```python
   copy_op = CopyOp(
       id="copy_0",
       input_value="src_data",
       output_value="dst_data",
       cross_device=True
   )
   ```

4. **Signal**: Send synchronization signal
   ```python
   signal_op = SignalOp(
       id="signal_0",
       signal_id="sync_1",
       target_ranks=[1, 2, 3]
   )
   ```

5. **Wait Signal**: Wait for synchronization signal
   ```python
   wait_op = WaitSignalOp(
       id="wait_0",
       signal_id="sync_1",
       source_ranks=[0]
   )
   ```

### Layer 3: Hardware Primitives

#### CUDA Primitives
- **multimem.reduce**: CUDA multi-memory reduction
- **warp-level operations**: Warp-level synchronization
- **shared.memory.ops**: Shared memory operations

#### RDMA Primitives
- **verbs.rdma.write**: RDMA write operations
- **verbs.rdma.read**: RDMA read operations
- **verbs.send.recv**: Send/receive operations

## JSON Serialization

The IR system uses JSON as the interchange format between Python and C++ runtime:

### Serialize IR to JSON

```python
from pccl.ir.json_serializer import serialize_to_file

# Save primitive IR to JSON file
serialize_to_file(primitive_graph, "my_operation.json")
```

### JSON Structure

```json
{
  "ir_type": "primitive",
  "values": {
    "input_tensor": {
      "id": "input_tensor",
      "dtype": "float32",
      "shape": [1024],
      "device_id": 0,
      "device_type": "cuda",
      "metadata": {}
    }
  },
  "operations": {
    "reduce_0": {
      "id": "reduce_0",
      "op_type": "reduce",
      "inputs": ["input_tensor", "received_data"],
      "outputs": ["reduced_data"],
      "attributes": {
        "reduce_op": "sum",
        "device_id": 0,
        "device_type": "cuda"
      },
      "metadata": {}
    }
  },
  "metadata": {
    "graph_id": "my_graph",
    "lowering_info": {
      "pass": "collective_to_primitive",
      "target_device": "cuda"
    }
  }
}
```

## C++ Runtime Execution

### Runtime API

```cpp
#include "runtime/json_scheduler.h"

int main() {
    pccl::runtime::JSONScheduler scheduler;

    // Load and execute IR
    if (!scheduler.load_graph_from_file("my_operation.json")) {
        return 1;
    }

    // Execute synchronously
    bool success = scheduler.execute_graph_sync();

    // Get statistics
    auto stats = scheduler.get_execution_statistics();

    return success ? 0 : 1;
}
```

### Hardware Executor Support

```cpp
// Create hardware-specific executors
auto cuda_executor = pccl::runtime::HardwareExecutorFactory::create(
    pccl::DeviceType::CUDA, 0
);

auto rdma_executor = pccl::runtime::HardwareExecutorFactory::create(
    pccl::DeviceType::RDMA, 0
);

// Check availability
if (cuda_executor->is_available()) {
    // Use CUDA executor
}
```

## Advanced Features

### Custom Pass Development

```python
from pccl.passes import register_pass

@register_pass("my_custom_pass")
class MyCustomPass(Pass):
    def execute(self, ir, context):
        # Custom transformation logic
        transformed_ir = self.apply_custom_logic(ir)
        return PassResult(success=True, ir=transformed_ir)
```

### Performance Profiling

```python
# Enable detailed profiling
context = PassContext(enable_profiling=True)
result = manager.execute_pass("collective_to_primitive", ir, context)

# Get detailed statistics
stats = manager.get_execution_statistics()
print(f"Execution time: {result.execution_time:.4f}s")
print(f"Operations executed: {len(result.ir.operations)}")
```

### Error Handling and Diagnostics

```python
if not result.success:
    print("Lowering failed:")
    for diagnostic in result.diagnostics:
        print(f"  {diagnostic}")

    # Get detailed error information
    if hasattr(result, 'metadata'):
        error_info = result.metadata.get('error_details', {})
        print(f"Error details: {error_info}")
```

## Best Practices

### Performance Optimization

1. **Choose the right algorithm**: Select algorithms based on data size and topology
2. **Use hardware-aware lowering**: Leverage device-specific features
3. **Enable caching**: Cache lowering results for repeated operations
4. **Profile and analyze**: Use built-in profiling to identify bottlenecks

### Debugging Tips

1. **Validate IR at each layer**: Use IR validation tools
2. **Visualize lowering**: Use IR visualization to understand transformations
3. **Check JSON serialization**: Validate JSON output manually
4. **Test with small graphs**: Start with simple cases before complex operations

### Integration with ML Frameworks

```python
# PyTorch integration example
import torch
import pccl

# Convert PyTorch tensor to PCCL
pccl_tensor = pccl.from_torch(torch_tensor)

# Apply communication operation
result = pccl.allreduce(pccl_tensor, op='sum')

# Convert back to PyTorch
torch_result = pccl.to_torch(result)
```

## Examples and Tutorials

See the `example/` directory for comprehensive examples:

- `example/collective_lowering_demo.py` - Basic lowering demonstration
- `example/three_layer_ir_demo.py` - Complete three-layer usage
- `example/performance_benchmark.py` - Performance comparison
- `example/custom_pass_development.py` - Custom pass tutorial

## API Reference

### Core Classes

- **IRGraph**: IR graph container
- **IRValue**: Data value in IR
- **IROperation**: Operation in IR
- **Pass**: Base transformation pass
- **PassManager**: Pass execution manager
- **JSONScheduler**: C++ runtime scheduler

### Pass Types

- **PassType.LAYER1_TO_LAYER2**: Collective to primitive
- **PassType.LAYER2_TO_LAYER3**: Primitive to hardware
- **PassType.OPTIMIZATION**: IR optimization passes

### Device Types

- **DeviceType.CPU**: CPU execution
- **DeviceType.CUDA**: NVIDIA GPU execution
- **DeviceType.RDMA**: RDMA network execution
- **DeviceType.ROCM**: AMD GPU execution

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure PCCL is properly installed
2. **Build failures**: Check CUDA toolkit and dependencies
3. **Runtime errors**: Validate IR graphs before execution
4. **Performance issues**: Use profiling to identify bottlenecks

### Getting Help

- Check the documentation at `docs/`
- Run the test suite with `python -m pytest test/`
- Examine example code in `example/`
- Review lowering results with visualization tools

## Future Development

The three-layer IR architecture is designed for extensibility:

- **New collective operations**: Easily add new collective algorithms
- **Hardware support**: Extend to new hardware platforms
- **Optimization passes**: Develop custom optimization strategies
- **Integration frameworks**: Add support for other ML frameworks

This architecture provides a solid foundation for the future of high-performance communication libraries.