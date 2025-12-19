# PCCL Chunk-Level Communication Control

PCCL provides fine-grained control over data communication at the chunk level, allowing you to precisely control how individual portions of your tensors are transmitted, reduced, and routed across devices.

## Overview

### What are Chunks?

Chunks are smaller, independent portions of a larger tensor. By splitting communication operations at the chunk level, you can:

- **Route different chunks to different destinations**
- **Apply different algorithms to different chunk sizes**
- **Overlap communication with computation**
- **Optimize based on network topology**
- **Implement custom communication patterns**

### Architecture

```
Tensor
├── Chunk 0 ──► Route A ──► Device 1
├── Chunk 1 ──► Route B ──► Device 2
├── Chunk 2 ──► Route C ──► Device 3
└── Chunk 3 ──► Route D ──► Device 0
```

## Basic Chunk Control

### 1. Manual Chunk Creation and Routing

```python
import torch
from pccl.lang.chunk_communication import ChunkCommunication
from pccl.lang.operator import Device

# Create devices
devices = [Device(rank=i, device_type="cuda") for i in range(4)]

# Initialize chunk communication
chunk_comm = ChunkCommunication(devices)

# Create tensor and split into chunks
tensor = torch.randn(1024, device="cuda")
chunk_size = 256
chunks = chunk_comm.create_chunks(tensor, chunk_size)

# Route different chunks to different destinations
chunk_comm.send_chunk("chunk_0", 0, 1, tag=100)  # chunk_0: rank 0 → rank 1
chunk_comm.send_chunk("chunk_1", 0, 2, tag=101)  # chunk_1: rank 0 → rank 2
chunk_comm.send_chunk("chunk_2", 0, 3, tag=102)  # chunk_2: rank 0 → rank 3
chunk_comm.send_chunk("chunk_3", 0, 1, tag=103)  # chunk_3: rank 0 → rank 1

# Execute all operations
results = chunk_comm.execute()
```

### 2. Selective Chunk Transfer

```python
from pccl.lang import communication

@communication
class SelectiveTransfer:
    def selective_communication(self, tensor, chunk_mapping):
        """
        Transfer specific chunks to different destinations
        chunk_mapping: {chunk_index: destination_rank}
        """
        chunk_size = 64
        chunks = self._split_tensor(tensor, chunk_size)

        for chunk_idx, chunk in enumerate(chunks):
            if chunk_idx in chunk_mapping:
                dest_rank = chunk_mapping[chunk_idx]

                if self.rank == 0:  # Source rank
                    from pccl.lang import send
                    send_op = send(
                        destination=dest_rank,
                        tag=chunk_idx,
                        data=chunk
                    )
                    result = send_op.execute(chunk)
                    print(f"Sent chunk {chunk_idx} to rank {dest_rank}")

                elif self.rank == dest_rank:  # Destination rank
                    from pccl.lang import recv
                    recv_op = recv(
                        source=0,
                        tag=chunk_idx,
                        shape=chunk.shape,
                        dtype=chunk.dtype
                    )
                    result = recv_op.execute()
                    print(f"Received chunk {chunk_idx} from rank 0")

# Usage
pattern = SelectiveTransfer()
tensor = torch.randn(256, device="cuda")
chunk_mapping = {0: 1, 1: 2, 2: 3}  # chunk_0→rank1, chunk_1→rank2, chunk_2→rank3
result = pattern.selective_communication(tensor, chunk_mapping)
```

## Advanced Chunk Patterns

### 1. Multi-Strategy AllReduce

Different chunk sizes benefit from different algorithms:

```python
@communication
class AdaptiveAllReduce:
    def adaptive_allreduce(self, tensor, participants):
        """Apply different algorithms based on chunk size"""
        chunk_configs = [
            (32, "tree"),        # Small chunks: tree reduction
            (128, "ring"),       # Medium chunks: ring allreduce
            (512, "hierarchical"), # Large chunks: hierarchical
            (2048, "direct")     # Very large: direct reduction
        ]

        results = []

        for chunk_size, algorithm in chunk_configs:
            chunks = self._split_tensor(tensor, chunk_size)

            for chunk in chunks:
                from pccl.lang import allreduce
                op = allreduce(
                    algorithm=algorithm,
                    participants=participants,
                    reduce_op="sum"
                )
                reduced_chunk = op.execute(chunk)
                results.append(reduced_chunk)

        return torch.cat(results).reshape(tensor.shape)
```

### 2. Topology-Aware Routing

Route chunks based on network topology:

```python
@communication
class TopologyAwareRouting:
    def topology_aware_communication(self, tensor, participants):
        """Route chunks based on network topology"""
        chunk_size = 256
        chunks = self._split_tensor(tensor, chunk_size)

        # Simulate topology information
        topology = {
            'bandwidth_matrix': self._get_bandwidth_matrix(len(participants)),
            'node_mapping': {i: i // 2 for i in participants}  # 2 ranks per node
        }

        results = []

        for chunk_idx, chunk in enumerate(chunks):
            # Select optimal route based on topology
            route = self._select_optimal_route(self.rank, participants, topology)

            # Execute routed communication
            result = self._execute_route(chunk, route)
            results.append(result)

        return torch.cat(results).reshape(tensor.shape)

    def _select_optimal_route(self, source_rank, participants, topology):
        """Choose route that maximizes bandwidth or minimizes latency"""
        # Prefer same-node communication for better performance
        my_node = topology['node_mapping'][source_rank]

        for rank in participants:
            if rank != source_rank and topology['node_mapping'][rank] == my_node:
                return [source_rank, rank]  # Same-node route

        # Fall back to highest bandwidth connection
        bandwidths = topology['bandwidth_matrix']
        best_rank = max(participants, key=lambda r: bandwidths[source_rank][r])
        return [source_rank, best_rank]
```

### 3. Pipeline Communication

Overlap communication with computation:

```python
@communication
class PipelineCommunication:
    def pipeline_allreduce(self, tensor, participants, num_stages=4):
        """Pipelined AllReduce with compute-communication overlap"""
        chunk_size = tensor.numel() // num_stages
        chunks = self._split_tensor(tensor, chunk_size)

        pipeline_results = [None] * len(chunks)

        for stage in range(num_stages):
            # Process chunks in different pipeline stages
            for chunk_idx in range(stage, len(chunks), num_stages):
                chunk = chunks[chunk_idx]

                # Start communication
                if self.rank == 0:
                    from pccl.lang import send
                    send_future = send(
                        destination=participants[1],
                        tag=f"pipe_{chunk_idx}_{stage}",
                        data=chunk
                    ).execute_async()
                else:
                    from pccl.lang import recv
                    recv_future = recv(
                        source=0,
                        tag=f"pipe_{chunk_idx}_{stage}",
                        shape=chunk.shape,
                        dtype=chunk.dtype
                    ).execute_async()

                # Overlap with computation on next chunk
                if chunk_idx + num_stages < len(chunks):
                    next_chunk = chunks[chunk_idx + num_stages]
                    computed_chunk = self._compute_on_chunk(next_chunk)

                # Wait for communication to complete
                if self.rank == 0:
                    pipeline_results[chunk_idx] = send_future.wait()
                else:
                    pipeline_results[chunk_idx] = recv_future.wait()

        return torch.cat(pipeline_results).reshape(tensor.shape)
```

### 4. Hierarchical Chunk Processing

Two-level hierarchy for large-scale systems:

```python
@communication
class HierarchicalChunkProcessing:
    def hierarchical_allreduce(self, tensor, participants):
        """Two-level hierarchy: intra-node + inter-node"""
        chunk_size = 512
        chunks = self._split_tensor(tensor, chunk_size)

        # Assume 2 ranks per node
        ranks_per_node = 2
        my_node = self.rank // ranks_per_node
        local_rank = self.rank % ranks_per_node

        results = []

        for chunk in chunks:
            # Phase 1: Intra-node reduction
            local_participants = [
                my_node * ranks_per_node + i
                for i in range(ranks_per_node)
                if my_node * ranks_per_node + i in participants
            ]

            if len(local_participants) > 1:
                chunk = self._intra_node_reduce(chunk, local_participants)

            # Phase 2: Inter-node reduction (node leaders only)
            if local_rank == 0:  # Node leaders
                node_leaders = [p for p in participants if p % ranks_per_node == 0]
                if len(node_leaders) > 1:
                    chunk = self._inter_node_reduce(chunk, node_leaders)

            # Phase 3: Broadcast within nodes
            if local_rank == 0:
                for rank in local_participants:
                    if rank != self.rank:
                        self._send_chunk(chunk, rank)
            else:
                chunk = self._recv_chunk(local_participants[0])

            results.append(chunk)

        return torch.cat(results).reshape(tensor.shape)
```

## ChunkManager Usage

For complex scenarios with multiple communication groups:

```python
from pccl.lang.chunk_communication import ChunkManager

# Create chunk manager
devices = [Device(rank=i, device_type="cuda") for i in range(4)]
manager = ChunkManager(devices)

# Create multiple communication groups
collective_group = manager.create_group("collective")
p2p_group = manager.create_group("p2p")
streaming_group = manager.create_group("streaming")

# Set up different operations in each group
tensor = torch.randn(1024, device="cuda")

# Collective operations
chunks = collective_group.create_chunks(tensor, 256)
collective_group.reduce_chunks(collective_group.list_chunks(), [0, 1, 2, 3])

# Point-to-point operations
p2p_chunks = p2p_group.create_chunks(tensor, 512)
p2p_group.send_chunk("chunk_0", 0, 1)
p2p_group.recv_chunk("chunk_1", 2, 3, (512,), torch.float32)

# Execute all groups
results = manager.execute_all_groups()

# Get performance metrics
for group_name, group in manager.chunk_groups.items():
    metrics = group.get_performance_metrics()
    print(f"{group_name} metrics: {metrics}")
```

## Direct IR-Level Control

For maximum flexibility, use the IR layer directly:

```python
from pccl.lang import IRBuilder

# Create IR builder
builder = IRBuilder()
devices = [Device(rank=i, device_type="cuda") for i in range(4)]

# Create tensor and chunks
data = torch.randn(512, device="cuda")
chunk_size = 128

chunks = []
for i in range(0, data.numel(), chunk_size):
    chunk_data = data.flatten()[i:i+chunk_size]
    chunk = builder.create_value(chunk_data, devices[0])
    chunks.append(chunk)

# Define custom chunk routing
operations = []
for i, chunk in enumerate(chunks):
    # Route chunks in a custom pattern
    dest_rank = (i * 2) % len(devices)

    send_op = builder.create_send(
        source=chunk,
        source_device=devices[0],
        destination_device=devices[dest_rank],
        tag=i
    )
    operations.append(send_op)

# Create and execute graph
graph = builder.create_graph(operations)
executor = pccl.GraphExecutor(devices)
result = executor.execute(graph)
```

## Performance Optimization Tips

### 1. Chunk Size Selection

```python
def optimize_chunk_size(tensor_size, network_bandwidth, network_latency):
    """Calculate optimal chunk size based on network conditions"""

    # Small chunks: minimize latency (latency-bound)
    if network_latency > 100:  # High latency
        return min(128, tensor_size // 8)

    # Large chunks: maximize bandwidth (bandwidth-bound)
    elif network_bandwidth > 50:  # High bandwidth
        return min(2048, tensor_size // 4)

    # Balanced approach
    else:
        return min(512, tensor_size // 6)
```

### 2. Algorithm Selection by Chunk Size

| Chunk Size | Recommended Algorithm | Reason |
|------------|----------------------|--------|
| < 1KB      | Tree Reduction       | Minimizes latency, good for small data |
| 1KB - 8KB  | Ring AllReduce       | Balanced latency/bandwidth |
| 8KB - 64KB | Hierarchical         | Good for multi-node systems |
| > 64KB     | Direct Reduction     | Maximizes bandwidth utilization |

### 3. Memory Considerations

```python
def memory_aware_chunking(tensor, available_memory_mb):
    """Chunk size based on available memory"""
    tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)

    # Reserve 20% of memory for overhead
    usable_memory = available_memory_mb * 0.8

    # Calculate chunks that fit in memory
    max_chunk_size_mb = usable_memory // 4  # Allow 4 concurrent chunks
    elements_per_chunk = int(max_chunk_size_mb * 1024 * 1024 / tensor.element_size())

    return min(elements_per_chunk, tensor.numel())
```

## Examples

### 1. Custom AllReduce Implementation

```python
# See example/chunk_level_communication.py
from example.chunk_level_communication import ChunkBasedAllReduce

pattern = ChunkBasedAllReduce()
tensor = torch.randn(1024, device="cuda")
result = pattern.custom_ring_allreduce(tensor, [0, 1, 2, 3])
```

### 2. Advanced Patterns

```python
# See example/advanced_chunk_patterns.py
from example.advanced_chunk_patterns import (
    CollectiveOptimizedChunkPatterns,
    TopologyAwareRouting,
    AdaptiveChunkSizing
)

# Multi-strategy AllReduce
optimized = CollectiveOptimizedChunkPatterns()
result = optimized.optimized_allreduce(tensor, participants)

# Topology-aware routing
routing = TopologyAwareRouting()
result = routing.topology_aware_routing(tensor, participants)

# Adaptive chunk sizing
adaptive = AdaptiveChunkSizing()
result = adaptive.adaptive_chunk_allreduce(tensor, participants)
```

## Integration with Existing Code

Chunk-level control integrates seamlessly with PCCL's high-level DSL:

```python
import pccl
from pccl.lang import communication, allreduce

@communication
class HybridPattern:
    # Use high-level AllReduce for some operations
    normal_allreduce = allreduce(
        algorithm="ring",
        participants=[0, 1, 2, 3],
        reduce_op="sum"
    )

    def custom_chunk_operation(self, tensor):
        # Use chunk-level control for custom patterns
        return self._custom_chunk_routing(tensor)
```

## Error Handling and Validation

```python
def safe_chunk_communication(chunk_comm, tensor, chunk_size):
    """Safe chunk communication with validation"""
    try:
        # Validate inputs
        if tensor.numel() == 0:
            raise ValueError("Empty tensor")
        if chunk_size <= 0:
            raise ValueError("Invalid chunk size")

        # Create chunks
        chunks = chunk_comm.create_chunks(tensor, chunk_size)

        # Validate chunk creation
        total_elements = sum(chunk.size for chunk in chunks)
        if total_elements != tensor.numel():
            raise ValueError("Chunk size mismatch")

        # Execute operations
        results = chunk_comm.execute()

        # Validate results
        if len(results) != len(chunks):
            raise ValueError("Result count mismatch")

        return results

    except Exception as e:
        print(f"Chunk communication failed: {e}")
        chunk_comm.clear_operations()
        return None
```

## Best Practices

1. **Chunk Size Selection**: Choose chunk sizes based on network conditions and data characteristics
2. **Algorithm Matching**: Use different algorithms for different chunk sizes
3. **Memory Management**: Be aware of memory constraints when creating many chunks
4. **Error Handling**: Always validate chunk operations and handle failures gracefully
5. **Performance Monitoring**: Use ChunkManager metrics to optimize communication patterns
6. **Topology Awareness**: Consider network topology when routing chunks

Chunk-level control in PCCL provides maximum flexibility for implementing custom communication patterns while maintaining the simplicity of the high-level DSL for standard operations.