"""
Advanced Chunk Communication Patterns
Demonstrates sophisticated chunk-level routing and optimization
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import pccl
from pccl.lang import communication, send, recv, allreduce
from pccl.lang.chunk_communication import ChunkCommunication, ChunkManager
from pccl.lang.operator import Device

@communication
class CollectiveOptimizedChunkPatterns:
    

    def optimized_allreduce(self, tensor, participants):
        
        chunk_configs = [
            (32, "tree"),      # Small chunks: tree reduction
            (128, "ring"),     # Medium chunks: ring allreduce
            (512, "hierarchical"),  # Large chunks: hierarchical
            (2048, "direct")   # Very large: direct reduction
        ]

        results = []
        processed_elements = 0

        for chunk_size, strategy in chunk_configs:
            if processed_elements >= tensor.numel():
                break

            remaining = tensor.numel() - processed_elements
            current_chunk_size = min(chunk_size, remaining)

            chunk_data = tensor.flatten()[processed_elements:processed_elements + current_chunk_size]

            if strategy == "tree":
                result = self._tree_reduce_chunk(chunk_data, participants)
            elif strategy == "ring":
                ring_op = allreduce(
                    algorithm="ring",
                    participants=participants,
                    reduce_op="sum"
                )
                result = ring_op.execute(chunk_data)
            elif strategy == "hierarchical":
                result = self._hierarchical_reduce_chunk(chunk_data, participants)
            elif strategy == "direct":
                result = self._direct_reduce_chunk(chunk_data, participants)

            results.append(result)
            processed_elements += current_chunk_size

        return torch.cat(results).reshape(tensor.shape)

    def _tree_reduce_chunk(self, chunk, participants):
        
        fanout = 2
        level = 0

        current_participants = participants.copy()
        reduced_chunk = chunk.clone()

        while len(current_participants) > 1:
            next_level = []

            for i in range(0, len(current_participants), fanout):
                group = current_participants[i:i+fanout]

                if self.rank == group[0]:
                    for child_rank in group[1:]:
                        recv_op = recv(
                            source=child_rank,
                            tag=f"tree_{level}_{child_rank}",
                            shape=chunk.shape,
                            dtype=chunk.dtype
                        )
                        child_chunk = recv_op.execute()
                        reduced_chunk += child_chunk
                    next_level.append(group[0])
                elif self.rank in group[1:]:
                    parent_rank = group[0]
                    send_op = send(
                        destination=parent_rank,
                        tag=f"tree_{level}_{self.rank}",
                        data=reduced_chunk
                    )
                    send_op.execute(reduced_chunk)
                    break

            current_participants = next_level
            level += 1
        if self.rank == participants[0]:
            for rank in participants[1:]:
                send_op = send(
                    destination=rank,
                    tag=f"tree_bcast_{level}",
                    data=reduced_chunk
                )
                send_op.execute(reduced_chunk)
        else:
            recv_op = recv(
                source=participants[0],
                tag=f"tree_bcast_{level}",
                shape=reduced_chunk.shape,
                dtype=reduced_chunk.dtype
            )
            reduced_chunk = recv_op.execute()

        return reduced_chunk

    def _hierarchical_reduce_chunk(self, chunk, participants):
        
        ranks_per_node = 2
        my_node = self.rank // ranks_per_node
        local_rank = self.rank % ranks_per_node

        local_participants = [my_node * ranks_per_node + i for i in range(ranks_per_node)
                             if my_node * ranks_per_node + i in participants]

        if len(local_participants) > 1:
            chunk = self._tree_reduce_chunk(chunk, local_participants)

        local_leaders = [p for p in participants if p % ranks_per_node == 0]
        if self.rank in local_leaders and len(local_leaders) > 1:
            chunk = self._tree_reduce_chunk(chunk, local_leaders)
        if self.rank in local_leaders:
            for rank in local_participants:
                if rank != self.rank:
                    send_op = send(
                        destination=rank,
                        tag=f"hier_bcast_{my_node}",
                        data=chunk
                    )
                    send_op.execute(chunk)
        else:
            recv_op = recv(
                source=local_leaders[0],
                tag=f"hier_bcast_{my_node}",
                shape=chunk.shape,
                dtype=chunk.dtype
            )
            chunk = recv_op.execute()

        return chunk

    def _direct_reduce_chunk(self, chunk, participants):
        
        if self.rank == participants[0]:
            reduced_chunk = chunk.clone()
            for rank in participants[1:]:
                recv_op = recv(
                    source=rank,
                    tag=f"direct_reduce_{rank}",
                    shape=chunk.shape,
                    dtype=chunk.dtype
                )
                received_chunk = recv_op.execute()
                reduced_chunk += received_chunk

            for rank in participants[1:]:
                send_op = send(
                    destination=rank,
                    tag=f"direct_result",
                    data=reduced_chunk
                )
                send_op.execute(reduced_chunk)

            return reduced_chunk
        else:
            send_op = send(
                destination=participants[0],
                tag=f"direct_reduce_{self.rank}",
                data=chunk
            )
            send_op.execute(chunk)

            recv_op = recv(
                source=participants[0],
                tag=f"direct_result",
                shape=chunk.shape,
                dtype=chunk.dtype
            )
            return recv_op.execute()

@communication
class ChunkRoutingStrategies:
    

    def topology_aware_routing(self, tensor, participants, topology_info=None):
        
        chunk_size = 256
        chunks = self._split_tensor(tensor, chunk_size)

        if topology_info is None:
            topology_info = {
                'bandwidth_matrix': self._generate_bandwidth_matrix(len(participants)),
                'latency_matrix': self._generate_latency_matrix(len(participants)),
                'node_mapping': {i: i // 2 for i in participants}
            }

        results = []

        for chunk_idx, chunk in enumerate(chunks):
            route = self._select_optimal_route(
                self.rank, participants, chunk_idx, topology_info
            )

            result = self._execute_routed_communication(chunk, route, topology_info)
            results.append(result)

        return torch.cat(results).reshape(tensor.shape)

    def _select_optimal_route(self, source_rank, participants, chunk_idx, topology_info):
        
        chunk_size = 128 + chunk_idx * 64  # Varying chunk sizes

        if chunk_size < 1024:  # Small chunks: minimize latency
            return self._minimize_latency_route(source_rank, participants, topology_info)
        else:  # Large chunks: maximize bandwidth
            return self._maximize_bandwidth_route(source_rank, participants, topology_info)

    def _minimize_latency_route(self, source_rank, participants, topology_info):
        
        latencies = topology_info['latency_matrix']
        node_mapping = topology_info['node_mapping']

        my_node = node_mapping[source_rank]

        for rank in participants:
            if rank != source_rank and node_mapping[rank] == my_node:
                return [source_rank, rank]
        min_latency = float('inf')
        best_route = [source_rank]

        for rank in participants:
            if rank != source_rank:
                if latencies[source_rank][rank] < min_latency:
                    min_latency = latencies[source_rank][rank]
                    best_route = [source_rank, rank]

        return best_route

    def _maximize_bandwidth_route(self, source_rank, participants, topology_info):
        
        bandwidths = topology_info['bandwidth_matrix']

        max_bandwidth = 0
        best_route = [source_rank]

        for rank in participants:
            if rank != source_rank:
                if bandwidths[source_rank][rank] > max_bandwidth:
                    max_bandwidth = bandwidths[source_rank][rank]
                    best_route = [source_rank, rank]

        return best_route

    def _execute_routed_communication(self, chunk, route, topology_info):
        
        current_chunk = chunk.clone()

        for i in range(len(route) - 1):
            current_rank = route[i]
            next_rank = route[i + 1]

            if self.rank == current_rank:
                send_op = send(
                    destination=next_rank,
                    tag=f"route_{id(chunk)}_{i}",
                    data=current_chunk
                )
                send_op.execute(current_chunk)
                break
            elif self.rank == next_rank:
                recv_op = recv(
                    source=current_rank,
                    tag=f"route_{id(chunk)}_{i}",
                    shape=current_chunk.shape,
                    dtype=current_chunk.dtype
                )
                current_chunk = recv_op.execute()

        return current_chunk

    def _generate_bandwidth_matrix(self, size):
        
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i == j:
                    matrix[i][j] = 100.0
                elif i // 2 == j // 2:  # Same node
                    matrix[i][j] = 50.0
                else:  # Different nodes
                    matrix[i][j] = 10.0
        return matrix

    def _generate_latency_matrix(self, size):
        
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i == j:
                    matrix[i][j] = 0.0
                elif i // 2 == j // 2:  # Same node
                    matrix[i][j] = 5.0
                else:  # Different nodes
                    matrix[i][j] = 50.0
        return matrix

@communication
class ChunkStreamingPatterns:
    

    def streaming_allreduce(self, tensor_stream, participants, buffer_size=4):
        
        results = []
        buffer = []

        for tensor in tensor_stream:
            buffer.append(tensor)

            if len(buffer) == buffer_size:
                buffered_tensor = torch.cat(buffer)
                reduced = self._process_buffer(buffered_tensor, participants)
                results.extend(torch.split(reduced, [t.numel() for t in buffer]))
                buffer.clear()
        if buffer:
            buffered_tensor = torch.cat(buffer)
            reduced = self._process_buffer(buffered_tensor, participants)
            results.extend(torch.split(reduced, [t.numel() for t in buffer]))

        return results

    def _process_buffer(self, buffered_tensor, participants):
        
        size = buffered_tensor.numel()

        if size < 1024:
            algorithm = "tree"
        elif size < 4096:
            algorithm = "ring"
        else:
            algorithm = "hierarchical"

        allreduce_op = allreduce(
            algorithm=algorithm,
            participants=participants,
            reduce_op="sum"
        )
        return allreduce_op.execute(buffered_tensor)

@communication
class AdaptiveChunkSizing:
    

    def adaptive_chunk_allreduce(self, tensor, participants):
        
        network_conditions = self._monitor_network_conditions()

        chunk_sizes = self._compute_optimal_chunk_sizes(tensor, network_conditions)

        results = []
        offset = 0

        for chunk_size in chunk_sizes:
            if offset >= tensor.numel():
                break

            actual_chunk_size = min(chunk_size, tensor.numel() - offset)
            chunk = tensor.flatten()[offset:offset + actual_chunk_size]

            algorithm = self._select_algorithm(chunk_size, network_conditions)

            allreduce_op = allreduce(
                algorithm=algorithm,
                participants=participants,
                reduce_op="sum"
            )

            reduced_chunk = allreduce_op.execute(chunk)
            results.append(reduced_chunk)
            offset += actual_chunk_size

        return torch.cat(results).reshape(tensor.shape)

    def _monitor_network_conditions(self):
        
        return {
            'bandwidth': 10.0,  # GB/s
            'latency': 50.0,    # microseconds
            'congestion': 0.3,  # 0-1 scale
            'available_bandwidth': 7.0  # GB/s (considering congestion)
        }

    def _compute_optimal_chunk_sizes(self, tensor, conditions):
        
        bandwidth = conditions['available_bandwidth']
        latency = conditions['latency']

        if bandwidth > 50:
            base_size = 2048
        elif bandwidth > 10:
            base_size = 1024
        elif bandwidth > 1:
            base_size = 512
        else:
            base_size = 256

        if latency > 100:
            base_size = min(base_size, 128)
        elif latency > 50:
            base_size = min(base_size, 256)

        congestion_factor = 1.0 - conditions['congestion'] * 0.5
        base_size = int(base_size * congestion_factor)

        chunk_sizes = []
        num_chunks = max(1, tensor.numel() // base_size)

        for i in range(num_chunks):
            variation = 0.75 + (i % 5) * 0.125
            chunk_sizes.append(int(base_size * variation))

        return chunk_sizes

    def _select_algorithm(self, chunk_size, conditions):
        
        congestion = conditions['congestion']
        bandwidth = conditions['bandwidth']

        if chunk_size < 256:
            return "tree"  # Small chunks: tree reduction
        elif congestion > 0.5 or bandwidth < 1:
            return "hierarchical"  # Congested or low bandwidth: hierarchical
        else:
            return "ring"  # Normal conditions: ring allreduce

def demonstrate_advanced_patterns():
    
    print("=== Advanced Chunk Communication Patterns ===\n")

    devices = [Device(rank=i, device_type="cuda") for i in range(4)]

    print("1. Collective-Optimized Chunk Patterns")
    print("-" * 40)
    optimized = CollectiveOptimizedChunkPatterns()
    tensor = torch.randn(2048, device="cuda")

    try:
        result = optimized.optimized_allreduce(tensor, [0, 1, 2, 3])
        print(f"✓ Optimized AllReduce completed")
        print(f"  Used multiple strategies: tree, ring, hierarchical, direct")
        print(f"  Input shape: {tensor.shape}, Output shape: {result.shape}")
    except Exception as e:
        print(f"✗ Optimized AllReduce failed: {e}")

    print("\n2. Topology-Aware Chunk Routing")
    print("-" * 35)
    routing = ChunkRoutingStrategies()
    tensor = torch.randn(1024, device="cuda")

    try:
        result = routing.topology_aware_routing(tensor, [0, 1, 2, 3])
        print(f"✓ Topology-aware routing completed")
        print(f"  Selected routes based on latency/bandwidth optimization")
    except Exception as e:
        print(f"✗ Topology-aware routing failed: {e}")

    print("\n3. Streaming Chunk Patterns")
    print("-" * 30)
    streaming = ChunkStreamingPatterns()
    tensor_stream = [torch.randn(128, device="cuda") for _ in range(8)]

    try:
        results = streaming.streaming_allreduce(tensor_stream, [0, 1, 2, 3])
        print(f"✓ Streaming AllReduce completed")
        print(f"  Processed {len(tensor_stream)} streaming chunks")
        print(f"  Buffer size: 4 chunks")
    except Exception as e:
        print(f"✗ Streaming AllReduce failed: {e}")

    print("\n4. Adaptive Chunk Sizing")
    print("-" * 25)
    adaptive = AdaptiveChunkSizing()
    tensor = torch.randn(1536, device="cuda")

    try:
        result = adaptive.adaptive_chunk_allreduce(tensor, [0, 1, 2, 3])
        print(f"✓ Adaptive chunk AllReduce completed")
        print(f"  Network-aware chunk sizing based on conditions")
        print(f"  Dynamic algorithm selection")
    except Exception as e:
        print(f"✗ Adaptive chunk AllReduce failed: {e}")

def demonstrate_chunk_manager_usage():
    
    print("\n\n=== ChunkManager Advanced Usage ===\n")

    devices = [Device(rank=i, device_type="cuda") for i in range(4)]
    manager = ChunkManager(devices)

    print("1. Multiple Communication Groups")
    print("-" * 35)

    collective_group = manager.create_group("collective")
    point_to_point_group = manager.create_group("p2p")
    broadcast_group = manager.create_group("broadcast")

    tensor = torch.randn(1024, device="cuda")

    chunks = collective_group.create_chunks(tensor, 256, ["c0", "c1", "c2", "c3"])
    collective_group.reduce_chunks(["c0", "c1", "c2", "c3"], [0, 1, 2, 3])

    p2p_chunks = point_to_point_group.create_chunks(tensor, 512, ["p0", "p1"])
    point_to_point_group.send_chunk("p0", 0, 1, tag=100)
    point_to_point_group.recv_chunk("p0", 0, 1, (512,), torch.float32, tag=100)

    bcast_chunks = broadcast_group.create_chunks(tensor, 1024, ["b0"])
    broadcast_group.broadcast_chunk("b0", 0, [0, 1, 2, 3])
    try:
        results = manager.execute_all_groups()
        print(f"✓ Multiple groups executed successfully")
        for group_name, group_results in results.items():
            print(f"  {group_name}: {len(group_results)} chunks processed")
    except Exception as e:
        print(f"✗ Multiple group execution failed: {e}")

    print("\n2. Performance Metrics Analysis")
    print("-" * 35)

    for group_name in ["collective", "p2p", "broadcast"]:
        group = manager.get_group(group_name)
        if group:
            metrics = group.get_performance_metrics()
            print(f"\n{group_name.upper()} Group Performance:")
            for op_name, time_taken in metrics.items():
                print(f"  {op_name}: {time_taken:.4f}s")

    print("\n3. Chunk Pipeline with Dependencies")
    print("-" * 40)

    pipeline_group = manager.create_group("pipeline")
    tensor = torch.randn(2048, device="cuda")

    chunks = pipeline_group.create_chunks(tensor, 256)

    for i, chunk_id in enumerate(pipeline_group.list_chunks()):
        dest_rank = (i + 1) % 4
        pipeline_group.send_chunk(chunk_id, 0, dest_rank, tag=i*10)

        if i > 0:
            prev_chunk_id = pipeline_group.list_chunks()[i-1]
            pipeline_group.reduce_chunks([prev_chunk_id, chunk_id], [0, dest_rank])
    manager.clear_all_groups()
    print("\n✓ All groups cleared")

if __name__ == "__main__":
    demonstrate_advanced_patterns()
    demonstrate_chunk_manager_usage()

    print("\n\n=== Key Advanced Features Demonstrated ===")
    print("✓ Multi-strategy AllReduce with chunk-aware optimization")
    print("✓ Topology-aware routing based on network conditions")
    print("✓ Streaming patterns for continuous data flow")
    print("✓ Adaptive chunk sizing with network monitoring")
    print("✓ ChunkManager for complex multi-group scenarios")
    print("✓ Performance metrics collection and analysis")
    print("✓ Chunk pipelines with operation dependencies")
    print("\nAdvanced chunk patterns enable sophisticated communication optimization!")