"""
PCCL Chunk-Level Communication Examples
Demonstrates fine-grained control over individual chunks of data
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import pccl
from pccl.lang import communication, send, recv, allreduce, broadcast, IRBuilder
from pccl.lang.chunk import Chunk
from pccl.lang.operator import Device

class ChunkController:
    def __init__(self, devices: List[Device]):
        self.devices = devices
        self.chunks = {}
        self.operations = []

    def create_chunks(self, tensor: torch.Tensor, chunk_size: int) -> List[Chunk]:
        
        chunks = []
        flat_tensor = tensor.flatten()
        num_chunks = (flat_tensor.numel() + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, flat_tensor.numel())

            chunk_data = flat_tensor[start_idx:end_idx]
            chunk = Chunk(
                data=chunk_data,
                device=self.devices[0],
                chunk_id=f"chunk_{i}",
                offset=start_idx,
                size=chunk_data.numel()
            )
            chunks.append(chunk)
            self.chunks[f"chunk_{i}"] = chunk

        return chunks

    def send_chunk(self, chunk: Chunk, source_rank: int, dest_rank: int, tag: int = 0):
        
        send_op = {
            'type': 'send',
            'chunk': chunk,
            'source': self.devices[source_rank],
            'destination': self.devices[dest_rank],
            'tag': tag
        }
        self.operations.append(send_op)
        return send_op

    def recv_chunk(self, chunk_id: str, source_rank: int, dest_rank: int,
                   size: int, shape: Tuple, dtype: torch.dtype, tag: int = 0):
        
        recv_chunk = Chunk.create_empty(
            size=size,
            device=self.devices[dest_rank],
            chunk_id=f"{chunk_id}_recv",
            shape=shape,
            dtype=dtype
        )

        recv_op = {
            'type': 'recv',
            'chunk': recv_chunk,
            'source': self.devices[source_rank],
            'destination': self.devices[dest_rank],
            'tag': tag
        }
        self.operations.append(recv_op)
        return recv_op

    def reduce_chunks(self, chunks: List[Chunk], ranks: List[int], reduce_op: str = "sum"):
        
        reduce_op_dict = {
            'type': 'reduce',
            'chunks': chunks,
            'ranks': ranks,
            'reduce_op': reduce_op
        }
        self.operations.append(reduce_op_dict)
        return reduce_op_dict

@communication
class BasicChunkTransfer:
    

    def selective_transfer(self, tensor, source_rank, dest_ranks, chunk_mapping):
        """
        Transfer specific chunks to different destinations
        chunk_mapping: {chunk_index: destination_rank}
        """
        chunk_size = 64
        chunks = self._split_tensor(tensor, chunk_size)

        results = {}

        for chunk_idx, chunk in enumerate(chunks):
            if chunk_idx in chunk_mapping:
                dest_rank = chunk_mapping[chunk_idx]

                if self.rank == source_rank:
                    send_op = send(
                        destination=dest_rank,
                        tag=chunk_idx,
                        data=chunk
                    )
                    results[f"chunk_{chunk_idx}"] = send_op.execute(chunk)
                    print(f"Rank {source_rank}: Sent chunk {chunk_idx} to rank {dest_rank}")

                elif self.rank == dest_rank:
                    recv_op = recv(
                        source=source_rank,
                        tag=chunk_idx,
                        shape=chunk.shape,
                        dtype=chunk.dtype
                    )
                    results[f"chunk_{chunk_idx}"] = recv_op.execute()
                    print(f"Rank {dest_rank}: Received chunk {chunk_idx} from rank {source_rank}")

        return results

@communication
class ChunkBasedAllReduce:
    

    def custom_ring_allreduce(self, tensor, participants):
        
        chunk_size = 128
        num_chunks = (tensor.numel() + chunk_size - 1) // chunk_size
        chunks = self._split_tensor(tensor, chunk_size)

        results = []

        for step in range(len(participants)):
            for chunk_idx, chunk in enumerate(chunks):
                target_rank = participants[(self.rank + step) % len(participants)]
                source_rank = participants[(self.rank - step) % len(participants)]

                if chunk_idx % len(participants) == self.rank:
                    if step < len(participants) - 1:
                        send_op = send(
                            destination=target_rank,
                            tag=f"scatter_{chunk_idx}_{step}",
                            data=chunk
                        )

                        recv_op = recv(
                            source=source_rank,
                            tag=f"scatter_{chunk_idx}_{step}",
                            shape=chunk.shape,
                            dtype=chunk.dtype
                        )

                        received_data = recv_op.execute()
                        reduced_chunk = chunk + received_data
                        results.append(reduced_chunk)
                    else:
                        results.append(chunk)
        final_chunks = [None] * num_chunks
        for step in range(len(participants)):
            for chunk_idx in range(num_chunks):
                owner_rank = chunk_idx % len(participants)

                if self.rank == owner_rank:
                    target_rank = participants[(self.rank + 1) % len(participants)]
                    send_op = send(
                        destination=target_rank,
                        tag=f"gather_{chunk_idx}_{step}",
                        data=results[chunk_idx]
                    )
                    send_op.execute(results[chunk_idx])

                else:
                    source_rank = participants[(self.rank - 1) % len(participants)]
                    recv_op = recv(
                        source=source_rank,
                        tag=f"gather_{chunk_idx}_{step}",
                        shape=results[chunk_idx].shape,
                        dtype=results[chunk_idx].dtype
                    )
                    final_chunks[chunk_idx] = recv_op.execute()
        return torch.cat(final_chunks).reshape(tensor.shape)

@communication
class HierarchicalChunkCommunication:
    

    def hierarchical_transfer(self, tensor, world_size):
        
        chunk_size = 256
        chunks = self._split_tensor(tensor, chunk_size)

        ranks_per_node = 4
        my_node = self.rank // ranks_per_node
        local_rank = self.rank % ranks_per_node

        results = []

        for chunk_idx, chunk in enumerate(chunks):
            if chunk_idx % 2 == 0:
                if local_rank == 0:
                    for rank in range(1, ranks_per_node):
                        recv_op = recv(
                            source=my_node * ranks_per_node + rank,
                            tag=f"local_{chunk_idx}",
                            shape=chunk.shape,
                            dtype=chunk.dtype
                        )
                        local_chunk = recv_op.execute()
                        chunk = chunk + local_chunk

                    for rank in range(1, ranks_per_node):
                        send_op = send(
                            destination=my_node * ranks_per_node + rank,
                            tag=f"local_bcast_{chunk_idx}",
                            data=chunk
                        )
                        send_op.execute(chunk)
                    results.append(chunk)

                else:
                    send_op = send(
                        destination=my_node * ranks_per_node,
                        tag=f"local_{chunk_idx}",
                        data=chunk
                    )
                    send_op.execute(chunk)

                    recv_op = recv(
                        source=my_node * ranks_per_node,
                        tag=f"local_bcast_{chunk_idx}",
                        shape=chunk.shape,
                        dtype=chunk.dtype
                    )
                    results.append(recv_op.execute())

            else:
                if local_rank == 0:
                    target_node = (my_node + 1) % (world_size // ranks_per_node)
                    target_rank = target_node * ranks_per_node

                    if my_node == 0:
                        send_op = send(
                            destination=target_rank,
                            tag=f"inter_{chunk_idx}",
                            data=chunk
                        )
                        send_op.execute(chunk)
                        results.append(chunk)
                    elif my_node == 1:
                        recv_op = recv(
                            source=target_rank - ranks_per_node,
                            tag=f"inter_{chunk_idx}",
                            shape=chunk.shape,
                            dtype=chunk.dtype
                        )
                        results.append(recv_op.execute())
                else:
                    results.append(chunk)

        return torch.cat(results).reshape(tensor.shape)

@communication
class AdaptiveChunkRouting:
    

    def adaptive_communication(self, tensor, participants):
        
        chunk_sizes = [32, 64, 128, 256, 512]
        results = []

        for chunk_size in chunk_sizes:
            chunks = self._split_tensor(tensor, chunk_size)

            for chunk_idx, chunk in enumerate(chunks):
                if chunk.numel() <= 1024:
                    method = "direct"
                elif chunk.numel() <= 4096:
                    method = "tree"
                else:
                    method = "ring"

                result = self._communicate_chunk(chunk, method, participants, chunk_idx)
                results.append(result)

        return torch.cat(results).reshape(tensor.shape)

    def _communicate_chunk(self, chunk, method, participants, chunk_idx):
        
        if method == "direct":
            if self.rank == 0:
                for rank in participants[1:]:
                    send_op = send(
                        destination=rank,
                        tag=f"direct_{chunk_idx}",
                        data=chunk
                    )
                    send_op.execute(chunk)
                return chunk
            else:
                recv_op = recv(
                    source=0,
                    tag=f"direct_{chunk_idx}",
                    shape=chunk.shape,
                    dtype=chunk.dtype
                )
                return recv_op.execute()

        elif method == "tree":
            return self._tree_reduce_chunk(chunk, participants, chunk_idx)

        elif method == "ring":
            ring_op = allreduce(
                algorithm="ring",
                participants=participants,
                reduce_op="sum"
            )
            return ring_op.execute(chunk)

    def _tree_reduce_chunk(self, chunk, participants, chunk_idx):
        
        tree_fanout = 2
        level = 0

        while len(participants) > 1:
            next_participants = []

            for i in range(0, len(participants), tree_fanout):
                group = participants[i:i+tree_fanout]

                if i // tree_fanout == self.rank // (tree_fanout ** (level + 1)):
                    if self.rank == group[0]:
                        for child_rank in group[1:]:
                            recv_op = recv(
                                source=child_rank,
                                tag=f"tree_{chunk_idx}_{level}",
                                shape=chunk.shape,
                                dtype=chunk.dtype
                            )
                            child_chunk = recv_op.execute()
                            chunk = chunk + child_chunk
                        next_participants.append(group[0])
                    else:
                        parent_rank = group[0]
                        send_op = send(
                            destination=parent_rank,
                            tag=f"tree_{chunk_idx}_{level}",
                            data=chunk
                        )
                        send_op.execute(chunk)

            participants = next_participants
            level += 1

            if len(participants) == 0:
                break

        return chunk

@communication
class PipelineChunkCommunication:
    

    def pipeline_allreduce(self, tensor, participants, num_stages=4):
        
        chunk_size = tensor.numel() // num_stages
        chunks = self._split_tensor(tensor, chunk_size)

        pipeline_results = [None] * len(chunks)

        for stage in range(num_stages):
            for chunk_idx in range(stage, len(chunks), num_stages):
                chunk = chunks[chunk_idx]

                if chunk_idx < len(chunks) - 1:
                    if self.rank == 0:
                        send_op = send(
                            destination=participants[1],
                            tag=f"pipe_{chunk_idx}_{stage}",
                            data=chunk
                        )
                        send_future = send_op.execute_async(chunk)

                        if chunk_idx + num_stages < len(chunks):
                            next_chunk = chunks[chunk_idx + num_stages]
                            computed_chunk = self._compute_on_chunk(next_chunk)

                        pipeline_results[chunk_idx] = send_future.wait()
                    else:
                        recv_op = recv(
                            source=0,
                            tag=f"pipe_{chunk_idx}_{stage}",
                            shape=chunk.shape,
                            dtype=chunk.dtype
                        )
                        recv_future = recv_op.execute_async()

                        if chunk_idx + num_stages < len(chunks):
                            next_chunk = chunks[chunk_idx + num_stages]
                            computed_chunk = self._compute_on_chunk(next_chunk)

                        pipeline_results[chunk_idx] = recv_future.wait()
                else:
                    allreduce_op = allreduce(
                        algorithm="ring",
                        participants=participants,
                        reduce_op="sum"
                    )
                    pipeline_results[chunk_idx] = allreduce_op.execute(chunk)

        return torch.cat(pipeline_results).reshape(tensor.shape)

    def _compute_on_chunk(self, chunk):
        
        return chunk * 2  # Simple computation

def demonstrate_chunk_control():
    
    print("=== PCCL Chunk-Level Communication Examples ===\n")

    devices = [Device(rank=i, device_type="cuda") for i in range(4)]

    print("1. Basic Chunk Transfer Example")
    print("-" * 40)
    basic_transfer = BasicChunkTransfer()
    tensor = torch.randn(256, device="cuda")
    chunk_mapping = {0: 1, 1: 2, 2: 3, 3: 1}  # chunk_0->rank1, chunk_1->rank2, etc.

    try:
        result = basic_transfer.selective_transfer(tensor, 0, [1, 2, 3], chunk_mapping)
        print(f"✓ Basic transfer completed")
    except Exception as e:
        print(f"✗ Basic transfer failed: {e}")

    print("\n2. Custom Ring AllReduce with Chunk Control")
    print("-" * 50)
    ring_allreduce = ChunkBasedAllReduce()
    tensor = torch.randn(512, device="cuda")

    try:
        result = ring_allreduce.custom_ring_allreduce(tensor, [0, 1, 2, 3])
        print(f"✓ Custom ring allreduce completed")
        print(f"  Input shape: {tensor.shape}")
        print(f"  Output shape: {result.shape}")
    except Exception as e:
        print(f"✗ Custom ring allreduce failed: {e}")

    print("\n3. Hierarchical Chunk Communication")
    print("-" * 40)
    hierarchical = HierarchicalChunkCommunication()
    tensor = torch.randn(1024, device="cuda")

    try:
        result = hierarchical.hierarchical_transfer(tensor, 8)
        print(f"✓ Hierarchical communication completed")
        print(f"  Processed {tensor.numel() // 256} chunks")
    except Exception as e:
        print(f"✗ Hierarchical communication failed: {e}")

    print("\n4. Adaptive Chunk Routing")
    print("-" * 30)
    adaptive = AdaptiveChunkRouting()
    tensor = torch.randn(2048, device="cuda")

    try:
        result = adaptive.adaptive_communication(tensor, [0, 1, 2, 3])
        print(f"✓ Adaptive routing completed")
        print(f"  Used multiple chunk sizes: [32, 64, 128, 256, 512]")
        print(f"  Applied different methods: [direct, tree, ring]")
    except Exception as e:
        print(f"✗ Adaptive routing failed: {e}")

    print("\n5. Pipeline Chunk Communication")
    print("-" * 35)
    pipeline = PipelineChunkCommunication()
    tensor = torch.randn(1024, device="cuda")

    try:
        result = pipeline.pipeline_allreduce(tensor, [0, 1, 2, 3], num_stages=4)
        print(f"✓ Pipeline communication completed")
        print(f"  Stages: 4")
        print(f"  Chunk overlap: Enabled")
    except Exception as e:
        print(f"✗ Pipeline communication failed: {e}")

    print("\n6. Direct IR-Level Chunk Control")
    print("-" * 35)
    try:
        builder = IRBuilder()
        data = torch.randn(512, device="cuda")
        chunk_size = 128

        chunks = []
        for i in range(0, data.numel(), chunk_size):
            chunk_tensor = data.flatten()[i:i+chunk_size]
            chunk = builder.create_value(chunk_tensor, devices[0])
            chunks.append(chunk)

        operations = []
        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                send_op = builder.create_send(
                    source=chunk,
                    source_device=devices[0],
                    destination_device=devices[1],
                    tag=i
                )
                operations.append(send_op)
            else:
                send_op = builder.create_send(
                    source=chunk,
                    source_device=devices[0],
                    destination_device=devices[2],
                    tag=i
                )
                operations.append(send_op)

        graph = builder.create_graph(operations)
        print(f"✓ IR-level chunk control completed")
        print(f"  Created {len(chunks)} chunks")
        print(f"  Generated {len(operations)} operations")
    except Exception as e:
        print(f"✗ IR-level chunk control failed: {e}")

def demonstrate_chunk_performance_patterns():
    
    print("\n\n=== Performance-Oriented Chunk Patterns ===\n")

    print("1. Bandwidth-Optimized Large Chunks")
    print("-" * 40)
    large_chunk_tensor = torch.randn(10240, device="cuda")
    print(f"Large chunk size: {large_chunk_tensor.numel() * 4 / 1024:.1f} KB")
    print("Recommended: Direct transfer, minimal overhead")

    print("\n2. Latency-Optimized Small Chunks")
    print("-" * 38)
    small_chunk_tensor = torch.randn(64, device="cuda")
    print(f"Small chunk size: {small_chunk_tensor.numel() * 4:.0f} B")
    print("Recommended: Batch multiple chunks, tree reduction")

    print("\n3. Balanced Medium Chunks")
    print("-" * 28)
    medium_chunk_tensor = torch.randn(512, device="cuda")
    print(f"Medium chunk size: {medium_chunk_tensor.numel() * 4 / 1024:.1f} KB")
    print("Recommended: Ring allreduce, pipeline overlap")

if __name__ == "__main__":
    demonstrate_chunk_control()
    demonstrate_chunk_performance_patterns()

    print("\n\n=== Key Features Demonstrated ===")
    print("✓ Selective chunk routing to different destinations")
    print("✓ Custom allreduce implementation with chunk control")
    print("✓ Hierarchical communication patterns")
    print("✓ Adaptive routing based on chunk characteristics")
    print("✓ Pipeline communication with compute-communication overlap")
    print("✓ Direct IR-level chunk manipulation")
    print("✓ Performance-oriented chunk sizing strategies")
    print("\nChunk-level communication provides maximum control over data movement!")