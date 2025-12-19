import torch
import typing
from typing import List, Dict, Tuple, Optional, Union, Any
from .config import CommunicationConfig, DeviceConfig
from .compiler import Compiler, CompiledOperator
from .chunk import Chunk
from .operator import Device

class ChunkCommunication:
    def __init__(self, devices: List[Device], topology=None):
        self.devices = devices
        self.topology = topology
        self.compiler = Compiler()
        self.chunks: Dict[str, Chunk] = {}
        self.operations: List[Dict[str, Any]] = []
        self.communication_context = {}
        self.performance_metrics = {}

    def create_chunks(self, tensor: torch.Tensor, chunk_size: int,
                     chunk_ids: Optional[List[str]] = None) -> List[Chunk]:
        chunks = []
        flat_tensor = tensor.flatten()
        num_chunks = (flat_tensor.numel() + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, flat_tensor.numel())

            chunk_data = flat_tensor[start_idx:end_idx]
            chunk_id = chunk_ids[i] if chunk_ids and i < len(chunk_ids) else f"chunk_{i}"

            chunk = Chunk(
                data=chunk_data,
                device=self.devices[0],
                chunk_id=chunk_id,
                offset=start_idx,
                size=chunk_data.numel()
            )
            chunks.append(chunk)
            self.chunks[chunk_id] = chunk

        return chunks

    def send_chunk(self, chunk_id: str, source_rank: int, dest_rank: int,
                  tag: int = 0, priority: int = 0) -> Dict[str, Any]:
        if chunk_id not in self.chunks:
            raise ValueError(f"Chunk {chunk_id} not found")

        chunk = self.chunks[chunk_id]
        send_op = {
            'type': 'send',
            'chunk_id': chunk_id,
            'source_rank': source_rank,
            'dest_rank': dest_rank,
            'tag': tag,
            'priority': priority,
            'size': chunk.size,
            'offset': chunk.offset,
            'dtype': chunk.dtype,
            'timestamp': self._get_timestamp()
        }
        self.operations.append(send_op)
        return send_op

    def recv_chunk(self, chunk_id: str, source_rank: int, dest_rank: int,
                  shape: Tuple[int, ...], dtype: torch.dtype, tag: int = 0,
                  priority: int = 0) -> Dict[str, Any]:
        recv_op = {
            'type': 'recv',
            'chunk_id': chunk_id,
            'source_rank': source_rank,
            'dest_rank': dest_rank,
            'tag': tag,
            'priority': priority,
            'shape': shape,
            'dtype': dtype,
            'timestamp': self._get_timestamp()
        }
        self.operations.append(recv_op)
        return recv_op

    def reduce_chunks(self, chunk_ids: List[str], ranks: List[int],
                     reduce_op: str = "sum", algorithm: str = "ring") -> Dict[str, Any]:
        reduce_op_dict = {
            'type': 'reduce',
            'chunk_ids': chunk_ids,
            'ranks': ranks,
            'reduce_op': reduce_op,
            'algorithm': algorithm,
            'timestamp': self._get_timestamp()
        }
        self.operations.append(reduce_op_dict)
        return reduce_op_dict

    def broadcast_chunk(self, chunk_id: str, root_rank: int, ranks: List[int]) -> Dict[str, Any]:
        bcast_op = {
            'type': 'broadcast',
            'chunk_id': chunk_id,
            'root_rank': root_rank,
            'ranks': ranks,
            'timestamp': self._get_timestamp()
        }
        self.operations.append(bcast_op)
        return bcast_op

    def pipeline_chunks(self, chunk_sequence: List[str], stages: int) -> List[Dict[str, Any]]:
        pipeline_ops = []
        for stage in range(stages):
            for i, chunk_id in enumerate(chunk_sequence):
                if i % stages == stage:
                    stage_op = {
                        'type': 'pipeline_stage',
                        'chunk_id': chunk_id,
                        'stage': stage,
                        'total_stages': stages,
                        'timestamp': self._get_timestamp()
                    }
                    pipeline_ops.append(stage_op)
                    self.operations.append(stage_op)
        return pipeline_ops

    def custom_routing(self, chunk_routes: Dict[str, List[int]]) -> List[Dict[str, Any]]:
        route_ops = []
        for chunk_id, route in chunk_routes.items():
            for i, dest_rank in enumerate(route):
                route_op = {
                    'type': 'route',
                    'chunk_id': chunk_id,
                    'dest_rank': dest_rank,
                    'hop': i,
                    'timestamp': self._get_timestamp()
                }
                route_ops.append(route_op)
                self.operations.append(route_op)
        return route_ops

    def aggregate_chunks(self, chunk_ids: List[str], aggregation_method: str = "concat") -> Dict[str, Any]:
        agg_op = {
            'type': 'aggregate',
            'chunk_ids': chunk_ids,
            'method': aggregation_method,
            'timestamp': self._get_timestamp()
        }
        self.operations.append(agg_op)
        return agg_op

    def split_chunk(self, chunk_id: str, split_sizes: List[int]) -> List[Dict[str, Any]]:
        split_ops = []
        for i, size in enumerate(split_sizes):
            split_op = {
                'type': 'split',
                'chunk_id': chunk_id,
                'split_id': f"{chunk_id}_split_{i}",
                'size': size,
                'timestamp': self._get_timestamp()
            }
            split_ops.append(split_op)
            self.operations.append(split_op)
        return split_ops

    def execute(self, async_execution: bool = False) -> Dict[str, torch.Tensor]:
        execution_plan = self.compiler.compile_chunk_operations(self.operations)

        if async_execution:
            return self._execute_async(execution_plan)
        else:
            return self._execute_sync(execution_plan)

    def _execute_sync(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        results = {}
        start_time = self._get_timestamp()

        for op in execution_plan:
            op_start = self._get_timestamp()

            if op['type'] == 'send':
                result = self._execute_send(op)
            elif op['type'] == 'recv':
                result = self._execute_recv(op)
            elif op['type'] == 'reduce':
                result = self._execute_reduce(op)
            elif op['type'] == 'broadcast':
                result = self._execute_broadcast(op)
            elif op['type'] == 'pipeline_stage':
                result = self._execute_pipeline_stage(op)
            elif op['type'] == 'route':
                result = self._execute_route(op)
            elif op['type'] == 'aggregate':
                result = self._execute_aggregate(op)
            elif op['type'] == 'split':
                result = self._execute_split(op)
            else:
                raise ValueError(f"Unknown operation type: {op['type']}")

            op_end = self._get_timestamp()
            self._record_performance_metric(op, op_start, op_end)

            if op.get('chunk_id'):
                results[op['chunk_id']] = result

        total_time = self._get_timestamp() - start_time
        self.performance_metrics['total_execution_time'] = total_time

        return results

    def _execute_async(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        futures = {}
        results = {}

        for op in execution_plan:
            if op['type'] in ['send', 'recv']:
                # Non-blocking operations
                future = self._execute_async_operation(op)
                futures[op['chunk_id']] = future
            else:
                # Blocking operations
                result = self._execute_sync({op})
                if op.get('chunk_id'):
                    results[op['chunk_id']] = result

        # Wait for all async operations to complete
        for chunk_id, future in futures.items():
            results[chunk_id] = future.wait()

        return results

    def _execute_send(self, op: Dict[str, Any]) -> torch.Tensor:
        chunk = self.chunks[op['chunk_id']]
        # Simulate send operation
        return chunk.data

    def _execute_recv(self, op: Dict[str, Any]) -> torch.Tensor:
        # Simulate receive operation
        shape = op['shape']
        dtype = op['dtype']
        return torch.randn(shape, dtype=dtype, device=self.devices[op['dest_rank']].device)

    def _execute_reduce(self, op: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        results = {}
        for chunk_id in op['chunk_ids']:
            chunk = self.chunks[chunk_id]
            # Simulate reduction across ranks
            results[chunk_id] = chunk.data * len(op['ranks'])
        return results

    def _execute_broadcast(self, op: Dict[str, Any]) -> torch.Tensor:
        chunk = self.chunks[op['chunk_id']]
        # Simulate broadcast
        return chunk.data.clone()

    def _execute_pipeline_stage(self, op: Dict[str, Any]) -> torch.Tensor:
        chunk = self.chunks[op['chunk_id']]
        # Simulate pipeline stage with computation
        return chunk.data * 2

    def _execute_route(self, op: Dict[str, Any]) -> torch.Tensor:
        chunk = self.chunks[op['chunk_id']]
        # Simulate routing through intermediate hops
        return chunk.data

    def _execute_aggregate(self, op: Dict[str, Any]) -> torch.Tensor:
        chunks_data = [self.chunks[chunk_id].data for chunk_id in op['chunk_ids']]
        if op['method'] == "concat":
            return torch.cat(chunks_data)
        elif op['method'] == "sum":
            return sum(chunks_data)
        else:
            return torch.cat(chunks_data)

    def _execute_split(self, op: Dict[str, Any]) -> torch.Tensor:
        chunk = self.chunks[op['chunk_id']]
        # Simulate chunk splitting
        return chunk.data[:op['size']]

    def _execute_async_operation(self, op: Dict[str, Any]):
        # Simulate async operation future
        class MockFuture:
            def __init__(self, result):
                self._result = result
            def wait(self):
                return self._result

        result = self._execute_sync({op})
        return MockFuture(result)

    def _get_timestamp(self) -> float:
        import time
        return time.time()

    def _record_performance_metric(self, op: Dict[str, Any], start_time: float, end_time: float):
        metric_key = f"{op['type']}_{op.get('chunk_id', 'unknown')}"
        self.performance_metrics[metric_key] = end_time - start_time

    def get_chunk_info(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        if chunk_id not in self.chunks:
            return None

        chunk = self.chunks[chunk_id]
        return {
            'chunk_id': chunk.chunk_id,
            'size': chunk.size,
            'offset': chunk.offset,
            'dtype': chunk.dtype,
            'device': chunk.device
        }

    def list_chunks(self) -> List[str]:
        return list(self.chunks.keys())

    def clear_operations(self):
        self.operations.clear()
        self.performance_metrics.clear()

    def get_performance_metrics(self) -> Dict[str, float]:
        return self.performance_metrics.copy()

class ChunkManager:
    def __init__(self, devices: List[Device]):
        self.devices = devices
        self.chunk_groups: Dict[str, ChunkCommunication] = {}

    def create_group(self, group_name: str, topology=None) -> ChunkCommunication:
        if group_name in self.chunk_groups:
            return self.chunk_groups[group_name]

        group = ChunkCommunication(self.devices, topology)
        self.chunk_groups[group_name] = group
        return group

    def get_group(self, group_name: str) -> Optional[ChunkCommunication]:
        return self.chunk_groups.get(group_name)

    def execute_all_groups(self) -> Dict[str, Dict[str, torch.Tensor]]:
        results = {}
        for group_name, group in self.chunk_groups.items():
            results[group_name] = group.execute()
        return results

    def clear_all_groups(self):
        for group in self.chunk_groups.values():
            group.clear_operations()
        self.chunk_groups.clear()

__all__ = [
    "ChunkCommunication",
    "ChunkManager"
]