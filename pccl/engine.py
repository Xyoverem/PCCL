"""PCCL Engine - Python wrapper for the C++ runtime (simplified)."""

import os

import torch
import torch.distributed as dist
import pccl.engine_c
from pccl.utils.to_tensor import to_tensor, to_str


class Engine:
    def __init__(self):
        self._engine = pccl.engine_c.Engine.get_instance()

    def register_operation(self, name, filename):
        return self._engine.regOp(name, filename)

    def execute_operation(self, name, input_tensor, output_tensor=None):
        if output_tensor is None:
            output_tensor = torch.empty_like(input_tensor)
        self._engine.exeOp(name, input_tensor, output_tensor)
        return output_tensor

    def execute_operation_async(self, name, input_tensor, output_tensor=None):
        if output_tensor is None:
            output_tensor = torch.empty_like(input_tensor)
        self._engine.exeOpAsync(name, input_tensor, output_tensor)
        return output_tensor

    def sync_operation(self, name):
        self._engine.syncOp(name)

    def reset_signals(self, name):
        self._engine.resetSignals(name)

    def get_endpoint(self):
        return self._engine.exportEndpoint()

    def update_endpoint(self, rank, endpoint):
        return self._engine.updateEndpoint(rank, endpoint)


_engine_instance = None


def _debug_endpoint_init(stage):
    if os.environ.get("PCCL_DEBUG_ENDPOINT_INIT") != "1":
        return
    rank = dist.get_rank() if dist.is_initialized() else -1
    print("PCCL_ENDPOINT_INIT rank={} {}".format(rank, stage), flush=True)

def get_engine() -> Engine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = Engine()
    return _engine_instance

def register_operation(name, filename):
    return get_engine().register_operation(name, filename)

def execute_operation(name, input_tensor, output_tensor=None):
    return get_engine().execute_operation(name, input_tensor, output_tensor)

def execute_operation_async(name, input_tensor, output_tensor=None):
    return get_engine().execute_operation_async(name, input_tensor, output_tensor)

def sync_operation(name):
    return get_engine().sync_operation(name)

def reset_signals(name):
    return get_engine().reset_signals(name)

def get_endpoint():
    return get_engine().get_endpoint()

def initialize_engine(group: dist.ProcessGroup):
    engine = get_engine()
    _debug_endpoint_init("engine_ready")
    endpoint = engine.get_endpoint()
    _debug_endpoint_init("endpoint_exported")
    backend = dist.get_backend(group)
    device = None
    if str(backend).lower() == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())

    t_endpoint = to_tensor(endpoint).to(device=device)
    _debug_endpoint_init("endpoint_tensor_ready")
    buffer_size = t_endpoint.numel()
    t_buffer_size = torch.zeros((1), dtype=torch.int32, device=device)
    buffer_sizes = torch.zeros((group.size()), dtype=torch.int32, device=device)
    t_buffer_size[0] = buffer_size
    dist.all_gather_into_tensor(buffer_sizes, t_buffer_size, group)
    _debug_endpoint_init("endpoint_sizes_gathered")
    max_size = int(buffer_sizes.max().item())
    all_endpoints_tensor = torch.zeros(
        (group.size() * max_size), dtype=torch.uint8, device=device)
    local_padded = torch.zeros(max_size, dtype=torch.uint8, device=device)
    local_padded[:t_endpoint.numel()] = t_endpoint
    dist.all_gather_into_tensor(all_endpoints_tensor, local_padded, group)
    _debug_endpoint_init("endpoint_payloads_gathered")

    all_endpoints_tensor = all_endpoints_tensor.view(group.size(), max_size)

    for rank in range(group.size()):
        if rank == group.rank():
            continue
        raw_tensor = all_endpoints_tensor[rank]
        actual_size = buffer_sizes[rank].item()
        valid_tensor = raw_tensor[:actual_size]
        endpoint_str = to_str(valid_tensor)
        engine.update_endpoint(rank, endpoint_str)
    _debug_endpoint_init("remote_endpoints_updated")

    dist.barrier()
    _debug_endpoint_init("barrier_complete")
