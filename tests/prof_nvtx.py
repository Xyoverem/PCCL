"""Profile PCCL TMA allreduce with NVTX + nsys across multiple sizes."""
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pccl import DeviceType, build_graph, compile_to_json_file


def ring_allreduce_tma(rank, world_size, tensor_size):
    chunk_size = tensor_size // world_size
    def build(op):
        op.tensor(dtype="float32", shape=(tensor_size,))
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size
        op.notify(signal_id=100, target_rank=next_rank)
        op.wait_notify(signal_id=100, source_rank=prev_rank)
        for step in range(world_size - 1):
            reduce_chunk = (rank - step - 1) % world_size
            off = reduce_chunk * chunk_size
            if step > 0:
                op.wait_notify(signal_id=0, source_rank=prev_rank)
            op.tma_reduce(reduce_op="sum", source_rank=prev_rank,
                          src_offset=off, dst_offset=off, remote_offset=off, count=chunk_size)
            op.notify(signal_id=0, target_rank=next_rank)
        for step in range(world_size - 1):
            recv_chunk = (rank - step) % world_size
            off = recv_chunk * chunk_size
            op.wait_notify(signal_id=0, source_rank=prev_rank)
            op.tma_copy(source_rank=prev_rank, src_offset=off, dst_offset=off, size=chunk_size)
            op.notify(signal_id=0, target_rank=next_rank)
        op.wait_notify(signal_id=0, source_rank=prev_rank)
    return build_graph(f"ring_ar_tma_rank{rank}", build, device=DeviceType.CUDA)


SIZES = [
    ("64KB",   64 * 1024),
    ("1MB",    1 * 1024 * 1024),
    ("16MB",   16 * 1024 * 1024),
    ("128MB",  128 * 1024 * 1024),
    ("512MB",  512 * 1024 * 1024),
]

WARMUP = 3
ITERS = 20


def worker(rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", rank=rank, world_size=world_size)

    import pccl.engine
    pccl.engine.get_engine()
    dist.barrier()
    pccl.engine.initialize_engine(dist.group.WORLD)
    dist.barrier()

    output_dir = Path(__file__).parent / "generated_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, data_bytes in SIZES:
        tensor_size = data_bytes // 4
        if tensor_size % world_size != 0:
            tensor_size = (tensor_size // world_size) * world_size

        op_name = f"prof_{label}"
        graph = ring_allreduce_tma(rank, world_size, tensor_size)
        json_file = str(output_dir / f"prof_{label}_rank{rank}.json")
        compile_to_json_file(graph, json_file)
        pccl.engine.register_operation(op_name, json_file)
        dist.barrier()

        inp = torch.ones(tensor_size, dtype=torch.float32, device=f"cuda:{rank}") * (rank + 1)
        out = torch.zeros_like(inp)

        for _ in range(WARMUP):
            pccl.engine.execute_operation_async(op_name, inp, out)
            pccl.engine.sync_operation(op_name)
        dist.barrier()
        torch.cuda.synchronize()

        for _ in range(ITERS):
            pccl.engine.execute_operation_async(op_name, inp, out)
            pccl.engine.sync_operation(op_name)
        torch.cuda.synchronize()
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, 29501), nprocs=2, join=True)
