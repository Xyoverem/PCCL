"""Correctness tests for collective operations: allreduce, reduce_scatter, allgather, alltoall."""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pccl import DeviceType, build_graph, compile_to_json_file
from pccl.dsl.decorators import CommunicationOp, Stream
from pccl.dsl.algorithms import RingAllreduce


def ring_allreduce_tma(rank, world_size, tensor_size):
    chunk_size = tensor_size // world_size
    with CommunicationOp(name=f"test_ar_rank{rank}", device=DeviceType.CUDA) as op:
        op.tensor(dtype="bfloat16", shape=(tensor_size,))
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
                          src_offset=off, dst_offset=off,
                          remote_offset=off, count=chunk_size)
            op.notify(signal_id=0, target_rank=next_rank)

        for step in range(world_size - 1):
            recv_chunk = (rank - step) % world_size
            off = recv_chunk * chunk_size
            op.wait_notify(signal_id=0, source_rank=prev_rank)
            op.tma_copy(source_rank=prev_rank,
                        src_offset=off, dst_offset=off, size=chunk_size)
            op.notify(signal_id=0, target_rank=next_rank)

        op.wait_notify(signal_id=0, source_rank=prev_rank)
    return op.graph


def ring_copy_only(rank, world_size, tensor_size):
    """Allgather-only: just TMA copies, no reduce."""
    chunk_size = tensor_size // world_size
    with CommunicationOp(name=f"test_copy_rank{rank}", device=DeviceType.CUDA) as op:
        op.tensor(dtype="bfloat16", shape=(tensor_size,))
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size

        op.notify(signal_id=100, target_rank=next_rank)
        op.wait_notify(signal_id=100, source_rank=prev_rank)

        for step in range(world_size - 1):
            recv_chunk = (rank - step) % world_size
            off = recv_chunk * chunk_size
            if step > 0:
                op.wait_notify(signal_id=0, source_rank=prev_rank)
            op.tma_copy(source_rank=prev_rank,
                        src_offset=off, dst_offset=off, size=chunk_size)
            op.notify(signal_id=0, target_rank=next_rank)

        op.wait_notify(signal_id=0, source_rank=prev_rank)
    return op.graph


def ring_reduce_only(rank, world_size, tensor_size):
    """Reduce-scatter only: just TMA reduces."""
    chunk_size = tensor_size // world_size
    with CommunicationOp(name=f"test_reduce_rank{rank}", device=DeviceType.CUDA) as op:
        op.tensor(dtype="bfloat16", shape=(tensor_size,))
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
                          src_offset=off, dst_offset=off,
                          remote_offset=off, count=chunk_size)
            op.notify(signal_id=0, target_rank=next_rank)

        op.wait_notify(signal_id=0, source_rank=prev_rank)
    return op.graph


def run_test(op_name, graph_fn, rank, world_size, tensor_size, inp, expected_fn, output_dir):
    graph = graph_fn(rank, world_size, tensor_size)
    json_file = str(output_dir / f"{op_name}_rank{rank}.json")
    compile_to_json_file(graph, json_file)

    import pccl.engine
    pccl.engine.register_operation(op_name, json_file)
    dist.barrier()

    out = torch.zeros_like(inp)

    pccl.engine.execute_operation_async(op_name, inp, out)
    pccl.engine.sync_operation(op_name)
    torch.cuda.synchronize()

    expected = expected_fn(rank, world_size, inp)
    mismatches = (out != expected).sum().item()
    total = out.numel()

    # All ranks print for debugging
    if mismatches > 0 or True:
        chunk = total // world_size
        print(f"  [{op_name}] rank={rank}: mismatches={mismatches}/{total} ({mismatches/total*100:.1f}%)")
        for c in range(world_size):
            start = c * chunk
            sample = out[start:start+3].tolist()
            exp_sample = expected[start:start+3].tolist()
            print(f"    chunk[{c}] [{start}:{start+3}]: got={sample}, expected={exp_sample}")

    dist.barrier()
    return mismatches


def worker(rank, world_size, master_port):
    os.environ.update({"MASTER_ADDR": "localhost", "MASTER_PORT": str(master_port),
                        "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size)})
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", rank=rank, world_size=world_size)

    import pccl.engine
    pccl.engine.get_engine()
    dist.barrier()
    pccl.engine.initialize_engine(dist.group.WORLD)
    dist.barrier()

    output_dir = Path(__file__).parent / "generated_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_size = 1024 * 1024  # 1M elements = 2MB in BF16
    tensor_size = (tensor_size // world_size) * world_size
    inp = torch.ones(tensor_size, dtype=torch.bfloat16, device=f"cuda:{rank}") * (rank + 1)

    if rank == 0:
        print(f"\nCorrectness test: {world_size} GPUs, {tensor_size} BF16 elements ({tensor_size*2/1024/1024:.1f} MB)")
        print(f"Input: rank i fills with (i+1), expected allreduce = sum(1..{world_size}) = {world_size*(world_size+1)//2}")

    # Test 1: TMA copy only (allgather)
    # Each rank writes its own chunk. After copy, each rank should see all chunks.
    def copy_expected(r, ws, inp_t):
        # The ring copy reads from prev_rank. For 2 GPUs:
        # rank 0 copies chunk (0-0)%2=0 from rank 1 → chunk[0] = 2.0
        # rank 1 copies chunk (1-0)%2=1 from rank 0 → chunk[1] = 1.0
        # So each rank's copied chunk gets the prev_rank's value
        chunk = tensor_size // ws
        result = inp_t.clone()  # start with own values
        prev_rank = (r - 1 + ws) % ws
        for step in range(ws - 1):
            recv_chunk = (r - step) % ws
            result[recv_chunk * chunk:(recv_chunk + 1) * chunk] = prev_rank + 1
        return result

    run_test("test_copy", ring_copy_only, rank, world_size, tensor_size, inp, copy_expected, output_dir)

    # Reset signals between tests
    pccl.engine.reset_signals("test_copy")
    dist.barrier()

    # Test 2: Full allreduce
    def allreduce_expected(r, ws, inp_t):
        return torch.full_like(inp_t, ws * (ws + 1) / 2)

    run_test("test_allreduce", ring_allreduce_tma, rank, world_size, tensor_size, inp, allreduce_expected, output_dir)

    pccl.engine.reset_signals("test_allreduce")
    dist.barrier()

    # Test 3: Reduce-scatter only
    # After reduce-scatter, rank r's chunk (rank - 0 - 1 + ws) % ws should have the sum
    def reduce_expected(r, ws, inp_t):
        chunk = tensor_size // ws
        expected_sum = ws * (ws + 1) / 2
        result = inp_t.clone()
        # Ring reduce-scatter: rank r reduces chunk (r-step-1)%ws for step in range(ws-1)
        # For 2 GPUs: rank 0 reduces chunk 1, rank 1 reduces chunk 0
        # The reduced chunk gets sum, other chunks stay as input
        reduced_chunk = (r - 1 + ws) % ws  # For ws-1=1 step: chunk (r-0-1)%ws
        result[reduced_chunk * chunk:(reduced_chunk + 1) * chunk] = expected_sum
        return result

    run_test("test_reduce", ring_reduce_only, rank, world_size, tensor_size, inp, reduce_expected, output_dir)

    pccl.engine.reset_signals("test_reduce")
    dist.barrier()

    # ---- Tests using DSL algorithm builders with proper collective_type ----
    alg = RingAllreduce()

    # Test 4: reduce_scatter via DSL builder
    def build_rs(r, ws, ts):
        return alg.build_reduce_scatter(r, ws, ts, dtype="bfloat16", executor="tma")

    chunk_size = tensor_size // world_size
    rs_out_tensor = torch.zeros(chunk_size, dtype=torch.bfloat16, device=f"cuda:{rank}")

    def rs_expected(r, ws, inp_t):
        expected_sum = ws * (ws + 1) / 2
        return torch.full((chunk_size,), expected_sum, dtype=torch.bfloat16, device=inp_t.device)

    graph = build_rs(rank, world_size, tensor_size)
    json_file = str(output_dir / f"test_rs_rank{rank}.json")
    compile_to_json_file(graph, json_file)
    pccl.engine.register_operation("test_rs", json_file)
    dist.barrier()
    pccl.engine.execute_operation_async("test_rs", inp, rs_out_tensor)
    pccl.engine.sync_operation("test_rs")
    torch.cuda.synchronize()
    expected = rs_expected(rank, world_size, inp)
    mismatches = (rs_out_tensor != expected).sum().item()
    if rank == 0:
        print(f"  [test_rs] reduce_scatter: mismatches={mismatches}/{chunk_size}")
        print(f"    got={rs_out_tensor[:3].tolist()}, expected={expected[:3].tolist()}")
    assert mismatches == 0, f"reduce_scatter failed on rank {rank}: {mismatches} mismatches"
    pccl.engine.reset_signals("test_rs")
    dist.barrier()

    # Test 5: allgather via DSL builder
    def build_ag(r, ws, ts):
        return alg.build_allgather(r, ws, ts, dtype="bfloat16", executor="tma")

    ag_inp = torch.full((chunk_size,), rank + 1, dtype=torch.bfloat16, device=f"cuda:{rank}")
    ag_out = torch.zeros(tensor_size, dtype=torch.bfloat16, device=f"cuda:{rank}")

    def ag_expected(r, ws, inp_t):
        result = torch.zeros(tensor_size, dtype=torch.bfloat16, device=inp_t.device)
        for i in range(ws):
            result[i * chunk_size:(i + 1) * chunk_size] = i + 1
        return result

    graph = build_ag(rank, world_size, tensor_size)
    json_file = str(output_dir / f"test_ag_rank{rank}.json")
    compile_to_json_file(graph, json_file)
    pccl.engine.register_operation("test_ag", json_file)
    dist.barrier()
    pccl.engine.execute_operation_async("test_ag", ag_inp, ag_out)
    pccl.engine.sync_operation("test_ag")
    torch.cuda.synchronize()
    expected = ag_expected(rank, world_size, ag_inp)
    mismatches = (ag_out != expected).sum().item()
    if rank == 0:
        print(f"  [test_ag] allgather: mismatches={mismatches}/{tensor_size}")
        for c in range(world_size):
            s = c * chunk_size
            print(f"    chunk[{c}]: got={ag_out[s:s+3].tolist()}, expected={expected[s:s+3].tolist()}")
    assert mismatches == 0, f"allgather failed on rank {rank}: {mismatches} mismatches"
    pccl.engine.reset_signals("test_ag")
    dist.barrier()

    # Test 6: alltoall via DSL builder
    def build_a2a(r, ws, ts):
        return alg.build_alltoall(r, ws, ts, dtype="bfloat16", executor="tma")

    a2a_inp = torch.zeros(tensor_size, dtype=torch.bfloat16, device=f"cuda:{rank}")
    for i in range(world_size):
        a2a_inp[i * chunk_size:(i + 1) * chunk_size] = rank * 100 + i
    a2a_out = torch.zeros(tensor_size, dtype=torch.bfloat16, device=f"cuda:{rank}")

    def a2a_expected(r, ws, inp_t):
        result = torch.zeros(tensor_size, dtype=torch.bfloat16, device=inp_t.device)
        for i in range(ws):
            result[i * chunk_size:(i + 1) * chunk_size] = i * 100 + r
        return result

    graph = build_a2a(rank, world_size, tensor_size)
    json_file = str(output_dir / f"test_a2a_rank{rank}.json")
    compile_to_json_file(graph, json_file)
    pccl.engine.register_operation("test_a2a", json_file)
    dist.barrier()
    pccl.engine.execute_operation_async("test_a2a", a2a_inp, a2a_out)
    pccl.engine.sync_operation("test_a2a")
    torch.cuda.synchronize()
    expected = a2a_expected(rank, world_size, a2a_inp)
    mismatches = (a2a_out != expected).sum().item()
    if rank == 0:
        print(f"  [test_a2a] alltoall: mismatches={mismatches}/{tensor_size}")
        for c in range(world_size):
            s = c * chunk_size
            print(f"    chunk[{c}]: got={a2a_out[s:s+3].tolist()}, expected={expected[s:s+3].tolist()}")
    assert mismatches == 0, f"alltoall failed on rank {rank}: {mismatches} mismatches"

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mp.spawn(worker, args=(nproc, 29503), nprocs=nproc, join=True)
