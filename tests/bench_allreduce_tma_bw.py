"""
PCCL AllReduce Bandwidth Benchmark

Measures alg_bw and bus_bw = alg_bw * 2*(n-1)/n for ring allreduce
across a sweep of tensor sizes.  Supports multiple performance options.

Performance options:
  --executor {tma,sm}      Executor type (default: tma)
  --channels N             Number of channels (default: 1)
  --disable-fused          Disable fused step executor (use DAG kernel)
  --no-nccl                Skip NCCL reference benchmark
  --sizes S1,S2,...        Comma-separated sizes in bytes (e.g. 4096,1048576)
  --min-size / --max-size  Auto-generate power-of-2 size sweep
  --warmup N               Warmup iterations (default: 5)
  --iters N                Benchmark iterations (default: 20)
  --dtype {bf16,f32}       Data type (default: bf16)

Usage (spawn):
    python tests/bench_allreduce_tma_bw.py --nproc=2
    python tests/bench_allreduce_tma_bw.py --nproc=8 --executor=sm --disable-fused
    python tests/bench_allreduce_tma_bw.py --nproc=4 --channels=4 --sizes=67108864,536870912

Usage (torchrun):
    torchrun --nproc_per_node=2 tests/bench_allreduce_tma_bw.py --torchrun
"""

import os
import sys
import math
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import DeviceType, build_graph, compile_to_json_file, Compiler
from pccl.dsl.decorators import CommunicationOp, Stream
from pccl.dsl.pipeline import Pipeline
from pccl.dsl.superopt import NVLINK_TOPOLOGY, H100_PROFILE
from pccl.dsl.algorithms import ALGORITHMS, select_algorithm


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def ring_allreduce(rank: int, world_size: int, tensor_size: int,
                   num_channels: int = 1, executor: str = "tma",
                   pipeline_depth: int = 1):
    """Build ring allreduce graph with configurable executor, channels, and pipeline.

    When ``pipeline_depth > 1`` the Pipeline construct is used to overlap
    consecutive reduce-scatter / allgather steps across channels.
    """
    if pipeline_depth > 1:
        return _pipelined_ring_allreduce(
            rank, world_size, tensor_size, executor, pipeline_depth)

    chunk_per_ch = tensor_size // num_channels
    chunk_per_rank = chunk_per_ch // world_size
    name = f"ar_{executor}_c{num_channels}_rank{rank}"

    use_tma = (executor == "tma")

    with CommunicationOp(name=name, device=DeviceType.CUDA) as op:
        dtype = "bfloat16" if _ARGS.dtype == "bf16" else "float32"
        op.tensor(dtype=dtype, shape=(tensor_size,))
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size

        for c in range(num_channels):
            base = c * chunk_per_ch
            sig_base = c * 1000

            with Stream(f"ch{c}"):
                op.set_channel(c)

                op.notify(signal_id=sig_base + 100, target_rank=next_rank)
                op.wait_notify(signal_id=sig_base + 100, source_rank=prev_rank)

                # Reduce-scatter
                for step in range(world_size - 1):
                    chunk_idx = (rank - step - 1) % world_size
                    off = base + chunk_idx * chunk_per_rank
                    if step > 0:
                        op.wait_notify(signal_id=sig_base, source_rank=prev_rank)
                    if use_tma:
                        op.tma_reduce(reduce_op="sum", source_rank=prev_rank,
                                      src_offset=off, dst_offset=off,
                                      remote_offset=off, count=chunk_per_rank)
                    else:
                        op.sm_reduce(reduce_op="sum", source_rank=prev_rank,
                                     src_offset=off, dst_offset=off,
                                     remote_offset=off, count=chunk_per_rank)
                    op.notify(signal_id=sig_base, target_rank=next_rank)

                # Allgather
                for step in range(world_size - 1):
                    chunk_idx = (rank - step) % world_size
                    off = base + chunk_idx * chunk_per_rank
                    op.wait_notify(signal_id=sig_base, source_rank=prev_rank)
                    if use_tma:
                        op.tma_copy(source_rank=prev_rank,
                                    src_offset=off, dst_offset=off,
                                    size=chunk_per_rank)
                    else:
                        op.sm_copy(source_rank=prev_rank,
                                   src_offset=off, dst_offset=off,
                                   size=chunk_per_rank)
                    op.notify(signal_id=sig_base, target_rank=next_rank)

                op.wait_notify(signal_id=sig_base, source_rank=prev_rank)

        return op.get_graph()


def _pipelined_ring_allreduce(rank: int, world_size: int, tensor_size: int,
                              executor: str, pipeline_depth: int):
    """Ring allreduce with step-level pipelining via Pipeline construct."""
    chunk_per_rank = tensor_size // world_size
    name = f"ar_pipe{pipeline_depth}_{executor}_rank{rank}"
    use_tma = (executor == "tma")

    with CommunicationOp(name=name, device=DeviceType.CUDA) as op:
        dtype = "bfloat16" if _ARGS.dtype == "bf16" else "float32"
        op.tensor(dtype=dtype, shape=(tensor_size,))
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size

        pipe = Pipeline(depth=pipeline_depth)
        sig_counter = 0

        with pipe.bind(op):
            # Initial barrier — one per pipeline channel
            for c in range(pipeline_depth):
                with pipe.stage():
                    sig_counter += 1
                    op.notify(signal_id=sig_counter, target_rank=next_rank)
                    op.wait_notify(signal_id=sig_counter, source_rank=prev_rank)

            # Reduce-scatter steps
            for step in range(world_size - 1):
                with pipe.stage():
                    chunk_idx = (rank - step - 1) % world_size
                    off = chunk_idx * chunk_per_rank
                    if step > 0:
                        sig_counter += 1
                        op.wait_notify(signal_id=sig_counter, source_rank=prev_rank)
                    if use_tma:
                        op.tma_reduce(reduce_op="sum", source_rank=prev_rank,
                                      src_offset=off, dst_offset=off,
                                      remote_offset=off, count=chunk_per_rank)
                    else:
                        op.sm_reduce(reduce_op="sum", source_rank=prev_rank,
                                     src_offset=off, dst_offset=off,
                                     remote_offset=off, count=chunk_per_rank)
                    sig_counter += 1
                    op.notify(signal_id=sig_counter, target_rank=next_rank)

            # Allgather steps
            for step in range(world_size - 1):
                with pipe.stage():
                    chunk_idx = (rank - step) % world_size
                    off = chunk_idx * chunk_per_rank
                    sig_counter += 1
                    op.wait_notify(signal_id=sig_counter, source_rank=prev_rank)
                    if use_tma:
                        op.tma_copy(source_rank=prev_rank,
                                    src_offset=off, dst_offset=off,
                                    size=chunk_per_rank)
                    else:
                        op.sm_copy(source_rank=prev_rank,
                                   src_offset=off, dst_offset=off,
                                   size=chunk_per_rank)
                    sig_counter += 1
                    op.notify(signal_id=sig_counter, target_rank=next_rank)

            # Final drain — wait for last signal on each channel
            for c in range(pipeline_depth):
                with pipe.stage():
                    sig_counter += 1
                    op.wait_notify(signal_id=sig_counter, source_rank=prev_rank)

        return op.get_graph()


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def fmt_size(nbytes):
    if nbytes >= 1 << 30: return f"{nbytes / (1 << 30):.1f}GB"
    if nbytes >= 1 << 20: return f"{nbytes / (1 << 20):.0f}MB"
    if nbytes >= 1 << 10: return f"{nbytes / (1 << 10):.0f}KB"
    return f"{nbytes}B"


def bench_pccl(label, graph_fn, rank, world_size, sizes_bytes,
               dtype, elem_bytes, warmup, iters, output_dir,
               superopt=False, num_channels=1):
    """Benchmark a PCCL graph variant across all sizes."""
    import pccl.engine
    results = []
    for data_bytes in sizes_bytes:
        tensor_size = data_bytes // elem_bytes
        tensor_size = (tensor_size // world_size) * world_size
        data_bytes = tensor_size * elem_bytes

        op_name = f"{label}_{data_bytes}"
        graph = graph_fn(rank, world_size, tensor_size)
        json_file = str(output_dir / f"{op_name}_rank{rank}.json")
        if superopt:
            compiler = Compiler(
                enable_superopt=True,
                enable_channelize=(num_channels > 1),
                num_channels=num_channels,
                topology=NVLINK_TOPOLOGY,
                device_profile=H100_PROFILE,
                data_size_hint=data_bytes,
            )
            compiler.compile_to_json(graph, json_file)
        else:
            compile_to_json_file(graph, json_file)

        success = pccl.engine.register_operation(op_name, json_file)
        if not success:
            if rank == 0: print(f"  SKIP {fmt_size(data_bytes):>8s} - register failed")
            continue

        dist.barrier()
        inp = torch.ones(tensor_size, dtype=dtype, device=f"cuda:{rank}") * (rank + 1)
        out = torch.zeros(tensor_size, dtype=dtype, device=f"cuda:{rank}")

        for _ in range(warmup):
            pccl.engine.execute_operation_async(op_name, inp, out)
            pccl.engine.sync_operation(op_name)

        dist.barrier()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            pccl.engine.execute_operation_async(op_name, inp, out)
            pccl.engine.sync_operation(op_name)
        t1.record()
        torch.cuda.synchronize()

        ms = t0.elapsed_time(t1)
        avg_us = (ms / iters) * 1e3
        alg_bw = data_bytes / (ms / iters / 1e3) / 1e9
        bus_bw = alg_bw * 2 * (world_size - 1) / world_size

        results.append((data_bytes, avg_us, alg_bw, bus_bw))
        pccl.engine.reset_signals(op_name)
        dist.barrier()

    return results


def bench_nccl(rank, world_size, sizes_bytes, dtype, elem_bytes, warmup, iters):
    """Benchmark NCCL allreduce for reference."""
    results = []
    for data_bytes in sizes_bytes:
        tensor_size = data_bytes // elem_bytes
        tensor_size = (tensor_size // world_size) * world_size
        data_bytes = tensor_size * elem_bytes

        inp = torch.ones(tensor_size, dtype=dtype, device=f"cuda:{rank}") * (rank + 1)

        for _ in range(warmup):
            dist.all_reduce(inp.clone(), op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        dist.barrier()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            dist.all_reduce(inp.clone(), op=dist.ReduceOp.SUM)
        t1.record()
        torch.cuda.synchronize()

        ms = t0.elapsed_time(t1)
        avg_us = (ms / iters) * 1e3
        alg_bw = data_bytes / (ms / iters / 1e3) / 1e9
        bus_bw = alg_bw * 2 * (world_size - 1) / world_size

        results.append((data_bytes, avg_us, alg_bw, bus_bw))
        dist.barrier()

    return results


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

_ARGS = None  # populated by parse_args


def run_bench(rank: int, world_size: int):
    args = _ARGS
    import pccl.engine
    pccl.engine.get_engine()
    dist.barrier()
    pccl.engine.initialize_engine(dist.group.WORLD)
    dist.barrier()

    # Apply perf env vars
    if args.disable_fused:
        os.environ["PCCL_DISABLE_FUSED"] = "1"

    # Build size sweep
    if args.sizes:
        sizes_bytes = [int(s) for s in args.sizes.split(",")]
    else:
        sizes_bytes = []
        b = args.min_size
        while b <= args.max_size:
            sizes_bytes.append(b)
            b *= 2

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    elem_bytes = 2 if args.dtype == "bf16" else 4

    output_dir = Path(__file__).parent / "generated_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine channel configs to test
    all_results = {}

    # Pipeline mode
    if args.pipeline_depth > 1:
        label = f"ar_{args.executor}_pipe{args.pipeline_depth}"
        graph_fn = lambda r, ws, ts: ring_allreduce(
            r, ws, ts, executor=args.executor,
            pipeline_depth=args.pipeline_depth)
        all_results[label] = bench_pccl(
            label, graph_fn, rank, world_size, sizes_bytes,
            dtype, elem_bytes, args.warmup, args.iters, output_dir,
            superopt=args.superopt)
    elif args.algorithm != "ring":
        # Algorithm template mode
        def _make_graph_fn(alg_name, exec_name):
            def _fn(r, ws, ts):
                if alg_name == "auto":
                    alg = select_algorithm(ws, ts * elem_bytes)
                else:
                    alg = ALGORITHMS[alg_name]()
                return alg.build_allreduce(
                    rank=r, world_size=ws, tensor_size=ts,
                    dtype="bfloat16" if args.dtype == "bf16" else "float32",
                    executor=exec_name)
            return _fn
        label = f"ar_{args.algorithm}_{args.executor}"
        graph_fn = _make_graph_fn(args.algorithm, args.executor)
        all_results[label] = bench_pccl(
            label, graph_fn, rank, world_size, sizes_bytes,
            dtype, elem_bytes, args.warmup, args.iters, output_dir,
            superopt=args.superopt)
    else:
        # Channel configs (original path)
        channel_configs = [int(c) for c in args.channels.split(",")]
        for nc in channel_configs:
            label = f"ar_{args.executor}_c{nc}"
            graph_fn = lambda r, ws, ts, _nc=nc: ring_allreduce(
                r, ws, ts, num_channels=_nc, executor=args.executor)
            all_results[label] = bench_pccl(
                label, graph_fn, rank, world_size, sizes_bytes,
                dtype, elem_bytes, args.warmup, args.iters, output_dir,
                superopt=args.superopt, num_channels=nc)

    nccl_results = None
    if not args.no_nccl:
        nccl_results = bench_nccl(rank, world_size, sizes_bytes,
                                  dtype, elem_bytes, args.warmup, args.iters)

    # Print results
    if rank == 0:
        fused_str = "DAG" if args.disable_fused else "fused"
        opt_str = "superopt" if args.superopt else "no-opt"
        alg_str = args.algorithm
        print(f"\n{'=' * 120}")
        print(f"  PCCL AllReduce Benchmark  |  {world_size} GPUs  |  "
              f"executor={args.executor}  kernel={fused_str}  "
              f"channels={args.channels}  algorithm={alg_str}  opt={opt_str}  "
              f"dtype={args.dtype}  iters={args.iters}")
        print(f"{'=' * 120}")

        # Header
        hdr = f"{'Size':>8s} |"
        for label in all_results:
            short = label.replace(f"ar_{args.executor}_", "")
            hdr += f" {short+' lat':>11s} {'bus_bw':>9s} |"
        if nccl_results:
            hdr += f" {'NCCL lat':>11s} {'bus_bw':>9s} | {'vs NCCL':>7s}"
        print(hdr)
        print(f"{'-' * 120}")

        labels = list(all_results.keys())
        n_sizes = len(all_results[labels[0]]) if labels else 0
        for i in range(n_sizes):
            first = all_results[labels[0]]
            db = first[i][0]
            line = f"{fmt_size(db):>8s} |"
            best_bb = 0
            for label in labels:
                _, us, _, bb = all_results[label][i]
                line += f" {us:10.1f}us {bb:8.1f}G |"
                best_bb = max(best_bb, bb)
            if nccl_results:
                _, nccl_us, _, nccl_bb = nccl_results[i]
                ratio = best_bb / nccl_bb if nccl_bb > 0 else 0
                line += f" {nccl_us:10.1f}us {nccl_bb:8.1f}G | {ratio:6.2f}x"
            print(line)

        print(f"{'=' * 120}")

    dist.barrier()


def worker(rank: int, world_size: int, master_port: int, args_ns):
    global _ARGS
    _ARGS = args_ns
    os.environ.update({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(master_port),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "NCCL_NVLS_ENABLE": "0",
    })
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", rank=rank, world_size=world_size)
    run_bench(rank, world_size)
    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="PCCL AllReduce Bandwidth Benchmark")
    p.add_argument("--nproc", type=int, default=2, help="Number of GPUs (spawn mode)")
    p.add_argument("--port", type=int, default=29500)
    p.add_argument("--torchrun", action="store_true", help="Use torchrun mode")

    # Performance options
    p.add_argument("--executor", choices=["tma", "sm"], default="tma",
                   help="Executor type: tma (default) or sm")
    p.add_argument("--channels", default="1",
                   help="Comma-separated channel counts to test (default: 1)")
    p.add_argument("--pipeline-depth", type=int, default=1,
                   help="Pipeline depth (>1 enables step-level overlap, default: 1)")
    p.add_argument("--disable-fused", action="store_true",
                   help="Disable fused step executor, use DAG kernel")
    p.add_argument("--no-nccl", action="store_true",
                   help="Skip NCCL reference benchmark")
    p.add_argument("--superopt", action="store_true",
                   help="Enable e-graph superoptimizer (executor upgrades + structural dedup)")
    p.add_argument("--algorithm", choices=["ring", "rhd", "tree", "auto"], default="ring",
                   help="Allreduce algorithm: ring (default), rhd (recursive halving-doubling), "
                        "tree (binary tree), auto (topology-aware selection)")

    # Size options
    p.add_argument("--sizes", default=None,
                   help="Comma-separated sizes in bytes (overrides min/max)")
    p.add_argument("--min-size", type=int, default=4 * 1024,
                   help="Minimum size in bytes for auto sweep (default: 4KB)")
    p.add_argument("--max-size", type=int, default=512 * 1024 * 1024,
                   help="Maximum size in bytes for auto sweep (default: 512MB)")

    # Iteration options
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    p.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    p.add_argument("--dtype", choices=["bf16", "f32"], default="bf16",
                   help="Data type (default: bf16)")

    return p.parse_args()


def main():
    global _ARGS
    _ARGS = parse_args()

    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if _ARGS.torchrun or "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="cuda:nccl,cpu:gloo")
        run_bench(rank, world_size)
        dist.destroy_process_group()
    else:
        print(f"Spawning {_ARGS.nproc} workers (port={_ARGS.port})")
        mp.spawn(worker, args=(_ARGS.nproc, _ARGS.port, _ARGS),
                 nprocs=_ARGS.nproc, join=True)


if __name__ == "__main__":
    main()
