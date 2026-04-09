"""
PCCL Algorithm Comparison Benchmark

Compares ring, recursive halving-doubling (RHD), tree, auto-selected
algorithms, and optionally NVLS against NCCL across different GPU counts.
Supports multiple collective types: allreduce, reduce_scatter, allgather, alltoall.

Usage:
    python tests/bench_algo_comparison.py --nproc=2,4,8
    python tests/bench_algo_comparison.py --nproc=8 --executor=sm
    python tests/bench_algo_comparison.py --nproc=2,4,8 --sizes=65536,1048576,67108864,536870912
    python tests/bench_algo_comparison.py --nproc=2,8 --superopt --no-nccl
    python tests/bench_algo_comparison.py --nproc=2 --no-nvls
    python tests/bench_algo_comparison.py --nproc=2,4,8 --collective=reduce_scatter
    python tests/bench_algo_comparison.py --nproc=8 --collective=allgather,alltoall
"""

import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pccl import DeviceType, compile_to_json_file, Compiler
from pccl.dsl.superopt import NVLINK_TOPOLOGY, H100_PROFILE
from pccl.dsl.algorithms import ALGORITHMS, select_algorithm

_ARGS = None


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _eligible_algos(world_size: int) -> list[str]:
    algos = ["ring"]
    if _is_power_of_2(world_size):
        algos += ["rhd", "tree"]
    algos.append("auto")
    return algos


def _build_graph(algo_name, rank, world_size, tensor_size, dtype, executor,
                 collective="allreduce"):
    if algo_name == "auto":
        elem_bytes = 2 if dtype == "bfloat16" else 4
        alg = select_algorithm(world_size, tensor_size * elem_bytes)
    elif algo_name == "nvls":
        alg = ALGORITHMS["ring"]()
    else:
        alg = ALGORITHMS[algo_name]()

    build_fn = {
        "allreduce": alg.build_allreduce,
        "reduce_scatter": alg.build_reduce_scatter,
        "allgather": alg.build_allgather,
        "alltoall": alg.build_alltoall,
    }[collective]
    return build_fn(
        rank=rank, world_size=world_size, tensor_size=tensor_size,
        dtype=dtype, executor=executor)


def fmt_size(b: int) -> str:
    if b >= 1024**3: return f"{b / 1024**3:.0f}GB"
    if b >= 1024**2: return f"{b / 1024**2:.0f}MB"
    if b >= 1024:    return f"{b / 1024:.0f}KB"
    return f"{b}B"


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def bench_pccl_algo(algo_name, rank, world_size, sizes_bytes,
                    dtype_torch, elem_bytes, warmup, iters,
                    output_dir, executor, dtype_str, superopt,
                    collective="allreduce"):
    import pccl.engine
    results = []
    dtype_dsl = "bfloat16" if dtype_str == "bf16" else "float32"

    for data_bytes in sizes_bytes:
        tensor_size = data_bytes // elem_bytes
        tensor_size = (tensor_size // world_size) * world_size
        data_bytes = tensor_size * elem_bytes

        op_name = f"cmp_{collective}_{algo_name}_{world_size}g_{data_bytes}"
        try:
            graph = _build_graph(algo_name, rank, world_size,
                                 tensor_size, dtype_dsl, executor,
                                 collective=collective)
        except (ValueError, NotImplementedError):
            results.append((data_bytes, float("inf"), 0.0, 0.0))
            continue

        json_file = str(output_dir / f"{op_name}_rank{rank}.json")
        if superopt:
            compiler = Compiler(
                enable_superopt=True,
                topology=NVLINK_TOPOLOGY,
                device_profile=H100_PROFILE,
                data_size_hint=data_bytes,
            )
            compiler.compile_to_json(graph, json_file)
        else:
            compile_to_json_file(graph, json_file)

        success = pccl.engine.register_operation(op_name, json_file)
        if not success:
            results.append((data_bytes, float("inf"), 0.0, 0.0))
            continue

        dist.barrier()

        if collective == "allgather":
            chunk_size = tensor_size // world_size
            inp = torch.ones(chunk_size, dtype=dtype_torch, device=f"cuda:{rank}") * (rank + 1)
            out = torch.zeros(tensor_size, dtype=dtype_torch, device=f"cuda:{rank}")
        elif collective == "reduce_scatter":
            chunk_size = tensor_size // world_size
            inp = torch.ones(tensor_size, dtype=dtype_torch, device=f"cuda:{rank}") * (rank + 1)
            out = torch.zeros(chunk_size, dtype=dtype_torch, device=f"cuda:{rank}")
        else:
            inp = torch.ones(tensor_size, dtype=dtype_torch, device=f"cuda:{rank}") * (rank + 1)
            out = torch.zeros(tensor_size, dtype=dtype_torch, device=f"cuda:{rank}")

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
        if collective == "allreduce":
            bus_bw = alg_bw * 2 * (world_size - 1) / world_size
        else:
            bus_bw = alg_bw * (world_size - 1) / world_size

        results.append((data_bytes, avg_us, alg_bw, bus_bw))
        pccl.engine.reset_signals(op_name)
        dist.barrier()

    return results


def bench_nccl(rank, world_size, sizes_bytes, dtype_torch, elem_bytes, warmup, iters,
               collective="allreduce"):
    results = []
    for data_bytes in sizes_bytes:
        tensor_size = data_bytes // elem_bytes
        tensor_size = (tensor_size // world_size) * world_size
        data_bytes = tensor_size * elem_bytes

        inp = torch.ones(tensor_size, dtype=dtype_torch, device=f"cuda:{rank}") * (rank + 1)

        if collective == "allreduce":
            def run():
                dist.all_reduce(inp.clone(), op=dist.ReduceOp.SUM)
            bus_factor = 2 * (world_size - 1) / world_size
        elif collective == "reduce_scatter":
            chunk = tensor_size // world_size
            out = torch.zeros(chunk, dtype=dtype_torch, device=f"cuda:{rank}")
            def run():
                dist.reduce_scatter_tensor(out, inp.clone(), op=dist.ReduceOp.SUM)
            bus_factor = (world_size - 1) / world_size
        elif collective == "allgather":
            chunk = tensor_size // world_size
            inp_chunk = inp[:chunk].contiguous()
            out = torch.zeros(tensor_size, dtype=dtype_torch, device=f"cuda:{rank}")
            def run():
                dist.all_gather_into_tensor(out, inp_chunk)
            bus_factor = (world_size - 1) / world_size
        elif collective == "alltoall":
            out = torch.zeros(tensor_size, dtype=dtype_torch, device=f"cuda:{rank}")
            def run():
                dist.all_to_all_single(out, inp.clone())
            bus_factor = (world_size - 1) / world_size
        else:
            raise ValueError(f"Unknown collective: {collective}")

        for _ in range(warmup):
            run()
        torch.cuda.synchronize()

        dist.barrier()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            run()
        t1.record()
        torch.cuda.synchronize()

        ms = t0.elapsed_time(t1)
        avg_us = (ms / iters) * 1e3
        alg_bw = data_bytes / (ms / iters / 1e3) / 1e9
        bus_bw = alg_bw * bus_factor
        results.append((data_bytes, avg_us, alg_bw, bus_bw))
        dist.barrier()

    return results


# ---------------------------------------------------------------------------
# Per-world-size driver
# ---------------------------------------------------------------------------

def _print_table(world_size, algos, algo_results, nccl_results, args,
                 nvls_results=None):
    opt_str = "superopt" if args.superopt else "no-opt"
    collective = getattr(args, "collective", "allreduce")
    extra_cols = (1 if nvls_results else 0) + (1 if nccl_results else 0)
    W = 24 * len(algos) + 24 * extra_cols + 30

    sep = "=" * W

    print(f"\n{sep}")
    print(f"  {collective.upper()} Comparison  |  {world_size} GPUs  |  "
          f"executor={args.executor}  opt={opt_str}  dtype={args.dtype}  "
          f"warmup={args.warmup}  iters={args.iters}")
    print(sep)

    hdr = f"{'Size':>8s} |"
    for algo in algos:
        hdr += f"  {algo.upper():>4s} lat   bus_bw |"
    if nvls_results:
        hdr += f"  NVLS lat   bus_bw |"
    if nccl_results:
        if algos:
            hdr += f"  NCCL lat   bus_bw | Best  ratio"
        else:
            hdr += f"  NCCL lat   bus_bw |"
    print(hdr)
    print("-" * W)

    n_sizes = len(nccl_results) if nccl_results else len(algo_results[algos[0]]) if algos else 0
    for i in range(n_sizes):
        if algos:
            db = algo_results[algos[0]][i][0]
        elif nccl_results:
            db = nccl_results[i][0]
        else:
            break
        line = f"{fmt_size(db):>8s} |"

        best_algo, best_bw = "", 0.0
        for algo in algos:
            _, us, _, bb = algo_results[algo][i]
            if us == float("inf"):
                line += f"  {'N/A':>8s} {'N/A':>7s} |"
            else:
                line += f" {us:8.1f}us {bb:6.1f}G |"
                if bb > best_bw:
                    best_bw = bb
                    best_algo = algo

        if nvls_results:
            _, nvls_us, _, nvls_bb = nvls_results[i]
            line += f" {nvls_us:8.1f}us {nvls_bb:6.1f}G |"
            if nvls_bb > best_bw:
                best_bw = nvls_bb
                best_algo = "nvls"

        if nccl_results:
            _, nccl_us, _, nccl_bb = nccl_results[i]
            if algos:
                ratio = best_bw / nccl_bb if nccl_bb > 0 else 0
                line += f" {nccl_us:8.1f}us {nccl_bb:6.1f}G | {best_algo:<5s} {ratio:.2f}x"
            else:
                line += f" {nccl_us:8.1f}us {nccl_bb:6.1f}G |"

        print(line)

    print(sep)


def run_comparison(rank: int, world_size: int):
    args = _ARGS
    collective = getattr(args, "collective", "allreduce")

    import pccl.engine
    pccl.engine.get_engine()
    dist.barrier()
    pccl.engine.initialize_engine(dist.group.WORLD)
    dist.barrier()

    if args.disable_fused:
        os.environ["PCCL_DISABLE_FUSED"] = "1"

    if args.sizes:
        sizes_bytes = [int(s) for s in args.sizes.split(",")]
    else:
        sizes_bytes = []
        b = args.min_size
        while b <= args.max_size:
            sizes_bytes.append(b)
            b *= 2

    dtype_torch = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    elem_bytes = 2 if args.dtype == "bf16" else 4

    output_dir = Path(__file__).parent / "generated_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    algos = _eligible_algos(world_size)
    if collective != "allreduce":
        algos = ["ring"]

    algo_results = {}
    for algo in algos:
        if rank == 0:
            print(f"  [{world_size}GPU] benchmarking PCCL {algo.upper()} {collective} ...",
                  flush=True)
        algo_results[algo] = bench_pccl_algo(
            algo, rank, world_size, sizes_bytes, dtype_torch, elem_bytes,
            args.warmup, args.iters, output_dir, args.executor, args.dtype,
            args.superopt, collective=collective)

    nccl_results = None
    if not args.no_nccl:
        if rank == 0:
            print(f"  [{world_size}GPU] benchmarking NCCL {collective} ...", flush=True)
        nccl_results = bench_nccl(rank, world_size, sizes_bytes,
                                  dtype_torch, elem_bytes, args.warmup, args.iters,
                                  collective=collective)

    if rank == 0:
        nvls_results = None
        if collective == "allreduce":
            nvls_results = _load_nvls_results(world_size, output_dir, sizes_bytes)
        _print_table(world_size, algos, algo_results, nccl_results, args,
                     nvls_results=nvls_results)
        _save_results(world_size, algos, algo_results, nccl_results, output_dir,
                      nvls_results=nvls_results)

    dist.barrier()


def _save_results(world_size, algos, algo_results, nccl_results, output_dir,
                  nvls_results=None):
    n_sizes = (len(algo_results[algos[0]]) if algos
               else len(nccl_results) if nccl_results else 0)
    summary = {"world_size": world_size, "sizes": []}
    for i in range(n_sizes):
        if algos:
            entry = {"data_bytes": algo_results[algos[0]][i][0]}
        elif nccl_results:
            entry = {"data_bytes": nccl_results[i][0]}
        else:
            break
        for algo in algos:
            _, us, ab, bb = algo_results[algo][i]
            entry[algo] = {"latency_us": us, "alg_bw": ab, "bus_bw": bb}
        if nvls_results:
            _, us, ab, bb = nvls_results[i]
            entry["nvls"] = {"latency_us": us, "alg_bw": ab, "bus_bw": bb}
        if nccl_results:
            _, us, ab, bb = nccl_results[i]
            entry["nccl"] = {"latency_us": us, "alg_bw": ab, "bus_bw": bb}
        summary["sizes"].append(entry)
    result_file = output_dir / f"comparison_{world_size}gpu.json"
    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2)


def _load_nvls_results(world_size, output_dir, sizes_bytes):
    """Load NVLS results saved by the separate worker_nvls process."""
    nvls_file = output_dir / f"nvls_{world_size}gpu.json"
    if not nvls_file.exists():
        return None
    with open(nvls_file) as f:
        data = json.load(f)
    by_size = {d["data_bytes"]: d for d in data}
    results = []
    for sb in sizes_bytes:
        if sb in by_size:
            d = by_size[sb]
            results.append((d["data_bytes"], d["latency_us"],
                            d["alg_bw"], d["bus_bw"]))
        else:
            results.append((sb, float("inf"), 0.0, 0.0))
    return results


# ---------------------------------------------------------------------------
# Cross-GPU summary
# ---------------------------------------------------------------------------

def _print_cross_gpu_summary(gpu_counts, output_dir):
    summaries = {}
    for ws in gpu_counts:
        path = output_dir / f"comparison_{ws}gpu.json"
        if path.exists():
            with open(path) as f:
                summaries[ws] = json.load(f)

    if len(summaries) < 2:
        return

    W = 28 * len(summaries) + 15
    sep = "=" * W
    print(f"\n{sep}")
    print(f"  Cross-GPU Summary  |  Best PCCL bus_bw vs NCCL per size")
    print(sep)

    hdr = f"{'Size':>8s} |"
    for ws in summaries:
        hdr += f"  ----- {ws} GPUs ------  |"
    print(hdr)

    sub = f"{'':>8s} |"
    for _ in summaries:
        sub += f"  {'best':>5s} {'bw(G)':>6s} {'ratio':>6s}  |"
    print(sub)
    print("-" * W)

    first_ws = list(summaries.keys())[0]
    for entry in summaries[first_ws]["sizes"]:
        db = entry["data_bytes"]
        line = f"{fmt_size(db):>8s} |"
        for ws, s in summaries.items():
            row = next((e for e in s["sizes"] if e["data_bytes"] == db), None)
            if not row:
                line += f"  {'':>5s} {'N/A':>6s} {'':>6s}  |"
                continue
            best_algo, best_bw = "", 0.0
            for algo in ["ring", "rhd", "tree", "auto", "nvls"]:
                if algo in row:
                    bb = row[algo]["bus_bw"]
                    if bb > best_bw:
                        best_bw = bb
                        best_algo = algo
            nccl_bw = row.get("nccl", {}).get("bus_bw", 0)
            ratio = best_bw / nccl_bw if nccl_bw > 0 else 0
            line += f"  {best_algo:>5s} {best_bw:5.1f}G {ratio:5.2f}x  |"
        print(line)

    print(sep)


# ---------------------------------------------------------------------------
# Worker / process management
# ---------------------------------------------------------------------------

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
    dist.init_process_group(backend="cuda:nccl,cpu:gloo",
                            rank=rank, world_size=world_size)
    run_comparison(rank, world_size)
    dist.destroy_process_group()


def worker_nvls(rank: int, world_size: int, master_port: int, args_ns):
    """Separate worker that initialises the engine with NVLS enabled.

    NVLS activation is checked once at engine init time, so a fresh
    process is required to toggle it independently of the main run.
    """
    global _ARGS
    _ARGS = args_ns
    os.environ.update({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(master_port),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "NCCL_NVLS_ENABLE": "0",
        "PCCL_NVLS_ENABLE": "1",
    })
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo",
                            rank=rank, world_size=world_size)
    _run_nvls_comparison(rank, world_size)
    dist.destroy_process_group()


def _run_nvls_comparison(rank: int, world_size: int):
    """Benchmark NVLS using a ring allreduce graph (collective_type=allreduce
    triggers the NVLS kernel path when the manager is initialised)."""
    args = _ARGS
    import pccl.engine
    pccl.engine.get_engine()
    dist.barrier()
    pccl.engine.initialize_engine(dist.group.WORLD)
    dist.barrier()

    if args.sizes:
        sizes_bytes = [int(s) for s in args.sizes.split(",")]
    else:
        sizes_bytes = []
        b = args.min_size
        while b <= args.max_size:
            sizes_bytes.append(b)
            b *= 2

    dtype_torch = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    elem_bytes = 2 if args.dtype == "bf16" else 4

    output_dir = Path(__file__).parent / "generated_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"  [{world_size}GPU] benchmarking NVLS ...", flush=True)

    results = bench_pccl_algo(
        "nvls", rank, world_size, sizes_bytes, dtype_torch, elem_bytes,
        args.warmup, args.iters, output_dir, args.executor, args.dtype,
        args.superopt)

    if rank == 0:
        nvls_file = output_dir / f"nvls_{world_size}gpu.json"
        data = []
        for db, us, ab, bb in results:
            data.append({"data_bytes": db, "latency_us": us,
                         "alg_bw": ab, "bus_bw": bb})
        with open(nvls_file, "w") as f:
            json.dump(data, f, indent=2)

    dist.barrier()


def parse_args():
    p = argparse.ArgumentParser(
        description="PCCL Algorithm Comparison Benchmark")
    p.add_argument("--nproc", default="2,4,8",
                   help="Comma-separated GPU counts to test (default: 2,4,8)")
    p.add_argument("--port", type=int, default=29500)

    p.add_argument("--executor", choices=["tma", "sm"], default="tma")
    p.add_argument("--collective", default="allreduce",
                   help="Collective type(s) to benchmark, comma-separated "
                        "(allreduce,reduce_scatter,allgather,alltoall)")
    p.add_argument("--disable-fused", action="store_true",
                   help="Disable fused step executor (use DAG kernel)")
    p.add_argument("--no-nccl", action="store_true",
                   help="Skip NCCL reference benchmark")
    p.add_argument("--no-nvls", action="store_true",
                   help="Skip NVLS benchmark")
    p.add_argument("--superopt", action="store_true",
                   help="Enable e-graph superoptimizer")

    p.add_argument("--sizes", default=None,
                   help="Comma-separated sizes in bytes")
    p.add_argument("--min-size", type=int, default=4 * 1024,
                   help="Min size for auto sweep (default: 4KB)")
    p.add_argument("--max-size", type=int, default=512 * 1024 * 1024,
                   help="Max size for auto sweep (default: 512MB)")

    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    return p.parse_args()


def main():
    args = parse_args()
    gpu_counts = [int(x) for x in args.nproc.split(",")]
    collectives = [c.strip() for c in args.collective.split(",")]
    num_gpus = torch.cuda.device_count()
    gpu_counts = [ws for ws in gpu_counts if ws <= num_gpus]

    if not gpu_counts:
        print(f"No valid GPU counts. Available GPUs: {num_gpus}")
        return

    print(f"PCCL Algorithm Comparison Benchmark")
    print(f"GPU counts to test: {gpu_counts}  |  Available GPUs: {num_gpus}")
    print(f"Collectives: {collectives}")

    output_dir = Path(__file__).parent / "generated_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    port_offset = 0
    for coll in collectives:
        args.collective = coll
        print(f"\n{'='*60}")
        print(f"  Collective: {coll.upper()}")
        print(f"{'='*60}")

        for ws in gpu_counts:
            port = args.port + port_offset
            port_offset += 2

            if coll == "allreduce" and not args.no_nvls:
                nvls_port = port + 1
                print(f"\n>>> Starting {ws}-GPU NVLS benchmark (port {nvls_port}) ...")
                mp.spawn(worker_nvls, nprocs=ws, args=(ws, nvls_port, args))

            print(f"\n>>> Starting {ws}-GPU {coll} comparison (port {port}) ...")
            mp.spawn(worker, nprocs=ws, args=(ws, port, args))

    if "allreduce" in collectives:
        _print_cross_gpu_summary(gpu_counts, output_dir)


if __name__ == "__main__":
    main()
