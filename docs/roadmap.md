# PCCL Runtime Optimization Roadmap

## Hardware

- 8× H100 80GB HBM3, NV18 (18-link NVLink 4.0, 900 GB/s bidirectional per GPU)
- NVSwitch 3.0: full-bisection topology (every GPU has direct link to every other)
- HBM3 bandwidth ~3.35 TB/s per GPU
- 450 GB/s unidirectional NVLink peak per GPU (shared across all sub-links)

---

## Completed Phases

### Phase 1: TMA Tile Size + Pipeline Wait Reorder (DONE)

- Adaptive TMA tile sizing: f32 tile_outer 64→127 (~63KB/buffer), f16/bf16 tile_outer→256 (64KB)
- Pipeline wait reorder: moved `cp_async_bulk_wait_all()` before compute (overlap store with next load)
- **Result**: 130.4 → 136.7 GB/s at 512MB/2GPU (+5%)

### Phase 2: Output D2D Copy Elimination (DONE)

- TMA double-write: store to both `self_desc` and `output_desc` from same shared memory tile
- Removed host-side output `cudaMemcpyAsync`; input D2D copy retained (peers need `self_addr[1]`)
- Key insight: TMA stores go through DMA engine — second store per tile adds near-zero SM overhead
- **Result**: 136.7 → 158.1 GB/s at 512MB/2GPU (+16%)

### Phase 3: Multi-Channel Infrastructure (DONE)

- Per-channel state: `ChannelState` struct with `alignas(128)`, per-channel ready queues
- Block-partitioned kernel: `blocks_per_ch = gridDim.x / num_ch`
- DSL: `set_channel()` method, `Stream` per channel for independent dependency chains
- False sharing fix: `ChannelState` alignment to 128B cache lines
- **Result**: Infrastructure works correctly. No bandwidth gain from same-topology multi-channel ring (all channels contend for same links). 8GPU: 155→157.5 GB/s (1ch→4ch).

### Phase 3.5: Runtime Micro-Optimizations (DONE)

- Replaced `atomicAdd(&x, 0)` read pattern with volatile reads in persistent kernel hot loop (dequeue, completion check, active_valid polling) — avoids expensive L2 atomic pipeline round-trips
- Removed unnecessary `__syncthreads()` in `executeTmaCopy` TMA pipeline (between two thread-0-only operations)
- Removed redundant `__threadfence()` in `dequeueFromRingBuffer` (covered by later fence before `active_valid` publish)
- Added `__nanosleep(64)` backoff to busy-wait spin loops (main loop idle, signal wait) — reduces power/thermal throttling
- Reorganized `DeviceWorkspace` struct: per-call fields first (128B), static fields after. Cached ops copy 128B instead of 4KB per call (32× reduction)
- **Result**: No large-message regression. Small-message latency improved ~6% at 8GPU (e.g., 64KB: 395→371µs). 512MB: 155.3→156.3 GB/s (1ch), 157.5→158.5 GB/s (4ch).

### Phase 7: Host-Side Overhead Reduction (DONE)

- Merged setup_kernel into executor kernel: block-0 setup prologue + flag-based grid barrier, eliminating one kernel launch per call
- Selective H2D workspace copy: 128B per-call (done in Phase 3.5)
- **Result**: 8GPU 64KB latency 371→359µs (-3%). 2GPU 64KB latency 77µs. No large-message regression.

### Phase 9: Eliminate Chunk Fragmentation for TMA (DONE)

- Static per-block tile partitioning for TMA ops: each block claims slot via `atomicAdd`, computes contiguous tile range `(slot * total_elems) / blocks_per_ch`
- Re-entry prevention via claim counter (slot >= blocks_per_ch → skip)
- SM ops retain existing chunk-grab loop unchanged
- TMA reduce microbenchmark: 141.8 GB/s at 64 blocks (~75% of TMA copy 188.1 GB/s)
- **Result**: 8GPU 512MB: 156.6→198.0 GB/s 1ch (+26%), 158.9→202.1 GB/s 4ch (+27%). 2GPU 512MB: 158.6→192.3 GB/s (+21%). NCCL ratio 0.42×→0.54×.

### DSL Superoptimizer (INTEGRATED)

**Architecture**: SMT-verified offline rule discovery + e-graph equality saturation online + parametric cost extraction. The superopt pass converts IR subgraphs to e-graph nodes, applies domain rewrite rules (executor equivalences, structural dedup, fused-op introduction), and extracts the lowest-cost variant via critical-path DAG analysis with topology/device-aware cost parameters.

| Module | Role | Status |
|--------|------|--------|
| `enumerator.py` | k-op DAG skeleton generation (Bell partitions + channel assignments) | Working |
| `verifier.py` | Hybrid: concrete simulation (fast reject) + slot-based Z3 (formal proof) | Working |
| `cost_model.py` | Critical-path DAG latency, H100 profile, channel-aware, parametric | Working |
| `rule.py` | PatternNode/PatternEdge/RewriteRule with JSON serialization | Working |
| `rule_db.py` | Rule cache at `~/.pccl/superopt/` | Working |
| `pass_.py` | Compiler pass: e-graph-based subgraph optimization + structural rewrites | Working |
| `egraph.py` | E-graph with equality saturation + cost-based extraction | Working (active) |
| `egraph_bridge.py` | IR ↔ e-graph conversion (subgraph → ENode tree, extraction → IR) | Working |
| `domain_rules.py` | E-graph rewrite rules: executor equivalences, structural dedup, multimem | Working |
| `semantics.py` | Z3 Array-based formal semantics | Working |
| `channelize.py` | Auto-channelization pass (sequential chains → parallel channels) | Working |

Compiler pipeline: `DependencyAnalysis → DCE → [Superopt (e-graph)] → [Channelize] → DependencyAnalysis`. 84 superopt + 51 multichannel tests pass (135 total).

**What works end-to-end**: E-graph equality saturation with executor upgrades (SM→TMA, SM→multimem), structural dedup, parametric cost extraction, channel field propagation (IR→JSON→runtime), auto-channelization. Full pipeline validated: DSL graph → superopt → JSON v2 → RuntimeGraphGenerator.

**Algorithm search module** (`pccl/dsl/algorithms/`): Template library for collective algorithms — ring (O(2(N-1)), bandwidth-optimal), recursive halving-doubling (O(2 log N), Rabenseifner), binary tree (O(2 log N), latency-optimal). Topology-aware selector picks the best algorithm via heuristic or cost-model comparison. 34 algorithm tests pass.

### DSL-Runtime Alignment: Pipeline Construct (DONE)

Step-level pipelining support for multi-channel overlap.

**Pipeline construct**:
- `Pipeline` class (`pccl/dsl/pipeline.py`) with `bind()`/`stage()` context managers
- Automatic channel assignment (`channel = iteration % depth`) and `Stream` scoping for independent dependency chains
- Enables overlapping consecutive ring steps across channels without manual channel management

**Result**: 51 multichannel tests pass, including pipeline and noop functionality.

### Bug Fixes (DONE)

Three critical bugs discovered and fixed during multi-GPU testing:

**1. PERCALL_COPY_SIZE overflow (4-GPU hang)**
- Root cause: `PERCALL_COPY_SIZE = 232` exceeded the actual per-call section boundary in `DeviceWorkspace`, causing H2D copy to overwrite `ring_buffers_` and corrupt kernel state
- Fix: Reduced `PERCALL_COPY_SIZE` to `208` (verified via `offsetof(DeviceWorkspace, ring_buffers_)`) and added `static_assert` to prevent future regressions
- Symptom: TREE and RHD algorithms hung on iteration 2+ at 4 GPUs

**2. Signal-ID deadlock in RHD/tree algorithms**
- Root cause: RHD and tree algorithms used incrementing `signal_id` counters per rank, but the runtime maintains independent auto-incrementing signal counters for each `(source_rank, signal_id)` pair — unique IDs caused cross-rank mismatches
- Fix: All notify/wait operations use a fixed `signal_id = 0` instead of incrementing counters
- Symptom: Immediate deadlock at 4+ GPUs for tree and RHD algorithms

**3. Inconsistent kernel path selection via per-rank `isAllreduce()` inference**
- Root cause: `GraphBuilder::isAllreduce()` inferred allreduce status per-rank by checking for both reduce AND copy primitives. In tree@4GPU, only rank 2 (intermediate node) had both `tma.reduce` and `tma.copy` — different ranks selected different kernel paths, causing deadlock
- Fix: Added `collective_type` field end-to-end — from `PrimitiveIRGraph.collective_type` (DSL) → JSON `"collective_type": "allreduce"` → C++ `GraphBuilder::is_allreduce_` member. All ranks now consistently get the same allreduce status
- Files touched: `graph.py`, `json_generator.py`, `graph_builder.h`, `graph_builder.cc`, `engine_c.cc`, plus all algorithm files (`tree.py`, `ring.py`, `recursive_hd.py`)
- Symptom: tree@4GPU hung immediately due to inconsistent kernel path selection across ranks

### Architecture Refactoring (DONE)

Three-phase refactoring to establish a clean plugin-based architecture separating device-agnostic engine logic from device-specific execution.

**Phase A: Heterogeneous Device Support**
- Created `DevicePlugin`/`Channel` interface hierarchy (`include/plugins/device.h`)
- Implemented channel-based dispatch: SM, TMA, CE, MultiMem, RDMA channels each provide `parse()` for different execution backends
- Created centralized `config.h` with 20+ runtime constants
- Updated `graph_builder` to use channel-based dispatch
- Maintained backward compatibility while enabling future ROCm support

**Phase B: CUDA Logic Migration**
- Moved ~190 lines of NVLS CUDA driver API code from `engine_c.cc` to new `NvlsManager` in CUDA plugin (`csrc/plugins/cuda/`)
- Moved fused descriptor CUDA allocation logic to `CudaDevice::uploadFusedDescriptor`
- Renamed `csrc/plugins/cuda/template/` to `csrc/plugins/cuda/kernel/` and updated all includes
- Consolidated plugin registry by removing redundant `device_map` in `base.cc`
- Moved `engine_c.h` to `include/engine/engine.h` with forwarding header
- Removed CUDA driver API calls and CUDA-specific includes from `engine_c.cc`

**Phase C: Host Executor Extraction**
- Moved `HostProxy` (RDMA) from `csrc/engine/` to `csrc/plugins/host/` with forwarding header at `include/engine/host_proxy.h`
- Created `CeProxy` (`csrc/plugins/host/ce_proxy.h`, `ce_proxy.cc`) — host-side copy-engine executor using pinned shared-memory queue, dedicated CUDA stream, and poll thread (same pattern as RDMA HostProxy)
- CE operations now route through host `cudaMemcpyAsync` instead of incorrectly executing on SM threads via `executeCopy()`
- Added `getCeProxyState()` to `DevicePlugin` interface; integrated into `HostDevice` with lazy start
- GPU kernel (`cuda_executor.cu`): added `isCEType()` helper, CE proxy enqueue/completion block, removed CE from `isCopyType()`

**Result**: Engine (`engine_c.cc`) is now device-agnostic — all CUDA/RDMA/CE specifics live in `csrc/plugins/cuda/` and `csrc/plugins/host/`. Clean plugin boundaries enable future device backends (ROCm, etc.) without engine changes.

**Post-refactoring verification**: Fixed static initialization order fiasco in plugin registry (`PluginRegistry` construct-on-first-use pattern). Consolidated test suite from 14 files to 7 (merged redundant superopt tests, removed obsolete profiling scripts). Benchmark tool (`bench_allreduce_tma_bw.py`) supports per-run executor/channel/fused/dtype sweeps. Performance verified at 2GPU and 8GPU — **no regression** from refactoring:

| Config | bus_bw (post-refactor) | bus_bw (pre-refactor) | vs NCCL |
|--------|----------------------:|---------------------:|--------:|
| 2GPU 512MB TMA 1ch | 246 GB/s | ~228 GB/s | 0.94× |
| 2GPU 128MB TMA 1ch | 229 GB/s | ~213 GB/s | 0.94× |
| 8GPU 512MB TMA 1ch | 282 GB/s | ~259 GB/s | 0.90× |
| 8GPU 32MB TMA 1ch | 185 GB/s | ~175 GB/s | 0.74× |

*Note: Performance improved by two optimisations: (1) NVLS kernel was auto-activated by
`collective_type: "allreduce"` — fixed to opt-in via `PCCL_NVLS_ENABLE=1`.
(2) Partial D2D output copy: fused kernel's TMA dual-write already places (N-1)/N of the
result in the output tensor; only the locally-reduced 1/N gap is copied, saving ~(N-1)/N
of the post-kernel memcpy.*

---

## Current Performance

Canonical benchmark: `tests/bench_algo_comparison.py` via `tests/run_algo_comparison.sh` (all algorithms, best per size, vs NCCL with `NCCL_NVLS_ENABLE=0`). NVLS disabled by default (opt-in via `PCCL_NVLS_ENABLE=1`).

| Size | 2GPU best | ratio | 4GPU best | ratio | 8GPU best | ratio |
|------|----------:|------:|----------:|------:|----------:|------:|
| 4KB | tree 0.1G | 1.10× | tree 0.1G | 0.67× | tree 0.1G | 0.68× |
| 8KB | tree 0.3G | 0.84× | tree 0.2G | 0.63× | tree 0.2G | 0.47× |
| 16KB | tree 0.5G | 0.81× | tree 0.5G | 0.58× | tree 0.4G | 0.48× |
| 32KB | tree 1.0G | 0.84× | tree 1.0G | 0.61× | tree 0.9G | 0.51× |
| 64KB | tree 2.0G | 0.83× | tree 1.9G | 0.88× | tree 1.5G | 0.44× |
| 128KB | tree 4.0G | 0.76× | tree 3.8G | 1.03× | tree 3.4G | 0.39× |
| 256KB | tree 7.8G | 0.82× | tree 7.1G | 0.56× | tree 6.1G | 0.43× |
| 512KB | tree 14.4G | 0.75× | tree 12.8G | 0.54× | tree 11.0G | 0.41× |
| 1MB | tree 23.0G | 0.60× | rhd 21.3G | 0.45× | rhd 19.1G | 0.37× |
| 2MB | tree 40.7G | 0.60× | rhd 36.0G | 0.38× | rhd 27.0G | 0.28× |
| 4MB | tree 70.6G | 0.65× | rhd 60.5G | 0.45× | rhd 53.4G | 0.40× |
| 8MB | rhd 111.2G | 0.70× | rhd 105.5G | 0.56× | rhd 93.0G | 0.89× |
| 16MB | auto 142.8G | 0.75× | rhd 149.6G | 0.65× | rhd 137.6G | 0.77× |
| 32MB | rhd 173.5G | 0.82× | rhd 185.6G | 0.72× | rhd 184.7G | 0.74× |
| 64MB | rhd 205.9G | 0.89× | rhd 221.8G | 0.83× | rhd 222.4G | 0.77× |
| 128MB | auto 228.5G | 0.94× | rhd 250.8G | 0.88× | rhd 253.0G | 0.84× |
| 256MB | auto 239.1G | 0.95× | rhd 266.4G | 0.90× | rhd 272.2G | 0.88× |
| 512MB | auto 246.4G | 0.94× | rhd 274.8G | 0.91× | rhd 282.0G | 0.90× |

**Summary**: Large messages approaching NCCL: **0.94×** at 2GPU 512MB, **0.91×** at 4GPU, **0.90×** at 8GPU. The partial D2D output copy optimisation saves (N-1)/N of the post-kernel memcpy (e.g. at 8GPU only 1/8 of the buffer needs copying). Small/medium messages have a latency-dominated gap (0.37-0.65× at 1MB) — signal overhead and kernel launch dominate.

---

## Performance Gap Analysis

### Why PCCL is at 155 GB/s — DIAGNOSED

**Root cause: chunk-fragmented TMA pipelines.** The persistent kernel dispatches TMA work through a chunk mechanism (`min_chunks = blocks_per_ch * 2`). For 8GPU 512MB allreduce (64MB per step, 64 blocks, 1 channel):

```
min_chunks = 64 * 2 = 128
chunk_elems = 16M / 128 = 125K elements = 500KB per chunk ≈ 8 TMA tiles
```

Each block grabs small chunks via `atomicAdd`, pipelines through ~8 tiles, issues `__threadfence_system()`, then grabs the next chunk. This creates three compounding problems:

1. **Shallow pipelines**: 8 tiles per pipeline run cannot amortize NVLink startup latency (double-buffer needs depth ≥30 tiles for near-peak throughput)
2. **Excessive fencing**: 128 × `__threadfence_system()` per step (each ~400ns = 51µs overhead per step)
3. **Atomic contention**: 128 chunk grabs + 128 completion counts per step

The recent MAX_CHUNK_SIZE_BYTES 256KB→4MB change had minimal effect (+0.9%) because `min_chunks = blocks_per_ch * 2` already forces 128 small chunks regardless of the max.

**Microbenchmark proof** (`bench_tma_sm_overlap.cu`, 2× H100, 512MB GPU1→GPU0, **no chunking — each block gets one contiguous range**):

| TMA blocks | BW (GB/s) | SM blocks | BW (GB/s) |
|-----------:|----------:|----------:|----------:|
| 1 | 50.3 | 1 | 3.8 |
| 4 | 183.5 | 16 | 59.8 |
| 8 | 306.7 | 64 | 230.5 |
| 16 | **375.1** | 96 | 322.4 |
| 32 | 374.9 | 128 | 363.6 |

**TMA copy saturates at 16 blocks = 375 GB/s** (83% NVLink, matching NCCL). The key difference from PCCL: each block pipelines through hundreds of contiguous tiles with zero intermediate fences, vs PCCL's 8 tiles per chunk with a fence after each.

**Phase 4 re-analysis**: Multi-ring spread blocks across channels, giving each channel *fewer* blocks while keeping the same chunking pathology. Fewer blocks × same shallow chunks = worse, not better.

**Important caveat**: The microbenchmark only tests TMA **copy** (`tma_store_2d`). TMA **reduce** (`tma_load_2d` → FP32 add → `tma_store_2d`) has a compute dependency in the pipeline that may limit multi-block scaling differently. A TMA reduce benchmark is needed before projecting allreduce performance.

### Why NCCL gets ~315 GB/s

NCCL on 8× H100 NVSwitch (with `NCCL_NVLS_ENABLE=0`) achieves ~315 GB/s bus_bw at 512MB:
1. **Hardcoded single-kernel allreduce**: No runtime dispatch — step logic is compiled in, zero inter-step overhead
2. **LL128 protocol**: SM-based NVLink access with 128-byte inline reduction for small/medium messages
3. **Deep kernel pipelining**: Step N send overlaps with step N+1 receive within the same kernel

NCCL achieves ~315/450 = 70% NVLink efficiency (without NVLS multicast).

### Key insight: the fix is eliminating chunk fragmentation

PCCL can potentially approach NCCL's ~315 GB/s with the existing ring allreduce by replacing the chunk-grab loop with per-block tile partitioning. Each block gets a contiguous tile range (total_tiles / blocks_per_ch) and pipelines through it without intermediate fences. This is what the microbenchmark validates for copy — reduce scaling remains unverified.

### Dispatch Overhead Analysis — Ring Buffer + DAG Mechanism

#### Static DAG structure (2-GPU, 2-channel TMA allreduce)

The DSL compiler generates **88 primitives regardless of message size** — 16KB and 512MB use the exact same DAG:

| | Per Channel | Total (2ch) | % of total |
|---|---:|---:|---:|
| notify | 15 | 30 | 34% |
| wait_notify | 15 | 30 | 34% |
| tma.reduce | 7 | 14 | 16% |
| tma.copy | 7 | 14 | 16% |
| **Total** | **44** | **88** | |

Each ring step = 3 primitives (notify + wait + data_op). The 7 reduce + 7 copy per channel come from the DSL splitting each ring step's data into 7 sub-chunks at the DAG level. Every primitive triggers a full iteration of the main dispatch loop in `cuda_executor_kernel`.

#### Per-primitive main-loop overhead

Each of the 88 primitives forces ALL blocks through one full main-loop iteration:

| Operation | Cost (H100 est.) | Notes |
|---|---:|---|
| volatile read `completed_primitives` | ~80ns | L2 cache hit |
| `__syncthreads()` (exit broadcast) | ~200ns | All blocks synchronize |
| volatile read `active_valid` + `__syncthreads()` | ~280ns | Broadcast to all blocks |
| `dequeueFromRingBuffer` CAS loop | ~150ns | `atomicCAS` + volatile head/tail reads |
| ChunkState setup + `__threadfence()` | ~150ns | Global memory fence |
| `atomicExch(active_valid, 1)` | ~50ns | Publish to all blocks |
| `updateDependencies` | ~300ns | `atomicAdd` per successor + conditional enqueue w/ `atomicAdd` + `__threadfence` |
| `atomicAdd(completed_primitives)` + `atomicExch(active_valid, 0)` | ~100ns | Completion bookkeeping |
| **Total per primitive** | **~1.2µs** | |

#### Key inefficiencies

1. **68% of primitives are single-thread ops (notify/wait)**: 60 out of 88 primitives are signal operations executed by one thread on one block, yet ALL blocks go through the full dispatch loop (volatile reads, `__syncthreads`, dequeue attempt, etc.) — wasting SM cycles.

2. **Ring buffer is overkill**: `dequeueFromRingBuffer` uses `atomicCAS` in a while-loop, but there's never contention — only one thread ever dequeues from a channel queue. `enqueueToRingBuffer` uses `atomicAdd` + `__threadfence` for at most one producer. A simple array index would suffice.

3. **active_valid state machine forces global memory round-trips**: Each primitive cycles through `active_valid=1` (write) → all blocks read volatile → execute → `active_valid=0` (write). That's 2 global writes + N block reads per primitive.

4. **Static 88-primitive DAG even for tiny messages**: At 16KB with 2 channels, each sub-chunk is only 292 elements (584 bytes) — far below any useful TMA tile size.

#### Impact by message size

| Size | Transfer Time (225 GB/s) | DAG Overhead (88 × 1.2µs) | Overhead % |
|---|---:|---:|---:|
| 16 KB | 0.07 µs | ~106 µs | **>100,000%** |
| 256 KB | 1.1 µs | ~106 µs | **9,600%** |
| 1 MB | 4.4 µs | ~106 µs | **2,400%** |
| 16 MB | 71 µs | ~106 µs | **149%** |
| 64 MB | 284 µs | ~106 µs | **37%** |
| 512 MB | 2,276 µs | ~106 µs | **4.7%** |

At 512MB the overhead is tolerable (~5%). Below 64MB it dominates. Below 1MB it's catastrophic.

---

## Runtime Op Set

| Op | Type IDs | Implementation | Status |
|----|----------|---------------|--------|
| SM copy | 6-10 | `vector_copy` (global mem) | Working |
| SM reduce | 1-5 | `vector_add` (global mem) | Working |
| TMA copy | 28-32 | Pipelined via TMA engine + smem | Working |
| TMA reduce | 33-37 | Pipelined load-compute-store via TMA | Working |
| CE copy | 23-27 | Copy engine path | Working |
| Signal (notify/wait) | 21-22 | Atomic signal increment / spin-wait | Working |
| RDMA write/read | 40-41 | Via host proxy | Working |

---

## Planned Optimizations & Projected Impact

### Phase 4: Multi-Ring AllReduce — ATTEMPTED, NO GAIN

**Root cause re-diagnosed**: The failure wasn't NVSwitch crossbar (as originally hypothesized), but single-block TMA pipelines. Multi-channel didn't help because it spread blocks across channels without increasing per-primitive parallelism. See microbenchmark results in Performance Gap Analysis.

### Phase 4b: SM+TMA Hybrid AllReduce — SUPERSEDED

**Status**: No longer needed. Microbenchmark showed TMA alone reaches 375 GB/s with 16 blocks. SM+TMA hybrid showed some overlap capability (~210 GB/s for 50/50 split) but is inferior to multi-block TMA and much more complex.

### Phase 9: Eliminate Chunk Fragmentation for TMA — DONE

**Goal**: Replace the chunk-grab dispatch loop with per-block tile partitioning for TMA ops, enabling deep pipelines and eliminating per-chunk `__threadfence_system()`.

**Implementation**: For TMA ops, each block claims one slot via `atomicAdd(&current_chunk, 1)` and computes its contiguous tile range from `(slot * total_elems) / blocks_per_ch`. Re-entry after completion is prevented by the claim counter (slot >= blocks_per_ch → skip). SM ops retain the existing chunk-grab loop unchanged.

**Result (8GPU, 512MB):**

| Config | Before | After | Change | vs NCCL |
|--------|-------:|------:|-------:|--------:|
| 1ch | 156.6 GB/s | 198.0 GB/s | +26% | 0.53× |
| 4ch | 158.9 GB/s | 202.1 GB/s | +27% | 0.54× |
| 2GPU 1ch | 158.6 GB/s | 192.3 GB/s | +21% | 0.73× |
| 2GPU 2ch | — | 201.4 GB/s | — | 0.77× |

**TMA reduce microbenchmark** (2× H100, 512MB): TMA reduce scales to 149 GB/s at 64 blocks (~40% of TMA copy at 375 GB/s). BF16 native `__hadd2` gives identical throughput to FP32 — **compute is not the bottleneck**. The pipeline is bound by HBM read serialization (`self_buf` load), 4× `__syncthreads()` per tile, and `fence.proxy.async.shared::cta` overhead.

**Analysis**: The +27% gain confirms chunk fragmentation was a real bottleneck, but the gap to NCCL (0.54× vs 1.0×) remains large. Remaining bottlenecks: (1) TMA reduce pipeline serialization (SM compute in the critical path), (2) ring allreduce O(N) step count, (3) per-step dispatch overhead.

### Phase 9b: Hardware TMA Reduce via `cp.reduce.async.bulk` — DONE

**Goal**: Replace SM-based reduce (TMA load → SM compute → TMA store) with hardware TMA reduce (`cp.reduce.async.bulk.tensor.2d...add`), eliminating SM involvement from the reduce-scatter critical path.

**Implementation**:
- Added HW reduce path in `executeTmaReduce`: when `offset_1 == offset_2`, uses `tma_reduce_add_2d(self_desc, smem, ...)` instead of SM `smem_vector_add()` + `tma_store_2d()`
- Uses `cp.async.bulk.wait_group.read 0` (not the standard `.wait_group 0`) to wait for TMA reduce completion
- SM reduce fallback retained for `offset_1 != offset_2` (strided access pattern)

**Microbenchmark** (2× H100, 512MB, `bench_tma_sm_overlap.cu`):

| Blocks | SM Reduce (GB/s) | HW Reduce (GB/s) | Speedup |
|-------:|------------------:|------------------:|--------:|
| 1 | 9.5 | 9.3 | 1.0× |
| 4 | 36.5 | 36.6 | 1.0× |
| 16 | 118.1 | 243.2 | 2.1× |
| 32 | 143.3 | 347.5 | 2.4× |
| 64 | 149.0 | 375.0 | 2.5× |

HW reduce matches TMA copy speed (375 GB/s) at scale — the L2/MC atomic path has no SM compute overhead and saturates NVLink bandwidth identically to pure stores.

**Result (2GPU, 512MB):**

| Config | Before | After | Change | vs NCCL |
|--------|-------:|------:|-------:|--------:|
| 1ch | 200.6 GB/s | 225.2 GB/s | +12.3% | 0.86× |
| 2ch | 201.4 GB/s | 224.9 GB/s | +11.7% | 0.86× |

**Result (8GPU, 512MB):**

| Config | Before | After | Change | vs NCCL |
|--------|-------:|------:|-------:|--------:|
| 1ch | 198.0 GB/s | 239.3 GB/s | +20.9% | 0.63× |
| 2ch | — | 238.6 GB/s | — | 0.63× |
| 4ch | 202.1 GB/s | 238.3 GB/s | +17.9% | 0.63× |

**Analysis**: The 8GPU gain (+21%) is larger than 2GPU (+12%) because the ring has more reduce-scatter steps (7 vs 1), so accelerating each reduce step compounds. Multi-channel provides no additional benefit — all channels achieve ~239 GB/s.

### Phase 9c: Runtime Cleanup + Correctness Fix — DONE

**Goal**: Clean up dead code from Phase 9b transition, fix pre-existing correctness bug, and re-attempt output D2D copy elimination.

**Changes**:

1. **Dead code cleanup**: Removed all `output_desc`/`output_buffer_` infrastructure that Phase 9b made unused (6 components across 5 files: `tma_descriptor.h`, `workspace.h`, `engine_c.cc`, `tma_manager.cc/.h`). Reduces per-call complexity and H2D copy overhead.

2. **Correctness fix — TMA last-tile overflow**: When `chunk_count` is not a multiple of `tile_elems` (128×255=32640 for BF16), using `ceil(chunk_count / tile_elems)` for `num_subtiles` caused the last TMA tile to write a full `tile_elems` beyond the chunk boundary, corrupting adjacent chunks. Fix: floor division (`chunk_count / tile_elems`) for complete tiles only, plus SM `vector_copy`/`vector_add` tail handling for remaining elements. **Allreduce now 100% correct** at both 2GPU and 8GPU.

3. **Output D2D elimination re-attempt**: Restored Phase 2's dual TMA store in `executeTmaCopy` (write to both `self_desc` and `output_desc` from same smem tile). Added `output_desc`/`output_valid` to `TmaDescriptors`, `output_buffer_` to `DeviceWorkspace` per-call section. Removed host-side `cudaMemcpyAsync` for output. **Result: performance-neutral** — dual TMA store overhead offsets the saved D2D copy. No regression, no improvement.

**Result**: Correct allreduce, cleaner codebase, same performance (~225 GB/s 2GPU, ~239 GB/s 8GPU).

### Phase 8: Dispatch Overhead Reduction — Fused Step Executor — DONE

**Goal**: Replace the ring buffer + DAG dispatch engine with a flat fused-step loop, eliminating ~100µs of per-iteration overhead and fundamentally closing the small/medium message gap.

**Implementation**:
- New `fused_allreduce_kernel` alongside existing `cuda_executor_kernel`
- `GraphBuilder::buildFusedDescriptor()` detects linear-chain DAGs and collapses {notify, wait, data_op} triplets into compact `FusedStep` entries
- Flat per-channel step loop with inline signal/wait — no ring buffer, no CAS dequeue, no active_valid state machine, no dependency tracking
- Inter-block barrier via per-channel atomic counters ensures all blocks wait for signal completion before data ops
- Automatic fallback to `cuda_executor_kernel` for non-linear DAGs
- D2D output copy added post-kernel (ensures output tensor always has complete result)
- `PCCL_DISABLE_FUSED=1` env var for A/B testing

**Result (2GPU, measured):**

| Size | DAG kernel | Fused kernel | Speedup |
|------|----------:|------------:|--------:|
| 4 KB | 79.2µs | 37.5µs | 2.1× |
| 64 KB | 78.2µs | 47.5µs (2ch) | 1.6× |
| 256 KB | 77.8µs | 47.4µs | 1.6× |
| 1 MB | 84.0µs | 52.6µs | 1.6× |
| 16 MB | 158.0µs | 124.6µs | 1.3× |
| 128 MB | 666.9µs | 632.9µs | 1.05× |
| 512 MB | 2396µs (224 GB/s) | 2358µs (228 GB/s) | 1.02× |

**Result (8GPU, measured):**

| Size | DAG kernel | Fused kernel | Speedup | vs NCCL |
|------|----------:|------------:|--------:|--------:|
| 4 KB | 359µs | 164µs | 2.2× | 5.0× worse |
| 64 KB | 360µs | 169µs | 2.1× | 2.3× worse |
| 1 MB | 362µs | 173µs | 2.1× | 5.2× worse |
| 16 MB | 542µs | 332µs | 1.6× | 2.4× worse |
| 64 MB | 883µs | 639µs (4ch) | 1.4× | 1.8× worse |
| 256 MB | 2246µs (209 GB/s) | 2002µs (235 GB/s) | 1.12× | 0.64× |
| 512 MB | 3966µs (237 GB/s) | 3715µs (253 GB/s) | 1.07× | 0.67× |

**Analysis**: The fused kernel eliminated ~40µs (2GPU) / ~190µs (8GPU) of fixed dispatch overhead. At 8GPU, each ring iteration has 88 DAG primitives (14 data ops + 30 notify + 30 wait + scheduling), each costing ~1.2µs — totaling ~100µs. The fused kernel collapses these into 14 flat step iterations. The remaining ~165µs baseline at 8GPU (small messages) is: 14 signal roundtrips × ~10µs = ~140µs plus kernel launch, H2D copy, and D2D copies. This signal-dominated baseline cannot be reduced further with the ring algorithm — O(log N) algorithms (RHD/tree) help by reducing step count.

### Phase 6: Superoptimizer Enhancements

#### Phase 6a: E-graph Equality Saturation — DONE

Restructured the superoptimizer around e-graph equality saturation. Key changes:
- Extended `ENode` with `op_type`, `executor`, `channel`, `peer`, `dep_count` fields for communication DAG representation
- Built `egraph_bridge.py`: IR → ENode tree conversion (subgraph partitioning) and ENode extraction → IR reconstruction with `tensor_info` preservation
- Created `domain_rules.py`: 4 rule categories — executor equivalences (SM↔TMA↔multimem), structural dedup (identical-peer merge), channel normalization, fused-op introduction
- Made `cost_model.py` parametric: `TopologyProfile` + `GpuProfile` feed into `critical_path_cost()` for topology-aware cost extraction
- Rewrote `SuperoptPass` to use e-graph: build ENode trees per subgraph → apply domain rules via equality saturation → extract lowest-cost variant
- Updated offline rule generation (`enumerator.py`) with multimem op type support

**Result**: SM→TMA upgrades now fire through e-graph equivalence classes. Full pipeline validated: DSL → superopt → JSON v2 → RuntimeGraphGenerator.

#### Phase 6b: End-to-End Validation — DONE

- `--superopt` flag wired into `bench_allreduce_tma_bw.py` for A/B comparison
- 5 E2E integration tests validate: SM ops upgraded to TMA, JSON v2 structure correct, data executors are TMA/multimem (not SM), all operation fields present
- `Compiler(enable_superopt=True, topology=..., device_profile=..., data_size_hint=...)` API stable

#### Phase 6c: Algorithm Search — DONE

- Algorithm template module (`pccl/dsl/algorithms/`): `CollectiveAlgorithm` ABC with `build_allreduce()` interface
- Three algorithms implemented: Ring (O(2(N-1)), bandwidth-optimal), Recursive Halving-Doubling (O(2 log N), Rabenseifner), Binary Tree (O(2 log N), latency-optimal)
- Topology-aware selector: heuristic mode (ring for large msgs, RHD for small+high GPU count) and cost-model mode (builds candidate graphs, compares via `critical_path_cost()`)
- `--algorithm {ring,rhd,tree,auto}` wired into benchmark
- 34 algorithm unit tests covering graph structure, compilation, and selection logic

**Projected impact**: Algorithm search provides O(log N) alternatives to ring for small/medium messages. At 8GPU, RHD uses 6 steps vs ring's 14, potentially halving signal latency (~70µs vs ~140µs). Main value is programmability and automatic topology adaptation.

### ncu Kernel Profiling (BLOCKED)

ncu unavailable on current machine. Deferred until hardware access allows it.

---

## Can PCCL Beat NCCL?

### Current standing (bench_algo_comparison.py, NVLS disabled)

| GPUs | 512MB PCCL | 512MB NCCL | Ratio | Best spot |
|-----:|-----------:|------------------:|------:|-----------|
| 2 | 246.4 GB/s | ~263 GB/s | 0.94× | 256MB 0.95× |
| 4 | 274.8 GB/s | ~302 GB/s | 0.91× | 128KB 1.03× |
| 8 | 282.0 GB/s | ~315 GB/s | 0.90× | 8MB 0.89× |

### Honest assessment

**PCCL large messages: 0.90-0.94× NCCL.** Two key optimisations brought performance from 0.51× to the current level:

1. **NVLS opt-in** (0.51×→0.87×): NVLS kernel was auto-activating via `collective_type` but was 1.67× slower than the fused kernel. Fixed by requiring `PCCL_NVLS_ENABLE=1`.

2. **Partial D2D output copy** (0.87×→0.94×): The fused kernel's `executeTmaCopy` already dual-writes (N-1)/N of the result to the output tensor via TMA. Only the locally-reduced gap (1/N of the data) needs a D2D copy. The engine pre-computes the gap ranges from the fused descriptor's COPY-step coverage and copies only those ranges. At 8GPU 512MB this saves 7/8 of the 512MB output copy.

The remaining ~6-10% gap at 512MB is from:
- D2D input copy (full 512MB per allreduce = ~356µs)
- Partial D2D output copy (1/N of 512MB = ~45-178µs depending on N)
- Signal/fence overhead in the fused kernel

**Small/medium message regression**: Without NVLS, 1MB@4-8GPU is at 0.37-0.45× NCCL. The fused kernel's per-step overhead (~10µs signal roundtrip × N steps) dominates at small sizes. NCCL's LL128 protocol handles these sizes with ~30µs latency.

**Where PCCL can differentiate:**

1. **Non-standard topologies**: NCCL's heuristics are tuned for standard NVSwitch/PCIe configs. PCCL's DSL + superoptimizer can auto-optimize for irregular topologies (heterogeneous clusters, partial NVLink, asymmetric bandwidth) where NCCL falls back to suboptimal defaults.

2. **Programmability**: Custom collectives (sparse allreduce, mixed-precision reduction, topology-aware sharding) that NCCL doesn't support. The DSL makes these expressible without kernel-level programming.

3. **Cross-node unification**: Same persistent kernel model for intra-node (NVLink) and inter-node (RDMA), with the superoptimizer selecting the best algorithm for each topology segment.

### Next: CUDA performance optimization

The primary focus is closing the remaining ~6-10% gap to NCCL on large messages and fixing small/medium message latency:
1. **D2D input copy elimination** — the input is copied from user tensor to IPC working buffer (~356µs at 512MB). If the user can allocate from the IPC buffer directly, this copy becomes unnecessary. Alternatively, overlap the input copy with kernel execution.
2. **TMA pipeline depth** — current 2-buffer double-buffering may not hide NVLink latency; deeper pipelining (3-4 stages) could help
3. **Signal latency reduction** — `__threadfence_system()` + atomicAdd for notify, `__nanosleep(64)` for wait; explore lighter synchronization primitives
4. **NVLS kernel optimization** — the nvls_allreduce_kernel is 1.67× slower than fused at 2GPU; needs profiling and rewrite before re-enabling
5. **Small message latency** — fused kernel baseline is ~50µs at 2GPU (vs NCCL ~30µs); reduce per-step signal overhead

---

## Validation

Each phase must pass:
1. **Correctness**: `test_basic.py` (44), `test_superopt.py` (84), `test_multichannel.py` (51), `test_algorithms.py` (34) — 213 tests total. GPU correctness validated via `test_correctness.py` (multi-process, requires GPUs).
2. **Performance**: `bench_algo_comparison.py` via `run_algo_comparison.sh` at 2, 4, and 8 GPUs — no regression, improvement at target sizes. Tests all algorithms (ring/RHD/tree/auto) and picks best per size.
3. **Multi-GPU scaling**: Verify improvement scales with GPU count
