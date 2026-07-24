# PCCL Algorithm IR v1

更新时间：2026-07-24 17:53 +08:00

## 1. 接口位置

```text
Execution Plan
  op_type / algorithm_type / topology / barrier
          |
          v
Collective Algorithm IR
  rank / buffer / chunk / step / copy / reduce / dependency
          |
          v
AlgorithmIRLowerer(rank, executor)
          |
          v
PCCL PrimitiveIRGraph
  notify / wait_notify / sm.* / tma.*
          |
          v
PCCL JSON v2 / CUDA engine
```

Execution Plan 决定执行什么和何时切换；Algorithm IR 描述 collective 的硬件无关数据流；Primitive IR 才选择 SM、TMA、RDMA 等执行器。OCS barrier 是 phase 间 host-control boundary，不进入单个 collective 的 Algorithm IR。

该 Algorithm IR 是 PCCL 内部算法生成接口。AICCL/RLCCL 等上层生成器的公开兼容边界已固定为标准 MSCCL XML，由 [`MSCCLXMLAlgorithm`](MSCCL_COMPATIBILITY.md) 导入；上层不需要改写为 `AlgorithmIRBuilder`。

## 2. 数据模型

| 类型 | 语义 |
|---|---|
| `ChunkRef` | `(rank, buffer, index, count)` 逻辑 chunk 范围 |
| `AlgorithmBuffer` | `input/output/scratch` |
| `AlgorithmTransfer` | 从 source chunk 到 destination chunk 的 `copy/reduce` |
| `AlgorithmSync` | 一条 rank 间 signal edge |
| `AlgorithmStep` | 同一 step 中可并行执行的 transfer 或 sync 集合 |
| `CollectiveAlgorithmIR` | 所有 rank 的完整静态 collective schedule |

`AlgorithmTransfer` 同时表达逻辑 send/recv：`src.rank` 是发送侧，`dst.rank` 是接收侧，数据 primitive 在目标 rank 执行。IR 不绑定 push/pull transport。

## 3. 构图接口

```python
from pccl import AlgorithmBuffer, AlgorithmIRBuilder, AlgorithmIRLowerer

builder = AlgorithmIRBuilder(
    name="copy_example",
    collective_type="custom",
    world_size=2,
    chunks_per_rank=2,
)
step = builder.step("exchange")
step.copy(
    builder.chunk(0, 0, AlgorithmBuffer.INPUT),
    builder.chunk(1, 0, AlgorithmBuffer.OUTPUT),
)
algorithm = builder.build()

rank1_graph = AlgorithmIRLowerer().lower(
    algorithm, rank=1, tensor_size=1024, dtype="float32", executor="sm"
)
```

Builder 按 chunk 读写自动推导 dependency：

1. 读取 source chunk 依赖其 last writer；
2. 写入 destination chunk 依赖其 last writer 和所有 last readers；
3. 同一 step 观察上一个 step 的状态，step 内 transfer 视为并行；
4. 同一 step 重复写同一 chunk 直接拒绝；
5. canonical JSON 和 SHA-256 digest 用于算法 artifact 身份。

## 4. Lowering 规则

远端 transfer 默认降低为：

```text
source rank: notify(destination, signal_id)
destination: wait_notify(source, signal_id)
destination: copy/reduce(source buffer -> local buffer)
```

| Algorithm IR | PCCL lowering |
|---|---|
| `copy` | `tma.copy` 或 `sm.copy` |
| `reduce` | `tma.reduce` 或 `sm.reduce` |
| sync edge | `notify + wait_notify` |
| scratch destination | 强制 `sm.copy/reduce`，避免越过当前 TMA descriptor 范围 |

Lowerer 只提取指定 rank 的本地程序。跨 rank 数据依赖由 matched notify/wait 保证；同 rank 和跨 channel 的 prior dependency 会添加到 Primitive IR DAG。

## 5. 已生成算法

### Ring AllReduce

`build_ring_allreduce_ir(world_size)` 自动生成：

```text
N-1 reduce-scatter steps
N-1 allgather steps
1 completion sync step
```

4 rank、单 channel 下生成 20 个本地 primitive，与现有手写 Ring 模板的 primitive 类型、peer 和数据 offset 一致。

### Direct AllToAll

`build_direct_alltoall_ir(world_size)` 自动生成：

```text
N-1 input rendezvous steps
N-1 pairwise exchange steps
```

4 rank 下生成 9 个本地 primitive，与现有模板数量和 scratch offset 一致。

## 6. Execution Plan 接入

默认模式继续使用旧模板；生成模式需要显式开启：

```python
compiler = ExecutionPlanCompiler(algorithm_lowering="generated")
compiled = compiler.compile(
    plan, rank=0, tensor_size=4096, dtype="float32", executor="sm"
)
```

生成模式当前支持：

| `op_type` | `algorithm_type` |
|---|---|
| `allreduce` | `ring` |
| `alltoall` | `direct` |

不支持的组合明确报错，不静默回退到手写模板。

外部生成器输出标准 MSCCL XML 时使用：

```python
compiler = ExecutionPlanCompiler(
    algorithm_lowering="msccl",
    artifact_resolver=artifact_store.__getitem__,
)
```

`phase.artifact_id` 指向 XML schedule；phase/barrier/topology/route plan 仍由 OCS Execution Plan 描述。完整契约见 [`MSCCL_COMPATIBILITY.md`](MSCCL_COMPATIBILITY.md)。

## 7. 验证结果

### 正确性

- 本机仓库全量 Python 回归：`301 passed`。
- 独立语义解释器验证 2/4/8 rank Ring AllReduce 和 Direct AllToAll。
- 2 张 RTX A5000 的测试服务器上，手写模板与生成模式均通过 3 轮真实 PCCL C++ engine 测试：每轮执行 `AllReduce -> barrier -> AllToAll -> barrier -> AllReduce -> barrier`，累计 9 个 phase、9 个新 barrier/epoch，最终结果与 barrier 序列均正确。
- 发布前审计新增并行 step 读写同一 chunk 的冲突检查，以及 `world_size/signal_base` 等公开参数的严格整数类型检查；新核心模块通过 Black、Flake8 和 Mypy，Execution Plan 示例通过 JSON Schema Draft 2020-12 正例与负例验证。远端 Algorithm IR/Execution Plan 专项为 `36 passed`，更新代码后的 generated GPU smoke 再次通过。
- 2026-07-24 标准 MSCCL XML importer 增加 `rcs/rrs/rrcs` 融合指令兼容；本机全量回归更新为 `318 passed`。双 RTX A5000 上，默认融合 XML 通过 3 轮真实 C++ engine smoke，共执行 9 phase/9 barrier。

### 数据面 A/B

测量区间仅包含已注册 PCCL graph 的：

```text
execute_operation_async -> sync_operation
```

Execution Plan 解析、Algorithm IR lowering、JSON 编译、operation 注册、OCS barrier、controller 和真实交换机切换均不在计时区间。环境为 2 张 RTX A5000、float32、SM executor；每个 case 预热 20 次、测量 100 次并重复 5 轮，取两个 rank 中较慢者，再取 5 轮中位数。

关闭 fused executor 的结果：

| collective | payload | 手写模板 | 自动生成 | 生成/模板 |
|---|---:|---:|---:|---:|
| AllReduce | 1 MiB | 230.11 us | 230.50 us | 1.0017x |
| AllReduce | 16 MiB | 2598.04 us | 2587.44 us | 0.9959x |
| AllReduce | 64 MiB | 9947.48 us | 9948.06 us | 1.0001x |
| AllToAll | 1 MiB | 117.63 us | 117.77 us | 1.0012x |
| AllToAll | 16 MiB | 1349.27 us | 1348.72 us | 0.9996x |
| AllToAll | 64 MiB | 5180.57 us | 5177.88 us | 0.9995x |

六组 case 的最大绝对差异为约 `0.41%`。启用 engine fused 路径后六组 case 也全部通过，生成/模板为 `0.9987x` 至 `1.0006x`，最大绝对差异约 `0.13%`。当前证据支持“生成器复现了原模板的数据面执行计划，未引入可测性能回归”，不代表已经达到亚微秒 OCS 控制或物理切换时延。

标准 MSCCL XML importer 已加入同一 benchmark。关闭 engine fusion 时，MSCCL/template 在 6 个 case 为 `0.9995x..1.0009x`，最大绝对偏差约 `0.095%`；启用 fusion 时为 `0.9908x..1.0017x`，最大偏差是 1 MiB AllReduce 的约 `1.99 us`。这说明 importer 的 XML 解析和 lowering 均在注册前完成，不进入数据面热路径；当前 fixture 生成等价 DAG，因此该实验验证兼容路径开销，不是不同 collective schedule 的算法优劣对比。完整表格和原始结果见 [`MSCCL_COMPATIBILITY.md`](MSCCL_COMPATIBILITY.md)。

复现实验：

```bash
bash tests/run_ocs_algorithm_ir_gpu_smoke.sh generated 3 4096
bash tests/run_ocs_algorithm_ir_gpu_smoke.sh template 3 4096
bash tests/run_algorithm_ir_ab.sh --modes template,generated,msccl \
  --warmup 20 --iterations 100 --repeats 5
```

## 8. v1 限制

- 仅静态、确定性通信量；尚未表达 MoE 动态 count。
- 生成算法仅支持单 channel；IR 数据结构已保留 `channel`。
- 仅支持 `copy/reduce` 与显式 sync；`send/recv` 目前由 transfer 两端隐式表达。
- 尚未生成 tree、RHD、allgather 和 reduce-scatter。
- 尚未实现 topology-aware schedule synthesis、cost-model search 和生成器侧自动 kernel fusion；现有 PCCL engine 的 fused 路径已完成 A/B 验证。
- GPU 验证当前只有单机 2 rank，尚未覆盖 4/8 GPU、多机或真实 OCS/RDMA 网络。
