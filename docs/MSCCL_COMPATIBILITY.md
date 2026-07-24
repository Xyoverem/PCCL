# MSCCL Schedule 兼容接口

更新时间：2026-07-24 13:58 +08:00

## 1. 分层与边界

```text
AICCL / RLCCL / 外部算法生成器
  ├─ 标准 MSCCL XML collective schedule
  └─ OCS topology plan
                 |
                 v
OCS-PCCL Execution Plan
  phase.artifact_id ──> MSCCL XML
  phase.topology_id / barrier_after / route_plan
                 |
                 v
MSCCLXMLAlgorithm
  XML 校验、send/recv 配对、dependency lowering
                 |
                 v
PCCL rank-local PrimitiveIRGraph
                 |
                 v
PCCL runtime
  communication phase -> barrier -> OCS switch -> next phase
```

稳定对接面是标准 MSCCL XML，不要求 AICCL/RLCCL 改用 PCCL 私有 builder。`AlgorithmIRBuilder` 保留为 PCCL 内部算法生成和实验接口，不是上层生成器必须实现的协议。

MSCCL XML 只描述一个 collective 的静态 schedule。OCS 的 phase、epoch、barrier、物理拓扑切换和 link-ready 状态属于外层 Execution Plan，不写入 XML。

## 2. 上游输出

上层生成器每个 phase 输出两部分：

1. collective schedule：MSCCLang/MSCCL 生成的标准 XML。
2. OCS plan：当前 topology、目标 topology、route plan 和切换 boundary。

建议 MSCCLang 生成参数：

```python
program = MSCCLProgram(
    name,
    topology,
    collective,
    protocol="Simple",
    instr_fusion=False,
)
program.XML()
```

必须使用 `instr_fusion=False`。`rcs/rrs/rrcs` 已丢失一部分中间 buffer/依赖边界，当前 importer 不猜测恢复，PCCL 可以在生成 Primitive IR 后自行执行 fusion pass。

## 3. MSCCL XML 映射

标准结构：

```text
algo(name, proto, nchannels, nchunksperloop, ngpus, coll, inplace)
  gpu(id, i_chunks, o_chunks, s_chunks)
    tb(id, send, recv, chan)
      step(s, type, srcbuf, srcoff, dstbuf, dstoff,
           cnt, depid, deps, hasdep)
```

| MSCCL XML | PCCL Primitive IR |
|---|---|
| `s` | `notify(target_rank, signal_id)` |
| `r` | `wait_notify(source_rank, signal_id)` + `copy` |
| `rrc` | `wait_notify(source_rank, signal_id)` + `reduce` |
| `cpy` | local `copy` |
| `re` | local `reduce` |
| `nop` | executable `noop`，保留依赖位置 |
| TB 内 step 顺序 | 同一 PCCL stream 的顺序依赖 |
| `depid/deps` | 跨 stream DAG edge |
| `chan` | Primitive IR channel |
| matched `s/r` | 相同 `signal_id` |

Importer 校验 GPU/TB/step id、buffer chunk 边界、peer/channel、依赖引用和环、send/recv 数量与字段匹配。XML 内容归一化后计算 SHA-256 digest，空白变化不影响 artifact identity。

## 4. PCCL 接口

直接导入：

```python
from pccl import MSCCLXMLAlgorithm

schedule = MSCCLXMLAlgorithm.load("ring_allreduce.xml")
graph = schedule.lower(
    rank=0,
    tensor_size=4096,
    dtype="float32",
    executor="sm",
)
```

通过 Execution Plan 导入：

```python
from pccl import ExecutionPlanCompiler, OCSExecutionPlan

artifacts = {
    "aiccl:ring:2:v1": "artifacts/ring_allreduce_2.xml",
    "rlccl:a2a:2:v1": "artifacts/direct_alltoall_2.xml",
}

plan = OCSExecutionPlan.load("execution_plan.json")
compiled = ExecutionPlanCompiler(
    algorithm_lowering="msccl",
    artifact_resolver=artifacts.__getitem__,
).compile(
    plan,
    rank=0,
    tensor_size=4096,
    dtype="float32",
    executor="sm",
)
```

`artifact_resolver(artifact_id)` 可返回 XML 字符串、bytes、文件路径或 `MSCCLXMLAlgorithm`。编译器强制验证：

- `XML.ngpus == len(plan.rank_list)`；
- `XML.coll == phase.op_type`；
- 当前 rank 属于通信组；
- XML schedule 和 PCCL graph 均合法；
- 配置了 `graph_digest` 时，最终 PCCL JSON v2 digest 一致。

## 5. OCS 扩展

每个 phase 的两个产物通过 `artifact_id` 关联：

```json
{
  "phase_id": 0,
  "epoch": 100,
  "op_type": "allreduce",
  "algorithm_type": "ring",
  "backend": "pccl",
  "topology_id": 10,
  "artifact_id": "aiccl:ring:2:v1",
  "barrier_after": {
    "barrier_id": 500,
    "next_epoch": 101,
    "next_phase_id": 1,
    "switch_action": "APPLY_ROUTE",
    "route_plan": {
      "route_plan_id": "route-500",
      "route_mode": "STATIC_PLAN",
      "source_topology_id": 10,
      "target_topology_id": 11,
      "payload": {}
    }
  }
}
```

Runtime 顺序固定为：

```text
launch phase graph
  -> wait phase complete
  -> READY convergence
  -> controller commits OCS route
  -> wait LINK_ALIGNED
  -> RELEASE next epoch
  -> launch next phase graph
```

MSCCL 的 `depid/deps` 是单个 collective 内的 device dependency；OCS barrier 是两个 collective phase 间的 host/control-plane boundary，两者不能互换。

## 6. 当前兼容范围

| 能力 | 状态 |
|---|---|
| 标准 `algo/gpu/tb/step` 解析 | 已实现 |
| `s/r/rrc/cpy/re/nop` | 已实现 |
| `depid/deps`、channel、send/recv matching | 已实现 |
| `Simple` protocol lowering | 已实现 |
| `LL/LL128` 元数据解析 | 已实现，执行时拒绝 |
| `rcs/rrs/rrcs` fused step | 拒绝，要求 `instr_fusion=False` |
| Ring AllReduce 标准 XML | CPU 测试及双 RTX A5000 engine smoke 通过 |
| Direct AllToAll 标准 XML | CPU 测试及双 RTX A5000 engine smoke 通过 |
| 通用 MSCCL AllToAll output layout | 尚未实现 |
| MSCCL XML 内 OCS switch | 不支持，设计上放在 Execution Plan |

当前 PCCL engine 的 AllToAll 会从 `tensor_size + source_rank * chunk_size` 组装输出。因此 v1 importer 只接受 `nchunksperloop == ngpus` 的标准 Direct AllToAll，并把远端 output slot 映射到该 scratch layout。更一般的 chunk factor、独立 input/output/scratch buffer 需要先扩展 C++ engine buffer model。

## 7. 验证

- 官方格式 2-rank Ring AllReduce fixture。
- 官方格式 2-rank Direct AllToAll fixture。
- XML metadata、digest、buffer/peer/channel、依赖环、unmatched transfer 负例。
- send/recv signal 配对、跨 TB dependency、Runtime JSON v2 编译。
- `AllReduce -> barrier -> AllToAll -> barrier -> AllReduce -> barrier` 三 phase Execution Plan 导入。
- 本机全量回归：`313 passed`。
- 服务器 `121.48.163.223` 双 RTX A5000：3 轮真实 C++ engine smoke 通过，累计 9 phase/9 barrier；两 rank 结果均正确，barrier id 为 `0..8`。

首次 GPU smoke 暴露了一个 runtime 适配问题：MSCCL threadblock 语义为并发，但 PCCL executor 使用 JSON 插入顺序打破并列 DAG root；官方 XML 先列 receive-TB 时，阻塞 wait 会先于独立 notify。Importer 现在先发射 send-capable TB，仅改变无依赖 root 的 tie-break 顺序，不改变 TB 内顺序或 `depid/deps`。修复后单轮和三轮 smoke 均通过。

本轮尚未完成 `msccl` importer 与 `generated/template` 的数据面 A/B；此前性能数据不能直接作为 importer 性能结果。
