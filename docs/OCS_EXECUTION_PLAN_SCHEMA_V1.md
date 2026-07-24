# OCS-PCCL Execution Plan Schema v1

更新时间：2026-07-23

Schema：`ocs-pccl.execution-plan.v1`

机器可读定义：[`schemas/ocs_execution_plan_v1.schema.json`](../schemas/ocs_execution_plan_v1.schema.json)

完整示例：[`examples/ocs_execution_plan_v1.json`](../examples/ocs_execution_plan_v1.json)

## 1. 接口分层

OCS-PCCL 固定为三种不同接口，禁止混用字段语义：

| 接口 | 作用 | 当前载体 |
|---|---|---|
| Execution Plan | 控制器描述执行什么 collective、用什么算法/拓扑、何时切换 | 本文与 JSON Schema |
| Runtime Manifest | PCCL engine 执行的 primitive DAG 和 phase 切分 | JSON v2 / JSON v3 `phased_ocs` |
| Wire Protocol | READY / RELEASE / ABORT / ACK 的传输与幂等 | `OCS_control_protocol_v1.md` |

Execution Plan 是控制器与编译器的稳定契约；JSON v3 是 PCCL 的可执行产物；wire message 只传递 barrier identity、plan digest 和 route payload。

## 2. 计划结构

```text
ExecutionPlan(job/group/participants)
  phase 0: data op + algorithm + current topology
    barrier 0: epoch 100 -> 101, apply route
  phase 1: data op + algorithm + current topology
    barrier 1: epoch 101 -> 102, apply route
  phase 2: data op + algorithm + current topology
    barrier 2: epoch 102 -> 103, prepare next iteration
```

`phase` 描述当前数据面；`barrier_after` 描述该 phase 完成后的状态转换。这样不会再混淆 `topology_id` 是当前拓扑还是目标拓扑：

- `phase.topology_id`：当前 phase 实际使用的拓扑；
- `route_plan.source_topology_id`：提交前拓扑，必须等于当前 phase 的 `topology_id`；
- `route_plan.target_topology_id`：RELEASE 后下一 epoch 使用的拓扑。

## 3. 字段语义

| 字段 | 类型 | 约束与用途 |
|---|---|---|
| `schema_version` | string | 固定为 `ocs-pccl.execution-plan.v1` |
| `job_id` | string | 训练任务身份，隔离不同 job |
| `plan_id` | string | 一份不可变执行计划的身份；内容变化必须换 ID |
| `group_id` | uint | DP/TP/EP 等通信组身份 |
| `rank_list` | uint[] | 参与 rank 的唯一、有序列表，是成员关系的规范来源 |
| `participant_bitmap` | hex string | `rank_list` 的派生值；wire v1 目前限制为 64 rank |
| `phase_id` | uint | plan 内从 0 开始、连续且唯一 |
| `epoch` | uint | 当前 phase 的全局通信 epoch，不随 `phase_id` 回绕 |
| `op_type` | string | `allreduce/alltoall/.../custom`；扩展值使用 `x-` 前缀 |
| `algorithm_type` | string | `ring/rhd/tree/direct/...`；不是 kernel 文件名 |
| `backend` | enum | `pccl` 或 `torch` |
| `topology_id` | uint | 当前 phase 的逻辑/物理拓扑版本 |
| `artifact_id` | string/null | 自动算法生成器产生并验证的可执行 artifact |
| `graph_digest` | string/null | PCCL primitive DAG 的 SHA-256，防止同名算法实际图不同 |
| `barrier_id` | uint | `(job_id, group_id)` 内单调且不复用的 barrier 身份 |
| `next_epoch` | uint | barrier RELEASE 后进入的 epoch，必须大于当前 `epoch` |
| `next_phase_id` | uint | RELEASE 后应执行的 phase；最后一个 barrier 可指向下一轮 phase 0 |
| `switch_action` | enum | v1 仅支持 `KEEP` 或 `APPLY_ROUTE` |
| `route_plan` | object/null | `APPLY_ROUTE` 时必填；`KEEP` 时必须为 null |

`route_plan` 包含：

```text
route_plan_id
route_mode
source_topology_id
target_topology_id
payload
```

`payload` 是厂商/控制器可扩展对象，但必须参与 canonical plan digest。运行时不得只比较 `route_plan_id` 而忽略 payload。

## 4. 跨字段不变量

JSON Schema 验证基本类型；控制器、编译器和 runtime 还必须验证：

1. `rank_list` 严格递增，且计算出的 bitmap 等于 `participant_bitmap`；
2. `phase_id` 从 0 连续增长，不重复；
3. 每个非最终 phase 必须有 `barrier_after`；最终 phase 可用 barrier 进入下一轮；
4. `barrier_id` 在 `(job_id, group_id)` 生命周期内不复用；
5. `next_epoch > epoch`；非最终 phase 的下一 phase 必须满足 `epoch == next_epoch`，最终 barrier 进入下一轮时由下一份 materialized plan 验证；
6. `source_topology_id == phase.topology_id`；
7. `target_topology_id == next phase.topology_id`；跨 plan 时由下一 plan 验证；
8. 所有 rank 必须对同一 canonical plan digest 和 `graph_digest` 发送 READY；
9. 只有 controller 返回 `LINK_ALIGNED`，runtime 才能 RELEASE 下一 phase；
10. 同一 `(plan_id, phase_id, barrier_id, epoch)` 的重复消息是重传，不得重复切换。

## 5. 多轮规则

一轮完整序列为：

```text
phase 0 allreduce -> barrier 0
phase 1 alltoall  -> barrier 1
phase 2 allreduce -> barrier 2
```

下一轮必须使用新的 `plan_id` 或新的 materialized plan instance，并继续递增 `epoch` 和 `barrier_id`：

```text
round 0: epoch 0/1/2, barrier 0/1/2
round 1: epoch 3/4/5, barrier 3/4/5
round 2: epoch 6/7/8, barrier 6/7/8
```

不得每轮复用相同 barrier identity。PCCL/Torch plan builder 通过 `include_final_barrier=True` 生成第三个 boundary。

## 6. MSCCL 对照

### 高层 DSL

经典 MSCCLang 是 chunk-oriented DSL：`chunk(rank, buffer, index, count)` 获取逻辑 chunk，`copy()` 搬运 chunk，`reduce()` 在目标位置归约。编译器跟踪 buffer 的 last-writer/last-reader，自动形成 Chunk DAG，再 lower 为 Instruction DAG。

### 经典 MSCCL XML primitive

| XML type | 含义 |
|---|---|
| `s` / `r` | send / receive |
| `cpy` / `re` | local copy / reduce |
| `rrc` | receive + reduce + copy |
| `rcs` | receive + copy + send |
| `rrs` | receive + reduce + send |
| `rrcs` | receive + reduce + copy + send |
| `nop` | 承载额外依赖 |

XML 结构是 `algo -> gpu -> tb -> step`。同一 threadblock 内按 step 顺序执行；跨 threadblock 依赖由 `depid`（依赖 TB）和 `deps`（依赖 step）表示，`hasdep` 表示其他操作会等待本 step。send/recv matching 构成通信依赖。

### Barrier 与 topology switch

- 经典 MSCCL XML 没有显式 barrier primitive；它用顺序和 cross-TB dependency 完成单个静态 algorithm 内同步。
- MSCCL++ DSL 增加了 `barrier/signal/wait/put/get/flush`，其中 barrier 是同一 GPU kernel/program 内的 threadblock 同步。
- MSCCL/MSCCL++ 的 topology 是 algorithm 编译/选择时的约束；上述 barrier 都不执行 OCS 物理拓扑重配，也没有 READY、route commit、link-ready、RELEASE 语义。

因此 OCS-PCCL 需要在 MSCCL 数据面模型之外增加：`job_id/plan_id/phase_id/group_id`、participant scope、epoch transition、`switch_action/route_plan`、link-ready 状态和跨 rank plan digest。

参考：

- [GC3/MSCCLang 论文](https://arxiv.org/abs/2201.11840)
- [MSCCL-tools 官方仓库](https://github.com/Azure/msccl-tools)
- [MSCCL synthesis 说明](https://github.com/Azure/msccl-tools/blob/main/SYNTHESIS.md)

## 7. 现有代码映射

| Schema v1 | 当前实现 |
|---|---|
| phase data op | `TorchCollectivePhase` / `OCSCollectivePhase` |
| phase primitive artifact | PCCL JSON v2 |
| multi-phase manifest | JSON v3 `phased_ocs` |
| barrier transition | `OCSPlan` + `OcsBarrierNode` |
| schema parser | `OCSExecutionPlan.load/from_json/from_dict` |
| backend lowering | `ExecutionPlanCompiler.compile_pccl/compile_torch` |
| algorithm_type | 当前 `OCSPlan.algorithm` |
| epoch/next_epoch | 当前 `epoch_id/next_epoch_id` |
| route_plan | `OCSRoutePlanSpec`，投影为 wire v1 route id 与 canonical payload |

当前实现已经由 parser 将 barrier transition 投影成旧 `OCSPlan`，保持 runtime 和 wire v1 可用。`OCSPlan.topology_id/algorithm/backend` 表示 RELEASE 后目标 phase 的属性；完整 `plan_id/phase_id/switch_action/route_plan` 放入 canonical payload 并受 plan digest 保护。

## 8. 加载与编译接口

```python
from pccl import OCSExecutionPlan, ExecutionPlanCompiler

plan = OCSExecutionPlan.load("examples/ocs_execution_plan_v1.json")
compiled = ExecutionPlanCompiler().compile(
    plan,
    rank=0,
    tensor_size=4096,
    dtype="float32",
    executor="sm",
)
```

编译流程为：

```text
Execution Plan JSON
  -> 类型与跨字段校验
  -> phase algorithm lowering
  -> OCSCollectivePlan / TorchCollectivePlan
  -> 每 phase PCCL JSON v2 artifact
  -> OCS barrier READY / switch / LINK_ALIGNED / RELEASE
```

当前 PCCL lowering 支持 `allreduce/alltoall/allgather/reducescatter`，其中 `direct` 映射为 pairwise all-to-all，`auto` 对 all-reduce 调用现有 selector。`broadcast/reduce/custom/hierarchical` 会明确报未实现，不会静默回退。

`graph_digest` 非空时，编译器会计算生成的 JSON v2 canonical SHA-256 并拒绝不一致的 artifact。字符串 `route_plan_id` 使用稳定哈希投影到 wire v1 的 uint64 字段，原始 route id 和 payload 仍完整保留在 canonical payload 中。

MSCCL 风格 Algorithm IR v1 已提供 `chunk/step/copy/reduce/sync/dependency`，并可生成 Ring AllReduce 与 Direct AllToAll；详细接口见 [`ALGORITHM_IR_V1.md`](ALGORITHM_IR_V1.md)。tree/RHD 和其他 collective 仍由旧 Python 模板生成，尚未迁移。

对 AICCL/RLCCL 等外部生成器，`artifact_id` 现在直接引用标准 MSCCL XML schedule。`ExecutionPlanCompiler(algorithm_lowering="msccl", artifact_resolver=...)` 会导入 XML，校验 `ngpus/coll` 与 phase 一致，再 lower 为 PCCL Primitive IR。OCS topology plan 仍保留在 `topology_id/barrier_after/route_plan`，不写入 MSCCL XML。接口和兼容范围见 [`MSCCL_COMPATIBILITY.md`](MSCCL_COMPATIBILITY.md)。
