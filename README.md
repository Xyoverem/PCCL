# PCCL - Parallel Communication and Computing Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PCCL 是一个高性能的并行通信与计算库，专为多 GPU 环境设计，提供高效的集合通信原语和可编程的通信优化框架。

## 特性

- **高性能集合通信**：AllReduce、AllGather、ReduceScatter 等操作的优化实现
- **多执行后端支持**：
  - SM（Streaming Multiprocessor）操作
  - TMA（Tensor Memory Accelerator）硬件加速
  - CE（Copy Engine）复制引擎
  - RDMA 远程直接内存访问
- **DSL 与超级优化器**：基于 DSL 的通信图编译与自动优化
- **算法模板**：Ring、Recursive Halving-Doubling、Binary Tree 等多种算法
- **多通道支持**：利用多通道实现更好的带宽利用率
- **融合内核**：减少调度开销的融合执行模式

## 系统要求

- NVIDIA GPU（计算能力 90a 或更高，如 H100）
- CUDA Toolkit
- PyTorch
- RDMA 支持（可选，用于跨节点通信）

## 安装

```bash
# 从源码安装
pip install .

# 或开发模式安装
pip install -e .
```

## 快速开始

```python
import torch
import torch.distributed as dist
import pccl

# 初始化引擎
engine = pccl.get_engine()

# 注册并执行操作
pccl.register_operation("my_op", "graph.json")
input_tensor = torch.randn(1024, 1024, device='cuda')
output_tensor = pccl.execute_operation("my_op", input_tensor)
```

## 使用 DSL 构建通信图

```python
from pccl import (
    DeviceType, ExecutorType, TensorInfo, ReduceOp,
    PrimitiveIRGraph, Stream, CommunicationOp,
    Compiler
)

# 创建 IR 图
graph = PrimitiveIRGraph(
    collective_type="allreduce",
    device_type=DeviceType.GPU,
    num_ranks=8
)

# 定义张量信息
tensor = TensorInfo(
    shape=(1024, 1024),
    dtype="float32",
    device=DeviceType.GPU
)

# 构建通信图（示例：简单的 allreduce）
stream = Stream(channel=0)
comm_op = CommunicationOp(
    op_type="reduce",
    tensor=tensor,
    reduce_op=ReduceOp.SUM,
    peer=1
)
stream.add_op(comm_op)

# 编译
compiler = Compiler(enable_superopt=True)
json_str = pccl.compile_to_json_string(graph, compiler)
```

## 性能

在 H100 GPU 集群上的性能表现（与 NCCL 对比）：

| GPUs | 512MB PCCL | 512MB NCCL | Ratio |
|-----:|-----------:|-----------:|------:|
| 2 | 246.4 GB/s | ~263 GB/s | 0.94× |
| 4 | 274.8 GB/s | ~302 GB/s | 0.91× |
| 8 | 282.0 GB/s | ~315 GB/s | 0.90× |

## 项目结构

```
pccl/
├── pccl/              # Python 包
│   ├── __init__.py    # 主入口
│   ├── engine.py      # 引擎 Python 包装
│   └── dsl/           # DSL 与编译器
├── csrc/              # C++ 源码
│   ├── engine_c.cc    # C++ 引擎实现
│   └── plugins/       # 设备插件
├── include/           # C++ 头文件
├── tests/             # 测试
└── docs/              # 文档
```

## 运行测试

```bash
# 运行单元测试
pytest tests/

# 运行性能基准测试
python tests/bench_algo_comparison.py
```

## 文档

- [OCS-PCCL Execution Plan Schema v1](docs/OCS_EXECUTION_PLAN_SCHEMA_V1.md)
- [机器可读 JSON Schema](schemas/ocs_execution_plan_v1.schema.json)
- [三 phase、三 barrier 示例](examples/ocs_execution_plan_v1.json)
- [MSCCL 风格 Algorithm IR v1](docs/ALGORITHM_IR_V1.md)
- [MSCCL XML schedule 兼容接口](docs/MSCCL_COMPATIBILITY.md)

Execution Plan 可以直接编译为现有 PCCL/Torch phase runner 使用的计划：

```python
from pccl import OCSExecutionPlan, ExecutionPlanCompiler

plan = OCSExecutionPlan.load("examples/ocs_execution_plan_v1.json")
compiled = ExecutionPlanCompiler().compile(
    plan, rank=0, tensor_size=4096, dtype="float32", executor="sm"
)
```

实验性自动算法生成模式：

```python
compiler = ExecutionPlanCompiler(algorithm_lowering="generated")
```

AICCL/RLCCL 等上层生成器输出标准 MSCCL XML 时：

```python
compiler = ExecutionPlanCompiler(
    algorithm_lowering="msccl",
    artifact_resolver=artifact_store.__getitem__,
)
compiled = compiler.compile(
    plan, rank=0, tensor_size=4096, dtype="float32", executor="sm"
)
```

`artifact_id` 引用 collective schedule；OCS topology、epoch 和 barrier 由外层 Execution Plan 描述。

生成路径与原手写模板的数据面 A/B：

```bash
bash tests/run_algorithm_ir_ab.sh --warmup 20 --iterations 100 --repeats 5
```

计时只覆盖已注册 graph 的 `execute_operation_async + sync_operation`；完整设计、正确性和双 A5000 结果见 [Algorithm IR v1](docs/ALGORITHM_IR_V1.md)。

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 作者

- **nastyapple** - 主要开发者
