"""
Distributed Training Example with PCCL

This example demonstrates how to use PCCL's Python DSL to implement
distributed training with AllReduce operations for gradient synchronization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np

import pccl
from pccl.lang import (
    communication, allreduce, broadcast, hierarchical_topology,
    ConfigBuilder, TopologyDiscovery, TopologyOptimizer
)

@communication
class DistributedTrainingAllreduce:
    """AllReduce operation for gradient synchronization"""
    gradient_allreduce: pccl.Allreduce = allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=[0, 1, 2, 3],
        enable_overlap=True,
        buffer_size=64 * 1024 * 1024
    )

    param_broadcast: pccl.Broadcast = broadcast(
        root_rank=0,
        participants=[0, 1, 2, 3],
        buffer_size=32 * 1024 * 1024
    )

def simple_model():
    """Create a simple neural network for demonstration"""
    if not TORCH_AVAILABLE:
        class DummyModel:
            def __init__(self):
                self.parameters = lambda: [np.random.randn(100, 100) for _ in range(5)]
                def __call__(self, x):
                    return x
            def zero_grad(self):
                pass
        return DummyModel()

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    return SimpleModel()

def create_synthetic_data(batch_size=32, num_classes=10):
    """Create synthetic training data"""
    if TORCH_AVAILABLE:
        data = torch.randn(batch_size, 784)
        labels = torch.randint(0, num_classes, (batch_size,))
        return data, labels
    else:
        data = np.random.randn(batch_size, 784).astype(np.float32)
        labels = np.random.randint(0, num_classes, (batch_size,))
        return data, labels

def demonstrate_basic_communication():
    """Demonstrate basic PCCL communication operations"""
    print("=== Basic PCCL Communication Demo ===")

    devices = [0, 1, 2, 3]
    data_size = 1024 * 1024

    if TORCH_AVAILABLE:
        input_tensor = torch.randn(data_size // 4)
        output_tensor = torch.zeros_like(input_tensor)
    else:
        input_tensor = np.random.randn(data_size // 4).astype(np.float32)
        output_tensor = np.zeros_like(input_tensor)

    print(f"Input data shape: {input_tensor.shape}")
    print(f"Participants: {devices}")

    allreduce_op = pccl.Allreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=devices,
        buffer_size=data_size
    )

    try:
        plan = allreduce_op.compile(devices)
        print(f"Compiled execution plan with {len(plan.operations)} operations")

        cost_estimate = allreduce_op.estimate_cost(devices)
        print(f"Estimated execution cost: {cost_estimate}")

        result = allreduce_op.execute(input_tensor, participants=devices)
        print("AllReduce executed successfully")

    except Exception as e:
        print(f"Execution failed: {e}")

def demonstrate_topology_discovery():
    """Demonstrate automatic topology discovery"""
    print("\n=== Topology Discovery Demo ===")

    discovery = TopologyDiscovery()
    discovered_devices = discovery.discover_devices()

    print(f"Discovered {len(discovered_devices)} devices:")
    for device in discovered_devices:
        print(f"  Device {device.device_id}: {device.device_type}, "
              f"Bandwidth: {device.memory_bandwidth} GB/s")

    topology = discovery.discover_topology(num_devices=4)
    print(f"\nDiscovered topology type: {topology['type']}")
    print(f"Total bandwidth: {topology['metrics'].total_bandwidth:.1f} GB/s")
    print(f"Average latency: {topology['metrics'].average_latency:.1f} μs")

def demonstrate_hierarchical_topology():
    """Demonstrate hierarchical topology configuration"""
    print("\n=== Hierarchical Topology Demo ===")

    node_groups = [
        [0, 1],    # Node 0: GPUs 0,1
        [2, 3],    # Node 1: GPUs 2,3
        [4, 5],    # Node 2: GPUs 4,5
        [6, 7]     # Node 3: GPUs 6,7
    ]

    topology = hierarchical_topology(
        node_groups=node_groups,
        intra_bandwidth=50.0,  # NVLink bandwidth
        inter_bandwidth=10.0   # RDMA bandwidth
    )

    print(f"Created hierarchical topology for {len(node_groups)} nodes")
    print(f"Total devices: {len(topology['devices'])}")
    print(f"Intra-node bandwidth: 50.0 GB/s")
    print(f"Inter-node bandwidth: 10.0 GB/s")

    allreduce_config = ConfigBuilder.hierarchical_allreduce(
        participants=[0, 1, 2, 3, 4, 5, 6, 7],
        node_size=2
    )

    allreduce_op = pccl.Allreduce(
        reduce_op="sum",
        algorithm="rabenseifner",
        participants=[0, 1, 2, 3, 4, 5, 6, 7],
        topology=allreduce_config.topology
    )

    plan = allreduce_op.compile()
    print(f"Compiled hierarchical AllReduce plan")

def demonstrate_distributed_training():
    """Demonstrate distributed training workflow"""
    print("\n=== Distributed Training Demo ===")

    world_size = 4
    rank = 0

    model = simple_model()
    if TORCH_AVAILABLE:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = None

    print(f"Initialized model on rank {rank}")

    allreduce_op = DistributedTrainingAllreduce()
    training_plan = allreduce_op.compile(participants=list(range(world_size)))

    print(f"Compiled distributed training plan")

    num_epochs = 2
    num_batches = 3

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx in range(num_batches):
            if optimizer:
                optimizer.zero_grad()

            data, labels = create_synthetic_data()

            if TORCH_AVAILABLE:
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, labels)
                loss.backward()

                grads = [p.grad for p in model.parameters() if p.grad is not None]
            else:
                grads = [np.random.randn(100, 100) for _ in range(5)]
                loss = np.random.random()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: "
                  f"Loss = {loss.item() if TORCH_AVAILABLE else loss:.4f}")

            try:
                if grads:
                    grad_sizes = [g.size if TORCH_AVAILABLE else g.shape for g in grads]
                    total_grad_size = sum(np.prod(s) for s in grad_sizes)
                    print(f"  Total gradient size: {total_grad_size}")

                allreduce_result = allreduce_op.execute(grads[0] if grads else data)

                if optimizer:
                    optimizer.step()

                total_loss += loss.item() if TORCH_AVAILABLE else loss

            except Exception as e:
                print(f"  Training step failed: {e}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

def demonstrate_custom_operator():
    """Demonstrate creating custom communication operators"""
    print("\n=== Custom Operator Demo ===")

    custom_allreduce = pccl.Allreduce(
        reduce_op="sum",
        algorithm="tree",
        participants=[0, 1, 2, 3],
        buffer_size=256 * 1024 * 1024,
        enable_overlap=True
    )

    print(f"Created custom AllReduce with tree algorithm")
    print(f"Algorithm: {custom_allreduce.config.algorithm.name}")

    plan = custom_allreduce.compile()
    print(f"Custom operator plan compiled")

    metrics = plan.topology['metrics']
    print(f"Topology metrics:")
    print(f"  Total bandwidth: {metrics.total_bandwidth:.1f} GB/s")
    print(f"  Connectivity: {metrics.connectivity:.1f}%")

def demonstrate_pipeline_allreduce():
    """Demonstrate pipeline AllReduce with compute overlap"""
    print("\n=== Pipeline AllReduce Demo ===")

    pipeline_allreduce = pccl.PipelineAllreduce(
        reduce_op="sum",
        algorithm="ring",
        participants=[0, 1, 2, 3],
        compute_chunks=4,
        communication_chunks=2,
        buffer_size=128 * 1024 * 1024
    )

    print(f"Created Pipeline AllReduce:")
    print(f"  Compute chunks: {pipeline_allreduce.config.compute_chunks}")
    print(f"  Communication chunks: {pipeline_allreduce.config.communication_chunks}")
    print(f"  Overlap enabled: {pipeline_allreduce.config.enable_overlap}")

    try:
        plan = pipeline_allreduce.compile()
        print(f"Pipeline AllReduce plan compiled successfully")

        input_data = np.random.randn(1024, 1024).astype(np.float32)
        result = pipeline_allreduce.execute(input_data)
        print("Pipeline AllReduce executed successfully")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")

def main():
    """Main demonstration function"""
    print("PCCL Distributed Training Example")
    print("=" * 50)

    try:
        demonstrate_basic_communication()
        demonstrate_topology_discovery()
        demonstrate_hierarchical_topology()
        demonstrate_distributed_training()
        demonstrate_custom_operator()
        demonstrate_pipeline_allreduce()

        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")

    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()