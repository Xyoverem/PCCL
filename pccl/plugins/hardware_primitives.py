"""
Hardware primitives registry and L2 to L3 lowering interface.

This module implements the plugin-based lowering system from primitive IR
(Layer 2) to hardware-specific primitives (Layer 3).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any, Callable
from enum import Enum
import logging

from ..ir.primitive_ir import *
from ..ir.json_serializer import IRGraph, IROperation, IRValue
from ..ir.cuda_primitives import *
from ..ir.rdma_primitives import *
from ..passes.base import Pass, PassResult, PassContext

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    CUDA = "cuda"
    RDMA = "rdma"
    CPU = "cpu"
    ROCM = "rocm"


class HardwareLoweringPass(ABC):
    """Abstract base class for hardware-specific lowering passes."""

    def __init__(self, hardware_type: HardwareType):
        self.hardware_type = hardware_type

    @abstractmethod
    def lower_write(self, write_op: WriteOp, context: PassContext) -> List[IROperation]:
        """Lower WriteOp to hardware primitives."""
        pass

    @abstractmethod
    def lower_reduce(self, reduce_op: ReduceOp, context: PassContext) -> List[IROperation]:
        """Lower ReduceOp to hardware primitives."""
        pass

    @abstractmethod
    def lower_copy(self, copy_op: CopyOp, context: PassContext) -> List[IROperation]:
        """Lower CopyOp to hardware primitives."""
        pass

    @abstractmethod
    def lower_signal(self, signal_op: SignalOp, context: PassContext) -> List[IROperation]:
        """Lower SignalOp to hardware primitives."""
        pass

    @abstractmethod
    def lower_wait_signal(self, wait_op: WaitSignalOp, context: PassContext) -> List[IROperation]:
        """Lower WaitSignalOp to hardware primitives."""
        pass

    @abstractmethod
    def get_supported_operations(self) -> List[str]:
        """Return list of supported primitive operations."""
        pass


class CUDALoweringPass(HardwareLoweringPass):
    """CUDA-specific lowering from primitive IR to CUDA hardware primitives."""

    def __init__(self):
        super().__init__(HardwareType.CUDA)
        self.builder = CUDAHardwarePrimitiveIRBuilder()

    def lower_write(self, write_op: WriteOp, context: PassContext) -> List[IROperation]:
        """Lower WriteOp to CUDA memory operations."""
        ops = []

        # Determine if this is a device write or host-to-device transfer
        target_device = write_op.chunk.device

        if target_device.type == DeviceType.CUDA:
            # Device memory write
            write_primitive = self.builder.create_multimem_write(
                src=write_op.source,
                dst=write_op.chunk,
                elements=write_op.chunk.size // write_op.chunk.dtype.size
            )
            ops.append(write_primitive)

            # Add memory fence if needed
            fence = self.builder.create_memory_fence(
                scope="device",
                semantics="acquire_release"
            )
            ops.append(fence)

        else:
            # Host-to-device transfer
            h2d = self.builder.create_memcpy_h2d(
                host_src=write_op.source,
                device_dst=write_op.chunk,
                size=write_op.chunk.size
            )
            ops.append(h2d)

        return ops

    def lower_reduce(self, reduce_op: ReduceOp, context: PassContext) -> List[IROperation]:
        """Lower ReduceOp to CUDA reduction primitives."""
        ops = []
        inputs = reduce_op.inputs
        output = reduce_op.output

        # Use multi-memory reduction for device-local reduction
        if reduce_op.reduce_type == ReduceOp.REDUCE_SUM:
            if len(inputs) == 2:
                # Binary reduction
                reduce_primitive = self.builder.create_multimem_reduce(
                    src_a=inputs[0],
                    src_b=inputs[1],
                    dst=output,
                    reduce_op="add",
                    elements=output.size // output.dtype.size
                )
                ops.append(reduce_primitive)
            else:
                # Multi-input reduction - chain binary reductions
                current_result = inputs[0]
                for i in range(1, len(inputs)):
                    temp_result = Chunk(
                        f"reduce_temp_{i}",
                        output.size,
                        output.dtype,
                        output.device
                    )

                    reduce_primitive = self.builder.create_multimem_reduce(
                        src_a=current_result,
                        src_b=inputs[i],
                        dst=temp_result,
                        reduce_op="add",
                        elements=output.size // output.dtype.size
                    )
                    ops.append(reduce_primitive)
                    current_result = temp_result

                # Copy final result to output
                copy_primitive = self.builder.create_device_memcpy(
                    src=current_result,
                    dst=output,
                    size=output.size
                )
                ops.append(copy_primitive)

        elif reduce_op.reduce_type == ReduceOp.REDUCE_MAX:
            # Max reduction using warp-level primitives
            if len(inputs) == 2:
                reduce_primitive = self.builder.create_warp_level_reduce(
                    src=inputs[0],
                    dst=output,
                    reduce_op="max",
                    elements=output.size // output.dtype.size
                )
                ops.append(reduce_primitive)

        return ops

    def lower_copy(self, copy_op: CopyOp, context: PassContext) -> List[IROperation]:
        """Lower CopyOp to CUDA memory copy primitives."""
        ops = []
        src_device = copy_op.src_chunk.device
        dst_device = copy_op.dst_chunk.device

        if src_device.type == DeviceType.CUDA and dst_device.type == DeviceType.CUDA:
            if src_device.id == dst_device.id:
                # Device-local copy
                copy_primitive = self.builder.create_device_memcpy(
                    src=copy_op.src_chunk,
                    dst=copy_op.dst_chunk,
                    size=copy_op.size
                )
                ops.append(copy_primitive)
            else:
                # Peer-to-peer copy
                p2p_copy = self.builder.create_peer_to_peer_memcpy(
                    src=copy_op.src_chunk,
                    dst=copy_op.dst_chunk,
                    size=copy_op.size,
                    src_device=src_device.id,
                    dst_device=dst_device.id
                )
                ops.append(p2p_copy)

        elif src_device.type == DeviceType.CPU and dst_device.type == DeviceType.CUDA:
            # Host-to-device
            h2d = self.builder.create_memcpy_h2d(
                host_src=copy_op.src_chunk,
                device_dst=copy_op.dst_chunk,
                size=copy_op.size
            )
            ops.append(h2d)

        elif src_device.type == DeviceType.CUDA and dst_device.type == DeviceType.CPU:
            # Device-to-host
            d2h = self.builder.create_memcpy_d2h(
                device_src=copy_op.src_chunk,
                host_dst=copy_op.dst_chunk,
                size=copy_op.size
            )
            ops.append(d2h)

        return ops

    def lower_signal(self, signal_op: SignalOp, context: PassContext) -> List[IROperation]:
        """Lower SignalOp to CUDA signaling primitives."""
        ops = []

        # Use CUDA event for signaling
        signal_primitive = self.builder.create_event_signal(
            event_id=signal_op.signal_id,
            device=signal_op.device.id if signal_op.device else 0
        )
        ops.append(signal_primitive)

        return ops

    def lower_wait_signal(self, wait_op: WaitSignalOp, context: PassContext) -> List[IROperation]:
        """Lower WaitSignalOp to CUDA wait primitives."""
        ops = []

        # Use CUDA event wait
        wait_primitive = self.builder.create_event_wait(
            event_id=wait_op.signal_id,
            device=wait_op.device.id if wait_op.device else 0
        )
        ops.append(wait_primitive)

        return ops

    def get_supported_operations(self) -> List[str]:
        return ["write", "reduce", "copy", "signal", "wait_signal"]


class RDMALoweringPass(HardwareLoweringPass):
    """RDMA-specific lowering from primitive IR to RDMA hardware primitives."""

    def __init__(self):
        super().__init__(HardwareType.RDMA)
        self.builder = RDMAHardwarePrimitiveIRBuilder()

    def lower_write(self, write_op: WriteOp, context: PassContext) -> List[IROperation]:
        """Lower WriteOp to RDMA write operations."""
        ops = []

        # Create memory region for the target
        mr = self.builder.create_memory_region(
            name=f"mr_{write_op.chunk.name}",
            addr=write_op.chunk.name,
            size=write_op.chunk.size,
            access="RW"
        )
        ops.append(mr)

        # RDMA write operation
        rdma_write = self.builder.create_rdma_write(
            qp_id=0,  # Default queue pair
            remote_addr=write_op.chunk.name,
            length=write_op.chunk.size,
            src=write_op.source
        )
        ops.append(rdma_write)

        return ops

    def lower_reduce(self, reduce_op: ReduceOp, context: PassContext) -> List[IROperation]:
        """Lower ReduceOp to RDMA atomic operations."""
        ops = []

        if reduce_op.reduce_type == ReduceOp.REDUCE_SUM and len(reduce_op.inputs) == 2:
            # Use RDMA atomic fetch-add for sum reduction
            atomic_op = self.builder.create_atomic_fetch_add(
                qp_id=0,
                remote_addr=reduce_op.output.name,
                addend=reduce_op.inputs[1].name,
                result=reduce_op.output.name
            )
            ops.append(atomic_op)

        else:
            # For complex reductions, need to implement custom logic
            # For now, fall back to RDMA read + local compute + write
            for input_chunk in reduce_op.inputs:
                rdma_read = self.builder.create_rdma_read(
                    qp_id=0,
                    remote_addr=input_chunk.name,
                    length=input_chunk.size,
                    dst=f"temp_{input_chunk.name}"
                )
                ops.append(rdma_read)

        return ops

    def lower_copy(self, copy_op: CopyOp, context: PassContext) -> List[IROperation]:
        """Lower CopyOp to RDMA read/write operations."""
        ops = []

        # Create memory regions
        src_mr = self.builder.create_memory_region(
            name=f"mr_src_{copy_op.src_chunk.name}",
            addr=copy_op.src_chunk.name,
            size=copy_op.src_chunk.size,
            access="R"
        )
        dst_mr = self.builder.create_memory_region(
            name=f"mr_dst_{copy_op.dst_chunk.name}",
            addr=copy_op.dst_chunk.name,
            size=copy_op.dst_chunk.size,
            access="W"
        )
        ops.extend([src_mr, dst_mr])

        # RDMA write operation
        rdma_write = self.builder.create_rdma_write(
            qp_id=0,
            remote_addr=copy_op.dst_chunk.name,
            length=copy_op.size,
            src=copy_op.src_chunk.name
        )
        ops.append(rdma_write)

        return ops

    def lower_signal(self, signal_op: SignalOp, context: PassContext) -> List[IROperation]:
        """Lower SignalOp to RDMA send operation."""
        ops = []

        # Use RDMA send for signaling
        send_signal = self.builder.create_rdma_send(
            qp_id=0,
            message=f"signal_{signal_op.signal_id}",
            length=1
        )
        ops.append(send_signal)

        return ops

    def lower_wait_signal(self, wait_op: WaitSignalOp, context: PassContext) -> List[IROperation]:
        """Lower WaitSignalOp to RDMA receive operation."""
        ops = []

        # Poll completion queue for signal
        cq_poll = self.builder.create_completion_queue_poll(
            cq_id=0,
            num_entries=1
        )
        ops.append(cq_poll)

        return ops

    def get_supported_operations(self) -> List[str]:
        return ["write", "reduce", "copy", "signal", "wait_signal"]


class HardwarePrimitiveRegistry:
    """Global registry for hardware lowering passes."""

    def __init__(self):
        self._lowering_passes: Dict[HardwareType, HardwareLoweringPass] = {}
        self._capabilities: Dict[HardwareType, Dict[str, Any]] = {}

        # Register default passes
        self._register_default_passes()

    def _register_default_passes(self):
        """Register built-in hardware lowering passes."""
        self.register_lowering_pass(HardwareType.CUDA, CUDALoweringPass())
        self.register_lowering_pass(HardwareType.RDMA, RDMALoweringPass())

    def register_lowering_pass(self, hardware_type: HardwareType, pass_instance: HardwareLoweringPass):
        """Register a hardware lowering pass."""
        self._lowering_passes[hardware_type] = pass_instance

        # Store capabilities
        self._capabilities[hardware_type] = {
            "supported_operations": pass_instance.get_supported_operations(),
            "pass_class": pass_instance.__class__.__name__
        }

        logger.info(f"Registered {pass_instance.__class__.__name__} for {hardware_type.value}")

    def get_lowering_pass(self, hardware_type: HardwareType) -> Optional[HardwareLoweringPass]:
        """Get lowering pass for specific hardware type."""
        return self._lowering_passes.get(hardware_type)

    def get_supported_hardware_types(self) -> List[HardwareType]:
        """Get list of supported hardware types."""
        return list(self._lowering_passes.keys())

    def get_capabilities(self, hardware_type: HardwareType) -> Optional[Dict[str, Any]]:
        """Get capabilities for specific hardware type."""
        return self._capabilities.get(hardware_type)


class PrimitiveToHardwareLoweringPass(Pass):
    """Main L2 to L3 lowering pass that delegates to hardware-specific passes."""

    def __init__(self, hardware_type: HardwareType):
        super().__init__(
            name="primitive_to_hardware",
            description=f"Lower primitive IR to {hardware_type.value} hardware primitives"
        )
        self.hardware_type = hardware_type
        self.registry = HardwarePrimitiveRegistry()

        # Set capabilities
        self.capabilities.preserves_structure = False
        self.capabilities.supported_operations = {"write", "reduce", "copy", "signal", "wait_signal"}

    def apply(self, ir_graph: IRGraph, context: PassContext) -> PassResult:
        """Apply hardware-specific lowering to the IR graph."""
        # Get the appropriate lowering pass
        lowering_pass = self.registry.get_lowering_pass(self.hardware_type)
        if not lowering_pass:
            raise ValueError(f"No lowering pass registered for {self.hardware_type.value}")

        # Create new IR graph for hardware primitives
        hw_graph = IRGraph(
            values={},
            operations={},
            metadata={
                **ir_graph.metadata,
                "hardware_type": self.hardware_type.value,
                "ir_layer": "hardware_primitives"
            }
        )

        lowered_ops = []
        op_mapping = {}

        # Process each operation in the original graph
        for op_id, operation in ir_graph.operations.items():
            if operation.op_type == "write":
                # Reconstruct WriteOp from IROperation
                write_op = self._reconstruct_write_op(operation, ir_graph)
                hw_ops = lowering_pass.lower_write(write_op, context)

            elif operation.op_type == "reduce":
                # Reconstruct ReduceOp from IROperation
                reduce_op = self._reconstruct_reduce_op(operation, ir_graph)
                hw_ops = lowering_pass.lower_reduce(reduce_op, context)

            elif operation.op_type == "copy":
                # Reconstruct CopyOp from IROperation
                copy_op = self._reconstruct_copy_op(operation, ir_graph)
                hw_ops = lowering_pass.lower_copy(copy_op, context)

            elif operation.op_type == "signal":
                # Reconstruct SignalOp from IROperation
                signal_op = self._reconstruct_signal_op(operation, ir_graph)
                hw_ops = lowering_pass.lower_signal(signal_op, context)

            elif operation.op_type == "wait_signal":
                # Reconstruct WaitSignalOp from IROperation
                wait_op = self._reconstruct_wait_signal_op(operation, ir_graph)
                hw_ops = lowering_pass.lower_wait_signal(wait_op, context)

            else:
                # Unknown operation type, keep as-is
                hw_ops = [operation]

            # Add lowered operations to hardware graph
            for hw_op in hw_ops:
                hw_op_id = f"{op_id}_hw_{len(lowered_ops)}"
                hw_graph.operations[hw_op_id] = hw_op

                # Add values from hardware operation
                for input_name in hw_op.inputs:
                    if input_name not in hw_graph.values:
                        hw_graph.values[input_name] = IRValue(
                            name=input_name,
                            dtype="unknown",
                            shape=None
                        )

                for output_name in hw_op.outputs:
                    if output_name not in hw_graph.values:
                        hw_graph.values[output_name] = IRValue(
                            name=output_name,
                            dtype="unknown",
                            shape=None
                        )

                lowered_ops.append(hw_op_id)

            # Track mapping from original to hardware operations
            op_mapping[op_id] = lowered_ops

        # Create transformation statistics
        stats = {
            "original_operations": len(ir_graph.operations),
            "hardware_operations": len(hw_graph.operations),
            "hardware_type": self.hardware_type.value,
            "expansion_ratio": len(hw_graph.operations) / len(ir_graph.operations) if ir_graph.operations else 0,
            "operation_mapping": op_mapping
        }

        return PassResult(
            success=True,
            ir_graph=hw_graph,
            message=f"Successfully lowered to {self.hardware_type.value} hardware primitives",
            transformation_stats=stats
        )

    def _reconstruct_write_op(self, operation: IROperation, ir_graph: IRGraph) -> WriteOp:
        """Reconstruct WriteOp from IROperation."""
        # This is a simplified reconstruction - in practice, you'd need proper
        # serialization/deserialization of the original operation objects
        chunk = Chunk(
            name=operation.outputs[0] if operation.outputs else "output",
            size=operation.attributes.get("size", 0),
            dtype=DataType.FLOAT32,  # Default
            device=Device(0, DeviceType.CUDA)  # Default
        )

        return WriteOp(chunk=chunk, source=operation.inputs[0] if operation.inputs else None)

    def _reconstruct_reduce_op(self, operation: IROperation, ir_graph: IRGraph) -> ReduceOp:
        """Reconstruct ReduceOp from IROperation."""
        output = Chunk(
            name=operation.outputs[0] if operation.outputs else "output",
            size=operation.attributes.get("size", 0),
            dtype=DataType.FLOAT32,
            device=Device(0, DeviceType.CUDA)
        )

        inputs = []
        for input_name in operation.inputs:
            input_chunk = Chunk(
                name=input_name,
                size=output.size,
                dtype=output.dtype,
                device=output.device
            )
            inputs.append(input_chunk)

        return ReduceOp(
            inputs=inputs,
            output=output,
            reduce_type=operation.attributes.get("reduce_type", "sum")
        )

    def _reconstruct_copy_op(self, operation: IROperation, ir_graph: IRGraph) -> CopyOp:
        """Reconstruct CopyOp from IROperation."""
        src_chunk = Chunk(
            name=operation.inputs[0] if operation.inputs else "src",
            size=operation.attributes.get("size", 0),
            dtype=DataType.FLOAT32,
            device=Device(0, DeviceType.CUDA)
        )

        dst_chunk = Chunk(
            name=operation.outputs[0] if operation.outputs else "dst",
            size=operation.attributes.get("size", 0),
            dtype=DataType.FLOAT32,
            device=Device(0, DeviceType.CUDA)
        )

        return CopyOp(
            src_chunk=src_chunk,
            dst_chunk=dst_chunk,
            size=operation.attributes.get("size", 0)
        )

    def _reconstruct_signal_op(self, operation: IROperation, ir_graph: IRGraph) -> SignalOp:
        """Reconstruct SignalOp from IROperation."""
        return SignalOp(
            signal_id=operation.attributes.get("signal_id", 0),
            device=Device(0, DeviceType.CUDA)
        )

    def _reconstruct_wait_signal_op(self, operation: IROperation, ir_graph: IRGraph) -> WaitSignalOp:
        """Reconstruct WaitSignalOp from IROperation."""
        return WaitSignalOp(
            signal_id=operation.attributes.get("signal_id", 0),
            device=Device(0, DeviceType.CUDA)
        )


# Global registry instance
_hardware_registry = HardwarePrimitiveRegistry()


def register_lowering_pass(hardware_type: HardwareType, pass_instance: HardwareLoweringPass):
    """Register a hardware lowering pass globally."""
    _hardware_registry.register_lowering_pass(hardware_type, pass_instance)


def get_hardware_primitives_for_device(hardware_type: HardwareType) -> Optional[HardwareLoweringPass]:
    """Get hardware primitives for specific device type."""
    return _hardware_registry.get_lowering_pass(hardware_type)


def create_lowering_pass_for_device(hardware_type: HardwareType) -> PrimitiveToHardwareLoweringPass:
    """Create a lowering pass for specific hardware type."""
    return PrimitiveToHardwareLoweringPass(hardware_type)


def get_supported_hardware_types() -> List[HardwareType]:
    """Get list of supported hardware types."""
    return _hardware_registry.get_supported_hardware_types()