
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .json_serializer import (
    IRValue, IROperation, IRGraph, IRType,
    PrimitiveOpType, ReduceOpType, DeviceType
)


class CollectiveOpType(Enum):
    ALLREDUCE = "allreduce"
    BROADCAST = "broadcast"
    ALLGATHER = "allgather"
    REDUCE_SCATTER = "reduce_scatter"
    REDUCE = "reduce"
    ALLTOALL = "alltoall"


@dataclass
class CollectiveOperation:
    id: str
    op_type: CollectiveOpType
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any]

    def __post_init__(self):
        if "reduce_op" not in self.attributes and self.op_type in [
            CollectiveOpType.ALLREDUCE, CollectiveOpType.REDUCE, CollectiveOpType.REDUCE_SCATTER
        ]:
            self.attributes["reduce_op"] = ReduceOpType.SUM.value

        if "root_rank" not in self.attributes and self.op_type in [
            CollectiveOpType.BROADCAST, CollectiveOpType.REDUCE
        ]:
            self.attributes["root_rank"] = 0

        if "world_size" not in self.attributes:
            self.attributes["world_size"] = 2

        if "rank" not in self.attributes:
            self.attributes["rank"] = 0


class PrimitiveOperation(IROperation):

    def __init__(self,
                 id: str,
                 op_type: PrimitiveOpType,
                 inputs: List[str],
                 outputs: List[str],
                 attributes: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            id=id,
            op_type=op_type.value,
            inputs=inputs,
            outputs=outputs,
            attributes=attributes or {},
            metadata=metadata or {}
        )
        self.primitive_type = op_type


class WriteOp(PrimitiveOperation):

    def __init__(self,
                 id: str,
                 input_value: str,
                 output_value: str,
                 address: Optional[int] = None,
                 size: Optional[int] = None,
                 device_id: int = 0,
                 device_type: DeviceType = DeviceType.CPU):
        super().__init__(
            id=id,
            op_type=PrimitiveOpType.WRITE,
            inputs=[input_value],
            outputs=[output_value]
        )

        self.attributes.update({
            "address": address,
            "size": size,
            "device_id": device_id,
            "device_type": device_type.value
        })


class ReduceOp(PrimitiveOperation):


    def __init__(self,
                 id: str,
                 input_values: List[str],
                 output_value: str,
                 reduce_op: ReduceOpType = ReduceOpType.SUM,
                 device_id: int = 0,
                 device_type: DeviceType = DeviceType.CPU):
        super().__init__(
            id=id,
            op_type=PrimitiveOpType.REDUCE,
            inputs=input_values,
            outputs=[output_value]
        )

        self.attributes.update({
            "reduce_op": reduce_op.value,
            "device_id": device_id,
            "device_type": device_type.value,
            "num_inputs": len(input_values)
        })


class CopyOp(PrimitiveOperation):


    def __init__(self,
                 id: str,
                 input_value: str,
                 output_value: str,
                 src_device_id: int = 0,
                 dst_device_id: int = 0,
                 src_device_type: DeviceType = DeviceType.CPU,
                 dst_device_type: DeviceType = DeviceType.CPU,
                 cross_device: bool = False):
        super().__init__(
            id=id,
            op_type=PrimitiveOpType.COPY,
            inputs=[input_value],
            outputs=[output_value]
        )

        self.attributes.update({
            "src_device_id": src_device_id,
            "dst_device_id": dst_device_id,
            "src_device_type": src_device_type.value,
            "dst_device_type": dst_device_type.value,
            "cross_device": cross_device or (src_device_id != dst_device_id)
        })


class SignalOp(PrimitiveOperation):


    def __init__(self,
                 id: str,
                 signal_id: str,
                 target_ranks: Optional[List[int]] = None,
                 device_id: int = 0,
                 device_type: DeviceType = DeviceType.CPU):
        super().__init__(
            id=id,
            op_type=PrimitiveOpType.SIGNAL,
            inputs=[],
            outputs=[signal_id]
        )

        self.attributes.update({
            "signal_id": signal_id,
            "target_ranks": target_ranks or [],
            "device_id": device_id,
            "device_type": device_type.value
        })


class WaitSignalOp(PrimitiveOperation):


    def __init__(self,
                 id: str,
                 signal_id: str,
                 source_ranks: Optional[List[int]] = None,
                 timeout_ms: Optional[int] = None,
                 device_id: int = 0,
                 device_type: DeviceType = DeviceType.CPU):
        super().__init__(
            id=id,
            op_type=PrimitiveOpType.WAIT_SIGNAL,
            inputs=[signal_id],
            outputs=[]
        )

        self.attributes.update({
            "signal_id": signal_id,
            "source_ranks": source_ranks or [],
            "timeout_ms": timeout_ms,
            "device_id": device_id,
            "device_type": device_type.value
        })


@dataclass
class ChunkInfo:

    id: str
    offset: int
    size: int
    stride: int
    device_id: int = 0
    device_type: DeviceType = DeviceType.CPU

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "offset": self.offset,
            "size": self.size,
            "stride": self.stride,
            "device_id": self.device_id,
            "device_type": self.device_type.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkInfo':
        return cls(
            id=data["id"],
            offset=data["offset"],
            size=data["size"],
            stride=data["stride"],
            device_id=data.get("device_id", 0),
            device_type=DeviceType(data.get("device_type", "cpu"))
        )


class PrimitiveIRBuilder:


    def __init__(self, graph_id: str = "primitive_graph"):
        self.graph = IRGraph(ir_type=IRType.PRIMITIVE)
        self.graph.metadata["graph_id"] = graph_id
        self.value_counter = 0
        self.op_counter = 0

    def add_value(self,
                  dtype: str,
                  shape: List[int],
                  device_id: int = 0,
                  device_type: DeviceType = DeviceType.CPU,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
    
        value_id = f"value_{self.value_counter}"
        self.value_counter += 1

        value = IRValue(
            id=value_id,
            dtype=dtype,
            shape=shape,
            device_id=device_id,
            device_type=device_type,
            metadata=metadata or {}
        )

        self.graph.add_value(value)
        return value_id

    def add_write_op(self,
                     input_value: str,
                     address: Optional[int] = None,
                     size: Optional[int] = None,
                     device_id: int = 0,
                     device_type: DeviceType = DeviceType.CPU) -> str:
    
        op_id = f"write_{self.op_counter}"
        self.op_counter += 1

        output_value = self.add_value(
            dtype="memory",
            shape=[],
            device_id=device_id,
            device_type=device_type
        )

        op = WriteOp(
            id=op_id,
            input_value=input_value,
            output_value=output_value,
            address=address,
            size=size,
            device_id=device_id,
            device_type=device_type
        )

        self.graph.add_operation(op)
        return op_id

    def add_reduce_op(self,
                      input_values: List[str],
                      reduce_op: ReduceOpType = ReduceOpType.SUM,
                      device_id: int = 0,
                      device_type: DeviceType = DeviceType.CPU) -> str:
    
        op_id = f"reduce_{self.op_counter}"
        self.op_counter += 1

        first_input = self.graph.get_value(input_values[0])
        if first_input:
            output_value = self.add_value(
                dtype=first_input.dtype,
                shape=first_input.shape,
                device_id=device_id,
                device_type=device_type
            )
        else:
            output_value = self.add_value(
                dtype="float32",
                shape=[],
                device_id=device_id,
                device_type=device_type
            )

        op = ReduceOp(
            id=op_id,
            input_values=input_values,
            output_value=output_value,
            reduce_op=reduce_op,
            device_id=device_id,
            device_type=device_type
        )

        self.graph.add_operation(op)
        return op_id

    def add_copy_op(self,
                    input_value: str,
                    dst_device_id: int = 0,
                    dst_device_type: DeviceType = DeviceType.CPU,
                    src_device_id: Optional[int] = None,
                    src_device_type: Optional[DeviceType] = None) -> str:
    
        op_id = f"copy_{self.op_counter}"
        self.op_counter += 1

        src_val = self.graph.get_value(input_value)
        if src_val:
            actual_src_device_id = src_device_id or src_val.device_id
            actual_src_device_type = src_device_type or src_val.device_type
            dtype = src_val.dtype
            shape = src_val.shape
        else:
            actual_src_device_id = src_device_id or 0
            actual_src_device_type = src_device_type or DeviceType.CPU
            dtype = "float32"
            shape = []

        output_value = self.add_value(
            dtype=dtype,
            shape=shape,
            device_id=dst_device_id,
            device_type=dst_device_type
        )

        op = CopyOp(
            id=op_id,
            input_value=input_value,
            output_value=output_value,
            src_device_id=actual_src_device_id,
            dst_device_id=dst_device_id,
            src_device_type=actual_src_device_type,
            dst_device_type=dst_device_type
        )

        self.graph.add_operation(op)
        return op_id

    def add_signal_op(self,
                      signal_id: str,
                      target_ranks: Optional[List[int]] = None,
                      device_id: int = 0,
                      device_type: DeviceType = DeviceType.CPU) -> str:
    
        op_id = f"signal_{self.op_counter}"
        self.op_counter += 1

        op = SignalOp(
            id=op_id,
            signal_id=signal_id,
            target_ranks=target_ranks,
            device_id=device_id,
            device_type=device_type
        )

        self.graph.add_operation(op)
        return op_id

    def add_wait_signal_op(self,
                          signal_id: str,
                          source_ranks: Optional[List[int]] = None,
                          timeout_ms: Optional[int] = None,
                          device_id: int = 0,
                          device_type: DeviceType = DeviceType.CPU) -> str:
    
        op_id = f"wait_{self.op_counter}"
        self.op_counter += 1

        op = WaitSignalOp(
            id=op_id,
            signal_id=signal_id,
            source_ranks=source_ranks,
            timeout_ms=timeout_ms,
            device_id=device_id,
            device_type=device_type
        )

        self.graph.add_operation(op)
        return op_id

    def get_graph(self) -> IRGraph:
    
        return self.graph

    def get_value(self, value_id: str) -> Optional[IRValue]:
    
        return self.graph.get_value(value_id)

    def get_operation(self, op_id: str) -> Optional[IROperation]:
    
        return self.graph.get_operation(op_id)


def create_allreduce_primitive_pattern(input_value_id: str,
                                     world_size: int,
                                     rank: int,
                                     device_id: int = 0,
                                     device_type: DeviceType = DeviceType.CPU) -> IRGraph:

    builder = PrimitiveIRBuilder("ring_allreduce")

    current_value = input_value_id

    for step in range(world_size - 1):
        target_rank = (rank + step + 1) % world_size
        signal_id = f"scatter_signal_{step}"
        builder.add_signal_op(signal_id, target_ranks=[target_rank], device_id=device_id, device_type=device_type)

        source_rank = (rank - step - 1) % world_size
        wait_signal_id = f"scatter_wait_{step}"
        builder.add_wait_signal_op(wait_signal_id, source_ranks=[source_rank], device_id=device_id, device_type=device_type)

        if step == 0:
            # First step, reduce with received data
            received_value = builder.add_value(
                dtype="float32",  # Should match input dtype
                shape=[1024],     # Should match input shape
                device_id=device_id,
                device_type=device_type
            )
            current_value = builder.add_reduce_op([current_value, received_value], device_id=device_id, device_type=device_type)

    for step in range(world_size - 1):
        target_rank = (rank + step + 1) % world_size
        signal_id = f"gather_signal_{step}"
        builder.add_signal_op(signal_id, target_ranks=[target_rank], device_id=device_id, device_type=device_type)

        source_rank = (rank - step - 1) % world_size
        wait_signal_id = f"gather_wait_{step}"
        builder.add_wait_signal_op(wait_signal_id, source_ranks=[source_rank], device_id=device_id, device_type=device_type)

        if step < world_size - 1:
            # Copy received data to appropriate location
            received_value = builder.add_value(
                dtype="float32",
                shape=[1024],
                device_id=device_id,
                device_type=device_type
            )
            builder.add_copy_op(received_value, device_id=device_id, device_type=device_type)

    return builder.get_graph()