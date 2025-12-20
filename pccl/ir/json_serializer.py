
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from datetime import datetime


class IRType(Enum):

    COLLECTIVE = "collective"
    PRIMITIVE = "primitive"
    HARDWARE = "hardware"


class PrimitiveOpType(Enum):

    WRITE = "write"
    REDUCE = "reduce"
    COPY = "copy"
    SIGNAL = "signal"
    WAIT_SIGNAL = "wait_signal"


class ReduceOpType(Enum):

    SUM = "sum"
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    PRODUCT = "product"


class DeviceType(Enum):

    CPU = "cpu"
    CUDA = "cuda"
    RDMA = "rdma"
    ROCM = "rocm"


@dataclass
class IRValue:

    id: str
    dtype: str
    shape: List[int]
    device_id: int
    device_type: DeviceType
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.id == "":
            self.id = f"value_{uuid.uuid4().hex[:8]}"
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
    
        return {
            "id": self.id,
            "dtype": self.dtype,
            "shape": self.shape,
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IRValue':
    
        return cls(
            id=data["id"],
            dtype=data["dtype"],
            shape=data["shape"],
            device_id=data["device_id"],
            device_type=DeviceType(data["device_type"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class IROperation:

    id: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.id == "":
            self.id = f"op_{uuid.uuid4().hex[:8]}"
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
    
        return {
            "id": self.id,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IROperation':
    
        return cls(
            id=data["id"],
            op_type=data["op_type"],
            inputs=data["inputs"],
            outputs=data["outputs"],
            attributes=data["attributes"],
            metadata=data.get("metadata", {})
        )


@dataclass
class IRGraph:

    ir_type: IRType
    values: Dict[str, IRValue]
    operations: Dict[str, IROperation]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.metadata["created_at"] = datetime.now().isoformat()

    def add_value(self, value: IRValue):
    
        self.values[value.id] = value

    def add_operation(self, op: IROperation):
    
        self.operations[op.id] = op

    def get_value(self, value_id: str) -> Optional[IRValue]:
    
        return self.values.get(value_id)

    def get_operation(self, op_id: str) -> Optional[IROperation]:
    
        return self.operations.get(op_id)

    def to_dict(self) -> Dict[str, Any]:
    
        return {
            "ir_type": self.ir_type.value,
            "values": {vid: val.to_dict() for vid, val in self.values.items()},
            "operations": {oid: op.to_dict() for oid, op in self.operations.items()},
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IRGraph':
    
        values = {
            vid: IRValue.from_dict(vdata)
            for vid, vdata in data["values"].items()
        }
        operations = {
            oid: IROperation.from_dict(odata)
            for oid, odata in data["operations"].items()
        }

        return cls(
            ir_type=IRType(data["ir_type"]),
            values=values,
            operations=operations,
            metadata=data.get("metadata", {})
        )


class IRSerializer:


    def __init__(self, indent: int = 2):
        self.indent = indent

    def serialize_graph(self, graph: IRGraph) -> str:
    
        data = graph.to_dict()
        return json.dumps(data, indent=self.indent, ensure_ascii=False)

    def deserialize_graph(self, json_str: str) -> IRGraph:
    
        data = json.loads(json_str)
        return IRGraph.from_dict(data)

    def serialize_to_file(self, graph: IRGraph, filepath: str):
    
        data = graph.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=self.indent, ensure_ascii=False)

    def deserialize_from_file(self, filepath: str) -> IRGraph:
    
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return IRGraph.from_dict(data)

    def validate_json(self, json_str: str) -> List[str]:
    
        errors = []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
            return errors

        required_fields = ["ir_type", "values", "operations"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        if "ir_type" in data:
            try:
                IRType(data["ir_type"])
            except ValueError:
                errors.append(f"Invalid IR type: {data['ir_type']}")

        if "values" in data:
            for vid, value_data in data["values"].items():
                value_errors = self._validate_value_data(vid, value_data)
                errors.extend(value_errors)

        if "operations" in data:
            for oid, op_data in data["operations"].items():
                op_errors = self._validate_operation_data(oid, op_data)
                errors.extend(op_errors)

        return errors

    def _validate_value_data(self, value_id: str, data: Dict[str, Any]) -> List[str]:
    
        errors = []

        required_fields = ["id", "dtype", "shape", "device_id", "device_type"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Value '{value_id}': missing field '{field}'")

        if "device_type" in data:
            try:
                DeviceType(data["device_type"])
            except ValueError:
                errors.append(f"Value '{value_id}': invalid device type '{data['device_type']}'")

        return errors

    def _validate_operation_data(self, op_id: str, data: Dict[str, Any]) -> List[str]:
    
        errors = []

        required_fields = ["id", "op_type", "inputs", "outputs", "attributes"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Operation '{op_id}': missing field '{field}'")

        return errors

    def get_statistics(self, graph: IRGraph) -> Dict[str, Any]:
    
        stats = {
            "ir_type": graph.ir_type.value,
            "num_values": len(graph.values),
            "num_operations": len(graph.operations),
            "operation_types": {},
            "device_distribution": {},
            "total_elements": 0
        }

        for op in graph.operations.values():
            op_type = op.op_type
            stats["operation_types"][op_type] = stats["operation_types"].get(op_type, 0) + 1

        for value in graph.values.values():
            device_type = value.device_type.value
            stats["device_distribution"][device_type] = stats["device_distribution"].get(device_type, 0) + 1

            # Count total elements
            if value.shape:
                total = 1
                for dim in value.shape:
                    total *= dim
                stats["total_elements"] += total

        return stats


# Global serializer instance
_default_serializer = IRSerializer()

def serialize_graph(graph: IRGraph) -> str:

    return _default_serializer.serialize_graph(graph)

def deserialize_graph(json_str: str) -> IRGraph:

    return _default_serializer.deserialize_graph(json_str)

def serialize_to_file(graph: IRGraph, filepath: str):

    _default_serializer.serialize_to_file(graph, filepath)

def deserialize_from_file(filepath: str) -> IRGraph:

    return _default_serializer.deserialize_from_file(filepath)