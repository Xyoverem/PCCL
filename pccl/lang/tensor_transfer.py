import numpy as np
from typing import Any, Dict, List, Optional, Union
import json

class TensorTransferManager:
    """Manages tensor data transfer between Python and C++ memory"""

    def __init__(self):
        self.cpp_engine = None
        self.tensor_buffers = {}
        self.memory_pools = {}

    def initialize_cpp_engine(self):
        """Initialize C++ engine for tensor operations"""
        import pccl.engine_c as engine_c
        self.cpp_engine = engine_c.Engine(0, 1)
        self.cpp_engine.initEngine()
        return True

    def transfer_tensor_to_cpp(self, tensor: Any, tensor_id: str, device_id: int = 0, device_type: str = "cuda") -> Dict[str, Any]:
        """Transfer Python tensor to C++ memory"""
        if not self.cpp_engine:
            self.initialize_cpp_engine()

        if hasattr(tensor, 'numpy'):
            # PyTorch tensor
            np_array = tensor.numpy()
        elif hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
            # NumPy array
            np_array = tensor
        else:
            # Python array/list
            np_array = np.array(tensor, dtype=np.float32)

        # Get tensor information
        shape = list(np_array.shape)
        dtype = str(np_array.dtype)
        size = np_array.nbytes
        data_ptr = np_array.ctypes.data

        # Create C++ tensor wrapper
        tensor_info = {
            "id": tensor_id,
            "shape": shape,
            "dtype": dtype,
            "size": size,
            "device_id": device_id,
            "device_type": device_type,
            "data_ptr": data_ptr,
            "numpy_array": np_array  # Keep reference to prevent garbage collection
        }

        self.tensor_buffers[tensor_id] = tensor_info

        return {
            "success": True,
            "tensor_id": tensor_id,
            "shape": shape,
            "dtype": dtype,
            "size": size,
            "device_id": device_id,
            "device_type": device_type
        }

    def create_cpp_tensor_wrapper(self, tensor_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a C++ tensor wrapper for the given tensor info"""
        import pccl.engine_c as engine_c

        # Create Value object in C++
        if tensor_info["dtype"] == "float32":
            dtype = engine_c.DataType.FLOAT32
        elif tensor_info["dtype"] == "float64":
            dtype = engine_c.DataType.FLOAT64
        elif tensor_info["dtype"] == "int32":
            dtype = engine_c.DataType.INT32
        elif tensor_info["dtype"] == "int64":
            dtype = engine_c.DataType.INT64
        else:
            dtype = engine_c.DataType.FLOAT32  # Default

        cpp_value = engine_c.Value(dtype, tensor_info["shape"], tensor_info["device_id"])

        return {
            "success": True,
            "cpp_value": cpp_value,
            "tensor_info": tensor_info
        }

    def transfer_tensors_from_ir_context(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer tensors from IR execution context to C++"""
        transferred_tensors = {}

        for tensor_id, tensor_data in input_data.items():
            transfer_result = self.transfer_tensor_to_cpp(
                tensor_data,
                tensor_id,
                context.get("device_id", 0),
                context.get("device_type", "cuda")
            )

            if transfer_result["success"]:
                wrapper_result = self.create_cpp_tensor_wrapper(transfer_result)
                if wrapper_result["success"]:
                    transferred_tensors[tensor_id] = wrapper_result["cpp_value"]
                else:
                    print(f"Warning: Failed to create wrapper for tensor {tensor_id}: {wrapper_result['error']}")
            else:
                print(f"Warning: Failed to transfer tensor {tensor_id}: {transfer_result['error']}")

        return transferred_tensors

    def extract_tensor_data(self, cpp_value: Any) -> Any:
        """Extract tensor data from C++ Value object"""
        try:
            # This would need to be implemented based on the actual C++ Value API
            # For now, return a placeholder
            return None
        except Exception as e:
            print(f"Warning: Failed to extract tensor data: {str(e)}")
            return None

    def cleanup_tensor(self, tensor_id: str):
        """Clean up tensor buffer"""
        if tensor_id in self.tensor_buffers:
            del self.tensor_buffers[tensor_id]

    def cleanup_all_tensors(self):
        """Clean up all tensor buffers"""
        self.tensor_buffers.clear()
        self.memory_pools.clear()

    def get_tensor_info(self, tensor_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a transferred tensor"""
        return self.tensor_buffers.get(tensor_id)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        total_memory = sum(info["size"] for info in self.tensor_buffers.values())
        return {
            "num_tensors": len(self.tensor_buffers),
            "total_memory_bytes": total_memory,
            "total_memory_mb": total_memory / (1024 * 1024),
            "tensor_details": {
                tensor_id: {
                    "shape": info["shape"],
                    "dtype": info["dtype"],
                    "size_mb": info["size"] / (1024 * 1024)
                }
                for tensor_id, info in self.tensor_buffers.items()
            }
        }


class IRExecutionContext:
    """Extended execution context with tensor transfer capabilities"""

    def __init__(self, device_id: int = 0, device_type: str = "cuda"):
        self.device_id = device_id
        self.device_type = device_type
        self.tensor_manager = TensorTransferManager()
        self.input_tensors = {}
        self.output_tensors = {}
        self.execution_stats = {}

    def set_input_tensor(self, tensor_id: str, tensor_data: Any):
        """Set input tensor for execution"""
        transfer_result = self.tensor_manager.transfer_tensor_to_cpp(
            tensor_data, tensor_id, self.device_id, self.device_type
        )

        if transfer_result["success"]:
            self.input_tensors[tensor_id] = transfer_result
        else:
            raise ValueError(f"Failed to set input tensor {tensor_id}: {transfer_result['error']}")

    def prepare_execution_context(self) -> Dict[str, Any]:
        """Prepare execution context for C++ IR executor"""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "inputs": {tid: info for tid, info in self.input_tensors.items()},
            "async_execution": False,
            "timeout_ms": 30000
        }

    def extract_output_tensors(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract output tensors from execution result"""
        # This would need to be implemented based on actual execution result format
        return self.output_tensors

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return self.tensor_manager.get_memory_usage()

    def cleanup(self):
        """Clean up all resources"""
        self.tensor_manager.cleanup_all_tensors()
        self.input_tensors.clear()
        self.output_tensors.clear()


class HybridExecutionBridge:
    """Enhanced execution bridge with tensor transfer capabilities"""

    def __init__(self, hardware_type: str = "cuda", device_id: int = 0):
        self.hardware_type = hardware_type
        self.device_id = device_id

        import pccl.engine_c as engine_c
        self.cpp_engine = engine_c.Engine(0, 1)
        self.cpp_engine.initEngine()
        self.cpp_available = True

        self.tensor_manager = TensorTransferManager()
        self.tensor_manager.initialize_cpp_engine()

    def execute_ir_graph_with_tensors(self, ir_graph, input_tensors: Dict[str, Any]) -> Dict[str, Any]:
        """Execute IR graph with proper tensor data transfer"""
        # Create execution context with tensors
        context = IRExecutionContext(self.device_id, self.hardware_type)

        # Transfer input tensors
        for tensor_id, tensor_data in input_tensors.items():
            context.set_input_tensor(tensor_id, tensor_data)

        # Prepare execution context
        exec_context = context.prepare_execution_context()

        # Serialize IR graph
        from ..ir.json_serializer import serialize_graph
        json_ir = serialize_graph(ir_graph)

        # Execute on C++ engine
        stats = self.cpp_engine.executeIRGraph(json_ir, self.device_id, self.hardware_type)

        # Extract output tensors (placeholder implementation)
        output_tensors = context.extract_output_tensors(stats.__dict__ if hasattr(stats, '__dict__') else {})

        result = {
            'success': stats.success,
            'execution_time_ms': stats.execution_time_ms,
            'num_operations': stats.num_operations,
            'num_values': stats.num_values,
            'operation_counts': dict(stats.operation_counts),
            'operation_times': dict(stats.operation_times),
            'error_message': stats.error_message,
            'input_tensor_info': {tid: info for tid, info in context.input_tensors.items()},
            'output_tensors': output_tensors,
            'memory_usage': context.get_memory_usage()
        }

        # Cleanup
        context.cleanup()

        return result

    