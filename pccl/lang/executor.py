from typing import Dict, List, Any, Optional, Union
import time
import threading
import queue

from .compiler import ExecutionPlan
from .execution_manager import ExecutionManager
from .integrated_compiler import HardwareType
from .tensor_transfer import IRExecutionContext, TensorTransferManager

class ExecutionHandle:
    """Handle for asynchronous execution operations"""

    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self._completed = False
        self._result = None
        self._error = None
        self._completion_event = threading.Event()

    def set_result(self, result: Any):
        """Set the execution result"""
        self._result = result
        self._completed = True
        self._completion_event.set()

    def set_error(self, error: Exception):
        """Set the execution error"""
        self._error = error
        self._completed = True
        self._completion_event.set()

    def wait(self, timeout: Optional[float] = None) -> Any:
        """Wait for completion and return result"""
        self._completion_event.wait(timeout)
        if self._error:
            raise self._error
        return self._result

    def is_completed(self) -> bool:
        """Check if execution is completed"""
        return self._completed

class ExecutionEngine:
    """Execution engine for communication plans"""

    def __init__(self, hardware_type: HardwareType = HardwareType.CUDA, device_id: int = 0, use_cpp_execution: bool = True):
        self._execution_counter = 0
        self._active_executions: Dict[str, ExecutionHandle] = {}
        self._worker_thread = None
        self._task_queue = queue.Queue()
        self._shutdown = False

        self.hardware_type = hardware_type
        self.device_id = device_id
        self.use_cpp_execution = True

        self.execution_manager = ExecutionManager(hardware_type)
        self.tensor_manager = TensorTransferManager()
        self.tensor_manager.initialize_cpp_engine()
        self.cpp_available = True

        self._start_worker()

    def _start_worker(self):
        """Start the background worker thread"""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        """Background worker thread loop"""
        while not self._shutdown:
            try:
                task = self._task_queue.get(timeout=0.1)
                if task is None:
                    continue

                execution_id, plan, input_data = task
                handle = self._active_executions.get(execution_id)

                if handle:
                    try:
                        result = self._execute_plan_sync(plan, input_data)
                        handle.set_result(result)
                    except Exception as e:
                        handle.set_error(e)

                self._task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker thread error: {e}")

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        self._execution_counter += 1
        return f"exec_{self._execution_counter}_{int(time.time() * 1000000)}"

    def execute_plan_sync(self, plan: ExecutionPlan, input_data: Any) -> Any:
        """Execute a plan synchronously"""
        return self._execute_plan_sync(plan, input_data)

    def _execute_plan_sync(self, plan: ExecutionPlan, input_data: Any) -> Any:
        """Internal synchronous execution implementation"""
        result = self.execution_manager.execute(plan, input_data, self.hardware_type, self.device_id)
        if not result['success']:
            raise RuntimeError(f"C++ execution failed: {result.get('error_message', 'Unknown error')}")

        return input_data

    
    def execute_with_tensor_transfer(self, ir_graph, input_tensors: Dict[str, Any]) -> Dict[str, Any]:
        """Execute IR graph with tensor data transfer"""
        from .tensor_transfer import HybridExecutionBridge
        bridge = HybridExecutionBridge(self.hardware_type.value, self.device_id)
        return bridge.execute_ir_graph_with_tensors(ir_graph, input_tensors)

    def transfer_tensor_to_cpp(self, tensor: Any, tensor_id: str) -> Dict[str, Any]:
        """Transfer a single tensor to C++ memory"""
        return self.tensor_manager.transfer_tensor_to_cpp(
            tensor, tensor_id, self.device_id, self.hardware_type.value
        )

    def get_tensor_memory_info(self) -> Dict[str, Any]:
        """Get information about tensor memory usage"""
        return self.tensor_manager.get_memory_usage()

    def cleanup_tensor_memory(self):
        """Clean up tensor memory"""
        self.tensor_manager.cleanup_all_tensors()

    def execute_plan_async(self, plan: ExecutionPlan, input_data: Any) -> ExecutionHandle:
        """Execute a plan asynchronously"""
        execution_id = self._generate_execution_id()
        handle = ExecutionHandle(execution_id)
        self._active_executions[execution_id] = handle

        self._task_queue.put((execution_id, plan, input_data))

        return handle

    def wait_for_completion(self, handle: ExecutionHandle, timeout: Optional[float] = None) -> Any:
        """Wait for async execution to complete"""
        result = handle.wait(timeout)

        execution_id = handle.execution_id
        if execution_id in self._active_executions:
            del self._active_executions[execution_id]

        return result

    
    
    def shutdown(self):
        """Shutdown the execution engine"""
        self._shutdown = True
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

        # Cleanup tensor memory
        self.cleanup_tensor_memory()

        # Shutdown execution manager
        if self.execution_manager:
            self.execution_manager.shutdown()

_execution_engine = None

def get_execution_engine(hardware_type: HardwareType = HardwareType.CUDA, device_id: int = 0, use_cpp_execution: bool = True) -> ExecutionEngine:
    """Get the global execution engine instance"""
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = ExecutionEngine(hardware_type, device_id, use_cpp_execution)
    return _execution_engine

def execute_plan(plan: ExecutionPlan, input_data: Any, hardware_type: HardwareType = HardwareType.CUDA, device_id: int = 0) -> Any:
    """Execute a compiled execution plan"""
    engine = get_execution_engine(hardware_type, device_id)
    return engine.execute_plan_sync(plan, input_data)

def execute_plan_async(plan: ExecutionPlan, input_data: Any) -> ExecutionHandle:
    """Execute a compiled execution plan asynchronously"""
    engine = get_execution_engine()
    return engine.execute_plan_async(plan, input_data)

def wait_for_completion(handle: ExecutionHandle, timeout: Optional[float] = None) -> Any:
    """Wait for asynchronous execution to complete"""
    engine = get_execution_engine()
    return engine.wait_for_completion(handle, timeout)

def shutdown_execution_engine():
    """Shutdown the global execution engine"""
    global _execution_engine
    if _execution_engine is not None:
        _execution_engine.shutdown()
        _execution_engine = None

class PerformanceProfiler:
    """Performance profiler for communication operations"""

    def __init__(self):
        self.profiles = []

    def profile_execution(self, plan: ExecutionPlan, input_data: Any, num_iterations: int = 10) -> Dict[str, float]:
        """Profile execution of a plan"""
        times = []

        for i in range(num_iterations):
            start_time = time.perf_counter()
            execute_plan(plan, input_data)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        data_size = self._estimate_data_size(input_data)
        bandwidth = data_size / avg_time if avg_time > 0 else 0

        profile = {
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'bandwidth_gb_s': bandwidth / (1024**3),
            'data_size_bytes': data_size,
            'num_operations': len(plan.operations),
            'topology_type': plan.topology.get('type', 'unknown')
        }

        self.profiles.append(profile)
        return profile

    def _estimate_data_size(self, data: Any) -> int:
        """Estimate the size of input data"""
        if hasattr(data, 'nbytes'):
            return data.nbytes
        elif hasattr(data, 'shape') and hasattr(data, 'dtype'):
            import numpy as np
            if isinstance(data, np.ndarray):
                return data.nbytes
        elif isinstance(data, (list, tuple)):
            return len(str(data).encode())
        else:
            return len(str(data).encode())

    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of all profiles"""
        if not self.profiles:
            return {}

        avg_times = [p['avg_time_ms'] for p in self.profiles]
        bandwidths = [p['bandwidth_gb_s'] for p in self.profiles]

        return {
            'total_executions': len(self.profiles),
            'avg_time_ms': sum(avg_times) / len(avg_times),
            'min_time_ms': min(avg_times),
            'max_time_ms': max(avg_times),
            'avg_bandwidth_gb_s': sum(bandwidths) / len(bandwidths),
            'max_bandwidth_gb_s': max(bandwidths)
        }

    def clear_profiles(self):
        """Clear all stored profiles"""
        self.profiles.clear()

_profiler = None

def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance"""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler

def profile(name: str = ""):
    """Decorator for profiling function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            profile_name = name or func.__name__

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            profiler.profiles.append({
                'name': profile_name,
                'avg_time_ms': execution_time * 1000,
                'min_time_ms': execution_time * 1000,
                'max_time_ms': execution_time * 1000,
                'bandwidth_gb_s': 0,
                'data_size_bytes': 0,
                'num_operations': 1,
                'topology_type': 'unknown'
            })

            return result

        return wrapper
    return decorator