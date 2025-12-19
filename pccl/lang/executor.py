from typing import Dict, List, Any, Optional, Union
import time
import threading
import queue

from .compiler import ExecutionPlan

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

    def __init__(self):
        self._execution_counter = 0
        self._active_executions: Dict[str, ExecutionHandle] = {}
        self._worker_thread = None
        self._task_queue = queue.Queue()
        self._shutdown = False

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
        result = input_data

        for op in plan.operations:
            result = self._execute_operation(op, result, plan.topology)

        return result

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

    def _execute_operation(self, operation: Dict[str, Any], input_data: Any, topology: Dict[str, Any]) -> Any:
        """Execute a single operation"""
        op_type = operation.get('type', '')

        if op_type == 'allreduce':
            return self._execute_allreduce(operation, input_data, topology)
        elif op_type == 'broadcast':
            return self._execute_broadcast(operation, input_data, topology)
        elif op_type == 'allgather':
            return self._execute_allgather(operation, input_data, topology)
        elif op_type == 'reduce_scatter':
            return self._execute_reduce_scatter(operation, input_data, topology)
        elif op_type == 'fused_allreduce':
            return self._execute_fused_allreduce(operation, input_data, topology)
        else:
            return input_data

    def _execute_allreduce(self, operation: Dict[str, Any], input_data: Any, topology: Dict[str, Any]) -> Any:
        """Execute allreduce operation"""
        participants = operation.get('participants', [0, 1])
        algorithm = operation.get('algorithm', 'ring')
        reduce_op = operation.get('reduce_op', 'sum')
        buffer_size = operation.get('buffer_size', 128 * 1024 * 1024)

        if hasattr(input_data, 'shape'):
            import numpy as np
            if isinstance(input_data, np.ndarray):
                if algorithm == 'ring':
                    result = self._simulate_ring_allreduce(input_data, participants, reduce_op)
                elif algorithm == 'tree':
                    result = self._simulate_tree_allreduce(input_data, participants, reduce_op)
                else:
                    result = input_data.copy()
                return result

        return input_data

    def _execute_broadcast(self, operation: Dict[str, Any], input_data: Any, topology: Dict[str, Any]) -> Any:
        """Execute broadcast operation"""
        root_rank = operation.get('root_rank', 0)
        participants = operation.get('participants', [0, 1])

        if hasattr(input_data, 'shape'):
            import numpy as np
            if isinstance(input_data, np.ndarray):
                return input_data.copy()

        return input_data

    def _execute_allgather(self, operation: Dict[str, Any], input_data: Any, topology: Dict[str, Any]) -> Any:
        """Execute allgather operation"""
        participants = operation.get('participants', [0, 1])
        input_size = operation.get('input_size', 0)

        if hasattr(input_data, 'shape'):
            import numpy as np
            if isinstance(input_data, np.ndarray):
                gathered_data = np.concatenate([input_data] * len(participants))
                return gathered_data

        return input_data

    def _execute_reduce_scatter(self, operation: Dict[str, Any], input_data: Any, topology: Dict[str, Any]) -> Any:
        """Execute reduce-scatter operation"""
        participants = operation.get('participants', [0, 1])
        reduce_op = operation.get('reduce_op', 'sum')
        input_size = operation.get('input_size', 0)
        output_size = operation.get('output_size', 0)

        if hasattr(input_data, 'shape'):
            import numpy as np
            if isinstance(input_data, np.ndarray):
                chunk_size = input_size // len(participants)
                start_idx = 0
                end_idx = min(chunk_size, output_size)
                return input_data[start_idx:end_idx]

        return input_data

    def _execute_fused_allreduce(self, operation: Dict[str, Any], input_data: Any, topology: Dict[str, Any]) -> Any:
        """Execute fused allreduce operation"""
        operations = operation.get('operations', [])

        result = input_data
        for op in operations:
            result = self._execute_operation(op, result, topology)

        return result

    def _simulate_ring_allreduce(self, data: Any, participants: List[int], reduce_op: str) -> Any:
        """Simulate ring allreduce execution"""
        import numpy as np

        if isinstance(data, np.ndarray):
            result = data.copy()

            if reduce_op == 'sum':
                result *= len(participants)
            elif reduce_op == 'avg':
                result = result / len(participants)
            elif reduce_op == 'max':
                pass
            elif reduce_op == 'min':
                pass

            return result

        return data

    def _simulate_tree_allreduce(self, data: Any, participants: List[int], reduce_op: str) -> Any:
        """Simulate tree allreduce execution"""
        import numpy as np

        if isinstance(data, np.ndarray):
            result = data.copy()

            if reduce_op == 'sum':
                result *= len(participants)
            elif reduce_op == 'avg':
                result = result / len(participants)
            elif reduce_op == 'max':
                pass
            elif reduce_op == 'min':
                pass

            return result

        return data

    def shutdown(self):
        """Shutdown the execution engine"""
        self._shutdown = True
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

# Global execution engine instance
_execution_engine = None

def get_execution_engine() -> ExecutionEngine:
    """Get the global execution engine instance"""
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = ExecutionEngine()
    return _execution_engine

def execute_plan(plan: ExecutionPlan, input_data: Any) -> Any:
    """Execute a compiled execution plan"""
    engine = get_execution_engine()
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

# Global profiler instance
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