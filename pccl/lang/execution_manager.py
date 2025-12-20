import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
import time
import threading
import concurrent.futures

from .execution_bridge import ExecutionBridge
from .integrated_compiler import IntegratedCompiler, HardwareType
from .config import DeviceType


class ExecutionManager:
    def __init__(self, default_hardware_type: HardwareType = HardwareType.CUDA):
        self.default_hardware_type = default_hardware_type
        self.bridges = {}
        self.execution_history = []
        self._lock = threading.Lock()

    def get_bridge(self, hardware_type: HardwareType, device_id: int = 0) -> ExecutionBridge:
        key = (hardware_type.value, device_id)
        if key not in self.bridges:
            self.bridges[key] = ExecutionBridge(hardware_type, device_id)
        return self.bridges[key]

    def execute(self, config, input_data: Any, hardware_type: Optional[HardwareType] = None,
                device_id: int = 0, participants: Optional[List[int]] = None) -> Dict[str, Any]:
        if hardware_type is None:
            hardware_type = self.default_hardware_type

        bridge = self.get_bridge(hardware_type, device_id)

        start_time = time.time()

        result = bridge.compile_and_execute(config, input_data, participants)

        end_time = time.time()
        total_time = (end_time - start_time) * 1000

        result['total_time_ms'] = total_time
        result['hardware_type'] = hardware_type.value
        result['device_id'] = device_id

        with self._lock:
            self.execution_history.append({
                'timestamp': time.time(),
                'config': config,
                'hardware_type': hardware_type.value,
                'device_id': device_id,
                'result': result
            })

            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]

        return result

    def execute_batch(self, configs: List[Any], input_data_list: List[Any],
                     hardware_type: Optional[HardwareType] = None,
                     device_id: int = 0, participants: Optional[List[int]] = None,
                     max_workers: int = 4) -> List[Dict[str, Any]]:
        if hardware_type is None:
            hardware_type = self.default_hardware_type

        def execute_single(config, input_data):
            return self.execute(config, input_data, hardware_type, device_id, participants)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(execute_single, config, input_data)
                      for config, input_data in zip(configs, input_data_list)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        return results

    def execute_async(self, config, input_data: Any,
                     hardware_type: Optional[HardwareType] = None,
                     device_id: int = 0, participants: Optional[List[int]] = None,
                     callback: Optional[Callable] = None) -> concurrent.futures.Future:
        if hardware_type is None:
            hardware_type = self.default_hardware_type

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.execute, config, input_data, hardware_type, device_id, participants)

        if callback:
            future.add_done_callback(lambda f: callback(f.result()))

        return future

    def benchmark_config(self, config, input_data: Any, num_iterations: int = 10,
                        hardware_types: Optional[List[HardwareType]] = None,
                        device_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        if hardware_types is None:
            hardware_types = [self.default_hardware_type]
        if device_ids is None:
            device_ids = [0]

        benchmark_results = {}

        for hardware_type in hardware_types:
            for device_id in device_ids:
                key = f"{hardware_type.value}:{device_id}"
                times = []
                successes = 0

                for i in range(num_iterations):
                    result = self.execute(config, input_data, hardware_type, device_id)
                    if result['success']:
                        times.append(result['total_time_ms'])
                        successes += 1

                if times:
                    benchmark_results[key] = {
                        'success_rate': successes / num_iterations,
                        'avg_time_ms': np.mean(times),
                        'min_time_ms': np.min(times),
                        'max_time_ms': np.max(times),
                        'std_time_ms': np.std(times),
                        'num_iterations': num_iterations,
                        'successful_iterations': successes
                    }
                else:
                    benchmark_results[key] = {
                        'success_rate': 0.0,
                        'avg_time_ms': 0.0,
                        'min_time_ms': 0.0,
                        'max_time_ms': 0.0,
                        'std_time_ms': 0.0,
                        'num_iterations': num_iterations,
                        'successful_iterations': 0
                    }

        return benchmark_results

    def get_hardware_status(self) -> Dict[str, Any]:
        status = {}
        for (hw_type, device_id), bridge in self.bridges.items():
            status[f"{hw_type}:{device_id}"] = bridge.get_hardware_info()
        return status

    def get_execution_statistics(self) -> Dict[str, Any]:
        with self._lock:
            if not self.execution_history:
                return {
                    'total_executions': 0,
                    'success_rate': 0.0,
                    'avg_execution_time_ms': 0.0,
                    'hardware_usage': {}
                }

            total_executions = len(self.execution_history)
            successful_executions = sum(1 for entry in self.execution_history
                                      if entry['result']['success'])
            success_rate = successful_executions / total_executions

            execution_times = [entry['result']['total_time_ms']
                             for entry in self.execution_history
                             if entry['result']['success']]
            avg_execution_time = np.mean(execution_times) if execution_times else 0.0

            hardware_usage = {}
            for entry in self.execution_history:
                hw_type = entry['hardware_type']
                hardware_usage[hw_type] = hardware_usage.get(hw_type, 0) + 1

            return {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'success_rate': success_rate,
                'avg_execution_time_ms': avg_execution_time,
                'hardware_usage': hardware_usage
            }

    def clear_history(self):
        with self._lock:
            self.execution_history.clear()

    def optimize_config(self, config, input_data: Any,
                       optimization_target: str = 'execution_time') -> Dict[str, Any]:
        hardware_types = [HardwareType.CUDA, HardwareType.CPU]
        if HardwareType.RDMA in [ht for ht in HardwareType]:
            hardware_types.append(HardwareType.RDMA)

        benchmark_results = self.benchmark_config(config, input_data, num_iterations=5,
                                                hardware_types=hardware_types)

        best_config = None
        best_key = None
        best_value = float('inf') if optimization_target == 'execution_time' else 0

        for key, results in benchmark_results.items():
            if results['success_rate'] > 0:
                value = results['avg_time_ms'] if optimization_target == 'execution_time' else results['success_rate']
                if optimization_target == 'execution_time' and value < best_value:
                    best_value = value
                    best_key = key
                elif optimization_target == 'success_rate' and value > best_value:
                    best_value = value
                    best_config = results

        return {
            'optimization_target': optimization_target,
            'best_configuration': best_key,
            'best_value': best_value,
            'all_results': benchmark_results,
            'recommendation': f"Use {best_key} for optimal {optimization_target}"
        }

    def shutdown(self):
        for bridge in self.bridges.values():
            if hasattr(bridge, 'cpp_engine') and bridge.cpp_engine:
                try:
                    if hasattr(bridge.cpp_engine, 'exitCluster'):
                        bridge.cpp_engine.exitCluster()
                except:
                    pass

        self.bridges.clear()
        self.clear_history()