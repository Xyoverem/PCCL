import numpy as np
from typing import Any, Dict, List, Optional, Union
import json

from .integrated_compiler import IntegratedCompiler, HardwareType
from ..ir.json_serializer import IRGraph, serialize_graph


class ExecutionBridge:
    def __init__(self, hardware_type: HardwareType = HardwareType.CUDA, device_id: int = 0):
        self.hardware_type = hardware_type
        self.device_id = device_id
        self.device_type = hardware_type.value

        import pccl.engine_c as engine_c
        self.cpp_engine = engine_c.Engine(0, 1)
        self.cpp_engine.initEngine()
        self.cpp_available = True

        self.integrated_compiler = IntegratedCompiler(hardware_type)

    def execute_ir_graph(self, ir_graph: IRGraph, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        json_ir = serialize_graph(ir_graph)

        context = {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'inputs': input_data or {},
            'async_execution': False
        }

        stats = self.cpp_engine.executeIRGraph(
            json_ir,
            self.device_id,
            self.device_type
        )

        return {
            'success': stats.success,
            'execution_time_ms': stats.execution_time_ms,
            'num_operations': stats.num_operations,
            'num_values': stats.num_values,
            'operation_counts': dict(stats.operation_counts),
            'operation_times': dict(stats.operation_times),
            'error_message': stats.error_message
        }

    def compile_and_execute(self, config, input_data: Any, participants: Optional[List[int]] = None) -> Dict[str, Any]:
        integrated_plan = self.integrated_compiler.compile(config, participants)

        if not integrated_plan.lowering_successful:
            return {
                'success': False,
                'error_message': 'IR lowering failed',
                'lowering_stats': integrated_plan.lowering_stats
            }

        result = self.execute_ir_graph(integrated_plan.hardware_ir_graph, {'input': input_data})
        result['lowering_stats'] = integrated_plan.lowering_stats
        result['execution_plan'] = integrated_plan.execution_plan

        return result

    def execute_from_json(self, json_ir: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'inputs': input_data or {},
            'async_execution': False
        }

        stats = self.cpp_engine.executeIRGraph(json_ir, self.device_id, self.device_type)

        return {
            'success': stats.success,
            'execution_time_ms': stats.execution_time_ms,
            'num_operations': stats.num_operations,
            'num_values': stats.num_values,
            'operation_counts': dict(stats.operation_counts),
            'operation_times': dict(stats.operation_times),
            'error_message': stats.error_message
        }

    def is_hardware_execution_available(self) -> bool:
        return True

    def get_hardware_info(self) -> Dict[str, Any]:
        return {
            'hardware_type': self.device_type,
            'device_id': self.device_id,
            'status': 'hardware_available',
            'cpp_engine_initialized': True
        }