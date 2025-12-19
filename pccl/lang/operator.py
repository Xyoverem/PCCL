from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import functools
import inspect

from .config import OperatorConfig, AllreduceConfig, BroadcastConfig, AllgatherConfig, ReduceScatterConfig
from .compiler import DSLCompiler, ExecutionPlan

@dataclass
class CommunicationOperator:
    """Base class for communication operators"""
    config: OperatorConfig
    name: str
    compiled_plan: Optional[ExecutionPlan] = None
    _execution_handle: Optional[Any] = None

    def compile(self, participants: Optional[List[int]] = None) -> ExecutionPlan:
        """Compile the operator to an execution plan"""
        compiler = DSLCompiler()
        plan = compiler.compile(self.config, participants)
        optimized_plan = compiler.optimize(plan)
        self.compiled_plan = optimized_plan
        return optimized_plan

    def execute(self, input_data: Any, participants: Optional[List[int]] = None) -> Any:
        """Execute the communication operator"""
        if self.compiled_plan is None:
            self.compile(participants)

        from . import executor
        return executor.execute_plan(self.compiled_plan, input_data)

    def async_execute(self, input_data: Any, participants: Optional[List[int]] = None) -> Any:
        """Execute the operator asynchronously"""
        if self.compiled_plan is None:
            self.compile(participants)

        from . import executor
        self._execution_handle = executor.execute_plan_async(self.compiled_plan, input_data)
        return self._execution_handle

    def wait(self) -> Any:
        """Wait for asynchronous execution to complete"""
        if self._execution_handle is not None:
            from . import executor
            result = executor.wait_for_completion(self._execution_handle)
            self._execution_handle = None
            return result
        return None

    def estimate_cost(self, participants: Optional[List[int]] = None) -> Dict[str, float]:
        """Estimate execution cost"""
        if self.compiled_plan is None:
            self.compile(participants)

        compiler = DSLCompiler()
        return compiler.estimate_cost(self.compiled_plan)

class Allreduce(CommunicationOperator):
    """AllReduce collective communication operator"""

    def __init__(self, reduce_op="sum", algorithm="ring", participants=None,
                 buffer_size=128*1024*1024, enable_overlap=False, topology=None):
        from .config import ReduceOp, AlgorithmType

        reduce_op_map = {
            "sum": ReduceOp.SUM,
            "avg": ReduceOp.AVG,
            "max": ReduceOp.MAX,
            "min": ReduceOp.MIN
        }

        algorithm_map = {
            "ring": AlgorithmType.RING,
            "tree": AlgorithmType.TREE,
            "rabenseifner": AlgorithmType.RABENSEIFNER,
            "double_binary_tree": AlgorithmType.DOUBLE_BINARY_TREE
        }

        config = AllreduceConfig(
            reduce_op=reduce_op_map.get(reduce_op, ReduceOp.SUM),
            algorithm=algorithm_map.get(algorithm, AlgorithmType.RING),
            participants=participants or [],
            buffer_size=buffer_size,
            enable_overlap=enable_overlap,
            topology=topology
        )

        super().__init__(config, "allreduce")

class Broadcast(CommunicationOperator):
    """Broadcast collective communication operator"""

    def __init__(self, root_rank=0, participants=None, buffer_size=128*1024*1024):
        config = BroadcastConfig(
            root_rank=root_rank,
            participants=participants or [root_rank],
            buffer_size=buffer_size
        )

        super().__init__(config, "broadcast")

class Allgather(CommunicationOperator):
    """AllGather collective communication operator"""

    def __init__(self, participants=None, input_size=0, buffer_size=128*1024*1024):
        config = AllgatherConfig(
            participants=participants or [],
            input_size=input_size,
            buffer_size=buffer_size
        )

        super().__init__(config, "allgather")

class ReduceScatter(CommunicationOperator):
    """Reduce-Scatter collective communication operator"""

    def __init__(self, reduce_op="sum", participants=None, input_size=0,
                 output_size=0, buffer_size=128*1024*1024):
        from .config import ReduceOp

        reduce_op_map = {
            "sum": ReduceOp.SUM,
            "avg": ReduceOp.AVG,
            "max": ReduceOp.MAX,
            "min": ReduceOp.MIN
        }

        config = ReduceScatterConfig(
            reduce_op=reduce_op_map.get(reduce_op, ReduceOp.SUM),
            participants=participants or [],
            input_size=input_size,
            output_size=output_size,
            buffer_size=buffer_size
        )

        super().__init__(config, "reduce_scatter")

class Send(CommunicationOperator):
    """Send point-to-point communication operator"""

    def __init__(self, dst_rank, tag=0, buffer_size=128*1024*1024):
        from .config import SendRecvConfig, DeviceType

        config = SendRecvConfig(
            src_rank=0,
            dst_rank=dst_rank,
            tag=tag,
            buffer_size=buffer_size
        )

        super().__init__(config, "send")

class Recv(CommunicationOperator):
    """Receive point-to-point communication operator"""

    def __init__(self, src_rank, tag=0, buffer_size=128*1024*1024):
        from .config import SendRecvConfig, DeviceType

        config = SendRecvConfig(
            src_rank=src_rank,
            dst_rank=0,
            tag=tag,
            buffer_size=buffer_size
        )

        super().__init__(config, "recv")

class PipelineAllreduce(CommunicationOperator):
    """Pipeline AllReduce with compute-communication overlap"""

    def __init__(self, reduce_op="sum", algorithm="ring", participants=None,
                 compute_chunks=4, communication_chunks=2, buffer_size=128*1024*1024):
        from .config import ReduceOp, AlgorithmType, PipelineAllreduceConfig

        reduce_op_map = {
            "sum": ReduceOp.SUM,
            "avg": ReduceOp.AVG,
            "max": ReduceOp.MAX,
            "min": ReduceOp.MIN
        }

        algorithm_map = {
            "ring": AlgorithmType.RING,
            "tree": AlgorithmType.TREE,
            "rabenseifner": AlgorithmType.RABENSEIFNER,
            "double_binary_tree": AlgorithmType.DOUBLE_BINARY_TREE
        }

        config = PipelineAllreduceConfig(
            reduce_op=reduce_op_map.get(reduce_op, ReduceOp.SUM),
            algorithm=algorithm_map.get(algorithm, AlgorithmType.RING),
            participants=participants or [],
            compute_chunks=compute_chunks,
            communication_chunks=communication_chunks,
            buffer_size=buffer_size
        )

        super().__init__(config, "pipeline_allreduce")

class CompositeOperator(CommunicationOperator):
    """Composite operator that combines multiple communication operations"""

    def __init__(self, operators: List[CommunicationOperator], name: str = "composite"):
        self.operators = operators
        self.name = name

        dummy_config = operators[0].config if operators else AllreduceConfig()
        super().__init__(dummy_config, name)

    def compile(self, participants: Optional[List[int]] = None) -> ExecutionPlan:
        """Compile all operators and create a combined execution plan"""
        compiler = DSLCompiler()
        configs = [op.config for op in self.operators]
        combined_plan = compiler.compile_multiple(configs, participants)
        optimized_plan = compiler.optimize(combined_plan)
        self.compiled_plan = optimized_plan
        return optimized_plan

    def execute(self, input_data: Any, participants: Optional[List[int]] = None) -> Any:
        """Execute all operators in sequence"""
        if self.compiled_plan is None:
            self.compile(participants)

        from . import executor
        return executor.execute_plan(self.compiled_plan, input_data)

@dataclass
class CommunicationPattern:
    """High-level communication pattern that can be instantiated with different parameters"""
    operator_class: type
    default_params: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def instantiate(self, **kwargs) -> CommunicationOperator:
        """Create an operator instance with the given parameters"""
        params = {**self.default_params, **kwargs}
        return self.operator_class(**params)

def communication(operator_func: Callable = None, *, name: str = "", auto_compile: bool = True):
    """Decorator for declaring communication operators"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if isinstance(result, CommunicationOperator):
                if auto_compile:
                    result.compile()
                if name:
                    result.name = name
                elif not result.name:
                    result.name = func.__name__
                return result
            else:
                return result

        wrapper._is_communication_decorator = True
        wrapper._original_function = func
        return wrapper

    if operator_func is None:
        return decorator
    else:
        return decorator(operator_func)

class OperatorRegistry:
    """Registry for communication operators"""

    def __init__(self):
        self._operators: Dict[str, type] = {}
        self._patterns: Dict[str, CommunicationPattern] = {}

        self._register_builtin_operators()

    def register_operator(self, name: str, operator_class: type):
        """Register a new operator class"""
        self._operators[name] = operator_class

    def register_pattern(self, name: str, pattern: CommunicationPattern):
        """Register a communication pattern"""
        self._patterns[name] = pattern

    def get_operator(self, name: str) -> Optional[type]:
        """Get an operator class by name"""
        return self._operators.get(name)

    def get_pattern(self, name: str) -> Optional[CommunicationPattern]:
        """Get a communication pattern by name"""
        return self._patterns.get(name)

    def create_operator(self, name: str, **kwargs) -> Optional[CommunicationOperator]:
        """Create an operator instance"""
        operator_class = self.get_operator(name)
        if operator_class:
            return operator_class(**kwargs)

        pattern = self.get_pattern(name)
        if pattern:
            return pattern.instantiate(**kwargs)

        return None

    def list_operators(self) -> List[str]:
        """List all registered operator names"""
        return list(self._operators.keys()) + list(self._patterns.keys())

    def _register_builtin_operators(self):
        """Register built-in operators"""
        self.register_operator("allreduce", Allreduce)
        self.register_operator("broadcast", Broadcast)
        self.register_operator("allgather", Allgather)
        self.register_operator("reduce_scatter", ReduceScatter)
        self.register_operator("send", Send)
        self.register_operator("recv", Recv)
        self.register_operator("pipeline_allreduce", PipelineAllreduce)
        self.register_operator("composite", CompositeOperator)

        ring_pattern = CommunicationPattern(
            operator_class=Allreduce,
            default_params={"algorithm": "ring"},
            name="ring_allreduce"
        )
        self.register_pattern("ring_allreduce", ring_pattern)

        tree_pattern = CommunicationPattern(
            operator_class=Allreduce,
            default_params={"algorithm": "tree"},
            name="tree_allreduce"
        )
        self.register_pattern("tree_allreduce", tree_pattern)

# Global operator registry
registry = OperatorRegistry()

def compile(operator_or_config, participants: Optional[List[int]] = None) -> Union[ExecutionPlan, CommunicationOperator]:
    """Compile an operator or configuration to an execution plan"""
    if isinstance(operator_or_config, OperatorConfig):
        compiler = DSLCompiler()
        plan = compiler.compile(operator_or_config, participants)
        return compiler.optimize(plan)
    elif isinstance(operator_or_config, CommunicationOperator):
        return operator_or_config.compile(participants)
    elif isinstance(operator_or_config, type) and issubclass(operator_or_config, CommunicationOperator):
        return operator_or_config
    else:
        raise ValueError(f"Cannot compile type: {type(operator_or_config)}")

def execute(operator_or_plan, input_data: Any, participants: Optional[List[int]] = None) -> Any:
    """Execute a compiled operator or execution plan"""
    if isinstance(operator_or_plan, CommunicationOperator):
        return operator_or_plan.execute(input_data, participants)
    elif isinstance(operator_or_plan, ExecutionPlan):
        from . import executor
        return executor.execute_plan(operator_or_plan, input_data)
    else:
        raise ValueError(f"Cannot execute type: {type(operator_or_plan)}")