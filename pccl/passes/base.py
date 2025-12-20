"""
Base classes for PCCL Pass System

Defines the fundamental interfaces for IR transformation passes in the
three-layer IR architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Type
from dataclasses import dataclass
from enum import Enum
import time


class PassResult:

    def __init__(self,
                 success: bool,
                 ir: Any,
                 metadata: Optional[Dict[str, Any]] = None,
                 diagnostics: Optional[List[str]] = None):
        self.success = success
        self.ir = ir
        self.metadata = metadata or {}
        self.diagnostics = diagnostics or []
        self.execution_time = 0.0

    def add_diagnostic(self, message: str, level: str = "info"):
    
        self.diagnostics.append(f"[{level.upper()}] {message}")

    def __bool__(self) -> bool:
        return self.success


class PassContext:

    def __init__(self,
                 target_device: str = "auto",
                 optimization_level: str = "default",
                 enable_profiling: bool = False):
        self.target_device = target_device
        self.optimization_level = optimization_level
        self.enable_profiling = enable_profiling
        self._cache: Dict[str, Any] = {}
        self._dependencies: Set[str] = set()

    def get_cached(self, key: str) -> Optional[Any]:
    
        return self._cache.get(key)

    def set_cached(self, key: str, value: Any):
    
        self._cache[key] = value

    def add_dependency(self, pass_name: str):
    
        self._dependencies.add(pass_name)

    def get_dependencies(self) -> Set[str]:
    
        return self._dependencies.copy()


class PassType(Enum):

    LAYER1_TO_LAYER2 = "collective_to_primitive"
    LAYER2_TO_LAYER3 = "primitive_to_hardware"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    VERIFICATION = "verification"


@dataclass
class PassCapabilities:

    input_ir_types: List[str]
    output_ir_types: List[str]
    supported_devices: List[str]
    required_features: List[str]
    optional_features: List[str]


class Pass(ABC):


    def __init__(self,
                 name: str,
                 pass_type: PassType,
                 description: str = ""):
        self.name = name
        self.pass_type = pass_type
        self.description = description
        self._capabilities: Optional[PassCapabilities] = None

    @property
    def capabilities(self) -> PassCapabilities:
    
        if self._capabilities is None:
            self._capabilities = self.declare_capabilities()
        return self._capabilities

    @abstractmethod
    def declare_capabilities(self) -> PassCapabilities:
    
        pass

    @abstractmethod
    def execute(self, ir: Any, context: PassContext) -> PassResult:
    
        pass

    def validate_input(self, ir: Any, context: PassContext) -> bool:
    
        return True

    def setup(self, context: PassContext):
    
        pass

    def teardown(self, context: PassContext):
    
        pass

    def run(self, ir: Any, context: PassContext) -> PassResult:
    
        result = PassResult(success=False, ir=ir)

        try:
            start_time = time.time() if context.enable_profiling else 0

            # Setup
            self.setup(context)

            # Validate input
            if not self.validate_input(ir, context):
                result.add_diagnostic("Input validation failed", "error")
                return result

            # Execute pass
            result = self.execute(ir, context)
            result.execution_time = time.time() - start_time if context.enable_profiling else 0

        except Exception as e:
            result.success = False
            result.add_diagnostic(f"Pass execution failed: {str(e)}", "error")

        finally:
            # Teardown
            try:
                self.teardown(context)
            except Exception as e:
                result.add_diagnostic(f"Pass teardown failed: {str(e)}", "warning")

        return result

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.pass_type.value})"

    def __repr__(self) -> str:
        return self.__str__()


class CompositePass(Pass):


    def __init__(self, name: str, subpasses: List[Pass]):
        super().__init__(name, PassType.OPTIMIZATION)
        self.subpasses = subpasses

    def declare_capabilities(self) -> PassCapabilities:
    
        input_types = []
        output_types = []
        devices = []
        required_features = []

        for subpass in self.subpasses:
            input_types.extend(subpass.capabilities.input_ir_types)
            output_types.extend(subpass.capabilities.output_ir_types)
            devices.extend(subpass.capabilities.supported_devices)
            required_features.extend(subpass.capabilities.required_features)

        return PassCapabilities(
            input_ir_types=list(set(input_types)),
            output_ir_types=list(set(output_types)),
            supported_devices=list(set(devices)),
            required_features=list(set(required_features)),
            optional_features=[]
        )

    def execute(self, ir: Any, context: PassContext) -> PassResult:
    
        current_ir = ir
        all_diagnostics = []

        for subpass in self.subpasses:
            result = subpass.run(current_ir, context)
            all_diagnostics.extend(result.diagnostics)

            if not result.success:
                return PassResult(
                    success=False,
                    ir=current_ir,
                    diagnostics=all_diagnostics
                )

            current_ir = result.ir

        return PassResult(
            success=True,
            ir=current_ir,
            diagnostics=all_diagnostics
        )