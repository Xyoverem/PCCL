"""
Pass Registry for PCCL

Manages registration and discovery of transformation passes in the
three-layer IR architecture.
"""

from typing import Dict, List, Optional, Type, Set
from .base import Pass, PassType


class PassRegistry:
    """Global registry for all transformation passes"""

    _instance: Optional['PassRegistry'] = None
    _passes: Dict[str, Pass] = {}
    _pass_classes: Dict[str, Type[Pass]] = {}
    _passes_by_type: Dict[PassType, List[str]] = {}
    _dependencies: Dict[str, Set[str]] = {}

    def __new__(cls) -> 'PassRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register_pass(self, pass_obj: Pass) -> None:
        """Register a pass instance"""
        if pass_obj.name in self._passes:
            raise ValueError(f"Pass '{pass_obj.name}' is already registered")

        self._passes[pass_obj.name] = pass_obj

        # Index by type
        if pass_obj.pass_type not in self._passes_by_type:
            self._passes_by_type[pass_obj.pass_type] = []
        self._passes_by_type[pass_obj.pass_type].append(pass_obj.name)

    def register_pass_class(self, name: str, pass_class: Type[Pass]) -> None:
        """Register a pass class for dynamic instantiation"""
        if name in self._pass_classes:
            raise ValueError(f"Pass class '{name}' is already registered")

        self._pass_classes[name] = pass_class

    def get_pass(self, name: str) -> Optional[Pass]:
        """Get a registered pass instance"""
        return self._passes.get(name)

    def create_pass(self, name: str, **kwargs) -> Optional[Pass]:
        """Create a pass instance from registered class"""
        if name not in self._pass_classes:
            return None

        pass_class = self._pass_classes[name]
        return pass_class(**kwargs)

    def get_passes_by_type(self, pass_type: PassType) -> List[Pass]:
        """Get all passes of a specific type"""
        pass_names = self._passes_by_type.get(pass_type, [])
        return [self._passes[name] for name in pass_names]

    def get_all_passes(self) -> List[Pass]:
        """Get all registered passes"""
        return list(self._passes.values())

    def list_passes(self, pass_type: Optional[PassType] = None) -> List[str]:
        """List names of registered passes"""
        if pass_type is None:
            return list(self._passes.keys())

        pass_names = self._passes_by_type.get(pass_type, [])
        return pass_names.copy()

    def add_dependency(self, pass_name: str, depends_on: str) -> None:
        """Add a dependency between passes"""
        if pass_name not in self._dependencies:
            self._dependencies[pass_name] = set()
        self._dependencies[pass_name].add(depends_on)

    def get_dependencies(self, pass_name: str) -> Set[str]:
        """Get all dependencies of a pass"""
        return self._dependencies.get(pass_name, set())

    def get_dependents(self, pass_name: str) -> Set[str]:
        """Get all passes that depend on this pass"""
        dependents = set()
        for name, deps in self._dependencies.items():
            if pass_name in deps:
                dependents.add(name)
        return dependents

    def resolve_execution_order(self, pass_names: List[str]) -> List[str]:
        """Resolve execution order based on dependencies"""
        # Topological sort
        visited = set()
        temp_visited = set()
        result = []

        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving pass '{name}'")
            if name in visited or name not in pass_names:
                return

            temp_visited.add(name)

            # Visit dependencies first
            for dep in self.get_dependencies(name):
                if dep in pass_names:
                    visit(dep)

            temp_visited.remove(name)
            visited.add(name)
            result.append(name)

        for name in pass_names:
            visit(name)

        return result

    def find_compatible_passes(self,
                              input_ir_type: str,
                              output_ir_type: str,
                              target_device: Optional[str] = None) -> List[Pass]:
        """Find passes that can handle the given IR types and device"""
        compatible = []

        for pass_obj in self._passes.values():
            caps = pass_obj.capabilities

            # Check IR type compatibility
            if (input_ir_type not in caps.input_ir_types or
                output_ir_type not in caps.output_ir_types):
                continue

            # Check device compatibility
            if (target_device and target_device != "auto" and
                target_device not in caps.supported_devices and
                "auto" not in caps.supported_devices):
                continue

            compatible.append(pass_obj)

        return compatible

    def clear(self) -> None:
        """Clear all registered passes"""
        self._passes.clear()
        self._pass_classes.clear()
        self._passes_by_type.clear()
        self._dependencies.clear()

    def __len__(self) -> int:
        return len(self._passes)

    def __contains__(self, name: str) -> bool:
        return name in self._passes


def register_pass(name: str = None):
    """Decorator for registering passes"""
    def decorator(pass_class: Type[Pass]):
        pass_name = name or pass_class.__name__
        PassRegistry().register_pass_class(pass_name, pass_class)
        return pass_class
    return decorator


def get_registry() -> PassRegistry:
    """Get the global pass registry"""
    return PassRegistry()