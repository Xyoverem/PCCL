"""
Pass Manager for PCCL

Manages execution of transformation passes with dependency resolution,
caching, and error handling.
"""

from typing import List, Optional, Dict, Any, Set
from collections import defaultdict
import logging

from .base import Pass, PassResult, PassContext, PassType
from .registry import PassRegistry, get_registry


class PassManager:
    """Manages execution of transformation passes"""

    def __init__(self,
                 enable_caching: bool = True,
                 enable_profiling: bool = False,
                 strict_mode: bool = False):
        self.enable_caching = enable_caching
        self.enable_profiling = enable_profiling
        self.strict_mode = strict_mode
        self.registry = get_registry()
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, PassResult] = {}
        self._execution_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def execute_pass(self,
                    pass_name: str,
                    ir: Any,
                    context: Optional[PassContext] = None) -> PassResult:
        """Execute a single pass"""
        if context is None:
            context = PassContext(
                enable_profiling=self.enable_profiling
            )

        # Get the pass
        pass_obj = self.registry.get_pass(pass_name)
        if pass_obj is None:
            # Try to create from registered class
            pass_obj = self.registry.create_pass(pass_name)
            if pass_obj is None:
                return PassResult(
                    success=False,
                    ir=ir,
                    diagnostics=[f"Pass '{pass_name}' not found"]
                )

        # Check cache
        cache_key = self._get_cache_key(pass_name, ir, context)
        if self.enable_caching and cache_key in self._cache:
            self.logger.debug(f"Using cached result for pass '{pass_name}'")
            cached_result = self._cache[cache_key]
            return cached_result

        # Execute pass
        result = pass_obj.run(ir, context)

        # Cache result
        if self.enable_caching and result.success:
            self._cache[cache_key] = result

        # Update statistics
        self._update_stats(pass_name, result)

        return result

    def execute_pass_sequence(self,
                             pass_names: List[str],
                             ir: Any,
                             context: Optional[PassContext] = None) -> PassResult:
        """Execute a sequence of passes"""
        if context is None:
            context = PassContext(
                enable_profiling=self.enable_profiling
            )

        # Resolve execution order based on dependencies
        try:
            ordered_passes = self.registry.resolve_execution_order(pass_names)
        except ValueError as e:
            return PassResult(
                success=False,
                ir=ir,
                diagnostics=[f"Dependency resolution failed: {str(e)}"]
            )

        current_ir = ir
        all_diagnostics = []
        total_time = 0.0

        for pass_name in ordered_passes:
            self.logger.info(f"Executing pass: {pass_name}")

            result = self.execute_pass(pass_name, current_ir, context)
            all_diagnostics.extend(result.diagnostics)
            total_time += result.execution_time

            if not result.success:
                if self.strict_mode:
                    return PassResult(
                        success=False,
                        ir=current_ir,
                        diagnostics=all_diagnostics
                    )
                else:
                    self.logger.warning(f"Pass '{pass_name}' failed, continuing...")
                    continue

            current_ir = result.ir

        return PassResult(
            success=True,
            ir=current_ir,
            metadata={
                "total_passes": len(ordered_passes),
                "execution_time": total_time,
                "passes_executed": ordered_passes
            },
            diagnostics=all_diagnostics
        )

    def find_and_execute_lowering_path(self,
                                      input_ir_type: str,
                                      output_ir_type: str,
                                      ir: Any,
                                      target_device: str = "auto",
                                      context: Optional[PassContext] = None) -> PassResult:
        """Find and execute appropriate lowering path"""
        if context is None:
            context = PassContext(
                target_device=target_device,
                enable_profiling=self.enable_profiling
            )

        # Find compatible passes
        compatible_passes = self.registry.find_compatible_passes(
            input_ir_type, output_ir_type, target_device
        )

        if not compatible_passes:
            return PassResult(
                success=False,
                ir=ir,
                diagnostics=[
                    f"No compatible passes found for {input_ir_type} -> {output_ir_type} "
                    f"on device {target_device}"
                ]
            )

        # Try each compatible pass until one succeeds
        for pass_obj in compatible_passes:
            self.logger.debug(f"Trying pass '{pass_obj.name}' for lowering")

            result = self.execute_pass(pass_obj.name, ir, context)
            if result.success:
                return result

            self.logger.debug(f"Pass '{pass_obj.name}' failed: {result.diagnostics}")

        # All passes failed
        all_diagnostics = []
        for pass_obj in compatible_passes:
            test_result = self.execute_pass(pass_obj.name, ir, context)
            all_diagnostics.extend([f"Pass '{pass_obj.name}':"] + test_result.diagnostics)

        return PassResult(
            success=False,
            ir=ir,
            diagnostics=["All compatible passes failed:"] + all_diagnostics
        )

    def analyze_pass_dependencies(self) -> Dict[str, List[str]]:
        """Analyze and return pass dependency graph"""
        dependency_graph = {}
        for pass_name in self.registry.list_passes():
            dependencies = self.registry.get_dependencies(pass_name)
            dependency_graph[pass_name] = list(dependencies)
        return dependency_graph

    def validate_pass_execution_order(self, pass_names: List[str]) -> List[str]:
        """Validate that pass execution order is valid"""
        try:
            ordered_passes = self.registry.resolve_execution_order(pass_names)
            return []  # No errors
        except ValueError as e:
            return [str(e)]

    def get_execution_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all passes"""
        return dict(self._execution_stats)

    def clear_cache(self):
        """Clear the execution cache"""
        self._cache.clear()

    def clear_statistics(self):
        """Clear execution statistics"""
        self._execution_stats.clear()

    def _get_cache_key(self, pass_name: str, ir: Any, context: PassContext) -> str:
        """Generate cache key for pass execution"""
        # Simple implementation - in practice, you'd want a more sophisticated IR hash
        ir_hash = str(hash(str(ir)))
        context_hash = str(hash((
            context.target_device,
            context.optimization_level,
            frozenset(context.get_dependencies())
        )))
        return f"{pass_name}:{ir_hash}:{context_hash}"

    def _update_stats(self, pass_name: str, result: PassResult):
        """Update execution statistics"""
        if pass_name not in self._execution_stats:
            self._execution_stats[pass_name] = {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0.0,
                "avg_time": 0.0
            }

        stats = self._execution_stats[pass_name]
        stats["executions"] += 1
        stats["total_time"] += result.execution_time
        stats["avg_time"] = stats["total_time"] / stats["executions"]

        if result.success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

    def __str__(self) -> str:
        return (f"PassManager(caching={self.enable_caching}, "
                f"profiling={self.enable_profiling}, "
                f"registered_passes={len(self.registry)})")

    def __repr__(self) -> str:
        return self.__str__()