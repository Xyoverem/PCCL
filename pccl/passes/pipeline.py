"""
Pass Pipeline for PCCL

Defines reusable pass pipelines for common transformation scenarios
in the three-layer IR architecture.
"""

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass

from .base import Pass, PassResult, PassContext, PassType
from .manager import PassManager
from .registry import get_registry


@dataclass
class PipelineStage:
    """Single stage in a pass pipeline"""
    name: str
    pass_names: List[str]
    optional: bool = False
    condition: Optional[Callable[[Any], bool]] = None


class PassPipeline:
    """Configurable pipeline of transformation passes"""

    def __init__(self,
                 name: str,
                 stages: List[PipelineStage],
                 description: str = ""):
        self.name = name
        self.stages = stages
        self.description = description
        self.manager = PassManager()
        self.registry = get_registry()

    def execute(self,
                ir: Any,
                context: Optional[PassContext] = None) -> PassResult:
        """Execute the pipeline"""
        if context is None:
            context = PassContext()

        current_ir = ir
        all_diagnostics = []
        executed_stages = []

        for stage in self.stages:
            # Check stage condition
            if stage.condition and not stage.condition(current_ir):
                continue

            # Execute stage
            stage_diagnostics = [f"Executing pipeline stage '{stage.name}'"]
            result = self.manager.execute_pass_sequence(
                stage.pass_names, current_ir, context
            )

            stage_diagnostics.extend(result.diagnostics)

            if result.success:
                current_ir = result.ir
                executed_stages.append(stage.name)
                stage_diagnostics.append(f"Stage '{stage.name}' completed successfully")
            elif not stage.optional:
                # Required stage failed
                stage_diagnostics.append(f"Required stage '{stage.name}' failed")
                all_diagnostics.extend(stage_diagnostics)
                return PassResult(
                    success=False,
                    ir=current_ir,
                    diagnostics=all_diagnostics,
                    metadata={"executed_stages": executed_stages}
                )
            else:
                # Optional stage failed
                stage_diagnostics.append(f"Optional stage '{stage.name}' failed, continuing")
                current_ir = result.ir  # Use partial result

            all_diagnostics.extend(stage_diagnostics)

        return PassResult(
            success=True,
            ir=current_ir,
            diagnostics=all_diagnostics,
            metadata={"executed_stages": executed_stages}
        )

    def add_stage(self, stage: PipelineStage) -> 'PassPipeline':
        """Add a stage to the pipeline"""
        self.stages.append(stage)
        return self

    def remove_stage(self, stage_name: str) -> 'PassPipeline':
        """Remove a stage from the pipeline"""
        self.stages = [s for s in self.stages if s.name != stage_name]
        return self

    def get_stage(self, stage_name: str) -> Optional[PipelineStage]:
        """Get a stage by name"""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None

    def validate(self) -> List[str]:
        """Validate the pipeline configuration"""
        errors = []

        for stage in self.stages:
            # Check if all passes exist
            for pass_name in stage.pass_names:
                if pass_name not in self.registry:
                    errors.append(f"Stage '{stage.name}': Pass '{pass_name}' not found")

        return errors

    def __str__(self) -> str:
        return f"PassPipeline(name='{self.name}', stages={len(self.stages)})"

    def __repr__(self) -> str:
        return self.__str__()


# Predefined pipeline configurations
class StandardPipelines:
    """Standard pass pipelines for common use cases"""

    @staticmethod
    def collective_to_primitive_pipeline() -> PassPipeline:
        """Standard pipeline for lowering collective operations to primitive IR"""
        return PassPipeline(
            name="collective_to_primitive",
            description="Lower collective operations (AllReduce, Broadcast, etc.) to primitive IR",
            stages=[
                PipelineStage(
                    name="collective_decomposition",
                    pass_names=["collective_to_primitive"],
                    optional=False
                ),
                PipelineStage(
                    name="primitive_optimization",
                    pass_names=["primitive_fusion", "memory_optimization"],
                    optional=True
                ),
                PipelineStage(
                    name="primitive_validation",
                    pass_names=["primitive_validator"],
                    optional=False
                )
            ]
        )

    @staticmethod
    def primitive_to_hardware_pipeline(target_device: str = "auto") -> PassPipeline:
        """Standard pipeline for lowering primitive IR to hardware primitives"""
        return PassPipeline(
            name=f"primitive_to_hardware_{target_device}",
            description=f"Lower primitive IR to {target_device} hardware primitives",
            stages=[
                PipelineStage(
                    name="device_analysis",
                    pass_names=["device_analyzer"],
                    optional=False
                ),
                PipelineStage(
                    name="hardware_lowering",
                    pass_names=["primitive_to_hardware"],
                    optional=False
                ),
                PipelineStage(
                    name="hardware_optimization",
                    pass_names=["hardware_fusion", "scheduling_optimization"],
                    optional=True
                )
            ]
        )

    @staticmethod
    def full_lowering_pipeline(target_device: str = "auto") -> PassPipeline:
        """Complete pipeline from collective operations to hardware primitives"""
        return PassPipeline(
            name=f"full_lowering_{target_device}",
            description=f"Complete lowering from collective operations to {target_device} hardware",
            stages=[
                PipelineStage(
                    name="collective_lowering",
                    pass_names=["collective_to_primitive"],
                    optional=False
                ),
                PipelineStage(
                    name="primitive_optimization",
                    pass_names=["primitive_fusion", "memory_optimization"],
                    optional=True
                ),
                PipelineStage(
                    name="hardware_lowering",
                    pass_names=["primitive_to_hardware"],
                    optional=False
                ),
                PipelineStage(
                    name="final_optimization",
                    pass_names=["hardware_fusion", "scheduling_optimization"],
                    optional=True
                ),
                PipelineStage(
                    name="validation",
                    pass_names=["hardware_validator"],
                    optional=False
                )
            ]
        )

    @staticmethod
    def optimization_pipeline(optimization_level: str = "default") -> PassPipeline:
        """Optimization-only pipeline for IR transformation"""
        if optimization_level == "aggressive":
            passes = ["constant_folding", "loop_unrolling", "vectorization",
                     "memory_fusion", "instruction_scheduling"]
        elif optimization_level == "minimal":
            passes = ["basic_fusion", "dead_code_elimination"]
        else:  # default
            passes = ["primitive_fusion", "memory_optimization", "dependency_optimization"]

        return PassPipeline(
            name=f"optimization_{optimization_level}",
            description=f"IR optimization pipeline ({optimization_level} level)",
            stages=[
                PipelineStage(
                    name="ir_analysis",
                    pass_names=["dependency_analyzer", "performance_model"],
                    optional=True
                ),
                PipelineStage(
                    name="optimization",
                    pass_names=passes,
                    optional=False
                )
            ]
        )