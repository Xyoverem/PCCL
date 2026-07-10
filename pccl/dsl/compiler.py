"""
PCCL DSL Compiler.
Pipeline: DependencyAnalysis -> DCE -> [Superopt] -> [Channelize] -> Final DependencyAnalysis
"""

from typing import Optional, Dict, Any
from .graph import PrimitiveIRGraph
from .passes import DependencyAnalysisPass, DeadCodeEliminationPass, ChannelizePass
from .codegen import RuntimeGraphGenerator


class Compiler:
    def __init__(
        self,
        enable_dce: bool = True,
        enable_superopt: bool = False,
        enable_channelize: bool = False,
        num_channels: int = 2,
        gpu_profile: str = "h100",
        data_size_hint: int = 0,
        topology: object = None,
        device_profile: object = None,
    ):
        self.enable_dce = enable_dce
        self.enable_superopt = enable_superopt
        self.enable_channelize = enable_channelize
        self.num_channels = num_channels
        self.gpu_profile = gpu_profile
        self.data_size_hint = data_size_hint
        self.topology = topology
        self.device_profile = device_profile

    def compile(self, graph: PrimitiveIRGraph) -> PrimitiveIRGraph:
        DependencyAnalysisPass().run(graph)
        if self.enable_dce:
            DeadCodeEliminationPass().run(graph)
        if self.enable_superopt:
            if graph.has_ocs_barriers():
                raise ValueError(
                    "Superoptimization across OCS barriers is unsupported; "
                    "compile the generated OCS phases independently")
            from .superopt.pass_ import SuperoptPass
            kwargs = {
                "data_size_hint": self.data_size_hint,
            }
            if self.topology is not None:
                kwargs["topology"] = self.topology
            if self.device_profile is not None:
                kwargs["device_profile"] = self.device_profile
            SuperoptPass(**kwargs).run(graph)
        if self.enable_channelize:
            ChannelizePass(num_channels=self.num_channels).run(graph)
        DependencyAnalysisPass().run(graph)
        return graph

    def compile_to_json(self, graph: PrimitiveIRGraph, output_file: Optional[str] = None) -> str:
        compiled = self.compile(graph)
        gen = RuntimeGraphGenerator()
        if output_file:
            return gen.generate_to_file(compiled, output_file)
        return gen.generate_string(compiled)

    def compile_to_dict(self, graph: PrimitiveIRGraph) -> Dict[str, Any]:
        compiled = self.compile(graph)
        return RuntimeGraphGenerator().generate(compiled)


# --- Convenience functions ---

def compile_to_json_string(graph: PrimitiveIRGraph, indent: int = 2) -> str:
    compiled = Compiler().compile(graph)
    return RuntimeGraphGenerator(indent=indent).generate_string(compiled)


def compile_to_json_file(graph: PrimitiveIRGraph, output_file: str, indent: int = 2) -> str:
    compiled = Compiler().compile(graph)
    return RuntimeGraphGenerator(indent=indent).generate_to_file(compiled, output_file)
