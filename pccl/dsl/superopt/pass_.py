"""Superoptimization compiler pass — e-graph based.

Two-phase optimization:
  1. Executor upgrades (fast path, no e-graph needed)
  2. E-graph equality saturation:
     a. Convert IR -> e-graph
     b. Add executor equivalences (domain rules Category A)
     c. Saturate with structural rules (Category B) + SMT rules (Category C)
     d. Extract optimal DAG with parametric cost function
     e. Convert e-graph -> IR
"""

from typing import Dict, List, Set, Optional
from functools import partial

from ..graph import PrimitiveIRGraph
from ..nodes import (
    IRNode, IRNodeVariant, PrimitiveOpType, ExecutorType,
    SmCopyNode, TmaCopyNode, CeCopyNode,
    SmReduceNode, TmaReduceNode,
    MultimemReduceNode, MultimemStoreNode,
    RdmaWriteNode, RdmaReadNode,
    NotifyNode, WaitNotifyNode,
)
from .rule import RewriteRule, PatternNode, PatternEdge
from .rule_db import load_all_rules
from .enumerator import EXECUTOR_UPGRADES, LinkType
from .cost_model import (
    GpuProfile, H100_PROFILE, TopologyProfile, NVLINK_TOPOLOGY,
    critical_path_cost, egraph_node_cost,
)
from .egraph import EGraph, ENode
from .egraph_bridge import ir_to_egraph, egraph_to_ir
from .domain_rules import add_executor_equivalences, structural_rules, smt_rules_to_egraph


_OP_TYPE_TO_NODE_CLS = {
    PrimitiveOpType.SM_COPY: SmCopyNode,
    PrimitiveOpType.TMA_COPY: TmaCopyNode,
    PrimitiveOpType.CE_COPY: CeCopyNode,
    PrimitiveOpType.SM_REDUCE: SmReduceNode,
    PrimitiveOpType.TMA_REDUCE: TmaReduceNode,
    PrimitiveOpType.MULTIMEM_REDUCE: MultimemReduceNode,
    PrimitiveOpType.MULTIMEM_STORE: MultimemStoreNode,
    PrimitiveOpType.RDMA_WRITE: RdmaWriteNode,
    PrimitiveOpType.RDMA_READ: RdmaReadNode,
}

_SYNC_OPS = {PrimitiveOpType.NOTIFY, PrimitiveOpType.WAIT_NOTIFY, PrimitiveOpType.NOOP}


class SuperoptPass:
    def __init__(
        self,
        device_profile: GpuProfile = H100_PROFILE,
        topology: TopologyProfile = NVLINK_TOPOLOGY,
        rules: Optional[List[RewriteRule]] = None,
        data_size_hint: int = 0,
        max_k: int = 6,
        gpu_profile_name: str = "h100",
    ):
        self.profile = device_profile
        self.topology = topology
        self.data_size_hint = data_size_hint
        self.rules = rules if rules is not None else load_all_rules(max_k=max_k)

    def run(self, graph: PrimitiveIRGraph) -> PrimitiveIRGraph:
        if graph.has_ocs_barriers():
            raise ValueError(
                "Superoptimization across OCS barriers is unsupported; "
                "split the graph into OCS phases first")
        self._apply_executor_upgrades(graph)
        self._apply_egraph_optimization(graph)
        return graph

    def _apply_executor_upgrades(self, graph: PrimitiveIRGraph) -> bool:
        changed = False
        for node_id in list(graph.nodes.keys()):
            node = graph.nodes[node_id]
            if node.op_type not in EXECUTOR_UPGRADES:
                continue
            new_op_type = EXECUTOR_UPGRADES[node.op_type]
            new_node = self._clone_node_with_type(node, new_op_type)
            if new_node is not None:
                graph.nodes[node_id] = new_node
                changed = True
        return changed

    def _apply_egraph_optimization(self, graph: PrimitiveIRGraph) -> bool:
        if graph.size() < 2:
            return False

        eg, id_map, roots = ir_to_egraph(graph)
        if not roots:
            return False

        add_executor_equivalences(eg)

        struct_rules = structural_rules(eg)
        smt_rules = smt_rules_to_egraph(eg, self.rules)
        all_rules = struct_rules + smt_rules
        if all_rules:
            eg.saturate(all_rules, limit=20)

        cost_fn = partial(
            egraph_node_cost,
            device=self.profile,
            topology=self.topology,
        )

        new_graph = egraph_to_ir(eg, roots, cost_fn, original_graph=graph)

        if new_graph.size() == 0:
            return False

        graph.nodes.clear()
        graph.entry_points.clear()
        graph.exit_points.clear()
        for nid, node in new_graph.nodes.items():
            graph.nodes[nid] = node
        graph._update_boundary_points()
        return True

    def _clone_node_with_type(
        self,
        node: IRNodeVariant,
        new_op_type: PrimitiveOpType,
    ) -> Optional[IRNodeVariant]:
        params = node.to_params()
        cls = _OP_TYPE_TO_NODE_CLS.get(new_op_type)
        if cls is None:
            return None

        if cls in (SmCopyNode, TmaCopyNode, CeCopyNode, MultimemStoreNode):
            new_node = cls(
                source_rank=params.get("source_rank", 0),
                src_offset=params.get("src_offset", 0),
                dst_offset=params.get("dst_offset", 0),
                size=params.get("size", 0),
            )
        elif cls in (SmReduceNode, TmaReduceNode, MultimemReduceNode):
            new_node = cls(
                reduce_op=params.get("reduce_op", "sum"),
                source_rank=params.get("source_rank", 0),
                src_offset=params.get("src_offset", 0),
                dst_offset=params.get("dst_offset", 0),
                remote_offset=params.get("remote_offset", 0),
                count=params.get("count", 0),
            )
        else:
            return None

        new_node.op_id = node.op_id
        new_node.dependencies = list(node.dependencies)
        new_node.next_ops = list(node.next_ops)
        new_node.device = node.device
        new_node.channel = node.channel
        new_node.tensor_info = node.tensor_info
        return new_node
