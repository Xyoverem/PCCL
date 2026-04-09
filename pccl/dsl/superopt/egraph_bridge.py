"""Bridge between PrimitiveIRGraph (DAG of IRNodes) and EGraph.

ir_to_egraph: converts an IR graph into an EGraph, returning the graph,
              a mapping from IR node op_id to e-class ID, and root e-class IDs.

egraph_to_ir: extracts optimal IR nodes from a saturated EGraph using a cost
              function, wiring edges based on ENode children relationships.
"""

from typing import Dict, List, Tuple, Callable, Optional
from collections import deque

from ..graph import PrimitiveIRGraph
from ..nodes import (
    IRNodeVariant, PrimitiveOpType,
    SmCopyNode, TmaCopyNode, CeCopyNode,
    SmReduceNode, TmaReduceNode,
    MultimemReduceNode, MultimemStoreNode,
    RdmaWriteNode, RdmaReadNode,
    NotifyNode, WaitNotifyNode,
)
from .egraph import EGraph, ENode


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
    PrimitiveOpType.NOTIFY: NotifyNode,
    PrimitiveOpType.WAIT_NOTIFY: WaitNotifyNode,
}

_SYNC_OPS = {PrimitiveOpType.NOTIFY, PrimitiveOpType.WAIT_NOTIFY, PrimitiveOpType.NOOP}


def _params_to_frozen(params: dict) -> Tuple[Tuple[str, object], ...]:
    return tuple(sorted(params.items()))


def ir_to_egraph(
    graph: PrimitiveIRGraph,
) -> Tuple[EGraph, Dict[str, int], List[int]]:
    eg = EGraph()
    id_map: Dict[str, int] = {}

    topo = graph.topological_sort()

    for node in topo:
        children = tuple(
            id_map[dep_id]
            for dep_id in node.dependencies
            if dep_id in id_map
        )
        op_type_str = node.op_type.value if node.op_type else "noop"
        params = _params_to_frozen(node.to_params())
        enode = ENode(
            op_type=op_type_str,
            rank=0,
            children=children,
            params=params,
        )
        eid = eg.add(enode)
        id_map[node.op_id] = eid

    exit_ids = [
        id_map[nid]
        for nid in graph.exit_points
        if nid in id_map
    ]
    if not exit_ids:
        exit_ids = [
            id_map[n.op_id]
            for n in topo
            if not n.next_ops and n.op_id in id_map
        ]

    return eg, id_map, exit_ids


def _extract_dag(
    eg: EGraph,
    roots: List[int],
    cost_fn: Callable[[ENode, Dict[int, float]], float],
) -> Dict[int, ENode]:
    best_cost: Dict[int, float] = {}
    best_node: Dict[int, ENode] = {}

    changed = True
    for _ in range(len(eg.eclass_nodes) + 1):
        if not changed:
            break
        changed = False
        for eid, enodes in eg.eclass_nodes.items():
            eid = eg.uf.find(eid)
            for enode in enodes:
                child_ok = all(
                    eg.uf.find(c) in best_cost for c in enode.children
                )
                if not child_ok and enode.children:
                    continue
                child_cost_map = {
                    eg.uf.find(c): best_cost.get(eg.uf.find(c), float("inf"))
                    for c in enode.children
                }
                c = cost_fn(enode, child_cost_map)
                if eid not in best_cost or c < best_cost[eid]:
                    best_cost[eid] = c
                    best_node[eid] = enode
                    changed = True

    reachable: Dict[int, ENode] = {}
    queue = [eg.uf.find(r) for r in roots]
    visited = set()
    while queue:
        eid = queue.pop()
        if eid in visited:
            continue
        visited.add(eid)
        node = best_node.get(eid)
        if node is not None:
            reachable[eid] = node
            for c in node.children:
                queue.append(eg.uf.find(c))
    return reachable


def _create_ir_node(enode: ENode, ref_node: Optional[IRNodeVariant] = None) -> Optional[IRNodeVariant]:
    try:
        op_type = PrimitiveOpType(enode.op_type)
    except ValueError:
        return None

    cls = _OP_TYPE_TO_NODE_CLS.get(op_type)
    if cls is None:
        return None

    params = dict(enode.params)

    if ref_node is not None:
        ref_params = ref_node.to_params()
        for k, v in ref_params.items():
            if k not in params:
                params[k] = v

    if cls in (SmCopyNode, TmaCopyNode, CeCopyNode, MultimemStoreNode):
        return cls(
            source_rank=params.get("source_rank", 0),
            src_offset=params.get("src_offset", 0),
            dst_offset=params.get("dst_offset", 0),
            size=params.get("size", 0),
        )
    if cls in (SmReduceNode, TmaReduceNode, MultimemReduceNode):
        return cls(
            reduce_op=params.get("reduce_op", "sum"),
            source_rank=params.get("source_rank", 0),
            src_offset=params.get("src_offset", 0),
            dst_offset=params.get("dst_offset", 0),
            remote_offset=params.get("remote_offset", 0),
            count=params.get("count", 0),
        )
    if cls == RdmaWriteNode:
        return cls(
            target_rank=params.get("target_rank", 0),
            src_offset=params.get("src_offset", 0),
            dst_offset=params.get("dst_offset", 0),
            size=params.get("size", 0),
        )
    if cls == RdmaReadNode:
        return cls(
            source_rank=params.get("source_rank", 0),
            src_offset=params.get("src_offset", 0),
            dst_offset=params.get("dst_offset", 0),
            size=params.get("size", 0),
        )
    if cls == NotifyNode:
        return cls(
            signal_id=params.get("signal_id", 0),
            target_rank=params.get("target_rank", 0),
        )
    if cls == WaitNotifyNode:
        return cls(
            signal_id=params.get("signal_id", 0),
            source_rank=params.get("source_rank", 0),
        )
    return None


def egraph_to_ir(
    eg: EGraph,
    roots: List[int],
    cost_fn: Callable[[ENode, Dict[int, float]], float],
    original_graph: Optional[PrimitiveIRGraph] = None,
) -> PrimitiveIRGraph:
    extracted = _extract_dag(eg, roots, cost_fn)

    orig_nodes_by_eclass: Dict[int, IRNodeVariant] = {}
    if original_graph is not None:
        topo = original_graph.topological_sort()
        id_map: Dict[str, int] = {}
        for node in topo:
            children = tuple(
                id_map[d] for d in node.dependencies if d in id_map
            )
            op_str = node.op_type.value if node.op_type else "noop"
            en = ENode(op_str, 0, children, _params_to_frozen(node.to_params()))
            en = en.canonicalize(eg.uf)
            if en in eg.hashcons:
                eid = eg.uf.find(eg.hashcons[en])
                id_map[node.op_id] = eid
                orig_nodes_by_eclass.setdefault(eid, node)
            else:
                eid_guess = eg.uf.find(eg.next_id - 1) if eg.next_id > 0 else 0
                id_map[node.op_id] = eid_guess

    new_graph = PrimitiveIRGraph(
        graph_name=original_graph.graph_name if original_graph else "extracted",
    )

    eclass_to_ir_id: Dict[int, str] = {}
    for eid, enode in extracted.items():
        ref_node = orig_nodes_by_eclass.get(eid)
        ir_node = _create_ir_node(enode, ref_node)
        if ir_node is None:
            continue
        if ref_node is not None:
            ir_node.channel = ref_node.channel
            ir_node.device = ref_node.device
            ir_node.tensor_info = ref_node.tensor_info
        new_graph.add_node(ir_node)
        eclass_to_ir_id[eid] = ir_node.op_id

    for eid, enode in extracted.items():
        dst_id = eclass_to_ir_id.get(eid)
        if dst_id is None:
            continue
        for child_eid in enode.children:
            child_eid = eg.uf.find(child_eid)
            src_id = eclass_to_ir_id.get(child_eid)
            if src_id is not None and src_id != dst_id:
                try:
                    new_graph.add_edge(src_id, dst_id)
                except (KeyError, ValueError):
                    pass

    new_graph._update_boundary_points()
    return new_graph
