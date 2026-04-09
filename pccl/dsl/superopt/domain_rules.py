"""Hand-written domain rewrite rules for e-graph equality saturation.

Three categories:
  A. Executor equivalences -- same semantics, different executor.
     Applied as a pre-saturation merge step (direct iteration, not ematch).
  B. Structural rules -- duplicate elimination, identity elimination.
     Expressed as (PatternExpr, applier) pairs for EGraph.saturate().
  C. SMT-verified rule adapter -- converts cached RewriteRule objects into
     e-graph (PatternExpr, applier) pairs.
"""

from typing import List, Tuple, Callable, Optional, Set, Dict

from .egraph import EGraph, ENode, PatternExpr, Substitution, OP_CATEGORY_MAP
from .rule import RewriteRule, PatternNode
from .rule_db import load_all_rules


EXECUTOR_EQUIVALENCES: Dict[str, List[str]] = {
    "copy": ["sm.copy", "tma.copy", "ce.copy"],
    "reduce": ["sm.reduce", "tma.reduce"],
    "multimem_reduce": ["multimem.reduce"],
    "multimem_store": ["multimem.store"],
    "rdma_write": ["rdma.write"],
    "rdma_read": ["rdma.read"],
}

_OP_TO_CATEGORY: Dict[str, str] = {}
for cat, ops in EXECUTOR_EQUIVALENCES.items():
    for op in ops:
        _OP_TO_CATEGORY[op] = cat


def add_executor_equivalences(eg: EGraph) -> int:
    """For every ENode, add equivalent executor variants and merge.

    Returns the number of new merges performed.
    """
    merges = 0
    snapshot = list(eg.eclass_nodes.items())
    for eid, enodes in snapshot:
        for enode in list(enodes):
            cat = _OP_TO_CATEGORY.get(enode.op_type)
            if cat is None:
                continue
            equivalents = EXECUTOR_EQUIVALENCES.get(cat, [])
            for alt_op in equivalents:
                if alt_op == enode.op_type:
                    continue
                alt_enode = ENode(alt_op, enode.rank, enode.children, enode.params)
                alt_eid = eg.add(alt_enode)
                if eg.find(alt_eid) != eg.find(eid):
                    eg.merge(eid, alt_eid)
                    merges += 1
    if merges > 0:
        eg.rebuild()
    return merges


def structural_rules(eg: EGraph) -> List[Tuple[PatternExpr, Callable[[Substitution], Optional[int]]]]:
    """Build structural rewrite rules bound to the given EGraph.

    Rule B1 -- Duplicate elimination:
      op(op(x, P), P) where both ops have same category and same params
      -> op(x, P)
    We implement this by scanning for parent-child pairs with matching
    category and params, then merging parent e-class with child e-class.
    Since this is hard to express via PatternExpr alone (params must match),
    we use a single pass rule with a trivially-matching pattern and a
    smart applier that scans the graph internally.
    """

    def _dedup_applier(_sub: Substitution) -> Optional[int]:
        merges = 0
        snapshot = list(eg.eclass_nodes.items())
        for eid, enodes in snapshot:
            eid = eg.find(eid)
            for enode in list(enodes):
                if len(enode.children) == 0:
                    continue
                cat = _OP_TO_CATEGORY.get(enode.op_type)
                if cat is None:
                    continue
                for child_eid in enode.children:
                    child_eid = eg.find(child_eid)
                    child_enodes = list(eg.eclass_nodes.get(child_eid, set()))
                    for child_enode in child_enodes:
                        child_cat = _OP_TO_CATEGORY.get(child_enode.op_type)
                        if child_cat != cat:
                            continue
                        if child_enode.params != enode.params:
                            continue
                        if eg.find(eid) != eg.find(child_eid):
                            eg.merge(eid, child_eid)
                            merges += 1
        return merges if merges > 0 else None

    trigger = PatternExpr(var="__dedup_any")
    return [(trigger, _dedup_applier)]


def smt_rules_to_egraph(
    eg: EGraph,
    rules: Optional[List[RewriteRule]] = None,
    max_k: int = 6,
) -> List[Tuple[PatternExpr, Callable[[Substitution], Optional[int]]]]:
    """Convert cached SMT-verified RewriteRules into e-graph rewrite rules.

    Each RewriteRule says "source pattern == replacement pattern".  We convert
    this into a saturation rule: when we see the source pattern in the e-graph,
    add the replacement as an equivalent and merge.

    Only handles single-node rules (executor upgrades) and 2-node shrinking
    rules (duplicate elimination), since these are the most common shapes
    from k<=3 discovery.
    """
    if rules is None:
        rules = load_all_rules(max_k=max_k)

    egraph_rules = []
    for rule in rules:
        if len(rule.source_nodes) == 1 and len(rule.replacement_nodes) == 1:
            src = rule.source_nodes[0]
            repl = rule.replacement_nodes[0]
            if src.op_type == repl.op_type:
                continue
            pattern = PatternExpr(op_type=src.op_type.value, rank=src.rank)

            def _make_applier(repl_op_type, _eg=eg):
                def applier(sub: Substitution) -> Optional[int]:
                    for eid in list(_eg.eclass_nodes.keys()):
                        eid = _eg.find(eid)
                        for enode in list(_eg.eclass_nodes.get(eid, set())):
                            if enode.op_type != repl_op_type:
                                continue
                            if _eg.find(eid) != eid:
                                continue
                    return None
                return applier

            egraph_rules.append((pattern, _make_applier(repl.op_type.value)))

    return egraph_rules
