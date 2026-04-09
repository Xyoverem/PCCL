"""Pure Python e-graph implementation for equality saturation.

Components:
  - UnionFind: path-compressed union-find for e-class IDs
  - ENode: immutable, hashable operation node (with optional params)
  - EGraph: main class with add, merge, rebuild, ematch, saturate, extract
"""

from dataclasses import dataclass, field
from typing import (
    Dict, List, Tuple, Optional, Set, FrozenSet, Callable, Any, NamedTuple,
)
from collections import defaultdict

from ..nodes import PrimitiveOpType


OP_CATEGORY_MAP: Dict[str, str] = {
    "sm.copy": "copy",
    "tma.copy": "copy",
    "ce.copy": "copy",
    "sm.reduce": "reduce",
    "tma.reduce": "reduce",
    "multimem.reduce": "reduce",
    "multimem.store": "store",
    "rdma.write": "rdma_write",
    "rdma.read": "rdma_read",
    "notify": "notify",
    "wait_notify": "wait_notify",
    "noop": "noop",
}


class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def make_set(self, x: int):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> int:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return rx


@dataclass(frozen=True)
class ENode:
    op_type: str
    rank: int
    children: Tuple[int, ...] = ()
    params: Tuple[Tuple[str, Any], ...] = ()

    def canonicalize(self, uf: UnionFind) -> "ENode":
        new_children = tuple(uf.find(c) for c in self.children)
        return ENode(self.op_type, self.rank, new_children, self.params)

    @property
    def category(self) -> str:
        return OP_CATEGORY_MAP.get(self.op_type, self.op_type)


class Substitution(dict):
    pass


@dataclass
class PatternExpr:
    op_type: Optional[str] = None
    rank: Optional[int] = None
    children: List["PatternExpr"] = field(default_factory=list)
    var: Optional[str] = None
    op_category: Optional[str] = None

    def is_var(self) -> bool:
        return self.var is not None


class EGraph:
    def __init__(self):
        self.uf = UnionFind()
        self.eclass_nodes: Dict[int, Set[ENode]] = defaultdict(set)
        self.hashcons: Dict[ENode, int] = {}
        self.next_id = 0
        self.pending: List[Tuple[int, int]] = []
        self.dirty: bool = False

    def _alloc_id(self) -> int:
        eid = self.next_id
        self.next_id += 1
        self.uf.make_set(eid)
        return eid

    def add(self, enode: ENode) -> int:
        enode = enode.canonicalize(self.uf)
        if enode in self.hashcons:
            return self.uf.find(self.hashcons[enode])
        eid = self._alloc_id()
        self.hashcons[enode] = eid
        self.eclass_nodes[eid].add(enode)
        return eid

    def merge(self, id1: int, id2: int) -> int:
        id1 = self.uf.find(id1)
        id2 = self.uf.find(id2)
        if id1 == id2:
            return id1
        merged = self.uf.union(id1, id2)
        other = id2 if merged == id1 else id1
        self.eclass_nodes[merged] |= self.eclass_nodes[other]
        del self.eclass_nodes[other]
        self.dirty = True
        return merged

    def rebuild(self):
        if not self.dirty:
            return
        new_hashcons: Dict[ENode, int] = {}
        for enode, eid in self.hashcons.items():
            canonical = enode.canonicalize(self.uf)
            canonical_eid = self.uf.find(eid)
            if canonical in new_hashcons:
                existing = new_hashcons[canonical]
                if existing != canonical_eid:
                    self.merge(existing, canonical_eid)
            else:
                new_hashcons[canonical] = canonical_eid
        self.hashcons = new_hashcons

        new_classes: Dict[int, Set[ENode]] = defaultdict(set)
        for enode, eid in self.hashcons.items():
            new_classes[self.uf.find(eid)].add(enode)
        self.eclass_nodes = new_classes
        self.dirty = False

    def find(self, eid: int) -> int:
        return self.uf.find(eid)

    def ematch(self, pattern: PatternExpr) -> List[Substitution]:
        results = []
        for eid in list(self.eclass_nodes.keys()):
            for sub in self._match_eclass(pattern, eid):
                results.append(sub)
        return results

    def _match_eclass(self, pattern: PatternExpr, eid: int) -> List[Substitution]:
        eid = self.uf.find(eid)
        if pattern.is_var():
            sub = Substitution()
            sub[pattern.var] = eid
            return [sub]

        results = []
        for enode in self.eclass_nodes.get(eid, set()):
            if pattern.op_type is not None and enode.op_type != pattern.op_type:
                continue
            if pattern.op_category is not None and enode.category != pattern.op_category:
                continue
            if pattern.rank is not None and enode.rank != pattern.rank:
                continue
            if len(pattern.children) != len(enode.children):
                continue

            if not pattern.children:
                results.append(Substitution())
                continue

            child_matches = [
                self._match_eclass(cpat, self.uf.find(cid))
                for cpat, cid in zip(pattern.children, enode.children)
            ]
            for combo in _cartesian_subs(child_matches):
                results.append(combo)

        return results

    def saturate(
        self,
        rules: List[Tuple[PatternExpr, Callable[[Substitution], Optional[int]]]],
        limit: int = 30,
    ) -> int:
        iterations = 0
        for _ in range(limit):
            new_merges = 0
            for pattern, applier in rules:
                for sub in self.ematch(pattern):
                    result = applier(sub)
                    if result is not None:
                        new_merges += 1
            self.rebuild()
            iterations += 1
            if new_merges == 0:
                break
        return iterations

    def extract(
        self,
        root: int,
        cost_fn: Callable[[ENode, Dict[int, float]], float],
    ) -> Tuple[float, Optional[ENode]]:
        root = self.uf.find(root)
        best_cost: Dict[int, float] = {}
        best_node: Dict[int, Optional[ENode]] = {}

        changed = True
        for _ in range(len(self.eclass_nodes) + 1):
            if not changed:
                break
            changed = False
            for eid, enodes in self.eclass_nodes.items():
                eid = self.uf.find(eid)
                for enode in enodes:
                    child_costs_available = all(
                        self.uf.find(c) in best_cost for c in enode.children
                    )
                    if not child_costs_available and enode.children:
                        continue
                    child_cost_map = {
                        self.uf.find(c): best_cost.get(self.uf.find(c), float("inf"))
                        for c in enode.children
                    }
                    c = cost_fn(enode, child_cost_map)
                    if eid not in best_cost or c < best_cost[eid]:
                        best_cost[eid] = c
                        best_node[eid] = enode
                        changed = True

        return best_cost.get(root, float("inf")), best_node.get(root)

    def eclass_count(self) -> int:
        return len(self.eclass_nodes)

    def enode_count(self) -> int:
        return len(self.hashcons)


def _cartesian_subs(match_lists: List[List[Substitution]]) -> List[Substitution]:
    if not match_lists:
        return [Substitution()]
    result = [Substitution()]
    for matches in match_lists:
        new_result = []
        for existing in result:
            for m in matches:
                merged = _merge_subs(existing, m)
                if merged is not None:
                    new_result.append(merged)
        result = new_result
    return result


def _merge_subs(a: Substitution, b: Substitution) -> Optional[Substitution]:
    merged = Substitution(a)
    for k, v in b.items():
        if k in merged:
            if merged[k] != v:
                return None
        else:
            merged[k] = v
    return merged
