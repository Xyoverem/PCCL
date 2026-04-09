"""Rewrite rule representation and JSON serialization."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from ..nodes import PrimitiveOpType


@dataclass(frozen=True)
class PatternNode:
    op_type: PrimitiveOpType
    rank: int
    chunk_id: int = 0
    link_type: str = "intra"
    channel: int = 0
    param_vars: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type.value,
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "link_type": self.link_type,
            "channel": self.channel,
            "param_vars": dict(self.param_vars),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatternNode":
        return cls(
            op_type=PrimitiveOpType(d["op_type"]),
            rank=d["rank"],
            chunk_id=d.get("chunk_id", 0),
            link_type=d.get("link_type", "intra"),
            channel=d.get("channel", 0),
            param_vars=d.get("param_vars", {}),
        )


@dataclass(frozen=True)
class PatternEdge:
    src_idx: int
    dst_idx: int

    def to_dict(self) -> Dict[str, int]:
        return {"src_idx": self.src_idx, "dst_idx": self.dst_idx}

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "PatternEdge":
        return cls(src_idx=d["src_idx"], dst_idx=d["dst_idx"])


@dataclass
class ApplicabilityPredicate:
    param: str
    op: str
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"param": self.param, "op": self.op, "value": self.value}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ApplicabilityPredicate":
        return cls(param=d["param"], op=d["op"], value=d["value"])

    def evaluate(self, bindings: Dict[str, Any]) -> bool:
        if self.param not in bindings:
            return True
        v = bindings[self.param]
        if self.op == ">":
            return v > self.value
        if self.op == "<":
            return v < self.value
        if self.op == ">=":
            return v >= self.value
        if self.op == "<=":
            return v <= self.value
        if self.op == "==":
            return v == self.value
        if self.op == "!=":
            return v != self.value
        raise ValueError(f"Unknown predicate operator: {self.op}")


@dataclass
class RewriteRule:
    source_nodes: List[PatternNode]
    source_edges: List[PatternEdge]
    replacement_nodes: List[PatternNode]
    replacement_edges: List[PatternEdge]
    param_constraints: List[str] = field(default_factory=list)
    predicates: List[ApplicabilityPredicate] = field(default_factory=list)
    link_type: str = "intra"
    rule_id: str = ""

    def source_size(self) -> int:
        return len(self.source_nodes)

    def replacement_size(self) -> int:
        return len(self.replacement_nodes)

    def is_applicable(self, bindings: Dict[str, Any]) -> bool:
        return all(p.evaluate(bindings) for p in self.predicates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "link_type": self.link_type,
            "source_nodes": [n.to_dict() for n in self.source_nodes],
            "source_edges": [e.to_dict() for e in self.source_edges],
            "replacement_nodes": [n.to_dict() for n in self.replacement_nodes],
            "replacement_edges": [e.to_dict() for e in self.replacement_edges],
            "param_constraints": self.param_constraints,
            "predicates": [p.to_dict() for p in self.predicates],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewriteRule":
        return cls(
            rule_id=d.get("rule_id", ""),
            link_type=d.get("link_type", "intra"),
            source_nodes=[PatternNode.from_dict(n) for n in d["source_nodes"]],
            source_edges=[PatternEdge.from_dict(e) for e in d["source_edges"]],
            replacement_nodes=[PatternNode.from_dict(n) for n in d["replacement_nodes"]],
            replacement_edges=[PatternEdge.from_dict(e) for e in d["replacement_edges"]],
            param_constraints=d.get("param_constraints", []),
            predicates=[ApplicabilityPredicate.from_dict(p) for p in d.get("predicates", [])],
        )
