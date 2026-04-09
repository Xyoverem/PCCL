"""PCCL DSL Superoptimizer — SMT-based rule discovery + e-graph rewriting.

Pruning strategies:
  1. Minimality: skip replacements containing a known smaller equivalent
  2. Given k, search replacements of size k, k-1, ..., 1 (shrinking only)
  3. Trivial executor upgrades generated directly (no z3)
  4. Cost-dominated filtering before z3
  5. No downgrade: TMA always dominates SM, only upgrade direction

Multi-channel:
  - Phase 1: single-channel structural rules (existing)
  - Phase 2: channelization rules (1-ch source → multi-ch replacement)
"""

import time
import logging
from typing import List, Optional, Set
from collections import defaultdict

from .rule import RewriteRule, PatternNode, PatternEdge
from .rule_db import save_rules, load_rules as _load_rules
from .enumerator import enumerate_skeletons, trivial_executor_rules, LinkType
from .verifier import (
    check_equivalence, concrete_fingerprint,
    signatures_compatible,
)
from .cost_model import (
    H100_PROFILE, critical_path_cost,
    TopologyProfile, NVLINK_TOPOLOGY, RDMA_TOPOLOGY, PCIE_TOPOLOGY,
    GpuProfile, egraph_node_cost,
)
from .egraph import EGraph, ENode, PatternExpr, OP_CATEGORY_MAP
from .domain_rules import (
    add_executor_equivalences, structural_rules,
    smt_rules_to_egraph, EXECUTOR_EQUIVALENCES,
)
from .egraph_bridge import ir_to_egraph, egraph_to_ir

logger = logging.getLogger(__name__)


def discover_rules(
    k: int,
    timeout_per_pair: float = 2.0,
    link_type: LinkType = LinkType.INTRA,
    progress: bool = True,
    data_sizes_for_cost: tuple = (1024, 1024*1024, 64*1024*1024),
    max_channels: int = 1,
) -> List[RewriteRule]:
    """Discover rewrite rules for all source sizes from 1..k.

    For each source size sk in [k, k-1, ..., 1], searches equivalent
    replacements of size sk, sk-1, ..., 1.  The returned rule set is
    self-contained: iterative application never needs rules from a
    separate discover_rules call.

    When max_channels > 1, also discovers channelization rules where
    source patterns are single-channel and replacements use multiple
    channels for parallel execution on independent HW paths.
    """
    trivial = trivial_executor_rules() if link_type == LinkType.INTRA else []
    if progress:
        logger.info(f"Generated {len(trivial)} trivial executor-upgrade rules")

    skeletons_by_size = {}
    for sz in range(1, k + 1):
        skeletons_by_size[sz] = list(enumerate_skeletons(sz, link_type, max_channels=1))
        if progress:
            logger.info(f"  skeletons k={sz}: {len(skeletons_by_size[sz])}")

    fp_cache = {}
    for sz, skels in skeletons_by_size.items():
        for idx, (nodes, edges) in enumerate(skels):
            fp_cache[(sz, idx)] = concrete_fingerprint(nodes, edges)

    found_equiv_hashes: Set[int] = set()
    rules: List[RewriteRule] = list(trivial)
    stats = {"checked": 0, "skipped_sig": 0, "skipped_fp": 0,
             "skipped_cost": 0, "skipped_minimal": 0}
    t0 = time.time()

    for sk in range(k, 0, -1):
        sources = skeletons_by_size[sk]
        for rk in range(sk, 0, -1):
            repls = skeletons_by_size[rk]

            for src_idx, (src_nodes, src_edges) in enumerate(sources):
                src_fp = fp_cache[(sk, src_idx)]
                src_cost_max = max(
                    critical_path_cost(src_nodes, src_edges, H100_PROFILE, ds)
                    for ds in data_sizes_for_cost
                )

                for repl_idx, (repl_nodes, repl_edges) in enumerate(repls):
                    if not signatures_compatible(src_nodes, repl_nodes):
                        stats["skipped_sig"] += 1
                        continue

                    repl_fp = fp_cache[(rk, repl_idx)]
                    if src_fp is not None and repl_fp is not None and src_fp != repl_fp:
                        stats["skipped_fp"] += 1
                        continue

                    repl_cost_min = min(
                        critical_path_cost(repl_nodes, repl_edges, H100_PROFILE, ds)
                        for ds in data_sizes_for_cost
                    )
                    if repl_cost_min > src_cost_max * 1.05:
                        stats["skipped_cost"] += 1
                        continue

                    repl_hash = _skeleton_hash(repl_nodes, repl_edges)
                    if _is_subsumed(repl_nodes, repl_edges, found_equiv_hashes):
                        stats["skipped_minimal"] += 1
                        continue

                    stats["checked"] += 1
                    rule = check_equivalence(
                        src_nodes, src_edges,
                        repl_nodes, repl_edges,
                        timeout=timeout_per_pair,
                    )
                    if rule is not None:
                        rule.link_type = link_type.value
                        rules.append(rule)
                        found_equiv_hashes.add(repl_hash)

        if progress:
            elapsed = time.time() - t0
            logger.info(
                f"  after sk={sk}: {len(rules)} rules, "
                f"checked={stats['checked']}, "
                f"skipped(sig={stats['skipped_sig']}, fp={stats['skipped_fp']}, "
                f"cost={stats['skipped_cost']}, minimal={stats['skipped_minimal']}), "
                f"{elapsed:.1f}s"
            )

    if max_channels > 1:
        ch_rules = _discover_channelization_rules(
            k, timeout_per_pair, link_type, max_channels,
            data_sizes_for_cost, skeletons_by_size, fp_cache, progress,
        )
        rules.extend(ch_rules)

    if progress:
        elapsed = time.time() - t0
        logger.info(f"discover_rules done: {len(rules)} rules in {elapsed:.1f}s")

    save_rules(k, rules)
    return rules


def _discover_channelization_rules(
    k: int,
    timeout_per_pair: float,
    link_type: LinkType,
    max_channels: int,
    data_sizes_for_cost: tuple,
    src_skeletons_by_size: dict,
    src_fp_cache: dict,
    progress: bool,
) -> List[RewriteRule]:
    """Discover rules where source is single-channel, replacement is multi-channel."""
    rules = []
    stats = {"checked": 0, "skipped_sig": 0, "skipped_fp": 0, "skipped_cost": 0}
    t0 = time.time()

    mc_skeletons_by_size = {}
    for sz in range(2, k + 1):
        mc_skeletons_by_size[sz] = list(
            enumerate_skeletons(sz, link_type, max_channels=max_channels)
        )
        sc_hashes = {
            _skeleton_hash(n, e)
            for n, e in src_skeletons_by_size.get(sz, [])
        }
        mc_skeletons_by_size[sz] = [
            (n, e) for n, e in mc_skeletons_by_size[sz]
            if len(set(nd.channel for nd in n)) > 1
            and _skeleton_hash(n, e) not in sc_hashes
        ]
        if progress:
            logger.info(f"  multi-ch skeletons k={sz}: {len(mc_skeletons_by_size[sz])}")

    mc_fp_cache = {}
    for sz, skels in mc_skeletons_by_size.items():
        for idx, (nodes, edges) in enumerate(skels):
            mc_fp_cache[(sz, idx)] = concrete_fingerprint(nodes, edges)

    for sk in range(k, 1, -1):
        sources = src_skeletons_by_size.get(sk, [])
        mc_repls = mc_skeletons_by_size.get(sk, [])
        if not mc_repls:
            continue

        for src_idx, (src_nodes, src_edges) in enumerate(sources):
            src_fp = src_fp_cache.get((sk, src_idx))
            src_cost_max = max(
                critical_path_cost(src_nodes, src_edges, H100_PROFILE, ds)
                for ds in data_sizes_for_cost
            )

            for repl_idx, (repl_nodes, repl_edges) in enumerate(mc_repls):
                if not signatures_compatible(src_nodes, repl_nodes):
                    stats["skipped_sig"] += 1
                    continue

                repl_fp = mc_fp_cache.get((sk, repl_idx))
                if src_fp is not None and repl_fp is not None and src_fp != repl_fp:
                    stats["skipped_fp"] += 1
                    continue

                repl_cost_min = min(
                    critical_path_cost(repl_nodes, repl_edges, H100_PROFILE, ds)
                    for ds in data_sizes_for_cost
                )
                if repl_cost_min > src_cost_max * 1.05:
                    stats["skipped_cost"] += 1
                    continue

                stats["checked"] += 1
                rule = check_equivalence(
                    src_nodes, src_edges,
                    repl_nodes, repl_edges,
                    timeout=timeout_per_pair,
                )
                if rule is not None:
                    rule.link_type = link_type.value
                    rule.rule_id = f"channelize_{rule.rule_id}"
                    rules.append(rule)

    if progress:
        elapsed = time.time() - t0
        logger.info(
            f"  channelization rules: {len(rules)} found, "
            f"checked={stats['checked']}, "
            f"skipped(sig={stats['skipped_sig']}, fp={stats['skipped_fp']}, "
            f"cost={stats['skipped_cost']}), "
            f"{elapsed:.1f}s"
        )

    return rules


def _skeleton_hash(nodes: List[PatternNode], edges: List[PatternEdge]) -> int:
    node_keys = tuple((n.op_type.value, n.chunk_id, n.channel) for n in nodes)
    edge_keys = tuple(sorted((e.src_idx, e.dst_idx) for e in edges))
    return hash((node_keys, edge_keys))


def _is_subsumed(
    repl_nodes: List[PatternNode],
    repl_edges: List[PatternEdge],
    known_hashes: Set[int],
) -> bool:
    if len(repl_nodes) <= 1:
        return False
    for i in range(len(repl_nodes)):
        sub_nodes = [n for j, n in enumerate(repl_nodes) if j != i]
        idx_map = {}
        new_idx = 0
        for j in range(len(repl_nodes)):
            if j != i:
                idx_map[j] = new_idx
                new_idx += 1
        sub_edges = [
            PatternEdge(idx_map[e.src_idx], idx_map[e.dst_idx])
            for e in repl_edges
            if e.src_idx != i and e.dst_idx != i
        ]
        if _skeleton_hash(sub_nodes, sub_edges) in known_hashes:
            return True
    return False


def load_rules(k: int) -> List[RewriteRule]:
    return _load_rules(k)
