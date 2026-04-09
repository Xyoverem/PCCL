"""Auto-channelization pass.

Identifies sequential chains of same-type data operations on contiguous
or partitionable data ranges and splits them across independent hardware
channels for parallel execution.

Each channel is an independent HW path (NVLink sub-link, RDMA QP) with
dedicated bandwidth.  The pass only channelizes when the channel-aware
cost model confirms improvement.
"""

from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from ..graph import PrimitiveIRGraph
from ..nodes import (
    IRNode, IRNodeVariant, PrimitiveOpType,
    SmCopyNode, TmaCopyNode, CeCopyNode,
    SmReduceNode, TmaReduceNode,
    RdmaWriteNode, RdmaReadNode,
    NotifyNode, WaitNotifyNode,
)
from ..superopt.cost_model import GpuProfile, H100_PROFILE, critical_path_cost
from ..superopt.rule import PatternNode, PatternEdge

_DATA_OPS = {
    PrimitiveOpType.SM_COPY, PrimitiveOpType.TMA_COPY, PrimitiveOpType.CE_COPY,
    PrimitiveOpType.SM_REDUCE, PrimitiveOpType.TMA_REDUCE,
    PrimitiveOpType.RDMA_WRITE, PrimitiveOpType.RDMA_READ,
}
_COPY_OPS = {
    PrimitiveOpType.SM_COPY, PrimitiveOpType.TMA_COPY, PrimitiveOpType.CE_COPY,
    PrimitiveOpType.RDMA_WRITE, PrimitiveOpType.RDMA_READ,
}
_REDUCE_OPS = {PrimitiveOpType.SM_REDUCE, PrimitiveOpType.TMA_REDUCE}
_SYNC_OPS = {PrimitiveOpType.NOTIFY, PrimitiveOpType.WAIT_NOTIFY, PrimitiveOpType.NOOP}

_COPY_NODE_CLASSES = {SmCopyNode, TmaCopyNode, CeCopyNode, RdmaWriteNode, RdmaReadNode}
_REDUCE_NODE_CLASSES = {SmReduceNode, TmaReduceNode}


class ChannelizePass:
    def __init__(
        self,
        num_channels: int = 2,
        min_data_size: int = 4096,
        profile: GpuProfile = H100_PROFILE,
    ):
        self.num_channels = num_channels
        self.min_data_size = min_data_size
        self.profile = profile

    def run(self, graph: PrimitiveIRGraph) -> PrimitiveIRGraph:
        chains = self._find_channelizable_chains(graph)
        for chain in chains:
            self._try_channelize_chain(graph, chain)
        return graph

    def _find_channelizable_chains(
        self,
        graph: PrimitiveIRGraph,
    ) -> List[List[IRNodeVariant]]:
        topo = graph.topological_sort()
        visited = set()
        chains = []

        for node in topo:
            if node.op_id in visited or node.op_type not in _DATA_OPS:
                continue
            if node.channel != 0:
                continue

            chain = [node]
            visited.add(node.op_id)
            current = node

            while True:
                data_succ = self._next_data_op(graph, current)
                if data_succ is None:
                    break
                if data_succ.op_id in visited:
                    break
                if data_succ.op_type != node.op_type:
                    break
                if data_succ.channel != 0:
                    break
                if not self._is_contiguous(current, data_succ):
                    break
                chain.append(data_succ)
                visited.add(data_succ.op_id)
                current = data_succ

            if len(chain) > self.num_channels:
                chains.append(chain)

        return chains

    def _next_data_op(
        self,
        graph: PrimitiveIRGraph,
        node: IRNodeVariant,
    ) -> Optional[IRNodeVariant]:
        queue = list(node.next_ops)
        visited = set()
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            n = graph.nodes.get(nid)
            if n is None:
                continue
            if n.op_type in _SYNC_OPS:
                queue.extend(n.next_ops)
            elif n.op_type in _DATA_OPS:
                return n
        return None

    def _is_contiguous(self, a: IRNodeVariant, b: IRNodeVariant) -> bool:
        pa = a.to_params()
        pb = b.to_params()
        if "size" in pa and "size" in pb:
            a_end = pa.get("src_offset", 0) + pa["size"]
            b_start = pb.get("src_offset", 0)
            return b_start == a_end
        if "count" in pa and "count" in pb:
            a_end = pa.get("src_offset", 0) + pa["count"]
            b_start = pb.get("src_offset", 0)
            return b_start == a_end
        return False

    def _try_channelize_chain(
        self,
        graph: PrimitiveIRGraph,
        chain: List[IRNodeVariant],
    ) -> bool:
        n = len(chain)
        nc = self.num_channels
        if n <= nc:
            return False

        data_size = self._chain_data_size(chain)
        if data_size < self.min_data_size:
            return False

        single_pnodes = [
            PatternNode(op_type=nd.op_type, rank=0, chunk_id=0, channel=0)
            for nd in chain
        ]
        single_edges = [PatternEdge(i, i + 1) for i in range(n - 1)]
        single_cost = critical_path_cost(
            single_pnodes, single_edges, self.profile, data_size // n,
        )

        per_ch = (n + nc - 1) // nc
        multi_pnodes = []
        multi_edges = []
        for ch in range(nc):
            start = ch * per_ch
            end = min(start + per_ch, n)
            ch_start_idx = len(multi_pnodes)
            for j in range(start, end):
                multi_pnodes.append(
                    PatternNode(op_type=chain[j].op_type, rank=0, chunk_id=0, channel=ch)
                )
            for j in range(end - start - 1):
                multi_edges.append(PatternEdge(ch_start_idx + j, ch_start_idx + j + 1))

        multi_cost = critical_path_cost(
            multi_pnodes, multi_edges, self.profile, data_size // n,
        )

        if multi_cost >= single_cost:
            return False

        self._apply_channelization(graph, chain, nc)
        return True

    def _chain_data_size(self, chain: List[IRNodeVariant]) -> int:
        total = 0
        for nd in chain:
            params = nd.to_params()
            total += params.get("size", params.get("count", 0))
        return total

    def _apply_channelization(
        self,
        graph: PrimitiveIRGraph,
        chain: List[IRNodeVariant],
        nc: int,
    ):
        n = len(chain)
        per_ch = (n + nc - 1) // nc

        for i, node in enumerate(chain):
            ch = i // per_ch if per_ch > 0 else 0
            if ch >= nc:
                ch = nc - 1
            node.channel = ch

    def _data_size_of(self, node: IRNodeVariant) -> int:
        params = node.to_params()
        return params.get("size", params.get("count", 0))
