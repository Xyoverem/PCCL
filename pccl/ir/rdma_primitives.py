"""
RDMA Hardware Primitives

Defines RDMA-specific hardware primitives for Layer 3 of the three-layer IR architecture.
These primitives map to actual RDMA verbs operations and are executed by RDMA plugins.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .json_serializer import (
    IRValue, IROperation, IRGraph, IRType,
    DeviceType
)
from .primitive_ir import PrimitiveOperation


class RDMAOperationType(Enum):
    """RDMA verbs-based operation types"""
    RDMA_WRITE = "rdma_write"
    RDMA_READ = "rdma_read"
    SEND = "send"
    RECV = "recv"
    ATOMIC_FAI = "atomic_fai"  # Fetch-and-add
    ATOMIC_CAS = "atomic_cas"  # Compare-and-swap
    COMPARE_SWAP = "compare_swap"
    ATOMIC_SWAP = "atomic_swap"
    POST_RECV = "post_recv"
    POLL_CQ = "poll_cq"  # Poll completion queue


class RDMAMemoryRegionType(Enum):
    """RDMA memory region types"""
    MR_MR = "mr"           # Memory Region
    MR_MW = "mw"           # Memory Window
    MR_MR_GLOBAL = "mr_global"  # Global Memory Region


class RDMAProtectionDomain(Enum):
    """RDMA protection domain types"""
    PD_IB = "pd_ib"         # InfiniBand Protection Domain
    PD_ETH = "pd_eth"       # Ethernet Protection Domain


class RDMACompletionType(Enum):
    """RDMA completion queue entry types"""
    CQ_SEND = "send"
    CQ_RECV = "recv"
    CQ_RDMA_WRITE = "rdma_write"
    CQ_RDMA_READ = "rdma_read"
    CQ_ATOMIC = "atomic"


class RDMAQueuePairType(Enum):
    """RDMA queue pair types"""
    QP_RC = "rc"           # Reliable Connected
    QP_UD = "ud"           # Unreliable Datagram
    QP_UC = "uc"           # Unreliable Connected
    QP_RAW = "raw"         # Raw Ethernet


@dataclass
class RDMALocalKey:
    """RDMA local key information"""
    qpn: int  # Queue Pair Number
    lid: int  # Local Identifier
    psn: int  # Packet Sequence Number

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qpn": self.qpn,
            "lid": self.lid,
            "psn": self.psn
        }


@dataclass
class RDMARemoteKey:
    """RDMA remote key information"""
    qpn: int  # Remote Queue Pair Number
    lid: int  # Remote Identifier
    psn: int  # Remote Packet Sequence Number
    gid: int  # Global Identifier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qpn": self.qpn,
            "lid": self.lid,
            "psn": self.psn,
            "gid": self.gid
        }


@dataclass
class RDMAMemoryRegion:
    """RDMA memory region information"""
    mr_id: int
    address: int
    size: int
    access_flags: List[str]
    lkey: RDMALocalKey
    rkey: RDMARemoteKey
    mr_type: RDMAMemoryRegionType

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mr_id": self.mr_id,
            "address": self.address,
            "size": self.size,
            "access_flags": self.access_flags,
            "lkey": self.lkey.to_dict(),
            "rkey": self.rkey.to_dict(),
            "mr_type": self.mr_type.value
        }


class RDMAHardwarePrimitiveOperation(PrimitiveOperation):
    """Base class for RDMA hardware primitive operations"""

    def __init__(self,
                 id: str,
                 rdma_op_type: RDMAOperationType,
                 inputs: List[str],
                 outputs: List[str],
                 attributes: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            id=id,
            op_type=rdma_op_type.value,
            inputs=inputs,
            outputs=outputs,
            attributes=attributes or {},
            metadata=metadata or {}
        )
        self.rdma_op_type = rdma_op_type


class RDMAWriteOp(RDMAHardwarePrimitiveOperation):
    """RDMA write operation"""

    def __init__(self,
                 id: str,
                 source: str,
                 target_mr: str,
                 target_address: int,
                 size: int,
                 immediate: int = 0,
                 opcode: str = "IBV_WR_SEND",
                 send_flags: List[str] = None):
        super().__init__(
            id=id,
            rdma_op_type=RDMAOperationType.RDMA_WRITE,
            inputs=[source],
            outputs=[],
            attributes={
                "target_mr": target_mr,
                "target_address": target_address,
                "size": size,
                "immediate": immediate,
                "opcode": opcode,
                "send_flags": send_flags or [],
                "operation_type": "rdma_write"
            }
        )


class RDMAReadOp(RDMAHardwarePrimitiveOperation):
    """RDMA read operation"""

    def __init__(self,
                 id: str,
                 source_mr: str,
                 source_address: int,
                 target: str,
                 size: int,
                 immediate: int = 0,
                 opcode: str = "IBV_RDMA_READ",
                 recv_flags: List[str] = None):
        super().__init__(
            id=id,
            rdma_op_type=RDMAOperationType.RDMA_READ,
            inputs=[],
            outputs=[target],
            attributes={
                "source_mr": source_mr,
                "source_address": source_address,
                "size": size,
                "immediate": immediate,
                "opcode": opcode,
                "recv_flags": recv_flags or [],
                "operation_type": "rdma_read"
            }
        )


class RDMASendOp(RDMAHardwarePrimitiveOperation):
    """RDMA send operation"""

    def __init__(self,
                 id: str,
                 source: str,
                 destination_qp: int,
                 size: int,
                 immediate: int = 0,
                 opcode: str = "IBV_SEND",
                 send_flags: List[str] = None):
        super().__init__(
            id=id,
            rdma_op_type=RDMAOperationType.SEND,
            inputs=[source],
            outputs=[],
            attributes={
                "destination_qp": destination_qp,
                "size": size,
                "immediate": immediate,
                "opcode": opcode,
                "send_flags": send_flags or [],
                "operation_type": "send"
            }
        )


class RDMARecvOp(RDMAHardwarePrimitiveOperation):
    """RDMA receive operation"""

    def __init__(self,
                 id: str,
                 source_qp: int,
                 target: str,
                 size: int,
                 immediate: int = 0,
                 opcode: str = "IBV_RECV",
                 recv_flags: List[str] = None):
        super().__init__(
            id=id,
            rdma_op_type=RDMAOperationType.RECV,
            inputs=[],
            outputs=[target],
            attributes={
                "source_qp": source_qp,
                "size": size,
                "immediate": immediate,
                "opcode": opcode,
                "recv_flags": recv_flags or [],
                "operation_type": "recv"
            }
        )


class RDMAAtomicFAIOp(RDMAHardwarePrimitiveOperation):
    """RDMA atomic fetch-and-add operation"""

    def __init__(self,
                 id: str,
                 target_mr: str,
                 target_address: int,
                 addend_value: int,
                 compare: Optional[int] = None,
                 opcode: str = "IBV_ATOMIC_FADD"):
        super().__init__(
            id=id,
            rdma_op_type=RDMAOperationType.ATOMIC_FAI,
            inputs=[],
            outputs=[],
            attributes={
                "target_mr": target_mr,
                "target_address": target_address,
                "addend_value": append_value,
                "compare": compare,
                "opcode": opcode,
                "operation_type": "atomic_fai"
            }
        )


class RDMAAtomicCASOp(RDMAHardwarePrimitiveOperation):
    """RDMA atomic compare-and-swap operation"""

    def __init__(self,
                 id: str,
                 target_mr: str,
                 target_address: int,
                 compare_value: int,
                 swap_value: int,
                 opcode: str = "IBV_ATOMIC_CAS"):
        super().__init__(
            id=id,
            rdma_op_type=RDMAOperationType.ATOMIC_CAS,
            inputs=[],
            outputs=[],
            attributes={
                "target_mr": target_mr,
                "target_address": target_address,
                "compare_value": compare_value,
                "swap_value": swap_value,
                "opcode": opcode,
                "operation_type": "atomic_cas"
            }
        )


class RDMAPostRecvOp(RDMAHardwarePrimitiveOperation):
    """RDMA post receive operation"""

    def __init__(self,
                 id: str,
                 wr_id: str,
                 target: str,
                 size: int):
        super().__init__(
            id=id,
            rdma_op_type=RDMAOperationType.POST_RECV,
            inputs=[wr_id],
            outputs=[target],
            attributes={
                "wr_id": wr_id,
                "size": size,
                "operation_type": "post_recv"
            }
        )


class RDMAPollCQOp(RDMAHardwarePrimitiveOperation):
    """RDMA completion queue poll operation"""

    def __init__(self,
                 id: str,
                 cq_id: int,
                 min_completions: int = 1,
                 timeout_ms: int = 0):
        super().__init__(
            id=id,
            rdma_op_type=RDMAOperationType.POLL_CQ,
            inputs=[],
            outputs=[],
            attributes={
                "cq_id": cq_id,
                "min_completions": min_completions,
                "timeout_ms": timeout_ms,
                "operation_type": "poll_cq"
            }
        )


class RDMAHardwarePrimitiveIRBuilder:
    """Builder for creating RDMA hardware primitive IR graphs"""

    def __init__(self, graph_id: str = "rdma_hardware_ir"):
        self.graph = IRGraph(ir_type=IRType.HARDWARE, values={}, operations={})
        self.graph.metadata["graph_id"] = graph_id
        self.graph.metadata["device_type"] = "rdma"
        self.value_counter = 0
        self.op_counter = 0
        self.mr_counter = 0
        self.qp_counter = 0
        self.cq_counter = 0

    def add_rdma_value(self,
                         dtype: str,
                         shape: List[int],
                         address: int = 0,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add an RDMA-specific value to the graph"""
        value_id = f"rdma_value_{self.value_counter}"
        self.value_counter += 1

        value = IRValue(
            id=value_id,
            dtype=dtype,
            shape=shape,
            device_id=0,
            device_type=DeviceType.RDMA,
            metadata={
                "address": address,
                **(metadata or {})
            }
        )

        self.graph.add_value(value)
        return value_id

    def create_memory_region(self,
                            size: int,
                            address: int = 0,
                            access_flags: List[str] = None,
                            mr_type: RDMAMemoryRegionType = RDMAMemoryRegionType.MR_MR) -> str:
        """Create and register a memory region"""
        mr_id = self.mr_counter
        self.mr_counter += 1

        # Create local and remote keys
        lkey = RDMALocalKey(
            qpn=self.qp_counter,
            lid=self.qp_counter,
            psn=0
        )
        rkey = RDMARemoteKey(
            qpn=self.qp_counter,
            lid=self.qp_counter,
            psn=0,
            gid=0
        )

        mr_info = RDMAMemoryRegion(
            mr_id=mr_id,
            address=address,
            size=size,
            access_flags=access_flags or ["READ", "WRITE"],
            lkey=lkey,
            rkey=rkey,
            mr_type=mr_type
        )

        # Store MR info in metadata
        mr_metadata = mr_info.to_dict()
        mr_metadata["registered"] = True

        return f"mr_{mr_id}"

    def add_rdma_write(self,
                        source: str,
                        target_mr: str,
                        target_address: int,
                        size: int,
                        immediate: int = 0) -> str:
        """Add RDMA write operation"""
        op_id = f"rdma_write_{self.op_counter}"
        self.op_counter += 1

        op = RDMAWriteOp(
            id=op_id,
            source=source,
            target_mr=target_mr,
            target_address=target_address,
            size=size,
            immediate=immediate
        )

        self.graph.add_operation(op)
        return op_id

    def add_rdma_read(self,
                       source_mr: str,
                       source_address: int,
                       target: str,
                       size: int,
                       immediate: int = 0) -> str:
        """Add RDMA read operation"""
        op_id = f"rdma_read_{self.op_counter}"
        self.op_counter += 1

        op = RDMAReadOp(
            id=op_id,
            source_mr=source_mr,
            source_address=source_address,
            target=target,
            size=size,
            immediate=immediate
        )

        self.graph.add_operation(op)
        return op_id

    def add_send(self,
                 source: str,
                 destination_qp: int,
                 size: int,
                 immediate: int = 0) -> str:
        """Add RDMA send operation"""
        op_id = f"rdma_send_{self.op_counter}"
        self.op_counter += 1

        op = RDMASendOp(
            id=op_id,
            source=source,
            destination_qp=destination_qp,
            size=size,
            immediate=immediate
        )

        self.graph.add_operation(op)
        return op_id

    def add_recv(self,
                 source_qp: int,
                 target: str,
                 size: int,
                 immediate: int = 0) -> str:
        """Add RDMA receive operation"""
        op_id = f"rdma_recv_{self.op_counter}"
        self.op_counter += 1

        op = RDMARecvOp(
            id=op_id,
            source_qp=source_qp,
            target=target,
            size=size,
            immediate=immediate
        )

        self.graph.add_operation(op)
        return op_id

    def add_atomic_fai(self,
                       target_mr: str,
                       target_address: int,
                       append_value: int,
                       compare: Optional[int] = None) -> str:
        """Add RDMA atomic fetch-and-add operation"""
        op_id = f"rdma_atomic_fai_{self.op_counter}"
        self.op_counter += 1

        op = RDMAAtomicFAIOp(
            id=id,
            target_mr=target_mr,
            target_address=target_address,
            append_value=append_value,
            compare=compare
        )

        self.graph.add_operation(op)
        return op_id

    def add_atomic_cas(self,
                       target_mr: str,
                       target_address: int,
                       compare_value: int,
                       swap_value: int) -> str:
        """Add RDMA atomic compare-and-swap operation"""
        op_id = f"rdma_atomic_cas_{self.op_counter}"
        self.op_counter += 1

        op = RDMAAtomicCASOp(
            id=id,
            target_mr=target_mr,
            target_address=target_address,
            compare_value=compare_value,
            swap_value=swap_value
        )

        self.graph.add_operation(op)
        return op_id

    def add_post_recv(self,
                      wr_id: str,
                      target: str,
                      size: int) -> str:
        """Add RDMA post receive operation"""
        op_id = f"rdma_post_recv_{self.op_counter}"
        self.op_counter += 1

        op = RDMAPostRecvOp(
            id=op_id,
            wr_id=wr_id,
            target=target,
            size=size
        )

        self.graph.add_operation(op)
        return op_id

    def add_poll_cq(self,
                    cq_id: int,
                    min_completions: int = 1,
                    timeout_ms: int = 0) -> str:
        """Add RDMA completion queue poll operation"""
        op_id = f"rdma_poll_cq_{self.op_counter}"
        self.op_counter += 1

        op = RDMAPollCQOp(
            id=op_id,
            cq_id=cq_id,
            min_completions=min_completions,
            timeout_ms=timeout_ms
        )

        self.graph.add_operation(op)
        return op_id

    def get_graph(self) -> IRGraph:
        """Get the built hardware primitive IR graph"""
        return self.graph

    def get_value(self, value_id: str) -> Optional[IRValue]:
        """Get an RDMA value by ID"""
        return self.graph.get_value(value_id)

    def get_operation(self, op_id: str) -> Optional[IROperation]:
        """Get an RDMA operation by ID"""
        return self.graph.get_operation(op_id)


def create_rdma_ring_allreduce_example() -> IRGraph:
    """Create an example RDMA-based ring AllReduce hardware primitive pattern"""
    builder = RDMAHardwarePrimitiveIRBuilder("rdma_ring_allreduce")

    # Create memory regions for ring allreduce
    chunk_size = 1024
    memory_regions = []

    for i in range(4):
        mr_id = builder.create_memory_region(
            size=chunk_size * 4,  # 4 bytes per float
            access_flags=["READ", "WRITE"],
            mr_type=RDMAMemoryRegionType.MR_MR
        )
        memory_regions.append(mr_id)

    # Input values (simplified)
    input_values = []
    for i, mr_id in enumerate(memory_regions):
        input_val = builder.add_rdma_value(
            "float32",
            [chunk_size],
            address=i * chunk_size * 4,
            metadata={"rank": i, "mr_id": mr_id}
        )
        input_values.append(input_val)

    # Ring AllReduce: Scatter-Reduce phase
    current_values = input_values.copy()
    ring_operations = []

    for step in range(3):  # 3 rounds for 4 nodes
        # Send to next node
        target_qp = (step + 1) % 4
        send_ops = []
        for i, value in enumerate(current_values):
            send_op = builder.add_send(
                source=value,
                destination_qp=target_qp,
                size=chunk_size * 4
            )
            send_ops.append(send_op)

        # Receive from previous node
        source_qp = (step - 1) % 4
        recv_ops = []
        for i in range(4):
            received_val = builder.add_rdma_value(
                "float32",
                [chunk_size],
                metadata={"rank": i, "step": step}
            )
            recv_op = builder.add_recv(
                source_qp=source_qp,
                target=received_val,
                size=chunk_size * 4
            )
            recv_ops.append(received_val)

        # Atomic add (simplified reduction)
        reduced_ops = []
        for i in range(4):
            if i == 0:
                # Node 0 reduces its own data with received data
                reduced_op = builder.add_atomic_fai(
                    target_mr=memory_regions[i],
                    target_address=i * chunk_size * 4,
                    append_value=1,  # Simplified: just add 1
                )
                reduced_ops.append(reduced_op)
                current_values[i] = reduced_op
            else:
                # Other nodes wait and receive reduced data
                current_values[i] = builder.add_rdma_value(
                    "float32",
                    [chunk_size],
                    metadata={"rank": i, "reduced": True}
                )

        ring_operations.extend(send_ops + recv_ops + reduced_ops)

    # AllGather phase
    final_values = current_values.copy()
    gather_operations = []

    for step in range(3):
        # Send reduced data to next node
        target_qp = (step + 1) % 4
        send_ops = []
        for i, value in enumerate(final_values):
            send_op = builder.add_rdma_write(
                source=value,
                target_mr=memory_regions[target_qp],
                target_address=target_qp * chunk_size * 4,
                size=chunk_size * 4
            )
            send_ops.append(send_op)

        # Receive data from previous node
        source_qp = (step - 1) % 4
        recv_ops = []
        for i in range(4):
            if i != source_qp:
                received_val = builder.add_rdma_value(
                    "float32",
                    [chunk_size],
                    metadata={"rank": i, "gather_step": step}
                )
                recv_op = builder.add_rdma_read(
                    source_mr=memory_regions[i],
                    source_address=i * chunk_size * 4,
                    target=received_val,
                    size=chunk_size * 4
                )
                recv_ops.append(recv_op)
                final_values[i] = received_val

        gather_operations.extend(send_ops + recv_ops)

    # Poll completion queues to ensure all operations complete
    for cq_id in range(8):  # Assume we have multiple completion queues
        builder.add_poll_cq(cq_id, min_completions=1)

    return builder.get_graph()


def create_rdma_two_sided_communication() -> IRGraph:
    """Create a two-sided RDMA communication example"""
    builder = RDMAHardwarePrimitiveIRBuilder("rdma_two_sided_comm")

    # Create memory regions for both sides
    size = 1024
    mr1 = builder.create_memory_region(size, access_flags=["READ", "WRITE"])
    mr2 = builder.create_memory_region(size, access_flags=["READ", "WRITE"])

    # Create values
    val1 = builder.add_rdma_value("uint8", [size], address=0, metadata={"side": 1})
    val2 = builder.add_rdma_value("uint8", [size], address=0, metadata={"side": 2})

    # Two-sided RDMA exchange
    # Side 1 writes to Side 2
    write_op1 = builder.add_rdma_write(val1, mr2, 0, size)

    # Side 2 reads from Side 1
    read_op2 = builder.add_rdma_read(mr1, 0, val2, size)

    # Side 2 writes to Side 1
    write_op2 = builder.add_rdma_write(val2, mr1, 0, size)

    # Side 1 reads from Side 2
    read_op1 = builder.add_rdma_read(mr2, 0, val1, size)

    # Poll completion queues
    builder.add_poll_cq(0)
    builder.add_poll_cq(1)

    return builder.get_graph()