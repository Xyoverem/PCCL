from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .config import TopologyConfig, InterconnectType, TopologyType

@dataclass
class DeviceInfo:
    device_id: int
    device_type: str
    hostname: str
    memory_bandwidth: float
    compute_capability: float
    memory_size: int

@dataclass
class LinkInfo:
    src_device: int
    dst_device: int
    interconnect_type: InterconnectType
    bandwidth: float
    latency: float
    bidirectional: bool = True

@dataclass
class TopologyMetrics:
    total_bandwidth: float
    average_latency: float
    network_diameter: int
    bisection_bandwidth: float
    connectivity: float

class TopologyBuilder:
    """Build network topologies from device configurations"""

    @staticmethod
    def build_ring_topology(devices: List[int],
                           interconnect: InterconnectType = InterconnectType.PCIE,
                           bandwidth: float = 10.0,
                           latency: float = 1.0) -> Dict[str, Any]:
        """Build ring topology"""
        links = []
        for i in range(len(devices)):
            src = devices[i]
            dst = devices[(i + 1) % len(devices)]
            links.append(LinkInfo(src, dst, interconnect, bandwidth, latency))

        return {
            'type': TopologyType.RING,
            'devices': devices,
            'links': links,
            'metrics': TopologyBuilder._calculate_ring_metrics(devices, bandwidth, latency)
        }

    @staticmethod
    def build_tree_topology(devices: List[int],
                           branching_factor: int = 2,
                           interconnect: InterconnectType = InterconnectType.PCIE,
                           bandwidth: float = 10.0,
                           latency: float = 1.0) -> Dict[str, Any]:
        """Build tree topology"""
        links = []
        for i in range(1, len(devices)):
            child = devices[i]
            parent = devices[(i - 1) // branching_factor]
            links.append(LinkInfo(parent, child, interconnect, bandwidth, latency))

        return {
            'type': TopologyType.TREE,
            'devices': devices,
            'links': links,
            'branching_factor': branching_factor,
            'metrics': TopologyBuilder._calculate_tree_metrics(devices, branching_factor, bandwidth, latency)
        }

    @staticmethod
    def build_hierarchical_topology(node_groups: List[List[int]],
                                  intra_interconnect: InterconnectType = InterconnectType.NVLINK,
                                  inter_interconnect: InterconnectType = InterconnectType.RDMA,
                                  intra_bandwidth: float = 50.0,
                                  inter_bandwidth: float = 10.0,
                                  intra_latency: float = 0.5,
                                  inter_latency: float = 1.0) -> Dict[str, Any]:
        """Build hierarchical topology"""
        all_devices = []
        links = []
        node_info = []

        for node_idx, node_devices in enumerate(node_groups):
            all_devices.extend(node_devices)
            node_info.append({
                'node_id': node_idx,
                'devices': node_devices
            })

            for i in range(len(node_devices)):
                for j in range(i + 1, len(node_devices)):
                    src = node_devices[i]
                    dst = node_devices[j]
                    links.append(LinkInfo(src, dst, intra_interconnect, intra_bandwidth, intra_latency))

        for i in range(len(node_groups)):
            for j in range(i + 1, len(node_groups)):
                if node_groups[i] and node_groups[j]:
                    src = node_groups[i][0]
                    dst = node_groups[j][0]
                    links.append(LinkInfo(src, dst, inter_interconnect, inter_bandwidth, inter_latency))

        return {
            'type': TopologyType.HIERARCHICAL,
            'devices': all_devices,
            'links': links,
            'node_groups': node_groups,
            'node_info': node_info,
            'metrics': TopologyBuilder._calculate_hierarchical_metrics(node_groups, intra_bandwidth, inter_bandwidth, intra_latency, inter_latency)
        }

    @staticmethod
    def build_mesh_topology(device_grid: List[List[int]],
                           interconnect: InterconnectType = InterconnectType.PCIE,
                           bandwidth: float = 10.0,
                           latency: float = 1.0) -> Dict[str, Any]:
        """Build 2D mesh topology"""
        all_devices = []
        links = []

        rows = len(device_grid)
        cols = len(device_grid[0]) if rows > 0 else 0

        for row in device_grid:
            all_devices.extend(row)

        for i in range(rows):
            for j in range(cols):
                current = device_grid[i][j]

                if j > 0:
                    left = device_grid[i][j-1]
                    links.append(LinkInfo(current, left, interconnect, bandwidth, latency))

                if i > 0:
                    up = device_grid[i-1][j]
                    links.append(LinkInfo(current, up, interconnect, bandwidth, latency))

        return {
            'type': TopologyType.MESH,
            'devices': all_devices,
            'links': links,
            'grid_shape': (rows, cols),
            'metrics': TopologyBuilder._calculate_mesh_metrics(rows, cols, bandwidth, latency)
        }

    @staticmethod
    def _calculate_ring_metrics(devices: List[int], bandwidth: float, latency: float) -> TopologyMetrics:
        """Calculate metrics for ring topology"""
        n = len(devices)
        total_bandwidth = n * bandwidth
        average_latency = latency
        network_diameter = n // 2
        bisection_bandwidth = 2 * bandwidth
        connectivity = 100.0 if n <= 2 else 95.0

        return TopologyMetrics(total_bandwidth, average_latency, network_diameter, bisection_bandwidth, connectivity)

    @staticmethod
    def _calculate_tree_metrics(devices: List[int], branching_factor: int, bandwidth: float, latency: float) -> TopologyMetrics:
        """Calculate metrics for tree topology"""
        n = len(devices)
        tree_depth = int(np.log(n) / np.log(branching_factor)) + 1
        total_bandwidth = (n - 1) * bandwidth
        average_latency = latency * tree_depth
        network_diameter = 2 * tree_depth
        bisection_bandwidth = bandwidth * branching_factor
        connectivity = 100.0

        return TopologyMetrics(total_bandwidth, average_latency, network_diameter, bisection_bandwidth, connectivity)

    @staticmethod
    def _calculate_hierarchical_topology_metrics(node_groups: List[List[int]],
                                                intra_bandwidth: float, inter_bandwidth: float,
                                                intra_latency: float, inter_latency: float) -> TopologyMetrics:
        """Calculate metrics for hierarchical topology"""
        total_devices = sum(len(group) for group in node_groups)
        total_intra_links = sum(len(group) * (len(group) - 1) // 2 for group in node_groups)
        total_inter_links = len(node_groups) * (len(node_groups) - 1) // 2

        total_bandwidth = total_intra_links * intra_bandwidth + total_inter_links * inter_bandwidth
        average_latency = (intra_latency + inter_latency) / 2
        network_diameter = 4
        bisection_bandwidth = min(len(node_groups) * inter_bandwidth, intra_bandwidth)
        connectivity = 98.0

        return TopologyMetrics(total_bandwidth, average_latency, network_diameter, bisection_bandwidth, connectivity)

    @staticmethod
    def _calculate_hierarchical_metrics(node_groups: List[List[int]],
                                      intra_bandwidth: float, inter_bandwidth: float,
                                      intra_latency: float, inter_latency: float) -> TopologyMetrics:
        """Calculate metrics for hierarchical topology"""
        return TopologyBuilder._calculate_hierarchical_topology_metrics(
            node_groups, intra_bandwidth, inter_bandwidth, intra_latency, inter_latency
        )

    @staticmethod
    def _calculate_mesh_metrics(rows: int, cols: int, bandwidth: float, latency: float) -> TopologyMetrics:
        """Calculate metrics for mesh topology"""
        total_devices = rows * cols
        horizontal_links = rows * (cols - 1)
        vertical_links = (rows - 1) * cols
        total_links = horizontal_links + vertical_links

        total_bandwidth = total_links * bandwidth
        average_latency = latency
        network_diameter = rows + cols - 2
        bisection_bandwidth = min(rows, cols) * bandwidth
        connectivity = 90.0 if rows > 1 and cols > 1 else 100.0

        return TopologyMetrics(total_bandwidth, average_latency, network_diameter, bisection_bandwidth, connectivity)

class TopologyDiscovery:
    """Automatic topology discovery and characterization"""

    def __init__(self):
        self.devices = []
        self.links = []

    def detect_gpu_interconnect(self) -> InterconnectType:
        """Detect GPU interconnect type"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    return InterconnectType.NVLINK
                else:
                    return InterconnectType.PCIE
        except ImportError:
            pass

        return InterconnectType.PCIE

    def detect_network_topology(self, num_nodes: int) -> TopologyType:
        """Detect network topology type"""
        if num_nodes <= 2:
            return TopologyType.FULLY_CONNECTED
        elif num_nodes <= 8:
            return TopologyType.RING
        elif num_nodes <= 16:
            return TopologyType.TREE
        else:
            return TopologyType.HIERARCHICAL

    def profile_bandwidth(self, device1: int, device2: int) -> float:
        """Profile bandwidth between two devices"""
        return 10.0

    def profile_latency(self, device1: int, device2: int) -> float:
        """Profile latency between two devices"""
        return 1.0

    def discover_devices(self) -> List[DeviceInfo]:
        """Discover available devices"""
        devices = []

        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    device_info = DeviceInfo(
                        device_id=i,
                        device_type="CUDA",
                        hostname="localhost",
                        memory_bandwidth=900.0,
                        compute_capability=8.0,
                        memory_size=16 * 1024 * 1024 * 1024
                    )
                    devices.append(device_info)
        except ImportError:
            pass

        cpu_info = DeviceInfo(
            device_id=-1,
            device_type="CPU",
            hostname="localhost",
            memory_bandwidth=50.0,
            compute_capability=1.0,
            memory_size=32 * 1024 * 1024 * 1024
        )
        devices.append(cpu_info)

        return devices

    def discover_topology(self, num_devices: Optional[int] = None) -> Dict[str, Any]:
        """Discover the system topology"""
        devices = self.discover_devices()
        if num_devices is None:
            num_devices = len(devices)

        device_ids = list(range(num_devices))

        interconnect_type = self.detect_gpu_interconnect()
        network_topology = self.detect_network_topology(max(1, num_devices // 4))

        if num_devices <= 2:
            return TopologyBuilder.build_ring_topology(device_ids, interconnect_type)
        elif network_topology == TopologyType.HIERARCHICAL:
            node_size = min(4, num_devices // 2)
            node_groups = []
            for i in range(0, num_devices, node_size):
                node_groups.append(list(range(i, min(i + node_size, num_devices))))
            return TopologyBuilder.build_hierarchical_topology(node_groups)
        elif network_topology == TopologyType.TREE:
            return TopologyBuilder.build_tree_topology(device_ids, branching_factor=2)
        else:
            return TopologyBuilder.build_ring_topology(device_ids, interconnect_type)

class TopologyOptimizer:
    """Optimize topology configurations for performance"""

    @staticmethod
    def optimize_for_allreduce(topology: Dict[str, Any], data_size: int, num_participants: int) -> Dict[str, Any]:
        """Optimize topology for allreduce operations"""
        optimized_topology = topology.copy()

        if data_size > 1024 * 1024 and num_participants > 4:
            if optimized_topology['type'] == TopologyType.RING:
                optimized_topology = TopologyBuilder.build_tree_topology(
                    topology['devices'], branching_factor=2,
                    bandwidth=topology['metrics'].total_bandwidth / len(topology['devices'])
                )

        return optimized_topology

    @staticmethod
    def optimize_for_bandwidth(topology: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize topology for bandwidth-intensive operations"""
        return topology

    @staticmethod
    def optimize_for_latency(topology: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize topology for latency-sensitive operations"""
        return topology