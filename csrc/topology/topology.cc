#include <topology/topology.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <queue>
#include <limits>
#include <cmath>

namespace engine_c {

Topology::Topology() {}

void Topology::addDevice(const DeviceInfo& device) {
  devices_[device.device_id] = device;
}

void Topology::addLink(const LinkInfo& link) {
  auto key = std::make_pair(link.src_device, link.dst_device);
  links_[key] = link;

  if (link.bidirectional) {
    auto reverse_key = std::make_pair(link.dst_device, link.src_device);
    LinkInfo reverse_link = link;
    reverse_link.src_device = link.dst_device;
    reverse_link.dst_device = link.src_device;
    links_[reverse_key] = reverse_link;
  }
}

void Topology::addNode(const TopologyNode& node) {
  nodes_[node.node_id] = node;
}

const DeviceInfo* Topology::getDevice(int device_id) const {
  auto it = devices_.find(device_id);
  return (it != devices_.end()) ? &it->second : nullptr;
}

const LinkInfo* Topology::getLink(int src_device, int dst_device) const {
  auto it = links_.find(std::make_pair(src_device, dst_device));
  return (it != links_.end()) ? &it->second : nullptr;
}

const TopologyNode* Topology::getNode(int node_id) const {
  auto it = nodes_.find(node_id);
  return (it != nodes_.end()) ? &it->second : nullptr;
}

TopologyMetrics Topology::calculateMetrics() const {
  TopologyMetrics metrics = {};

  float total_bandwidth = 0.0f;
  float total_latency = 0.0f;
  int link_count = 0;

  for (const auto& pair : links_) {
    const LinkInfo& link = pair.second;
    total_bandwidth += link.bandwidth;
    total_latency += link.latency;
    link_count++;
  }

  metrics.total_bandwidth = total_bandwidth;
  metrics.average_latency = (link_count > 0) ? (total_latency / link_count) : 0.0f;

  auto shortest_paths = getShortestPaths();
  if (!shortest_paths.empty()) {
    int max_distance = 0;
    float total_pairs = 0;
    float connected_pairs = 0;

    for (size_t i = 0; i < shortest_paths.size(); ++i) {
      for (size_t j = 0; j < shortest_paths[i].size(); ++j) {
        if (i != j && shortest_paths[i][j] != std::numeric_limits<int>::max()) {
          max_distance = std::max(max_distance, shortest_paths[i][j]);
          connected_pairs++;
        }
        total_pairs++;
      }
    }

    metrics.network_diameter = max_distance;
    metrics.connectivity = static_cast<int>((connected_pairs / total_pairs) * 100);
  } else {
    metrics.network_diameter = 0;
    metrics.connectivity = 0;
  }

  int num_devices = devices_.size();
  if (num_devices >= 2) {
    int half_devices = num_devices / 2;
    std::vector<int> first_half, second_half;

    int count = 0;
    for (const auto& pair : devices_) {
      if (count < half_devices) {
        first_half.push_back(pair.first);
      } else {
        second_half.push_back(pair.first);
      }
      count++;
    }

    float min_cut_bandwidth = std::numeric_limits<float>::max();
    for (int src : first_half) {
      for (int dst : second_half) {
        const LinkInfo* link = getLink(src, dst);
        if (link) {
          min_cut_bandwidth = std::min(min_cut_bandwidth, link->bandwidth);
        }
      }
    }

    metrics.bisection_bandwidth = (min_cut_bandwidth != std::numeric_limits<float>::max()) ?
                                  min_cut_bandwidth : 0.0f;
  } else {
    metrics.bisection_bandwidth = 0.0f;
  }

  return metrics;
}

std::vector<std::vector<int>> Topology::getShortestPaths() const {
  std::vector<std::vector<int>> distances;
  auto graph = buildAdjacencyList();

  std::vector<int> device_ids;
  for (const auto& pair : devices_) {
    device_ids.push_back(pair.first);
  }

  for (int src : device_ids) {
    auto path_distances = dijkstra(src, graph);
    distances.push_back(path_distances);
  }

  return distances;
}

std::vector<int> Topology::getShortestPath(int src, int dst) const {
  auto graph = buildAdjacencyList();
  auto distances = dijkstra(src, graph);
  return distances;
}

bool Topology::isValid() const {
  if (devices_.empty()) return false;

  for (const auto& pair : links_) {
    const LinkInfo& link = pair.second;
    if (devices_.find(link.src_device) == devices_.end()) return false;
    if (devices_.find(link.dst_device) == devices_.end()) return false;
  }

  return true;
}

bool Topology::isFullyConnected() const {
  auto shortest_paths = getShortestPaths();
  if (shortest_paths.empty()) return false;

  for (size_t i = 0; i < shortest_paths.size(); ++i) {
    for (size_t j = 0; j < shortest_paths[i].size(); ++j) {
      if (i != j && shortest_paths[i][j] == std::numeric_limits<int>::max()) {
        return false;
      }
    }
  }

  return true;
}

void Topology::printTopology() const {
  std::cout << "Topology Information:\n";
  std::cout << "Devices (" << devices_.size() << "):\n";
  for (const auto& pair : devices_) {
    const DeviceInfo& device = pair.second;
    std::cout << "  Device " << device.device_id
              << " (" << device.device_name << ")\n";
  }

  std::cout << "Links (" << links_.size() << "):\n";
  for (const auto& pair : links_) {
    const LinkInfo& link = pair.second;
    std::cout << "  " << link.src_device << " -> " << link.dst_device
              << " (Bandwidth: " << link.bandwidth << " GB/s, "
              << "Latency: " << link.latency << " μs)\n";
  }

  auto metrics = calculateMetrics();
  std::cout << "Metrics:\n";
  std::cout << "  Total Bandwidth: " << metrics.total_bandwidth << " GB/s\n";
  std::cout << "  Average Latency: " << metrics.average_latency << " μs\n";
  std::cout << "  Network Diameter: " << metrics.network_diameter << "\n";
  std::cout << "  Bisection Bandwidth: " << metrics.bisection_bandwidth << " GB/s\n";
  std::cout << "  Connectivity: " << metrics.connectivity << "%\n";
}

std::vector<int> Topology::dijkstra(int src, const std::unordered_map<int, std::vector<std::pair<int, float>>>& graph) const {
  std::unordered_map<int, float> distances;
  std::unordered_set<int> visited;
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> pq;

  std::vector<int> device_ids;
  for (const auto& pair : devices_) {
    device_ids.push_back(pair.first);
    distances[pair.first] = std::numeric_limits<float>::max();
  }

  distances[src] = 0.0f;
  pq.push({0.0f, src});

  while (!pq.empty()) {
    int current = pq.top().second;
    pq.pop();

    if (visited.find(current) != visited.end()) continue;
    visited.insert(current);

    auto it = graph.find(current);
    if (it == graph.end()) continue;

    for (const auto& neighbor_pair : it->second) {
      int neighbor = neighbor_pair.first;
      float weight = neighbor_pair.second;

      if (distances[current] + weight < distances[neighbor]) {
        distances[neighbor] = distances[current] + weight;
        pq.push({distances[neighbor], neighbor});
      }
    }
  }

  std::vector<int> result;
  for (int device_id : device_ids) {
    if (distances[device_id] == std::numeric_limits<float>::max()) {
      result.push_back(std::numeric_limits<int>::max());
    } else {
      result.push_back(static_cast<int>(distances[device_id]));
    }
  }

  return result;
}

std::unordered_map<int, std::vector<std::pair<int, float>>> Topology::buildAdjacencyList() const {
  std::unordered_map<int, std::vector<std::pair<int, float>>> graph;

  for (const auto& pair : devices_) {
    graph[pair.first] = {};
  }

  for (const auto& pair : links_) {
    const LinkInfo& link = pair.second;
    graph[link.src_device].push_back({link.dst_device, link.latency});
  }

  return graph;
}

std::unique_ptr<Topology> TopologyBuilder::buildRingTopology(const std::vector<int>& device_ids,
                                                           InterconnectType interconnect,
                                                           float bandwidth,
                                                           float latency) {
  auto topology = std::make_unique<Topology>();

  for (int device_id : device_ids) {
    DeviceInfo device;
    device.device_id = device_id;
    device.device_type = DeviceType::CPU;
    device.device_name = "Device_" + std::to_string(device_id);
    device.memory_bandwidth = bandwidth;
    device.compute_capability = 1.0f;
    device.memory_size = 1024 * 1024 * 1024;
    device.numa_node = 0;

    topology->addDevice(device);
  }

  for (size_t i = 0; i < device_ids.size(); ++i) {
    int src = device_ids[i];
    int dst = device_ids[(i + 1) % device_ids.size()];

    LinkInfo link;
    link.src_device = src;
    link.dst_device = dst;
    link.interconnect_type = interconnect;
    link.bandwidth = bandwidth;
    link.latency = latency;
    link.bidirectional = true;

    topology->addLink(link);
  }

  return topology;
}

std::unique_ptr<Topology> TopologyBuilder::buildTreeTopology(const std::vector<int>& device_ids,
                                                           int branching_factor,
                                                           InterconnectType interconnect,
                                                           float bandwidth,
                                                           float latency) {
  auto topology = std::make_unique<Topology>();

  for (int device_id : device_ids) {
    DeviceInfo device;
    device.device_id = device_id;
    device.device_type = DeviceType::CPU;
    device.device_name = "Device_" + std::to_string(device_id);
    device.memory_bandwidth = bandwidth;
    device.compute_capability = 1.0f;
    device.memory_size = 1024 * 1024 * 1024;
    device.numa_node = 0;

    topology->addDevice(device);
  }

  for (size_t i = 1; i < device_ids.size(); ++i) {
    int child = device_ids[i];
    int parent = device_ids[(i - 1) / branching_factor];

    LinkInfo link;
    link.src_device = parent;
    link.dst_device = child;
    link.interconnect_type = interconnect;
    link.bandwidth = bandwidth;
    link.latency = latency;
    link.bidirectional = true;

    topology->addLink(link);
  }

  return topology;
}

std::unique_ptr<Topology> TopologyBuilder::buildMeshTopology(const std::vector<std::vector<int>>& device_grid,
                                                           InterconnectType interconnect,
                                                           float bandwidth,
                                                           float latency) {
  auto topology = std::make_unique<Topology>();

  for (const auto& row : device_grid) {
    for (int device_id : row) {
      DeviceInfo device;
      device.device_id = device_id;
      device.device_type = DeviceType::CPU;
      device.device_name = "Device_" + std::to_string(device_id);
      device.memory_bandwidth = bandwidth;
      device.compute_capability = 1.0f;
      device.memory_size = 1024 * 1024 * 1024;
      device.numa_node = 0;

      topology->addDevice(device);
    }
  }

  int rows = device_grid.size();
  if (rows == 0) return topology;

  int cols = device_grid[0].size();

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int current = device_grid[i][j];

      if (j > 0) {
        int left = device_grid[i][j - 1];
        LinkInfo link;
        link.src_device = current;
        link.dst_device = left;
        link.interconnect_type = interconnect;
        link.bandwidth = bandwidth;
        link.latency = latency;
        link.bidirectional = true;
        topology->addLink(link);
      }

      if (i > 0) {
        int up = device_grid[i - 1][j];
        LinkInfo link;
        link.src_device = current;
        link.dst_device = up;
        link.interconnect_type = interconnect;
        link.bandwidth = bandwidth;
        link.latency = latency;
        link.bidirectional = true;
        topology->addLink(link);
      }
    }
  }

  return topology;
}

std::unique_ptr<Topology> TopologyBuilder::buildHierarchicalTopology(const std::vector<std::vector<int>>& node_groups,
                                                                   InterconnectType intra_interconnect,
                                                                   InterconnectType inter_interconnect,
                                                                   float intra_bandwidth,
                                                                   float inter_bandwidth,
                                                                   float intra_latency,
                                                                   float inter_latency) {
  auto topology = std::make_unique<Topology>();

  for (size_t node_idx = 0; node_idx < node_groups.size(); ++node_idx) {
    TopologyNode node;
    node.node_id = static_cast<int>(node_idx);
    node.node_name = "Node_" + std::to_string(node_idx);
    node.device_ids = node_groups[node_idx];

    for (int device_id : node_groups[node_idx]) {
      DeviceInfo device;
      device.device_id = device_id;
      device.device_type = DeviceType::CPU;
      device.device_name = "Device_" + std::to_string(device_id);
      device.memory_bandwidth = intra_bandwidth;
      device.compute_capability = 1.0f;
      device.memory_size = 1024 * 1024 * 1024;
      device.numa_node = node_idx;

      topology->addDevice(device);

      for (int other_device : node_groups[node_idx]) {
        if (device_id < other_device) {
          LinkInfo intra_link;
          intra_link.src_device = device_id;
          intra_link.dst_device = other_device;
          intra_link.interconnect_type = intra_interconnect;
          intra_link.bandwidth = intra_bandwidth;
          intra_link.latency = intra_latency;
          intra_link.bidirectional = true;
          topology->addLink(intra_link);

          node.internal_links[device_id] = intra_link;
        }
      }
    }

    topology->addNode(node);
  }

  for (size_t i = 0; i < node_groups.size(); ++i) {
    for (size_t j = i + 1; j < node_groups.size(); ++j) {
      if (!node_groups[i].empty() && !node_groups[j].empty()) {
        LinkInfo inter_link;
        inter_link.src_device = node_groups[i][0];
        inter_link.dst_device = node_groups[j][0];
        inter_link.interconnect_type = inter_interconnect;
        inter_link.bandwidth = inter_bandwidth;
        inter_link.latency = inter_latency;
        inter_link.bidirectional = true;
        topology->addLink(inter_link);
      }
    }
  }

  return topology;
}

std::unique_ptr<Topology> TopologyBuilder::buildFullyConnectedTopology(const std::vector<int>& device_ids,
                                                                      InterconnectType interconnect,
                                                                      float bandwidth,
                                                                      float latency) {
  auto topology = std::make_unique<Topology>();

  for (int device_id : device_ids) {
    DeviceInfo device;
    device.device_id = device_id;
    device.device_type = DeviceType::CPU;
    device.device_name = "Device_" + std::to_string(device_id);
    device.memory_bandwidth = bandwidth;
    device.compute_capability = 1.0f;
    device.memory_size = 1024 * 1024 * 1024;
    device.numa_node = 0;

    topology->addDevice(device);
  }

  for (size_t i = 0; i < device_ids.size(); ++i) {
    for (size_t j = i + 1; j < device_ids.size(); ++j) {
      LinkInfo link;
      link.src_device = device_ids[i];
      link.dst_device = device_ids[j];
      link.interconnect_type = interconnect;
      link.bandwidth = bandwidth;
      link.latency = latency;
      link.bidirectional = true;

      topology->addLink(link);
    }
  }

  return topology;
}

TopologyManager::TopologyManager() {}

void TopologyManager::discoverTopology() {
  discoverCUDADevices();
  discoverRDMADevices();
  discoverCPUDevices();
  detectDeviceConnections();

  if (!available_devices_.empty()) {
    std::vector<int> device_ids;
    for (const auto& pair : available_devices_) {
      device_ids.push_back(pair.first);
    }

    auto topology = TopologyBuilder::buildRingTopology(device_ids);
    setActiveTopology(topology);
    addTopology("discovered", topology);
  }
}

void TopologyManager::loadTopology(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) return;

  auto topology = std::make_unique<Topology>();

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "DEVICE") {
      DeviceInfo device;
      iss >> device.device_id >> device.device_name >> device.memory_bandwidth
         >> device.compute_capability >> device.memory_size >> device.numa_node;
      topology->addDevice(device);
      available_devices_[device.device_id] = device;
    } else if (type == "LINK") {
      LinkInfo link;
      std::string interconnect_str;
      iss >> link.src_device >> link.dst_device >> interconnect_str
         >> link.bandwidth >> link.latency;

      if (interconnect_str == "NVLINK") {
        link.interconnect_type = InterconnectType::NVLINK;
      } else if (interconnect_str == "PCIE") {
        link.interconnect_type = InterconnectType::PCIE;
      } else if (interconnect_str == "RDMA") {
        link.interconnect_type = InterconnectType::RDMA;
      } else if (interconnect_str == "ETHERNET") {
        link.interconnect_type = InterconnectType::ETHERNET;
      } else if (interconnect_str == "INFINIBAND") {
        link.interconnect_type = InterconnectType::INFINIBAND;
      } else {
        link.interconnect_type = InterconnectType::UNKNOWN;
      }

      link.bidirectional = true;
      topology->addLink(link);
    }
  }

  setActiveTopology(topology);
  addTopology("loaded", topology);
}

void TopologyManager::saveTopology(const std::string& filepath) const {
  if (!active_topology_) return;

  std::ofstream file(filepath);
  if (!file.is_open()) return;

  file << "# PCCL Topology File\n";

  for (const auto& pair : active_topology_->getDevices()) {
    const DeviceInfo& device = pair.second;
    file << "DEVICE " << device.device_id << " " << device.device_name << " "
         << device.memory_bandwidth << " " << device.compute_capability << " "
         << device.memory_size << " " << device.numa_node << "\n";
  }

  for (const auto& pair : active_topology_->getLinks()) {
    const LinkInfo& link = pair.second;
    std::string interconnect_str = "UNKNOWN";
    switch (link.interconnect_type) {
      case InterconnectType::NVLINK: interconnect_str = "NVLINK"; break;
      case InterconnectType::PCIE: interconnect_str = "PCIE"; break;
      case InterconnectType::RDMA: interconnect_str = "RDMA"; break;
      case InterconnectType::ETHERNET: interconnect_str = "ETHERNET"; break;
      case InterconnectType::INFINIBAND: interconnect_str = "INFINIBAND"; break;
      default: break;
    }

    if (link.src_device < link.dst_device) {
      file << "LINK " << link.src_device << " " << link.dst_device << " "
           << interconnect_str << " " << link.bandwidth << " " << link.latency << "\n";
    }
  }
}

void TopologyManager::setActiveTopology(std::shared_ptr<Topology> topology) {
  active_topology_ = topology;
}

std::shared_ptr<Topology> TopologyManager::getTopology(const std::string& name) const {
  auto it = topologies_.find(name);
  return (it != topologies_.end()) ? it->second : nullptr;
}

void TopologyManager::addTopology(const std::string& name, std::shared_ptr<Topology> topology) {
  topologies_[name] = topology;
}

std::vector<std::string> TopologyManager::getAvailableTopologies() const {
  std::vector<std::string> names;
  for (const auto& pair : topologies_) {
    names.push_back(pair.first);
  }
  return names;
}

TopologyType TopologyManager::detectOptimalTopologyType(const std::vector<int>& device_ids) const {
  size_t num_devices = device_ids.size();

  if (num_devices <= 2) {
    return TopologyType::FULLY_CONNECTED;
  } else if (num_devices <= 8) {
    return TopologyType::RING;
  } else if (num_devices <= 16) {
    return TopologyType::TREE;
  } else {
    return TopologyType::HIERARCHICAL;
  }
}

std::shared_ptr<Topology> TopologyManager::buildOptimalTopology(const std::vector<int>& device_ids) const {
  TopologyType optimal_type = detectOptimalTopologyType(device_ids);

  switch (optimal_type) {
    case TopologyType::RING:
      return TopologyBuilder::buildRingTopology(device_ids);
    case TopologyType::TREE:
      return TopologyBuilder::buildTreeTopology(device_ids);
    case TopologyType::HIERARCHICAL: {
      size_t node_size = std::min<size_t>(4, device_ids.size() / 2);
      std::vector<std::vector<int>> node_groups;
      for (size_t i = 0; i < device_ids.size(); i += node_size) {
        std::vector<int> group(device_ids.begin() + i,
                              device_ids.begin() + std::min(i + node_size, device_ids.size()));
        node_groups.push_back(group);
      }
      return TopologyBuilder::buildHierarchicalTopology(node_groups);
    }
    case TopologyType::FULLY_CONNECTED:
      return TopologyBuilder::buildFullyConnectedTopology(device_ids);
    default:
      return TopologyBuilder::buildRingTopology(device_ids);
  }
}

bool TopologyManager::isDeviceAvailable(int device_id) const {
  return available_devices_.find(device_id) != available_devices_.end();
}

std::vector<int> TopologyManager::getAvailableDevices() const {
  std::vector<int> device_ids;
  for (const auto& pair : available_devices_) {
    device_ids.push_back(pair.first);
  }
  return device_ids;
}

void TopologyManager::updateDeviceMetrics(int device_id, float bandwidth, float compute_capability) {
  auto it = available_devices_.find(device_id);
  if (it != available_devices_.end()) {
    it->second.memory_bandwidth = bandwidth;
    it->second.compute_capability = compute_capability;
  }
}

void TopologyManager::discoverCUDADevices() {
}

void TopologyManager::discoverRDMADevices() {
}

void TopologyManager::discoverCPUDevices() {
  int num_cpus = 4;
  for (int i = 0; i < num_cpus; ++i) {
    DeviceInfo device;
    device.device_id = i;
    device.device_type = DeviceType::CPU;
    device.device_name = "CPU_" + std::to_string(i);
    device.memory_bandwidth = 50.0f;
    device.compute_capability = 1.0f;
    device.memory_size = 16 * 1024 * 1024 * 1024;
    device.numa_node = i % 2;

    available_devices_[i] = device;
  }
}

void TopologyManager::detectDeviceConnections() {
}

InterconnectType TopologyManager::detectInterconnectType(int src_device, int dst_device) const {
  return InterconnectType::PCIE;
}

float TopologyManager::measureBandwidth(int src_device, int dst_device) const {
  return 10.0f;
}

float TopologyManager::measureLatency(int src_device, int dst_device) const {
  return 1.0f;
}

}