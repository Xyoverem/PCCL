#pragma once

#include <base/registry.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <unordered_set>

namespace engine_c {

enum class InterconnectType {
  NVLINK,
  PCIE,
  RDMA,
  ETHERNET,
  INFINIBAND,
  UNKNOWN
};

enum class TopologyType {
  RING,
  TREE,
  MESH,
  TORUS,
  FAT_TREE,
  HIERARCHICAL,
  FULLY_CONNECTED
};

struct DeviceInfo {
  int device_id;
  DeviceType device_type;
  std::string device_name;
  float memory_bandwidth;
  float compute_capability;
  int memory_size;
  int numa_node;
};

struct LinkInfo {
  int src_device;
  int dst_device;
  InterconnectType interconnect_type;
  float bandwidth;
  float latency;
  bool bidirectional;
};

struct TopologyNode {
  int node_id;
  std::vector<int> device_ids;
  std::string node_name;
  std::unordered_map<int, LinkInfo> internal_links;
  std::unordered_map<int, LinkInfo> external_links;
};

struct TopologyMetrics {
  float total_bandwidth;
  float average_latency;
  int network_diameter;
  float bisection_bandwidth;
  int connectivity;
};

class Topology {
public:
  Topology();
  ~Topology() = default;

  void addDevice(const DeviceInfo& device);
  void addLink(const LinkInfo& link);
  void addNode(const TopologyNode& node);

  const DeviceInfo* getDevice(int device_id) const;
  const LinkInfo* getLink(int src_device, int dst_device) const;
  const TopologyNode* getNode(int node_id) const;

  const std::unordered_map<int, DeviceInfo>& getDevices() const { return devices_; }
  const std::unordered_map<std::pair<int, int>, LinkInfo>& getLinks() const { return links_; }
  const std::unordered_map<int, TopologyNode>& getNodes() const { return nodes_; }

  TopologyMetrics calculateMetrics() const;
  std::vector<std::vector<int>> getShortestPaths() const;
  std::vector<int> getShortestPath(int src, int dst) const;

  bool isValid() const;
  bool isFullyConnected() const;

  void printTopology() const;

private:
  std::unordered_map<int, DeviceInfo> devices_;
  std::unordered_map<std::pair<int, int>, LinkInfo> links_;
  std::unordered_map<int, TopologyNode> nodes_;

  std::vector<int> dijkstra(int src, const std::unordered_map<int, std::vector<std::pair<int, float>>>& graph) const;
  std::unordered_map<int, std::vector<std::pair<int, float>>> buildAdjacencyList() const;
};

class TopologyBuilder {
public:
  static std::unique_ptr<Topology> buildRingTopology(const std::vector<int>& device_ids,
                                                     InterconnectType interconnect = InterconnectType::PCIE,
                                                     float bandwidth = 10.0f,
                                                     float latency = 1.0f);

  static std::unique_ptr<Topology> buildTreeTopology(const std::vector<int>& device_ids,
                                                    int branching_factor = 2,
                                                    InterconnectType interconnect = InterconnectType::PCIE,
                                                    float bandwidth = 10.0f,
                                                    float latency = 1.0f);

  static std::unique_ptr<Topology> buildMeshTopology(const std::vector<std::vector<int>>& device_grid,
                                                    InterconnectType interconnect = InterconnectType::PCIE,
                                                    float bandwidth = 10.0f,
                                                    float latency = 1.0f);

  static std::unique_ptr<Topology> buildHierarchicalTopology(const std::vector<std::vector<int>>& node_groups,
                                                           InterconnectType intra_interconnect = InterconnectType::NVLINK,
                                                           InterconnectType inter_interconnect = InterconnectType::RDMA,
                                                           float intra_bandwidth = 50.0f,
                                                           float inter_bandwidth = 10.0f,
                                                           float intra_latency = 0.5f,
                                                           float inter_latency = 1.0f);

  static std::unique_ptr<Topology> buildFullyConnectedTopology(const std::vector<int>& device_ids,
                                                              InterconnectType interconnect = InterconnectType::PCIE,
                                                              float bandwidth = 10.0f,
                                                              float latency = 1.0f);
};

class TopologyManager {
public:
  TopologyManager();
  ~TopologyManager() = default;

  void discoverTopology();
  void loadTopology(const std::string& filepath);
  void saveTopology(const std::string& filepath) const;

  void setActiveTopology(std::shared_ptr<Topology> topology);
  std::shared_ptr<Topology> getActiveTopology() const { return active_topology_; }

  std::shared_ptr<Topology> getTopology(const std::string& name) const;
  void addTopology(const std::string& name, std::shared_ptr<Topology> topology);

  std::vector<std::string> getAvailableTopologies() const;

  TopologyType detectOptimalTopologyType(const std::vector<int>& device_ids) const;
  std::shared_ptr<Topology> buildOptimalTopology(const std::vector<int>& device_ids) const;

  bool isDeviceAvailable(int device_id) const;
  std::vector<int> getAvailableDevices() const;

  void updateDeviceMetrics(int device_id, float bandwidth, float compute_capability);

private:
  std::shared_ptr<Topology> active_topology_;
  std::unordered_map<std::string, std::shared_ptr<Topology>> topologies_;
  std::unordered_map<int, DeviceInfo> available_devices_;

  void discoverCUDADevices();
  void discoverRDMADevices();
  void discoverCPUDevices();
  void detectDeviceConnections();

  InterconnectType detectInterconnectType(int src_device, int dst_device) const;
  float measureBandwidth(int src_device, int dst_device) const;
  float measureLatency(int src_device, int dst_device) const;
};

}