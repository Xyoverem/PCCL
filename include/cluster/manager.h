#pragma once

#include <map>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <base/registry.h>

namespace engine_c {

struct NodeMeta {
  int rank;
  std::map<std::string, std::string> endpoint_configs;
  std::string host_id;
  int port;

  NodeMeta(int r = 0, std::string host = "", int p = 0)
    : rank(r), host_id(host), port(p) {}

  std::string serialize() const;
  static NodeMeta deserialize(const std::string& data);
};

struct ConnectionInfo {
  int rank;
  std::string endpoint;
  int socket_fd;
  bool is_connected;

  ConnectionInfo(int r = 0, std::string ep = "")
    : rank(r), endpoint(ep), socket_fd(-1), is_connected(false) {}
};

class ClusterManager {
public:
  ClusterManager(const std::map<std::string, std::string>& config);
  ~ClusterManager();

  // Core cluster management
  std::string exportEndpoint();
  void joinCluster(const std::string& master_endpoint);
  void exitCluster();

  // Node management
  const NodeMeta& getLocalMeta();
  const std::map<int, NodeMeta>& getAllNodes();
  bool addOrUpdateNode(const NodeMeta& node_meta);

  // Operator management
  void registerOperator(const std::string& name, const std::string& config);
  void unregisterOperator(const std::string& name);
  std::map<std::string, std::string> getClusterInfo();

  // Communication
  bool connectToNode(int rank, const std::string& endpoint);
  void disconnectFromNode(int rank);
  bool isConnectedTo(int rank);
  std::vector<int> getConnectedNodes();

  // Event handling
  void registerConnectionCallback(std::function<void(int)> callback);
  void registerDisconnectionCallback(std::function<void(int)> callback);

private:
  std::map<int, NodeMeta> nodes_;
  std::map<int, ConnectionInfo> connections_;
  std::map<std::string, std::string> config_;
  std::map<std::string, std::string> operators_;
  std::vector<std::function<void(int)>> connection_callbacks_;
  std::vector<std::function<void(int)>> disconnection_callbacks_;

  int local_rank_;
  int world_size_;
  bool is_master_;
  int listener_socket_;
  std::string local_endpoint_;

  std::mutex cluster_mutex_;
  std::unique_ptr<std::thread> listener_thread_;
  std::atomic<bool> shutdown_;

  // Internal methods
  void startListener();
  void stopListener();
  void handleNewConnections();
  void handleNodeConnection(int socket_fd, const std::string& endpoint);
  void sendNodeInfo(int socket_fd);
  void receiveNodeInfo(int socket_fd);
  std::string generateEndpoint();
  bool validateEndpoint(const std::string& endpoint);
  void broadcastNodeUpdate(const NodeMeta& node);
  void processCallbacks();
};

}





