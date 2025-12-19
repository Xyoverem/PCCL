#include <cluster/manager.h>
#include <string>
#include <nlohmann/json.hpp>
#include <common.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <iostream>
#include <algorithm>

namespace engine_c {

std::string NodeMeta::serialize() const {
  nlohmann::json j;
  j["rank"] = rank;
  j["host_id"] = host_id;
  j["port"] = port;
  j["endpoint_configs"] = endpoint_configs;

  return j.dump();
}

NodeMeta NodeMeta::deserialize(const std::string& data) {
  nlohmann::json j = nlohmann::json::parse(data);

  NodeMeta node;
  node.rank = j["rank"];
  node.host_id = j["host_id"];
  node.port = j["port"];
  node.endpoint_configs = j["endpoint_configs"];

  return node;
}

ClusterManager::ClusterManager(const std::map<std::string, std::string>& config)
    : local_rank_(0), world_size_(1), is_master_(false),
      listener_socket_(-1), shutdown_(false) {

  config_ = config;

  // Extract local rank from config
  auto rank_it = config.find("rank");
  if (rank_it != config.end()) {
    local_rank_ = std::stoi(rank_it->second);
  }

  // Create local node metadata
  NodeMeta local_node;
  local_node.rank = local_rank_;
  local_node.host_id = config.at("host_id");
  local_node.port = 0; // Will be set when listener starts
  local_node.endpoint_configs = config;

  nodes_[local_rank_] = local_node;
}

ClusterManager::~ClusterManager() {
  exitCluster();
}

std::string ClusterManager::exportEndpoint() {
  std::lock_guard<std::mutex> lock(cluster_mutex_);

  if (local_endpoint_.empty()) {
    local_endpoint_ = generateEndpoint();
  }

  return local_endpoint_;
}

void ClusterManager::joinCluster(const std::string& master_endpoint) {
  std::lock_guard<std::mutex> lock(cluster_mutex_);

  if (is_master_) {
    throw std::runtime_error("Already acting as master node");
  }

  // Connect to master node
  if (!connectToNode(0, master_endpoint)) {
    throw std::runtime_error("Failed to connect to master node at: " + master_endpoint);
  }

  // Send local node info to master
  int master_socket = connections_[0].socket_fd;
  sendNodeInfo(master_socket);

  // Wait for full cluster information
  receiveNodeInfo(master_socket);

  std::cout << "Joined cluster as rank " << local_rank_ << std::endl;
}

void ClusterManager::exitCluster() {
  shutdown_ = true;

  // Stop listener
  stopListener();

  // Close all connections
  for (auto& [rank, conn] : connections_) {
    if (conn.socket_fd != -1) {
      close(conn.socket_fd);
    }
  }

  connections_.clear();
  nodes_.clear();
}

const NodeMeta& ClusterManager::getLocalMeta() {
  std::lock_guard<std::mutex> lock(cluster_mutex_);
  return nodes_.at(local_rank_);
}

const std::map<int, NodeMeta>& ClusterManager::getAllNodes() {
  std::lock_guard<std::mutex> lock(cluster_mutex_);
  return nodes_;
}

bool ClusterManager::addOrUpdateNode(const NodeMeta& node_meta) {
  std::lock_guard<std::mutex> lock(cluster_mutex_);

  bool is_new = nodes_.find(node_meta.rank) == nodes_.end();
  nodes_[node_meta.rank] = node_meta;

  // Broadcast update to all other nodes
  if (is_new && node_meta.rank != local_rank_) {
    broadcastNodeUpdate(node_meta);
  }

  return is_new;
}

void ClusterManager::registerOperator(const std::string& name, const std::string& config) {
  std::lock_guard<std::mutex> lock(cluster_mutex_);
  operators_[name] = config;
}

void ClusterManager::unregisterOperator(const std::string& name) {
  std::lock_guard<std::mutex> lock(cluster_mutex_);
  operators_.erase(name);
}

std::map<std::string, std::string> ClusterManager::getClusterInfo() {
  std::lock_guard<std::mutex> lock(cluster_mutex_);

  std::map<std::string, std::string> info;
  info["rank"] = std::to_string(local_rank_);
  info["world_size"] = std::to_string(nodes_.size());
  info["is_master"] = is_master_ ? "true" : "false";

  // Add operator information
  for (const auto& [name, config] : operators_) {
    info["operator_" + name] = config;
  }

  return info;
}

bool ClusterManager::connectToNode(int rank, const std::string& endpoint) {
  if (!validateEndpoint(endpoint)) {
    return false;
  }

  // Parse endpoint (host:port format)
  size_t colon_pos = endpoint.find(':');
  if (colon_pos == std::string::npos) {
    return false;
  }

  std::string host = endpoint.substr(0, colon_pos);
  int port = std::stoi(endpoint.substr(colon_pos + 1));

  // Create socket
  int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd == -1) {
    return false;
  }

  // Set socket options
  int opt = 1;
  setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  // Connect to remote node
  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);

  struct hostent* host_entry = gethostbyname(host.c_str());
  if (!host_entry) {
    close(socket_fd);
    return false;
  }

  memcpy(&addr.sin_addr, host_entry->h_addr, host_entry->h_length);

  if (connect(socket_fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
    close(socket_fd);
    return false;
  }

  // Store connection info
  ConnectionInfo conn_info(rank, endpoint);
  conn_info.socket_fd = socket_fd;
  conn_info.is_connected = true;
  connections_[rank] = conn_info;

  std::cout << "Connected to node " << rank << " at " << endpoint << std::endl;

  // Trigger connection callback
  for (const auto& callback : connection_callbacks_) {
    callback(rank);
  }

  return true;
}

void ClusterManager::disconnectFromNode(int rank) {
  std::lock_guard<std::mutex> lock(cluster_mutex_);

  auto it = connections_.find(rank);
  if (it != connections_.end()) {
    if (it->second.socket_fd != -1) {
      close(it->second.socket_fd);
    }

    it->second.is_connected = false;
    connections_.erase(it);

    // Trigger disconnection callback
    for (const auto& callback : disconnection_callbacks_) {
      callback(rank);
    }

    std::cout << "Disconnected from node " << rank << std::endl;
  }
}

bool ClusterManager::isConnectedTo(int rank) {
  std::lock_guard<std::mutex> lock(cluster_mutex_);

  auto it = connections_.find(rank);
  return it != connections_.end() && it->second.is_connected;
}

std::vector<int> ClusterManager::getConnectedNodes() {
  std::lock_guard<std::mutex> lock(cluster_mutex_);

  std::vector<int> connected_ranks;
  for (const auto& [rank, conn] : connections_) {
    if (conn.is_connected) {
      connected_ranks.push_back(rank);
    }
  }

  return connected_ranks;
}

void ClusterManager::registerConnectionCallback(std::function<void(int)> callback) {
  connection_callbacks_.push_back(callback);
}

void ClusterManager::registerDisconnectionCallback(std::function<void(int)> callback) {
  disconnection_callbacks_.push_back(callback);
}

// Private methods implementation
void ClusterManager::startListener() {
  if (listener_socket_ != -1) {
    return; // Already listening
  }

  listener_socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (listener_socket_ == -1) {
    throw std::runtime_error("Failed to create listener socket");
  }

  int opt = 1;
  setsockopt(listener_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = 0; // Let OS choose port

  if (bind(listener_socket_, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
    close(listener_socket_);
    throw std::runtime_error("Failed to bind listener socket");
  }

  if (listen(listener_socket_, SOMAXCONN) == -1) {
    close(listener_socket_);
    throw std::runtime_error("Failed to listen on socket");
  }

  // Get actual port
  struct sockaddr_in bound_addr;
  socklen_t len = sizeof(bound_addr);
  getsockname(listener_socket_, (struct sockaddr*)&bound_addr, &len);
  int port = ntohs(bound_addr.sin_port);

  // Update local node port
  nodes_[local_rank_].port = port;
  local_endpoint_ = "0.0.0.0:" + std::to_string(port);

  // Start listener thread
  shutdown_ = false;
  listener_thread_ = std::make_unique<std::thread>(&ClusterManager::handleNewConnections, this);

  std::cout << "Listening for connections on " << local_endpoint_ << std::endl;
}

void ClusterManager::stopListener() {
  if (listener_socket_ != -1) {
    shutdown_ = true;

    if (listener_thread_ && listener_thread_->joinable()) {
      listener_thread_->join();
    }

    close(listener_socket_);
    listener_socket_ = -1;
  }
}

void ClusterManager::handleNewConnections() {
  while (!shutdown_) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    int client_socket = accept(listener_socket_, (struct sockaddr*)&client_addr, &client_len);
    if (client_socket == -1) {
      if (shutdown_) break;
      continue;
    }

    // Get client IP and port
    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
    int client_port = ntohs(client_addr.sin_port);

    std::string endpoint = std::string(client_ip) + ":" + std::to_string(client_port);
    handleNodeConnection(client_socket, endpoint);
  }
}

void ClusterManager::handleNodeConnection(int socket_fd, const std::string& endpoint) {
  // Receive node info
  receiveNodeInfo(socket_fd);

  // Send cluster info back
  sendNodeInfo(socket_fd);

  std::cout << "Node connected from " << endpoint << std::endl;
}

void ClusterManager::sendNodeInfo(int socket_fd) {
  std::string node_info = getLocalMeta().serialize();
  uint32_t size = node_info.size();

  // Send size first
  send(socket_fd, &size, sizeof(size), 0);

  // Send data
  send(socket_fd, node_info.c_str(), size, 0);
}

void ClusterManager::receiveNodeInfo(int socket_fd) {
  // Receive size
  uint32_t size = 0;
  recv(socket_fd, &size, sizeof(size), 0);

  // Receive data
  std::vector<char> buffer(size + 1);
  recv(socket_fd, buffer.data(), size, 0);
  buffer[size] = '\0';

  // Parse node info
  NodeMeta node = NodeMeta::deserialize(std::string(buffer.data()));

  // Add node to cluster
  addOrUpdateNode(node);
}

std::string ClusterManager::generateEndpoint() {
  return "127.0.0.1:0"; // Will be replaced by actual bound port
}

bool ClusterManager::validateEndpoint(const std::string& endpoint) {
  // Basic validation: host:port format
  size_t colon_pos = endpoint.find(':');
  if (colon_pos == std::string::npos || colon_pos == endpoint.length() - 1) {
    return false;
  }

  std::string port_str = endpoint.substr(colon_pos + 1);
  try {
    int port = std::stoi(port_str);
    return port > 0 && port <= 65535;
  } catch (...) {
    return false;
  }
}

void ClusterManager::broadcastNodeUpdate(const NodeMeta& node) {
  std::string update_data = node.serialize();
  uint32_t size = update_data.size();

  for (auto& [rank, conn] : connections_) {
    if (conn.is_connected && rank != node.rank) {
      send(conn.socket_fd, &size, sizeof(size), 0);
      send(conn.socket_fd, update_data.c_str(), size, 0);
    }
  }
}

}

