#include <network/network.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <chrono>
#include <thread>

namespace engine_c {
namespace network {

TcpConnection::TcpConnection() : socket_fd_(-1), status_(ConnectionStatus::DISCONNECTED) {}

TcpConnection::~TcpConnection() {
  disconnect();
}

bool TcpConnection::connect(const NetworkAddress& address) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (status_ != ConnectionStatus::DISCONNECTED) {
    return false;
  }

  socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd_ < 0) {
    return false;
  }

  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(address.port);

  if (inet_pton(AF_INET, address.ip.c_str(), &server_addr.sin_addr) <= 0) {
    close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  if (::connect(socket_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  int flags = fcntl(socket_fd_, F_GETFL, 0);
  fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);

  status_ = ConnectionStatus::CONNECTED;
  remote_addr_ = address;

  struct sockaddr_in local_addr;
  socklen_t addr_len = sizeof(local_addr);
  getsockname(socket_fd_, (struct sockaddr*)&local_addr, &addr_len);
  local_addr_ = NetworkAddress(inet_ntoa(local_addr.sin_addr), ntohs(local_addr.sin_port));

  return true;
}

void TcpConnection::disconnect() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (socket_fd_ >= 0) {
    close(socket_fd_);
    socket_fd_ = -1;
  }

  status_ = ConnectionStatus::DISCONNECTED;
  pending_operations_.clear();
}

ConnectionStatus TcpConnection::getStatus() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_;
}

bool TcpConnection::sendMessage(const MessageHeader& header, const void* data) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (status_ != ConnectionStatus::CONNECTED || socket_fd_ < 0) {
    return false;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  ssize_t bytes_sent = send(socket_fd_, &header, sizeof(header), MSG_NOSIGNAL);
  if (bytes_sent != sizeof(header)) {
    status_ = ConnectionStatus::ERROR;
    return false;
  }

  if (header.data_size > 0 && data) {
    bytes_sent = send(socket_fd_, data, header.data_size, MSG_NOSIGNAL);
    if (bytes_sent != static_cast<ssize_t>(header.data_size)) {
      status_ = ConnectionStatus::ERROR;
      return false;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  auto& manager = NetworkManager::getInstance();
  total_latency_ += duration.count() / 1000.0;
  message_count_++;

  return true;
}

bool TcpConnection::receiveMessage(MessageHeader& header, void* data, size_t max_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (status_ != ConnectionStatus::CONNECTED || socket_fd_ < 0) {
    return false;
  }

  ssize_t bytes_received = recv(socket_fd_, &header, sizeof(header), MSG_WAITALL);
  if (bytes_received != sizeof(header)) {
    if (bytes_received == 0) {
      status_ = ConnectionStatus::DISCONNECTED;
    } else {
      status_ = ConnectionStatus::ERROR;
    }
    return false;
  }

  if (header.data_size > 0 && data) {
    size_t bytes_to_receive = std::min(static_cast<size_t>(header.data_size), max_size);
    bytes_received = recv(socket_fd_, data, bytes_to_receive, MSG_WAITALL);
    if (bytes_received != static_cast<ssize_t>(bytes_to_receive)) {
      status_ = ConnectionStatus::ERROR;
      return false;
    }
  }

  return true;
}

bool TcpConnection::sendAsync(const MessageHeader& header, const void* data) {
  uint32_t message_id = next_message_id_++;

  std::lock_guard<std::mutex> lock(mutex_);
  pending_operations_[message_id] = true;

  return sendMessage(header, data);
}

bool TcpConnection::recvAsync(MessageHeader& header, void* data, size_t max_size) {
  uint32_t message_id = next_message_id_++;

  std::lock_guard<std::mutex> lock(mutex_);
  pending_operations_[message_id] = true;

  return receiveMessage(header, data, max_size);
}

bool TcpConnection::pollCompletion(std::vector<uint32_t>& completed_ids) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& op : pending_operations_) {
    if (op.second) {
      completed_ids.push_back(op.first);
    }
  }

  pending_operations_.clear();
  return !completed_ids.empty();
}

std::string TcpConnection::getLocalAddress() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return local_addr_.ip + ":" + std::to_string(local_addr_.port);
}

NetworkAddress TcpConnection::getRemoteAddress() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return remote_addr_;
}

NetworkManager& NetworkManager::getInstance() {
  static NetworkManager instance;
  return instance;
}

NetworkManager::~NetworkManager() {
  shutdown();
}

bool NetworkManager::initialize(NetworkType type) {
  if (initialized_) {
    return true;
  }

  network_type_ = type;

  if (type == NetworkType::TCP_SOCKET) {
    initialized_ = true;
    return true;
  } else if (type == NetworkType::RDMA_VERBS) {
    return false;
  }

  return false;
}

void NetworkManager::shutdown() {
  if (!initialized_) {
    return;
  }

  stopListening();

  std::lock_guard<std::mutex> lock(connections_mutex_);
  connections_.clear();
  initialized_ = false;
}

ConnectionPtr NetworkManager::createConnection() {
  if (!initialized_) {
    return nullptr;
  }

  ConnectionPtr conn;

  if (network_type_ == NetworkType::TCP_SOCKET) {
    conn = std::make_shared<TcpConnection>();
  }

  if (conn) {
    int conn_id = next_connection_id_++;
    std::lock_guard<std::mutex> lock(connections_mutex_);
    connections_[conn_id] = conn;
  }

  return conn;
}

bool NetworkManager::removeConnection(int connection_id) {
  std::lock_guard<std::mutex> lock(connections_mutex_);
  auto it = connections_.find(connection_id);
  if (it != connections_.end()) {
    it->second->disconnect();
    connections_.erase(it);
    return true;
  }
  return false;
}

ConnectionPtr NetworkManager::getConnection(int connection_id) {
  std::lock_guard<std::mutex> lock(connections_mutex_);
  auto it = connections_.find(connection_id);
  return (it != connections_.end()) ? it->second : nullptr;
}

bool NetworkManager::listenForConnections(const NetworkAddress& listen_addr) {
  if (listening_) {
    return false;
  }

  listening_ = true;
  return true;
}

void NetworkManager::stopListening() {
  listening_ = false;

  if (listener_thread_ && listener_thread_->joinable()) {
    listener_thread_->join();
  }

  listener_thread_.reset();
}

void NetworkManager::acceptConnections() {
}

void NetworkManager::setMessageCallback(MessageCallback callback) {
  message_callback_ = callback;
}

void NetworkManager::setErrorCallback(std::function<void(int, const std::string&)> callback) {
  error_callback_ = callback;
}

std::vector<int> NetworkManager::getActiveConnections() const {
  std::lock_guard<std::mutex> lock(connections_mutex_);

  std::vector<int> active_ids;
  for (const auto& conn : connections_) {
    if (conn.second->getStatus() == ConnectionStatus::CONNECTED) {
      active_ids.push_back(conn.first);
    }
  }

  return active_ids;
}

ConnectionStatus NetworkManager::getConnectionStatus(int connection_id) const {
  std::lock_guard<std::mutex> lock(connections_mutex_);
  auto it = connections_.find(connection_id);
  return (it != connections_.end()) ? it->second->getStatus() : ConnectionStatus::DISCONNECTED;
}

size_t NetworkManager::getTotalBytesSent() const {
  return total_bytes_sent_.load();
}

size_t NetworkManager::getTotalBytesReceived() const {
  return total_bytes_received_.load();
}

double NetworkManager::getAverageLatency() const {
  uint64_t count = message_count_.load();
  return count > 0 ? total_latency_.load() / count : 0.0;
}

}
}