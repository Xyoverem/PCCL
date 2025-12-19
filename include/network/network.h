#pragma once

#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstdint>

namespace engine_c {
namespace network {

enum class NetworkType {
  TCP_SOCKET,
  RDMA_VERBS,
  GPU_DIRECT
};

enum class ConnectionStatus {
  DISCONNECTED,
  CONNECTING,
  CONNECTED,
  ERROR
};

struct NetworkAddress {
  std::string ip;
  int port;
  std::string device;

  NetworkAddress() : port(0) {}
  NetworkAddress(const std::string& ip_addr, int port_num)
    : ip(ip_addr), port(port_num) {}
};

struct MessageHeader {
  uint32_t message_id;
  uint32_t data_size;
  uint32_t source_rank;
  uint32_t dest_rank;
  uint32_t tag;
  uint32_t flags;
  uint64_t timestamp;
};

class NetworkConnection {
public:
  virtual ~NetworkConnection() = default;

  virtual bool connect(const NetworkAddress& address) = 0;
  virtual void disconnect() = 0;
  virtual ConnectionStatus getStatus() const = 0;

  virtual bool sendMessage(const MessageHeader& header, const void* data) = 0;
  virtual bool receiveMessage(MessageHeader& header, void* data, size_t max_size) = 0;
  virtual bool sendAsync(const MessageHeader& header, const void* data) = 0;
  virtual bool recvAsync(MessageHeader& header, void* data, size_t max_size) = 0;
  virtual bool pollCompletion(std::vector<uint32_t>& completed_ids) = 0;

  virtual std::string getLocalAddress() const = 0;
  virtual NetworkAddress getRemoteAddress() const = 0;
  virtual NetworkType getType() const = 0;
};

using ConnectionPtr = std::shared_ptr<NetworkConnection>;
using MessageCallback = std::function<void(const MessageHeader&, const void*)>;

class NetworkManager {
public:
  static NetworkManager& getInstance();

  bool initialize(NetworkType type = NetworkType::TCP_SOCKET);
  void shutdown();

  ConnectionPtr createConnection();
  bool removeConnection(int connection_id);
  ConnectionPtr getConnection(int connection_id);

  bool listenForConnections(const NetworkAddress& listen_addr);
  void stopListening();
  void acceptConnections();

  void setMessageCallback(MessageCallback callback);
  void setErrorCallback(std::function<void(int, const std::string&)> callback);

  std::vector<int> getActiveConnections() const;
  ConnectionStatus getConnectionStatus(int connection_id) const;

  size_t getTotalBytesSent() const;
  size_t getTotalBytesReceived() const;
  double getAverageLatency() const;

private:
  NetworkManager() = default;
  ~NetworkManager();
  NetworkManager(const NetworkManager&) = delete;
  NetworkManager& operator=(const NetworkManager&) = delete;

  NetworkType network_type_;
  std::atomic<bool> initialized_{false};
  std::atomic<bool> listening_{false};

  mutable std::mutex connections_mutex_;
  std::unordered_map<int, ConnectionPtr> connections_;
  std::atomic<int> next_connection_id_{1};

  std::atomic<size_t> total_bytes_sent_{0};
  std::atomic<size_t> total_bytes_received_{0};
  std::atomic<double> total_latency_{0.0};
  std::atomic<uint64_t> message_count_{0};

  MessageCallback message_callback_;
  std::function<void(int, const std::string&)> error_callback_;
  std::unique_ptr<std::thread> listener_thread_;
};

class TcpConnection : public NetworkConnection {
public:
  TcpConnection();
  ~TcpConnection() override;

  bool connect(const NetworkAddress& address) override;
  void disconnect() override;
  ConnectionStatus getStatus() const override;

  bool sendMessage(const MessageHeader& header, const void* data) override;
  bool receiveMessage(MessageHeader& header, void* data, size_t max_size) override;
  bool sendAsync(const MessageHeader& header, const void* data) override;
  bool recvAsync(MessageHeader& header, void* data, size_t max_size) override;
  bool pollCompletion(std::vector<uint32_t>& completed_ids) override;

  std::string getLocalAddress() const override;
  NetworkAddress getRemoteAddress() const override;
  NetworkType getType() const override { return NetworkType::TCP_SOCKET; }

private:
  int socket_fd_;
  ConnectionStatus status_;
  NetworkAddress local_addr_;
  NetworkAddress remote_addr_;

  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, bool> pending_operations_;
  std::atomic<uint32_t> next_message_id_{1};
};

class RdmaConnection : public NetworkConnection {
public:
  RdmaConnection();
  ~RdmaConnection() override;

  bool connect(const NetworkAddress& address) override;
  void disconnect() override;
  ConnectionStatus getStatus() const override;

  bool sendMessage(const MessageHeader& header, const void* data) override;
  bool receiveMessage(MessageHeader& header, void* data, size_t max_size) override;
  bool sendAsync(const MessageHeader& header, const void* data) override;
  bool recvAsync(MessageHeader& header, void* data, size_t max_size) override;
  bool pollCompletion(std::vector<uint32_t>& completed_ids) override;

  std::string getLocalAddress() const override;
  NetworkAddress getRemoteAddress() const override;
  NetworkType getType() const override { return NetworkType::RDMA_VERBS; }

  bool registerMemoryRegion(void* addr, size_t size, uint32_t& lkey, uint32_t& rkey);
  bool unregisterMemoryRegion(void* addr);

private:
  struct RdmaContext {
    void* verbs_context;
    void* protection_domain;
    void* completion_queue;
    void* queue_pair;
    uint32_t qp_num;
    uint16_t lid;
  };

  RdmaContext rdma_ctx_;
  ConnectionStatus status_;
  NetworkAddress local_addr_;
  NetworkAddress remote_addr_;

  mutable std::mutex mutex_;
  std::unordered_map<void*, std::pair<uint32_t, uint32_t>> memory_regions_;
  std::unordered_map<uint32_t, bool> pending_operations_;
  std::atomic<uint32_t> next_message_id_{1};
};

}
}