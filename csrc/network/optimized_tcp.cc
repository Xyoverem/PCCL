#include <network/network.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>

namespace engine_c {
namespace network {

class OptimizedTcpConnection : public TcpConnection {
public:
  OptimizedTcpConnection();
  ~OptimizedTcpConnection() = default;

  bool connect(const NetworkAddress& address) override;
  bool sendMessage(const MessageHeader& header, const void* data) override;
  bool receiveMessage(MessageHeader& header, void* data, size_t max_size) override;

  bool enableTcpOptimizations();
  bool setBufferSize(size_t send_buffer_size, size_t recv_buffer_size);
  bool enableNoDelay();
  bool enableKeepAlive();

  double measureLatency();
  double measureThroughput(size_t message_size);

private:
  size_t send_buffer_size_;
  size_t recv_buffer_size_;
  bool tcp_nodelay_enabled_;
  bool keepalive_enabled_;
  std::vector<char> temp_buffer_;
};

OptimizedTcpConnection::OptimizedTcpConnection()
  : send_buffer_size_(1024 * 1024), recv_buffer_size_(1024 * 1024),
    tcp_nodelay_enabled_(false), keepalive_enabled_(false) {
  temp_buffer_.reserve(64 * 1024);
}

bool OptimizedTcpConnection::connect(const NetworkAddress& address) {
  bool result = TcpConnection::connect(address);

  if (result) {
    enableTcpOptimizations();
  }

  return result;
}

bool OptimizedTcpConnection::enableTcpOptimizations() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (socket_fd_ < 0) {
    return false;
  }

  int flags = 1;
  if (setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, &flags, sizeof(flags)) == 0) {
    tcp_nodelay_enabled_ = true;
  }

  if (setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF,
                 &send_buffer_size_, sizeof(send_buffer_size_)) == 0) {
  }

  if (setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF,
                 &recv_buffer_size_, sizeof(recv_buffer_size_)) == 0) {
  }

  int keepalive = 1;
  int keepalive_idle = 30;
  int keepalive_intvl = 5;
  int keepalive_cnt = 3;

  if (setsockopt(socket_fd_, SOL_SOCKET, SO_KEEPALIVE,
                 &keepalive, sizeof(keepalive)) == 0) {
    setsockopt(socket_fd_, IPPROTO_TCP, TCP_KEEPIDLE,
               &keepalive_idle, sizeof(keepalive_idle));
    setsockopt(socket_fd_, IPPROTO_TCP, TCP_KEEPINTVL,
               &keepalive_intvl, sizeof(keepalive_intvl));
    setsockopt(socket_fd_, IPPROTO_TCP, TCP_KEEPCNT,
               &keepalive_cnt, sizeof(keepalive_cnt));
    keepalive_enabled_ = true;
  }

  int flags = fcntl(socket_fd_, F_GETFL, 0);
  fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);

  return true;
}

bool OptimizedTcpConnection::setBufferSize(size_t send_buffer_size, size_t recv_buffer_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (socket_fd_ < 0) {
    return false;
  }

  send_buffer_size_ = send_buffer_size;
  recv_buffer_size_ = recv_buffer_size;

  return (setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF,
                     &send_buffer_size_, sizeof(send_buffer_size_)) == 0 &&
          setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF,
                     &recv_buffer_size_, sizeof(recv_buffer_size_)) == 0);
}

bool OptimizedTcpConnection::enableNoDelay() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (socket_fd_ < 0) {
    return false;
  }

  int flag = 1;
  bool result = (setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY,
                            &flag, sizeof(flag)) == 0);

  if (result) {
    tcp_nodelay_enabled_ = true;
  }

  return result;
}

bool OptimizedTcpConnection::enableKeepAlive() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (socket_fd_ < 0) {
    return false;
  }

  int flag = 1;
  bool result = (setsockopt(socket_fd_, SOL_SOCKET, SO_KEEPALIVE,
                            &flag, sizeof(flag)) == 0);

  if (result) {
    keepalive_enabled_ = true;
  }

  return result;
}

bool OptimizedTcpConnection::sendMessage(const MessageHeader& header, const void* data) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (status_ != ConnectionStatus::CONNECTED || socket_fd_ < 0) {
    return false;
  }

  struct iovec iov[2];
  iov[0].iov_base = const_cast<MessageHeader*>(&header);
  iov[0].iov_len = sizeof(header);
  iov[1].iov_base = const_cast<void*>(data);
  iov[1].iov_len = header.data_size;

  struct msghdr msg;
  memset(&msg, 0, sizeof(msg));
  msg.msg_iov = iov;
  msg.msg_iovlen = (header.data_size > 0) ? 2 : 1;

  ssize_t total_sent = 0;
  ssize_t expected_size = sizeof(header) + header.data_size;

  while (total_sent < expected_size) {
    ssize_t sent = sendmsg(socket_fd_, &msg, MSG_NOSIGNAL);
    if (sent < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        std::this_thread::yield();
        continue;
      } else {
        status_ = ConnectionStatus::ERROR;
        return false;
      }
    }
    total_sent += sent;

    if (total_sent < expected_size) {
      if (sent >= static_cast<ssize_t>(sizeof(header))) {
        size_t data_sent = sent - sizeof(header);
        iov[1].iov_base = static_cast<char*>(iov[1].iov_base) + data_sent;
        iov[1].iov_len -= data_sent;
      } else {
        iov[0].iov_base = static_cast<char*>(iov[0].iov_base) + sent;
        iov[0].iov_len -= sent;
      }
      msg.msg_iov = iov;
      msg.msg_iovlen = (iov[0].iov_len > 0) ? 2 : 1;
    }
  }

  return true;
}

bool OptimizedTcpConnection::receiveMessage(MessageHeader& header, void* data, size_t max_size) {
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

double OptimizedTcpConnection::measureLatency() {
  const int iterations = 100;
  const size_t message_size = 64;

  std::vector<char> send_data(message_size, 'x');
  std::vector<char> recv_data(message_size);

  MessageHeader header;
  header.message_id = 9999;
  header.data_size = message_size;
  header.source_rank = 0;
  header.dest_rank = 1;
  header.tag = 0;
  header.flags = 0;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    if (!sendMessage(header, send_data.data())) {
      return -1.0;
    }
    if (!receiveMessage(header, recv_data.data(), recv_data.size())) {
      return -1.0;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  return static_cast<double>(duration.count()) / iterations / 2.0;
}

double OptimizedTcpConnection::measureThroughput(size_t message_size) {
  const int iterations = 50;

  std::vector<char> send_data(message_size, 'x');
  std::vector<char> recv_data(message_size);

  MessageHeader header;
  header.message_id = 9998;
  header.data_size = message_size;
  header.source_rank = 0;
  header.dest_rank = 1;
  header.tag = 0;
  header.flags = 0;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    if (!sendMessage(header, send_data.data())) {
      return -1.0;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  double total_seconds = duration.count() / 1000000.0;
  double total_bytes = static_cast<double>(message_size * iterations);
  double mb_per_sec = (total_bytes / 1024.0 / 1024.0) / total_seconds;

  return mb_per_sec;
}

ConnectionPtr createOptimizedTcpConnection() {
  return std::make_shared<OptimizedTcpConnection>();
}

}
}