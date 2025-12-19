#include <communication/communicator.h>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstring>
#include <queue>

namespace engine_c {
namespace communication {

NetworkCommunicator::NetworkCommunicator()
  : rank_(0), world_size_(1), initialized_(false) {
  network_manager_ = std::make_unique<network::NetworkManager>();
  async_worker_running_ = true;
  async_worker_thread_ = std::thread(&NetworkCommunicator::asyncWorkerLoop, this);
}

NetworkCommunicator::~NetworkCommunicator() {
  finalize();
  async_worker_running_ = false;
  async_cv_.notify_all();
  if (async_worker_thread_.joinable()) {
    async_worker_thread_.join();
  }
}

bool NetworkCommunicator::initialize(int rank, int world_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    return true;
  }

  rank_ = rank;
  world_size_ = world_size;

  if (!network_manager_->initialize(network::NetworkType::TCP_SOCKET)) {
    return false;
  }

  connections_.clear();
  for (int i = 0; i < world_size_; ++i) {
    if (i != rank_) {
      auto conn = network_manager_->createConnection();
      if (conn) {
        connections_[i] = conn;
      }
    }
  }

  initialized_ = true;
  return true;
}

void NetworkCommunicator::finalize() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_) {
    return;
  }

  connections_.clear();
  network_manager_->shutdown();
  initialized_ = false;
}

bool NetworkCommunicator::allReduce(const void* sendbuf, void* recvbuf, size_t count,
                                  ReductionOp op, uint32_t tag) {
  auto start_time = std::chrono::high_resolution_clock::now();

  if (!initialized_ || !sendbuf || !recvbuf || count == 0) {
    return false;
  }

  bool result = false;

  if (collective_config_.algorithm == "ring" || collective_config_.algorithm == "auto") {
    result = ringAllReduce(sendbuf, recvbuf, count, op, tag);
  } else if (collective_config_.algorithm == "tree") {
    result = treeAllReduce(sendbuf, recvbuf, count, op, tag);
  } else if (collective_config_.algorithm == "rabenseifner") {
    result = rabenseifnerAllReduce(sendbuf, recvbuf, count, op, tag);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  last_operation_time_ = duration.count() / 1000.0;
  total_bytes_transferred_ += count * world_size_;
  total_latency_ += last_operation_time_;
  operation_count_++;

  return result;
}

bool NetworkCommunicator::ringAllReduce(const void* sendbuf, void* recvbuf, size_t count,
                                       ReductionOp op, uint32_t tag) {
  size_t chunk_size = count / world_size_;
  size_t remaining = count % world_size_;

  std::vector<char> temp_buffer(count * 2);
  std::memcpy(recvbuf, sendbuf, count);

  for (int step = 0; step < world_size_; ++step) {
    int send_to = (rank_ + 1) % world_size_;
    int recv_from = (rank_ - 1 + world_size_) % world_size_;

    size_t send_start = (step * chunk_size + std::min(step, static_cast<int>(remaining))) % count;
    size_t send_size = chunk_size + (step < remaining ? 1 : 0);
    if (send_start + send_size > count) {
      send_size = count - send_start;
    }

    size_t recv_start = send_start;
    size_t recv_size = send_size;

    network::MessageHeader header;
    header.message_id = tag;
    header.data_size = send_size;
    header.source_rank = rank_;
    header.dest_rank = send_to;
    header.tag = tag;
    header.flags = 0;
    header.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    auto send_conn = connections_[send_to];
    if (send_conn && send_conn->getStatus() == network::ConnectionStatus::CONNECTED) {
      send_conn->sendMessage(header, static_cast<const char*>(recvbuf) + send_start);
    }

    auto recv_conn = connections_[recv_from];
    if (recv_conn && recv_conn->getStatus() == network::ConnectionStatus::CONNECTED) {
      network::MessageHeader recv_header;
      if (recv_conn->receiveMessage(recv_header, temp_buffer.data(), temp_buffer.size())) {
        performReduction(static_cast<char*>(recvbuf) + recv_start,
                        temp_buffer.data(), recv_size, op);
      }
    }

    std::this_thread::yield();
  }

  return true;
}

bool NetworkCommunicator::treeAllReduce(const void* sendbuf, void* recvbuf, size_t count,
                                       ReductionOp op, uint32_t tag) {
  std::memcpy(recvbuf, sendbuf, count);

  int tree_parent = (rank_ == 0) ? -1 : (rank_ - 1) / 2;
  std::vector<int> tree_children;
  int left_child = rank_ * 2 + 1;
  int right_child = rank_ * 2 + 2;

  if (left_child < world_size_) {
    tree_children.push_back(left_child);
  }
  if (right_child < world_size_) {
    tree_children.push_back(right_child);
  }

  std::vector<char> temp_buffer(count);

  for (int child : tree_children) {
    network::MessageHeader header;
    header.message_id = tag;
    header.data_size = count;
    header.source_rank = child;
    header.dest_rank = rank_;
    header.tag = tag;
    header.flags = 0;

    auto conn = connections_[child];
    if (conn && conn->getStatus() == network::ConnectionStatus::CONNECTED) {
      if (conn->receiveMessage(header, temp_buffer.data(), count)) {
        performReduction(recvbuf, temp_buffer.data(), count, op);
      }
    }
  }

  if (tree_parent != -1) {
    network::MessageHeader header;
    header.message_id = tag;
    header.data_size = count;
    header.source_rank = rank_;
    header.dest_rank = tree_parent;
    header.tag = tag;
    header.flags = 0;
    header.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    auto conn = connections_[tree_parent];
    if (conn && conn->getStatus() == network::ConnectionStatus::CONNECTED) {
      conn->sendMessage(header, recvbuf);
    }
  }

  for (int child : tree_children) {
    network::MessageHeader header;
    header.message_id = tag;
    header.data_size = count;
    header.source_rank = rank_;
    header.dest_rank = child;
    header.tag = tag;
    header.flags = 0;
    header.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    auto conn = connections_[child];
    if (conn && conn->getStatus() == network::ConnectionStatus::CONNECTED) {
      conn->sendMessage(header, recvbuf);
    }
  }

  return true;
}

bool NetworkCommunicator::rabenseifnerAllReduce(const void* sendbuf, void* recvbuf, size_t count,
                                               ReductionOp op, uint32_t tag) {
  std::memcpy(recvbuf, sendbuf, count);

  int power_of_two = 1;
  while (power_of_two < world_size_) {
    power_of_two *= 2;
  }

  for (int step = 1; step < power_of_two; step *= 2) {
    int partner = rank_ ^ step;

    if (partner < world_size_) {
      size_t chunk_size = count / step;
      size_t offset = (rank_ / step) * chunk_size;

      network::MessageHeader header;
      header.message_id = tag;
      header.data_size = chunk_size;
      header.source_rank = rank_;
      header.dest_rank = partner;
      header.tag = tag;
      header.flags = 0;
      header.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      std::vector<char> temp_buffer(chunk_size);

      auto conn = connections_[partner];
      if (conn && conn->getStatus() == network::ConnectionStatus::CONNECTED) {
        conn->sendMessage(header, static_cast<const char*>(recvbuf) + offset);

        if (conn->receiveMessage(header, temp_buffer.data(), chunk_size)) {
          performReduction(static_cast<char*>(recvbuf) + offset,
                          temp_buffer.data(), chunk_size, op);
        }
      }
    }
  }

  return true;
}

void NetworkCommunicator::performReduction(void* dest, const void* src, size_t count, ReductionOp op) {
  switch (op) {
    case ReductionOp::SUM: {
      float* d = static_cast<float*>(dest);
      const float* s = static_cast<const float*>(src);
      for (size_t i = 0; i < count / sizeof(float); ++i) {
        d[i] += s[i];
      }
      break;
    }
    case ReductionOp::MAX: {
      float* d = static_cast<float*>(dest);
      const float* s = static_cast<const float*>(src);
      for (size_t i = 0; i < count / sizeof(float); ++i) {
        d[i] = std::max(d[i], s[i]);
      }
      break;
    }
    case ReductionOp::MIN: {
      float* d = static_cast<float*>(dest);
      const float* s = static_cast<const float*>(src);
      for (size_t i = 0; i < count / sizeof(float); ++i) {
        d[i] = std::min(d[i], s[i]);
      }
      break;
    }
    default:
      break;
  }
}

bool NetworkCommunicator::allGather(const void* sendbuf, void* recvbuf, size_t sendcount,
                                   size_t element_size, uint32_t tag) {
  return false;
}

bool NetworkCommunicator::reduceScatter(const void* sendbuf, void* recvbuf,
                                       const size_t recvcounts[], ReductionOp op,
                                       uint32_t tag) {
  return false;
}

bool NetworkCommunicator::broadcast(void* buffer, size_t count, int root,
                                   uint32_t tag) {
  return false;
}

bool NetworkCommunicator::reduce(const void* sendbuf, void* recvbuf, size_t count,
                                ReductionOp op, int root, uint32_t tag) {
  return false;
}

bool NetworkCommunicator::gather(const void* sendbuf, void* recvbuf, size_t sendcount,
                                size_t element_size, int root, uint32_t tag) {
  return false;
}

bool NetworkCommunicator::scatter(const void* sendbuf, void* recvbuf, size_t recvcount,
                                 size_t element_size, int root, uint32_t tag) {
  return false;
}

bool NetworkCommunicator::alltoall(const void* sendbuf, void* recvbuf, size_t sendcount,
                                  size_t element_size, uint32_t tag) {
  return false;
}

bool NetworkCommunicator::allReduceAsync(const void* sendbuf, void* recvbuf, size_t count,
                                        ReductionOp op, uint32_t tag, std::future<bool>& future) {
  auto async_op = std::make_unique<AsyncOperation>();
  async_op->type = CollectiveType::ALLREDUCE;
  async_op->tag = tag;
  async_op->start_time = std::chrono::high_resolution_clock::now();
  future = async_op->promise.get_future();

  std::lock_guard<std::mutex> lock(async_mutex_);
  async_tasks_.push([this, sendbuf, recvbuf, count, op, tag, async_op_ptr = async_op.release()]() {
    bool result = allReduce(sendbuf, recvbuf, count, op, tag);
    async_op_ptr->promise.set_value(result);
    delete async_op_ptr;
  });

  async_cv_.notify_one();
  return true;
}

bool NetworkCommunicator::allGatherAsync(const void* sendbuf, void* recvbuf, size_t sendcount,
                                        size_t element_size, uint32_t tag, std::future<bool>& future) {
  return false;
}

bool NetworkCommunicator::broadcastAsync(void* buffer, size_t count, int root,
                                        uint32_t tag, std::future<bool>& future) {
  return false;
}

bool NetworkCommunicator::send(const void* buffer, size_t count, int dest, int tag) {
  auto conn = connections_[dest];
  if (!conn || conn->getStatus() != network::ConnectionStatus::CONNECTED) {
    return false;
  }

  network::MessageHeader header;
  header.message_id = tag;
  header.data_size = count;
  header.source_rank = rank_;
  header.dest_rank = dest;
  header.tag = tag;
  header.flags = 0;
  header.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();

  return conn->sendMessage(header, buffer);
}

bool NetworkCommunicator::recv(void* buffer, size_t count, int source, int tag) {
  auto conn = connections_[source];
  if (!conn || conn->getStatus() != network::ConnectionStatus::CONNECTED) {
    return false;
  }

  network::MessageHeader header;
  return conn->receiveMessage(header, buffer, count);
}

bool NetworkCommunicator::sendAsync(const void* buffer, size_t count, int dest, int tag,
                                   std::future<bool>& future) {
  return false;
}

bool NetworkCommunicator::recvAsync(void* buffer, size_t count, int source, int tag,
                                   std::future<bool>& future) {
  return false;
}

bool NetworkCommunicator::flush() {
  return true;
}

bool NetworkCommunicator::barrier() {
  return true;
}

int NetworkCommunicator::getRank() const {
  return rank_;
}

int NetworkCommunicator::getWorldSize() const {
  return world_size_;
}

double NetworkCommunicator::getLastOperationTime() const {
  return last_operation_time_.load();
}

uint64_t NetworkCommunicator::getTotalBytesTransferred() const {
  return total_bytes_transferred_.load();
}

double NetworkCommunicator::getAverageLatency() const {
  uint64_t count = operation_count_.load();
  return count > 0 ? total_latency_.load() / count : 0.0;
}

double NetworkCommunicator::getThroughput() const {
  double total_time = total_latency_.load();
  return total_time > 0 ? (total_bytes_transferred_.load() / 1024.0 / 1024.0) / (total_time / 1000.0) : 0.0;
}

void NetworkCommunicator::setCollectiveConfig(const CollectiveConfig& config) {
  collective_config_ = config;
}

CollectiveConfig NetworkCommunicator::getCollectiveConfig() const {
  return collective_config_;
}

bool NetworkCommunicator::connectToPeers(const std::vector<network::NetworkAddress>& peer_addresses) {
  if (!initialized_ || peer_addresses.size() != static_cast<size_t>(world_size_ - 1)) {
    return false;
  }

  size_t peer_idx = 0;
  for (int i = 0; i < world_size_; ++i) {
    if (i != rank_) {
      auto conn = connections_[i];
      if (conn && peer_idx < peer_addresses.size()) {
        if (!conn->connect(peer_addresses[peer_idx])) {
          return false;
        }
        peer_idx++;
      }
    }
  }

  return true;
}

void NetworkCommunicator::setNetworkType(network::NetworkType type) {
  network_manager_->shutdown();
  network_manager_->initialize(type);
}

void NetworkCommunicator::asyncWorkerLoop() {
  while (async_worker_running_) {
    std::unique_lock<std::mutex> lock(async_mutex_);
    async_cv_.wait(lock, [this] { return !async_tasks_.empty() || !async_worker_running_; });

    while (!async_tasks_.empty()) {
      auto task = async_tasks_.front();
      async_tasks_.pop();
      lock.unlock();
      task();
      lock.lock();
    }
  }
}

CommunicatorPtr createCommunicator(network::NetworkType type) {
  auto comm = std::make_shared<NetworkCommunicator>();
  comm->setNetworkType(type);
  return comm;
}

}
}