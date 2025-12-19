#include <algorithms/communication_wrapper.h>
#include <chrono>
#include <cstring>

namespace engine_c {
namespace algorithms {

CommunicationWrapper::CommunicationWrapper() : initialized_(false), cache_valid_(false) {}

CommunicationWrapper::~CommunicationWrapper() {
  finalize();
}

bool CommunicationWrapper::initialize(int rank, int world_size,
                                     network::NetworkType type) {
  if (initialized_) {
    return true;
  }

  communicator_ = communication::createCommunicator(type);
  if (!communicator_) {
    return false;
  }

  if (!communicator_->initialize(rank, world_size)) {
    communicator_.reset();
    return false;
  }

  latency_cache_.resize(world_size, 0.0);
  bandwidth_cache_.resize(world_size, 0.0);

  initialized_ = true;
  return true;
}

void CommunicationWrapper::finalize() {
  if (initialized_ && communicator_) {
    communicator_->finalize();
    communicator_.reset();
    initialized_ = false;
    cache_valid_ = false;
  }
}

bool CommunicationWrapper::allReduceRing(const void* sendbuf, void* recvbuf, size_t count,
                                        communication::ReductionOp op, uint32_t tag) {
  if (!initialized_ || !communicator_) {
    return false;
  }

  communication::CollectiveConfig config;
  config.type = communication::CollectiveType::ALLREDUCE;
  config.reduction_op = op;
  config.algorithm = "ring";
  config.tag = tag;
  communicator_->setCollectiveConfig(config);

  return communicator_->allReduce(sendbuf, recvbuf, count, op, tag);
}

bool CommunicationWrapper::allReduceTree(const void* sendbuf, void* recvbuf, size_t count,
                                        communication::ReductionOp op, uint32_t tag) {
  if (!initialized_ || !communicator_) {
    return false;
  }

  communication::CollectiveConfig config;
  config.type = communication::CollectiveType::ALLREDUCE;
  config.reduction_op = op;
  config.algorithm = "tree";
  config.tag = tag;
  communicator_->setCollectiveConfig(config);

  return communicator_->allReduce(sendbuf, recvbuf, count, op, tag);
}

bool CommunicationWrapper::allReduceRabenseifner(const void* sendbuf, void* recvbuf, size_t count,
                                                communication::ReductionOp op, uint32_t tag) {
  if (!initialized_ || !communicator_) {
    return false;
  }

  communication::CollectiveConfig config;
  config.type = communication::CollectiveType::ALLREDUCE;
  config.reduction_op = op;
  config.algorithm = "rabenseifner";
  config.tag = tag;
  communicator_->setCollectiveConfig(config);

  return communicator_->allReduce(sendbuf, recvbuf, count, op, tag);
}

bool CommunicationWrapper::broadcast(void* buffer, size_t count, int root, uint32_t tag) {
  if (!initialized_ || !communicator_) {
    return false;
  }

  return communicator_->broadcast(buffer, count, root, tag);
}

bool CommunicationWrapper::send(const void* buffer, size_t count, int dest, uint32_t tag) {
  if (!initialized_ || !communicator_) {
    return false;
  }

  return communicator_->send(buffer, count, dest, tag);
}

bool CommunicationWrapper::recv(void* buffer, size_t count, int source, uint32_t tag) {
  if (!initialized_ || !communicator_) {
    return false;
  }

  return communicator_->recv(buffer, count, source, tag);
}

bool CommunicationWrapper::isInitialized() const {
  return initialized_;
}

int CommunicationWrapper::getRank() const {
  return initialized_ ? communicator_->getRank() : -1;
}

int CommunicationWrapper::getWorldSize() const {
  return initialized_ ? communicator_->getWorldSize() : 0;
}

void CommunicationWrapper::setLatencyCallback(std::function<double(int, int)> callback) {
  latency_callback_ = callback;
}

void CommunicationWrapper::setBandwidthCallback(std::function<double(int, int)> callback) {
  bandwidth_callback_ = callback;
}

double CommunicationWrapper::measureLatency(int peer_rank, size_t message_size) {
  if (!initialized_ || !communicator_ || peer_rank == getRank()) {
    return 0.0;
  }

  if (cache_valid_ && peer_rank < latency_cache_.size() && latency_cache_[peer_rank] > 0.0) {
    return latency_cache_[peer_rank];
  }

  if (latency_callback_) {
    double latency = latency_callback_(getRank(), peer_rank);
    if (latency > 0.0) {
      return latency;
    }
  }

  std::vector<char> send_buf(message_size, 'x');
  std::vector<char> recv_buf(message_size);

  int tag = 9999;

  if (getRank() < peer_rank) {
    auto start_time = std::chrono::high_resolution_clock::now();
    if (send(send_buf.data(), message_size, peer_rank, tag) &&
        recv(recv_buf.data(), message_size, peer_rank, tag)) {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      double latency = duration.count() / 2.0;

      if (peer_rank < latency_cache_.size()) {
        latency_cache_[peer_rank] = latency;
      }
      return latency;
    }
  } else {
    auto start_time = std::chrono::high_resolution_clock::now();
    if (recv(recv_buf.data(), message_size, peer_rank, tag) &&
        send(send_buf.data(), message_size, peer_rank, tag)) {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      double latency = duration.count() / 2.0;

      if (peer_rank < latency_cache_.size()) {
        latency_cache_[peer_rank] = latency;
      }
      return latency;
    }
  }

  return 0.0;
}

double CommunicationWrapper::measureBandwidth(int peer_rank, size_t message_size) {
  if (!initialized_ || !communicator_ || peer_rank == getRank()) {
    return 0.0;
  }

  if (cache_valid_ && peer_rank < bandwidth_cache_.size() && bandwidth_cache_[peer_rank] > 0.0) {
    return bandwidth_cache_[peer_rank];
  }

  if (bandwidth_callback_) {
    double bandwidth = bandwidth_callback_(getRank(), peer_rank);
    if (bandwidth > 0.0) {
      return bandwidth;
    }
  }

  std::vector<char> send_buf(message_size, 'x');
  std::vector<char> recv_buf(message_size);

  int tag = 9998;
  int iterations = 10;

  if (getRank() < peer_rank) {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      if (!send(send_buf.data(), message_size, peer_rank, tag + i) ||
          !recv(recv_buf.data(), message_size, peer_rank, tag + i)) {
        return 0.0;
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    double total_time_seconds = duration.count() / 1000000.0;
    double total_bytes = message_size * iterations;
    double bandwidth = (total_bytes / 1024.0 / 1024.0) / total_time_seconds;

    if (peer_rank < bandwidth_cache_.size()) {
      bandwidth_cache_[peer_rank] = bandwidth;
    }
    return bandwidth;
  } else {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      if (!recv(recv_buf.data(), message_size, peer_rank, tag + i) ||
          !send(send_buf.data(), message_size, peer_rank, tag + i)) {
        return 0.0;
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    double total_time_seconds = duration.count() / 1000000.0;
    double total_bytes = message_size * iterations;
    double bandwidth = (total_bytes / 1024.0 / 1024.0) / total_time_seconds;

    if (peer_rank < bandwidth_cache_.size()) {
      bandwidth_cache_[peer_rank] = bandwidth;
    }
    return bandwidth;
  }

  return 0.0;
}

void CommunicationWrapper::updateCaches() {
  if (!initialized_) {
    return;
  }

  int world_size = getWorldSize();
  for (int i = 0; i < world_size; ++i) {
    if (i != getRank()) {
      measureLatency(i, 1024);
      measureBandwidth(i, 1024 * 1024);
    }
  }
  cache_valid_ = true;
}

}
}