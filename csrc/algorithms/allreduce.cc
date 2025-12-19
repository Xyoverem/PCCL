#include <algorithms/allreduce.h>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace engine_c {

AllreduceImpl::AllreduceImpl(const AllreduceConfig& config)
    : config_(config), completed_(false) {}

RingAllreduce::RingAllreduce(const AllreduceConfig& config)
    : AllreduceImpl(config), rank_(0), world_size_(2), next_rank_(1), prev_rank_(1) {
  if (!config_.participants.empty()) {
    world_size_ = config_.participants.size();
    auto it = std::find(config_.participants.begin(), config_.participants.end(), rank_);
    if (it != config_.participants.end()) {
      size_t idx = it - config_.participants.begin();
      next_rank_ = config_.participants[(idx + 1) % world_size_];
      prev_rank_ = config_.participants[(idx + world_size_ - 1) % world_size_];
    }
  }
}

void RingAllreduce::execute(void* input, void* output, size_t data_size, DataType dtype) {
  memcpy(output, input, data_size);

  scatterReduceRing(output, data_size, dtype);
  allgatherRing(output, data_size, dtype);

  completed_ = true;
}

void RingAllreduce::executeAsync(void* input, void* output, size_t data_size, DataType dtype) {
  execute(input, output, data_size, dtype);
}

void RingAllreduce::waitCompletion() {
  while (!completed_) {
  }
}

void RingAllreduce::scatterReduceRing(void* buffer, size_t data_size, DataType dtype) {
  if (world_size_ <= 1) return;

  size_t chunk_size = data_size / world_size_;
  if (chunk_size == 0) chunk_size = 1;

  for (int i = 0; i < world_size_ - 1; ++i) {
    int send_to = (rank_ + 1) % world_size_;
    int recv_from = (rank_ + world_size_ - 1) % world_size_;

    size_t send_offset = ((rank_ - i - 1 + world_size_) % world_size_) * chunk_size;
    size_t recv_offset = ((rank_ - i - 2 + world_size_) % world_size_) * chunk_size;

    if (send_offset + chunk_size > data_size) send_offset = data_size - chunk_size;
    if (recv_offset + chunk_size > data_size) recv_offset = data_size - chunk_size;

    char* send_ptr = static_cast<char*>(buffer) + send_offset;
    char* recv_ptr = static_cast<char*>(buffer) + recv_offset;

    char* temp_buffer = new char[chunk_size];
    memcpy(temp_buffer, recv_ptr, chunk_size);

    switch (config_.reduce_op) {
      case ReduceOp::SUM:
        if (dtype == DataType::FLOAT32) {
          float* send_f = reinterpret_cast<float*>(send_ptr);
          float* temp_f = reinterpret_cast<float*>(temp_buffer);
          for (size_t j = 0; j < chunk_size / sizeof(float); ++j) {
            send_f[j] += temp_f[j];
          }
        }
        break;
      case ReduceOp::MAX:
        if (dtype == DataType::FLOAT32) {
          float* send_f = reinterpret_cast<float*>(send_ptr);
          float* temp_f = reinterpret_cast<float*>(temp_buffer);
          for (size_t j = 0; j < chunk_size / sizeof(float); ++j) {
            send_f[j] = std::max(send_f[j], temp_f[j]);
          }
        }
        break;
      case ReduceOp::MIN:
        if (dtype == DataType::FLOAT32) {
          float* send_f = reinterpret_cast<float*>(send_ptr);
          float* temp_f = reinterpret_cast<float*>(temp_buffer);
          for (size_t j = 0; j < chunk_size / sizeof(float); ++j) {
            send_f[j] = std::min(send_f[j], temp_f[j]);
          }
        }
        break;
      case ReduceOp::AVG:
        if (dtype == DataType::FLOAT32) {
          float* send_f = reinterpret_cast<float*>(send_ptr);
          float* temp_f = reinterpret_cast<float*>(temp_buffer);
          for (size_t j = 0; j < chunk_size / sizeof(float); ++j) {
            send_f[j] = (send_f[j] + temp_f[j]) * 0.5f;
          }
        }
        break;
    }

    delete[] temp_buffer;
  }
}

void RingAllreduce::allgatherRing(void* buffer, size_t data_size, DataType dtype) {
  if (world_size_ <= 1) return;

  size_t chunk_size = data_size / world_size_;
  if (chunk_size == 0) chunk_size = 1;

  for (int i = 0; i < world_size_ - 1; ++i) {
    int send_to = (rank_ + 1) % world_size_;
    int recv_from = (rank_ + world_size_ - 1) % world_size_;

    size_t send_offset = ((rank_ - i + world_size_) % world_size_) * chunk_size;
    size_t recv_offset = ((rank_ - i - 1 + world_size_) % world_size_) * chunk_size;

    if (send_offset + chunk_size > data_size) send_offset = data_size - chunk_size;
    if (recv_offset + chunk_size > data_size) recv_offset = data_size - chunk_size;

    char* send_ptr = static_cast<char*>(buffer) + send_offset;
    char* recv_ptr = static_cast<char*>(buffer) + recv_offset;

    char* temp_buffer = new char[chunk_size];
    memcpy(temp_buffer, send_ptr, chunk_size);
    memcpy(send_ptr, recv_ptr, chunk_size);
    memcpy(recv_ptr, temp_buffer, chunk_size);

    delete[] temp_buffer;
  }
}

TreeAllreduce::TreeAllreduce(const AllreduceConfig& config, int branching_factor)
    : AllreduceImpl(config), rank_(0), world_size_(2), branching_factor_(branching_factor),
      parent_rank_(-1) {
  if (!config_.participants.empty()) {
    world_size_ = config_.participants.size();
    rank_ = config_.participants[0];
  }

  buildTree();
}

void TreeAllreduce::buildTree() {
  child_ranks_.clear();

  if (world_size_ <= 1) return;

  std::vector<int> levels;
  int current_level_start = 0;
  int current_level_size = 1;

  levels.push_back(0);

  while (current_level_start + current_level_size < world_size_) {
    int next_level_start = current_level_start + current_level_size;
    int next_level_size = std::min(current_level_size * branching_factor_, world_size_ - next_level_start);

    for (int i = next_level_start; i < next_level_start + next_level_size; ++i) {
      levels.push_back((i - next_level_start) / branching_factor_ + current_level_start);
    }

    current_level_start = next_level_start;
    current_level_size = next_level_size;
  }

  if (rank_ > 0 && rank_ < levels.size()) {
    parent_rank_ = levels[rank_];
  }

  for (int i = 1; i < levels.size(); ++i) {
    if (levels[i] == rank_) {
      child_ranks_.push_back(i);
    }
  }
}

void TreeAllreduce::execute(void* input, void* output, size_t data_size, DataType dtype) {
  memcpy(output, input, data_size);

  reduceTree(output, data_size, dtype);
  broadcastTree(output, data_size, dtype);

  completed_ = true;
}

void TreeAllreduce::executeAsync(void* input, void* output, size_t data_size, DataType dtype) {
  execute(input, output, data_size, dtype);
}

void TreeAllreduce::waitCompletion() {
  while (!completed_) {
  }
}

void TreeAllreduce::reduceTree(void* buffer, size_t data_size, DataType dtype) {
  if (world_size_ <= 1) return;

  for (int child_rank : child_ranks_) {
    char* child_buffer = new char[data_size];
    memcpy(child_buffer, buffer, data_size);

    switch (config_.reduce_op) {
      case ReduceOp::SUM:
        if (dtype == DataType::FLOAT32) {
          float* buf_f = reinterpret_cast<float*>(buffer);
          float* child_f = reinterpret_cast<float*>(child_buffer);
          for (size_t i = 0; i < data_size / sizeof(float); ++i) {
            buf_f[i] += child_f[i];
          }
        }
        break;
      case ReduceOp::MAX:
        if (dtype == DataType::FLOAT32) {
          float* buf_f = reinterpret_cast<float*>(buffer);
          float* child_f = reinterpret_cast<float*>(child_buffer);
          for (size_t i = 0; i < data_size / sizeof(float); ++i) {
            buf_f[i] = std::max(buf_f[i], child_f[i]);
          }
        }
        break;
      case ReduceOp::MIN:
        if (dtype == DataType::FLOAT32) {
          float* buf_f = reinterpret_cast<float*>(buffer);
          float* child_f = reinterpret_cast<float*>(child_buffer);
          for (size_t i = 0; i < data_size / sizeof(float); ++i) {
            buf_f[i] = std::min(buf_f[i], child_f[i]);
          }
        }
        break;
      case ReduceOp::AVG:
        if (dtype == DataType::FLOAT32) {
          float* buf_f = reinterpret_cast<float*>(buffer);
          float* child_f = reinterpret_cast<float*>(child_buffer);
          int total_participants = child_ranks_.size() + 1;
          for (size_t i = 0; i < data_size / sizeof(float); ++i) {
            buf_f[i] = (buf_f[i] + child_f[i]) / static_cast<float>(total_participants);
          }
        }
        break;
    }

    delete[] child_buffer;
  }
}

void TreeAllreduce::broadcastTree(void* buffer, size_t data_size, DataType dtype) {
  if (world_size_ <= 1) return;

  for (int child_rank : child_ranks_) {
  }
}

RabenseifnerAllreduce::RabenseifnerAllreduce(const AllreduceConfig& config)
    : AllreduceImpl(config) {
  tree_allreduce_ = std::make_unique<TreeAllreduce>(config);
  ring_allreduce_ = std::make_unique<RingAllreduce>(config);
}

void RabenseifnerAllreduce::execute(void* input, void* output, size_t data_size, DataType dtype) {
  if (data_size < config_.buffer_size) {
    ring_allreduce_->execute(input, output, data_size, dtype);
  } else {
    tree_allreduce_->execute(input, output, data_size, dtype);
  }
  completed_ = true;
}

void RabenseifnerAllreduce::executeAsync(void* input, void* output, size_t data_size, DataType dtype) {
  execute(input, output, data_size, dtype);
}

void RabenseifnerAllreduce::waitCompletion() {
  while (!completed_) {
  }
}

std::unique_ptr<AllreduceImpl> AllreduceFactory::create(const AllreduceConfig& config) {
  switch (config.algorithm) {
    case AllreduceAlgorithm::RING:
      return createRing(config);
    case AllreduceAlgorithm::TREE:
      return createTree(config);
    case AllreduceAlgorithm::RABENSEIFNER:
      return createRabenseifner(config);
    case AllreduceAlgorithm::DOUBLE_BINARY_TREE:
      return createDoubleBinaryTree(config);
    default:
      return createRing(config);
  }
}

std::unique_ptr<AllreduceImpl> AllreduceFactory::createRing(const AllreduceConfig& config) {
  return std::make_unique<RingAllreduce>(config);
}

std::unique_ptr<AllreduceImpl> AllreduceFactory::createTree(const AllreduceConfig& config, int branching_factor) {
  return std::make_unique<TreeAllreduce>(config, branching_factor);
}

std::unique_ptr<AllreduceImpl> AllreduceFactory::createRabenseifner(const AllreduceConfig& config) {
  return std::make_unique<RabenseifnerAllreduce>(config);
}

std::unique_ptr<AllreduceImpl> AllreduceFactory::createDoubleBinaryTree(const AllreduceConfig& config) {
  return createTree(config, 2);
}

AllreduceAlgorithm AllreduceFactory::selectOptimalAlgorithm(size_t data_size, int world_size, bool bandwidth_limited) {
  if (world_size <= 2) {
    return AllreduceAlgorithm::RING;
  }

  if (bandwidth_limited) {
    return AllreduceAlgorithm::RABENSEIFNER;
  } else {
    return AllreduceAlgorithm::TREE;
  }
}

}