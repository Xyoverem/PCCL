#include <cluster/process_group.h>
#include <communication/communicator.h>
#include <algorithms/communication_wrapper.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <memory>

namespace engine_c {

ProcessGroup::ProcessGroup(const std::string& name, ProcessGroupType type, const std::vector<int>& ranks)
    : name_(name), type_(type), ranks_(ranks) {
  std::sort(ranks_.begin(), ranks_.end());
}

void ProcessGroup::addProcess(int rank, const ProcessInfo& info) {
  if (std::find(ranks_.begin(), ranks_.end(), rank) != ranks_.end()) {
    processes_[rank] = info;
  }
}

void ProcessGroup::removeProcess(int rank) {
  processes_.erase(rank);
}

const ProcessInfo* ProcessGroup::getProcessInfo(int rank) const {
  auto it = processes_.find(rank);
  return (it != processes_.end()) ? &it->second : nullptr;
}

bool ProcessGroup::isRankInGroup(int rank) const {
  return std::find(ranks_.begin(), ranks_.end(), rank) != ranks_.end();
}

int ProcessGroup::getRankIndex(int rank) const {
  auto it = std::find(ranks_.begin(), ranks_.end(), rank);
  return (it != ranks_.end()) ? static_cast<int>(it - ranks_.begin()) : -1;
}

CPUProcessGroup::CPUProcessGroup(const std::string& name, const std::vector<int>& ranks)
    : ProcessGroup(name, ProcessGroupType::NODE_LOCAL, ranks), current_rank_(0) {
  if (!ranks.empty()) {
    current_rank_ = ranks[0];
  }
}

void CPUProcessGroup::initialize() {
  communication_wrapper_ = std::make_unique<algorithms::CommunicationWrapper>();

  AllreduceConfig config;
  config.algorithm = AllreduceAlgorithm::RING;
  config.reduce_op = ReduceOp::SUM;
  config.participants = ranks_;
  config.buffer_size = 128 * 1024 * 1024;
  config.enable_overlap = false;
  config.pipeline_depth = 2;

  allreduce_impl_ = AllreduceFactory::create(config);
}

void CPUProcessGroup::allreduce(void* input, void* output, size_t data_size, DataType dtype, ReduceOp reduce_op) {
  if (communication_wrapper_ && communication_wrapper_->initialize(getRankIndex(current_rank_), ranks_.size(), network::NetworkType::TCP_SOCKET)) {
    communication::ReductionOp reduction_op;
    switch (reduce_op) {
      case ReduceOp::SUM:
        reduction_op = communication::ReductionOp::SUM;
        break;
      case ReduceOp::MAX:
        reduction_op = communication::ReductionOp::MAX;
        break;
      case ReduceOp::MIN:
        reduction_op = communication::ReductionOp::MIN;
        break;
      default:
        reduction_op = communication::ReductionOp::SUM;
        break;
    }

    communication_wrapper_->allReduceRing(input, output, data_size, reduction_op);
    communication_wrapper_->finalize();
  } else if (allreduce_impl_) {
    AllreduceConfig config = allreduce_impl_->getConfig();
    config.reduce_op = reduce_op;
    allreduce_impl_->execute(input, output, data_size, dtype);
  } else {
    memcpy(output, input, data_size);
  }
}

void CPUProcessGroup::allgather(void* input, void* output, size_t input_size, size_t output_size) {
  if (ranks_.size() <= 1) {
    memcpy(output, input, input_size);
    return;
  }

  char* output_ptr = static_cast<char*>(output);
  const char* input_ptr = static_cast<const char*>(input);

  for (size_t i = 0; i < ranks_.size(); ++i) {
    if (static_cast<int>(i) == getRankIndex(current_rank_)) {
      memcpy(output_ptr + i * input_size, input_ptr, input_size);
    }
  }
}

void CPUProcessGroup::broadcast(void* buffer, size_t size, int root_rank) {
  if (ranks_.size() <= 1 || current_rank_ == root_rank) {
    return;
  }

  const ProcessInfo* root_info = getProcessInfo(root_rank);
  if (root_info) {
  }
}

void CPUProcessGroup::reduce(void* input, void* output, size_t size, DataType dtype, ReduceOp reduce_op, int root_rank) {
  if (ranks_.size() <= 1) {
    memcpy(output, input, size);
    return;
  }

  memcpy(output, input, size);

  switch (reduce_op) {
    case ReduceOp::SUM:
      if (dtype == DataType::FLOAT32) {
        float* output_f = reinterpret_cast<float*>(output);
        for (size_t i = 1; i < ranks_.size(); ++i) {
          output_f[0] *= static_cast<float>(ranks_.size());
        }
      }
      break;
    case ReduceOp::AVG:
      if (dtype == DataType::FLOAT32) {
        float* output_f = reinterpret_cast<float*>(output);
        *output_f /= static_cast<float>(ranks_.size());
      }
      break;
    case ReduceOp::MAX:
    case ReduceOp::MIN:
      break;
  }
}

void CPUProcessGroup::reduceScatter(void* input, void* output, size_t input_size, size_t output_size, DataType dtype, ReduceOp reduce_op) {
  if (ranks_.size() <= 1) {
    memcpy(output, input, output_size);
    return;
  }

  size_t chunk_size = input_size / ranks_.size();
  int rank_index = getRankIndex(current_rank_);
  if (rank_index >= 0 && rank_index < static_cast<int>(ranks_.size())) {
    const char* input_ptr = static_cast<const char*>(input);
    memcpy(output, input_ptr + rank_index * chunk_size, std::min(output_size, chunk_size));
  }
}

void CPUProcessGroup::send(void* buffer, size_t size, int dst_rank, int tag) {
  if (communication_wrapper_ && communication_wrapper_->initialize(getRankIndex(current_rank_), ranks_.size(), network::NetworkType::TCP_SOCKET)) {
    int local_dst = getRankIndex(dst_rank);
    communication_wrapper_->send(buffer, size, local_dst, tag);
    communication_wrapper_->finalize();
  }
}

void CPUProcessGroup::recv(void* buffer, size_t size, int src_rank, int tag) {
  if (communication_wrapper_ && communication_wrapper_->initialize(getRankIndex(current_rank_), ranks_.size(), network::NetworkType::TCP_SOCKET)) {
    int local_src = getRankIndex(src_rank);
    communication_wrapper_->recv(buffer, size, local_src, tag);
    communication_wrapper_->finalize();
  }
}

void CPUProcessGroup::barrier() {
  if (communication_wrapper_ && communication_wrapper_->initialize(getRankIndex(current_rank_), ranks_.size(), network::NetworkType::TCP_SOCKET)) {
    int dummy = 0;
    communication_wrapper_->allReduceRing(&dummy, &dummy, sizeof(int), communication::ReductionOp::SUM);
    communication_wrapper_->finalize();
  }
}

CUDAProcessGroup::CUDAProcessGroup(const std::string& name, const std::vector<int>& ranks, const std::vector<int>& device_ids)
    : ProcessGroup(name, ProcessGroupType::DEVICE_LOCAL, ranks),
      device_ids_(device_ids), current_rank_(0), current_device_(0) {
  if (!ranks.empty()) {
    current_rank_ = ranks[0];
  }
  if (!device_ids_.empty()) {
    current_device_ = device_ids_[0];
  }
}

void CUDAProcessGroup::initialize() {
  communication_wrapper_ = std::make_unique<algorithms::CommunicationWrapper>();

  AllreduceConfig config;
  config.algorithm = AllreduceAlgorithm::RING;
  config.reduce_op = ReduceOp::SUM;
  config.participants = ranks_;
  config.buffer_size = 128 * 1024 * 1024;
  config.enable_overlap = true;
  config.pipeline_depth = 4;

  allreduce_impl_ = AllreduceFactory::create(config);
}

void CUDAProcessGroup::allreduce(void* input, void* output, size_t data_size, DataType dtype, ReduceOp reduce_op) {
  if (allreduce_impl_) {
    AllreduceConfig config = allreduce_impl_->getConfig();
    config.reduce_op = reduce_op;
    allreduce_impl_->execute(input, output, data_size, dtype);
  } else {
    copyFromDevice(output, input, data_size);
  }
}

void CUDAProcessGroup::allgather(void* input, void* output, size_t input_size, size_t output_size) {
  if (ranks_.size() <= 1) {
    copyFromDevice(output, input, input_size);
    return;
  }

  void* device_input = allocateDeviceBuffer(input_size);
  void* device_output = allocateDeviceBuffer(output_size);

  copyToDevice(device_input, input, input_size);

  char* device_output_ptr = static_cast<char*>(device_output);
  char* device_input_ptr = static_cast<char*>(device_input);

  for (size_t i = 0; i < ranks_.size(); ++i) {
    if (static_cast<int>(i) == getRankIndex(current_rank_)) {
      memcpy(device_output_ptr + i * input_size, device_input_ptr, input_size);
    }
  }

  copyFromDevice(output, device_output, output_size);

  freeDeviceBuffer(device_input);
  freeDeviceBuffer(device_output);
}

void CUDAProcessGroup::broadcast(void* buffer, size_t size, int root_rank) {
  if (ranks_.size() <= 1 || current_rank_ == root_rank) {
    return;
  }

  void* device_buffer = allocateDeviceBuffer(size);
  copyToDevice(device_buffer, buffer, size);

  const ProcessInfo* root_info = getProcessInfo(root_rank);
  if (root_info) {
  }

  copyFromDevice(buffer, device_buffer, size);
  freeDeviceBuffer(device_buffer);
}

void CUDAProcessGroup::reduce(void* input, void* output, size_t size, DataType dtype, ReduceOp reduce_op, int root_rank) {
  if (ranks_.size() <= 1) {
    copyFromDevice(output, input, size);
    return;
  }

  void* device_input = allocateDeviceBuffer(size);
  void* device_output = allocateDeviceBuffer(size);

  copyToDevice(device_input, input, size);
  memcpy(device_output, device_input, size);

  switch (reduce_op) {
    case ReduceOp::SUM:
      if (dtype == DataType::FLOAT32) {
        float* output_f = reinterpret_cast<float*>(device_output);
        for (size_t i = 1; i < ranks_.size(); ++i) {
          output_f[0] *= static_cast<float>(ranks_.size());
        }
      }
      break;
    case ReduceOp::AVG:
      if (dtype == DataType::FLOAT32) {
        float* output_f = reinterpret_cast<float*>(device_output);
        *output_f /= static_cast<float>(ranks_.size());
      }
      break;
    case ReduceOp::MAX:
    case ReduceOp::MIN:
      break;
  }

  copyFromDevice(output, device_output, size);

  freeDeviceBuffer(device_input);
  freeDeviceBuffer(device_output);
}

void CUDAProcessGroup::reduceScatter(void* input, void* output, size_t input_size, size_t output_size, DataType dtype, ReduceOp reduce_op) {
  if (ranks_.size() <= 1) {
    copyFromDevice(output, input, output_size);
    return;
  }

  void* device_input = allocateDeviceBuffer(input_size);
  void* device_output = allocateDeviceBuffer(output_size);

  copyToDevice(device_input, input, input_size);

  size_t chunk_size = input_size / ranks_.size();
  int rank_index = getRankIndex(current_rank_);
  if (rank_index >= 0 && rank_index < static_cast<int>(ranks_.size())) {
    char* device_input_ptr = static_cast<char*>(device_input);
    char* device_output_ptr = static_cast<char*>(device_output);
    memcpy(device_output_ptr, device_input_ptr + rank_index * chunk_size, std::min(output_size, chunk_size));
  }

  copyFromDevice(output, device_output, output_size);

  freeDeviceBuffer(device_input);
  freeDeviceBuffer(device_output);
}

void CUDAProcessGroup::send(void* buffer, size_t size, int dst_rank, int tag) {
  if (communication_wrapper_ && communication_wrapper_->initialize(getRankIndex(current_rank_), ranks_.size(), network::NetworkType::TCP_SOCKET)) {
    int local_dst = getRankIndex(dst_rank);
    communication_wrapper_->send(buffer, size, local_dst, tag);
    communication_wrapper_->finalize();
  }
}

void CUDAProcessGroup::recv(void* buffer, size_t size, int src_rank, int tag) {
  if (communication_wrapper_ && communication_wrapper_->initialize(getRankIndex(current_rank_), ranks_.size(), network::NetworkType::TCP_SOCKET)) {
    int local_src = getRankIndex(src_rank);
    communication_wrapper_->recv(buffer, size, local_src, tag);
    communication_wrapper_->finalize();
  }
}

void CUDAProcessGroup::barrier() {
  if (communication_wrapper_ && communication_wrapper_->initialize(getRankIndex(current_rank_), ranks_.size(), network::NetworkType::TCP_SOCKET)) {
    int dummy = 0;
    communication_wrapper_->allReduceRing(&dummy, &dummy, sizeof(int), communication::ReductionOp::SUM);
    communication_wrapper_->finalize();
  }
}

void* CUDAProcessGroup::allocateDeviceBuffer(size_t size) {
  return malloc(size);
}

void CUDAProcessGroup::freeDeviceBuffer(void* buffer) {
  free(buffer);
}

void CUDAProcessGroup::copyToDevice(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void CUDAProcessGroup::copyFromDevice(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

RDMAProcessGroup::RDMAProcessGroup(const std::string& name, const std::vector<int>& ranks, const std::vector<std::string>& endpoints)
    : ProcessGroup(name, ProcessGroupType::CUSTOM, ranks), endpoints_(endpoints), current_rank_(0) {
  if (!ranks.empty()) {
    current_rank_ = ranks[0];
  }
}

void RDMAProcessGroup::initialize() {
  communication_wrapper_ = std::make_unique<algorithms::CommunicationWrapper>();

  AllreduceConfig config;
  config.algorithm = AllreduceAlgorithm::TREE;
  config.reduce_op = ReduceOp::SUM;
  config.participants = ranks_;
  config.buffer_size = 256 * 1024 * 1024;
  config.enable_overlap = true;
  config.pipeline_depth = 8;

  allreduce_impl_ = AllreduceFactory::create(config);
}

void RDMAProcessGroup::allreduce(void* input, void* output, size_t data_size, DataType dtype, ReduceOp reduce_op) {
  if (allreduce_impl_) {
    AllreduceConfig config = allreduce_impl_->getConfig();
    config.reduce_op = reduce_op;
    allreduce_impl_->execute(input, output, data_size, dtype);
  } else {
    memcpy(output, input, data_size);
  }
}

void RDMAProcessGroup::allgather(void* input, void* output, size_t input_size, size_t output_size) {
  if (ranks_.size() <= 1) {
    memcpy(output, input, input_size);
    return;
  }

  char* output_ptr = static_cast<char*>(output);
  const char* input_ptr = static_cast<const char*>(input);

  for (size_t i = 0; i < ranks_.size(); ++i) {
    if (static_cast<int>(i) == getRankIndex(current_rank_)) {
      memcpy(output_ptr + i * input_size, input_ptr, input_size);
    }
  }
}

void RDMAProcessGroup::broadcast(void* buffer, size_t size, int root_rank) {
  if (ranks_.size() <= 1 || current_rank_ == root_rank) {
    return;
  }

  const ProcessInfo* root_info = getProcessInfo(root_rank);
  if (root_info) {
  }
}

void RDMAProcessGroup::reduce(void* input, void* output, size_t size, DataType dtype, ReduceOp reduce_op, int root_rank) {
  if (ranks_.size() <= 1) {
    memcpy(output, input, size);
    return;
  }

  memcpy(output, input, size);

  switch (reduce_op) {
    case ReduceOp::SUM:
      if (dtype == DataType::FLOAT32) {
        float* output_f = reinterpret_cast<float*>(output);
        for (size_t i = 1; i < ranks_.size(); ++i) {
          output_f[0] *= static_cast<float>(ranks_.size());
        }
      }
      break;
    case ReduceOp::AVG:
      if (dtype == DataType::FLOAT32) {
        float* output_f = reinterpret_cast<float*>(output);
        *output_f /= static_cast<float>(ranks_.size());
      }
      break;
    case ReduceOp::MAX:
    case ReduceOp::MIN:
      break;
  }
}

void RDMAProcessGroup::reduceScatter(void* input, void* output, size_t input_size, size_t output_size, DataType dtype, ReduceOp reduce_op) {
  if (ranks_.size() <= 1) {
    memcpy(output, input, output_size);
    return;
  }

  size_t chunk_size = input_size / ranks_.size();
  int rank_index = getRankIndex(current_rank_);
  if (rank_index >= 0 && rank_index < static_cast<int>(ranks_.size())) {
    const char* input_ptr = static_cast<const char*>(input);
    memcpy(output, input_ptr + rank_index * chunk_size, std::min(output_size, chunk_size));
  }
}

void RDMAProcessGroup::send(void* buffer, size_t size, int dst_rank, int tag) {
  if (communication_wrapper_ && communication_wrapper_->initialize(getRankIndex(current_rank_), ranks_.size(), network::NetworkType::RDMA_VERBS)) {
    int local_dst = getRankIndex(dst_rank);
    communication_wrapper_->send(buffer, size, local_dst, tag);
    communication_wrapper_->finalize();
  }
}

void RDMAProcessGroup::recv(void* buffer, size_t size, int src_rank, int tag) {
  if (communication_wrapper_ && communication_wrapper_->initialize(getRankIndex(current_rank_), ranks_.size(), network::NetworkType::RDMA_VERBS)) {
    int local_src = getRankIndex(src_rank);
    communication_wrapper_->recv(buffer, size, local_src, tag);
    communication_wrapper_->finalize();
  }
}

void RDMAProcessGroup::barrier() {
  if (communication_wrapper_ && communication_wrapper_->initialize(getRankIndex(current_rank_), ranks_.size(), network::NetworkType::RDMA_VERBS)) {
    int dummy = 0;
    communication_wrapper_->allReduceRing(&dummy, &dummy, sizeof(int), communication::ReductionOp::SUM);
    communication_wrapper_->finalize();
  }
}

ProcessGroupManager::ProcessGroupManager() : global_rank_(0), global_size_(1) {}

void ProcessGroupManager::initialize(int global_rank, int global_size) {
  global_rank_ = global_rank;
  global_size_ = global_size;

  createGlobalProcessGroup();
}

std::shared_ptr<ProcessGroup> ProcessGroupManager::createProcessGroup(const std::string& name,
                                                                      ProcessGroupType type,
                                                                      const std::vector<int>& ranks) {
  std::shared_ptr<ProcessGroup> group;

  switch (type) {
    case ProcessGroupType::NODE_LOCAL:
      group = std::make_shared<CPUProcessGroup>(name, ranks);
      break;
    case ProcessGroupType::DEVICE_LOCAL:
      std::vector<int> device_ids;
      for (int rank : ranks) {
        device_ids.push_back(rank);
      }
      group = std::make_shared<CUDAProcessGroup>(name, ranks, device_ids);
      break;
    default:
      group = std::make_shared<CPUProcessGroup>(name, ranks);
      break;
  }

  if (topology_manager_) {
    auto topology = topology_manager_->buildOptimalTopology(ranks);
    group->setTopology(topology);
  }

  std::static_pointer_cast<CPUProcessGroup>(group)->initialize();

  process_groups_[name] = group;
  return group;
}

std::shared_ptr<ProcessGroup> ProcessGroupManager::createCPUGroup(const std::string& name, const std::vector<int>& ranks) {
  return createProcessGroup(name, ProcessGroupType::NODE_LOCAL, ranks);
}

std::shared_ptr<ProcessGroup> ProcessGroupManager::createCUDAGroup(const std::string& name,
                                                                   const std::vector<int>& ranks,
                                                                   const std::vector<int>& device_ids) {
  auto group = std::make_shared<CUDAProcessGroup>(name, ranks, device_ids);

  if (topology_manager_) {
    auto topology = topology_manager_->buildOptimalTopology(ranks);
    group->setTopology(topology);
  }

  std::static_pointer_cast<CUDAProcessGroup>(group)->initialize();

  process_groups_[name] = group;
  return group;
}

std::shared_ptr<ProcessGroup> ProcessGroupManager::createRDMAGroup(const std::string& name,
                                                                   const std::vector<int>& ranks,
                                                                   const std::vector<std::string>& endpoints) {
  auto group = std::make_shared<RDMAProcessGroup>(name, ranks, endpoints);

  if (topology_manager_) {
    auto topology = topology_manager_->buildOptimalTopology(ranks);
    group->setTopology(topology);
  }

  std::static_pointer_cast<RDMAProcessGroup>(group)->initialize();

  process_groups_[name] = group;
  return group;
}

std::shared_ptr<ProcessGroup> ProcessGroupManager::getProcessGroup(const std::string& name) const {
  auto it = process_groups_.find(name);
  return (it != process_groups_.end()) ? it->second : nullptr;
}

void ProcessGroupManager::destroyProcessGroup(const std::string& name) {
  process_groups_.erase(name);
}

std::vector<std::string> ProcessGroupManager::getProcessGroupNames() const {
  std::vector<std::string> names;
  for (const auto& pair : process_groups_) {
    names.push_back(pair.first);
  }
  return names;
}

void ProcessGroupManager::setTopologyManager(std::shared_ptr<TopologyManager> topology_manager) {
  topology_manager_ = topology_manager;
}

void ProcessGroupManager::createGlobalProcessGroup() {
  std::vector<int> global_ranks;
  for (int i = 0; i < global_size_; ++i) {
    global_ranks.push_back(i);
  }

  global_group_ = createProcessGroup("global", ProcessGroupType::GLOBAL, global_ranks);
  process_groups_["global"] = global_group_;
}

}