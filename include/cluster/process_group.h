#pragma once

#include <base/registry.h>
#include <base/ir.h>
#include <algorithms/allreduce.h>
#include <algorithms/communication_wrapper.h>
#include <topology/topology.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <functional>

namespace engine_c {

struct ProcessInfo {
  int rank;
  std::string hostname;
  std::string endpoint;
  DeviceType device_type;
  int device_id;
  bool is_local;
};

enum class ProcessGroupType {
  GLOBAL,
  NODE_LOCAL,
  DEVICE_LOCAL,
  CUSTOM
};

class ProcessGroup {
public:
  ProcessGroup(const std::string& name, ProcessGroupType type, const std::vector<int>& ranks);
  virtual ~ProcessGroup() = default;

  const std::string& getName() const { return name_; }
  ProcessGroupType getType() const { return type_; }
  const std::vector<int>& getRanks() const { return ranks_; }
  int getSize() const { return ranks_.size(); }

  void addProcess(int rank, const ProcessInfo& info);
  void removeProcess(int rank);
  const ProcessInfo* getProcessInfo(int rank) const;

  virtual void allreduce(void* input, void* output, size_t data_size, DataType dtype, ReduceOp reduce_op) = 0;
  virtual void allgather(void* input, void* output, size_t input_size, size_t output_size) = 0;
  virtual void broadcast(void* buffer, size_t size, int root_rank) = 0;
  virtual void reduce(void* input, void* output, size_t size, DataType dtype, ReduceOp reduce_op, int root_rank) = 0;
  virtual void reduceScatter(void* input, void* output, size_t input_size, size_t output_size, DataType dtype, ReduceOp reduce_op) = 0;
  virtual void send(void* buffer, size_t size, int dst_rank, int tag) = 0;
  virtual void recv(void* buffer, size_t size, int src_rank, int tag) = 0;

  virtual void barrier() = 0;

  bool isRankInGroup(int rank) const;
  int getRankIndex(int rank) const;

  void setTopology(std::shared_ptr<Topology> topology) { topology_ = topology; }
  std::shared_ptr<Topology> getTopology() const { return topology_; }

protected:
  std::string name_;
  ProcessGroupType type_;
  std::vector<int> ranks_;
  std::unordered_map<int, ProcessInfo> processes_;
  std::shared_ptr<Topology> topology_;

  virtual void initialize() = 0;
};

class CPUProcessGroup : public ProcessGroup {
public:
  CPUProcessGroup(const std::string& name, const std::vector<int>& ranks);
  ~CPUProcessGroup() = default;

  void allreduce(void* input, void* output, size_t data_size, DataType dtype, ReduceOp reduce_op) override;
  void allgather(void* input, void* output, size_t input_size, size_t output_size) override;
  void broadcast(void* buffer, size_t size, int root_rank) override;
  void reduce(void* input, void* output, size_t size, DataType dtype, ReduceOp reduce_op, int root_rank) override;
  void reduceScatter(void* input, void* output, size_t input_size, size_t output_size, DataType dtype, ReduceOp reduce_op) override;
  void send(void* buffer, size_t size, int dst_rank, int tag) override;
  void recv(void* buffer, size_t size, int src_rank, int tag) override;

  void barrier() override;

protected:
  void initialize() override;

private:
  std::unique_ptr<AllreduceImpl> allreduce_impl_;
  std::unique_ptr<algorithms::CommunicationWrapper> communication_wrapper_;
  int current_rank_;
};

class CUDAProcessGroup : public ProcessGroup {
public:
  CUDAProcessGroup(const std::string& name, const std::vector<int>& ranks, const std::vector<int>& device_ids);
  ~CUDAProcessGroup() = default;

  void allreduce(void* input, void* output, size_t data_size, DataType dtype, ReduceOp reduce_op) override;
  void allgather(void* input, void* output, size_t input_size, size_t output_size) override;
  void broadcast(void* buffer, size_t size, int root_rank) override;
  void reduce(void* input, void* output, size_t size, DataType dtype, ReduceOp reduce_op, int root_rank) override;
  void reduceScatter(void* input, void* output, size_t input_size, size_t output_size, DataType dtype, ReduceOp reduce_op) override;
  void send(void* buffer, size_t size, int dst_rank, int tag) override;
  void recv(void* buffer, size_t size, int src_rank, int tag) override;

  void barrier() override;

protected:
  void initialize() override;

private:
  std::vector<int> device_ids_;
  std::unique_ptr<AllreduceImpl> allreduce_impl_;
  std::unique_ptr<algorithms::CommunicationWrapper> communication_wrapper_;
  int current_rank_;
  int current_device_;

  void* allocateDeviceBuffer(size_t size);
  void freeDeviceBuffer(void* buffer);
  void copyToDevice(void* dst, const void* src, size_t size);
  void copyFromDevice(void* dst, const void* src, size_t size);
};

class RDMAProcessGroup : public ProcessGroup {
public:
  RDMAProcessGroup(const std::string& name, const std::vector<int>& ranks, const std::vector<std::string>& endpoints);
  ~RDMAProcessGroup() = default;

  void allreduce(void* input, void* output, size_t data_size, DataType dtype, ReduceOp reduce_op) override;
  void allgather(void* input, void* output, size_t input_size, size_t output_size) override;
  void broadcast(void* buffer, size_t size, int root_rank) override;
  void reduce(void* input, void* output, size_t size, DataType dtype, ReduceOp reduce_op, int root_rank) override;
  void reduceScatter(void* input, void* output, size_t input_size, size_t output_size, DataType dtype, ReduceOp reduce_op) override;
  void send(void* buffer, size_t size, int dst_rank, int tag) override;
  void recv(void* buffer, size_t size, int src_rank, int tag) override;

  void barrier() override;

protected:
  void initialize() override;

private:
  std::vector<std::string> endpoints_;
  std::unique_ptr<AllreduceImpl> allreduce_impl_;
  std::unique_ptr<algorithms::CommunicationWrapper> communication_wrapper_;
  int current_rank_;
};

class ProcessGroupManager {
public:
  ProcessGroupManager();
  ~ProcessGroupManager() = default;

  void initialize(int global_rank, int global_size);

  std::shared_ptr<ProcessGroup> createProcessGroup(const std::string& name,
                                                   ProcessGroupType type,
                                                   const std::vector<int>& ranks);

  std::shared_ptr<ProcessGroup> createCPUGroup(const std::string& name, const std::vector<int>& ranks);
  std::shared_ptr<ProcessGroup> createCUDAGroup(const std::string& name,
                                                const std::vector<int>& ranks,
                                                const std::vector<int>& device_ids);
  std::shared_ptr<ProcessGroup> createRDMAGroup(const std::string& name,
                                                const std::vector<int>& ranks,
                                                const std::vector<std::string>& endpoints);

  std::shared_ptr<ProcessGroup> getProcessGroup(const std::string& name) const;
  std::shared_ptr<ProcessGroup> getGlobalProcessGroup() const { return global_group_; }

  void destroyProcessGroup(const std::string& name);

  std::vector<std::string> getProcessGroupNames() const;

  void setTopologyManager(std::shared_ptr<class TopologyManager> topology_manager);

  int getGlobalRank() const { return global_rank_; }
  int getGlobalSize() const { return global_size_; }

private:
  int global_rank_;
  int global_size_;
  std::shared_ptr<ProcessGroup> global_group_;
  std::unordered_map<std::string, std::shared_ptr<ProcessGroup>> process_groups_;
  std::shared_ptr<class TopologyManager> topology_manager_;

  void createGlobalProcessGroup();
};

}