#pragma once

#include <base/registry.h>
#include <base/ir.h>
#include <functional>
#include <memory>
#include <unordered_map>
#include <string>

namespace engine_c {

struct RingBuffer {
  ExecutorType producer_;
  ExecutorType consumer_;
  int elem_size_;
  int max_num_elems_;
  int max_flush_time_;
  int tail_;
  int head_;
  void *producer_buffer_;
  void *consumer_buffer_;
};

struct OperationLayoutHeader {
  int op_uid_;
  int required_executors_;
  int remaining_executors_;
  unsigned char pre_dependencies_;
  ExecutorType executor_type_;
  unsigned char op_size_;
  unsigned char num_next_ops_;
  void *op_buffer_;
  int *next_op_uids_head_;
};

struct GraphBufferLayout {
  unsigned int *completed_operator;
  unsigned int *total_operator;
  OperationLayoutHeader **operators;
  unsigned int num_operators;
  RingBuffer **ready_queues;
  unsigned int num_queues;
};

struct OperatorInfo {
  std::string name;
  OpType op_type;
  ExecutorType executor_type;
  std::function<std::shared_ptr<Op>(const std::unordered_map<std::string, std::string>&)> factory;
  std::unordered_map<std::string, std::string> default_params;
};

struct ExecutionContext {
  int rank;
  int world_size;
  class ClusterManager* cluster;
  class BufferManager* buffers;

  ExecutionContext() : rank(0), world_size(1), cluster(nullptr), buffers(nullptr) {}
};

class OperatorManager {
public:
  OperatorManager();
  ~OperatorManager() = default;

  void registerOperator(const OperatorInfo& info);
  void registerOperator(const std::string& name, OpType op_type, ExecutorType executor_type,
                       std::function<std::shared_ptr<Op>(const std::unordered_map<std::string, std::string>&)> factory,
                       const std::unordered_map<std::string, std::string>& default_params = {});

  bool isOperatorRegistered(const std::string& name) const;

  std::shared_ptr<Op> createOperator(const std::string& name, const std::unordered_map<std::string, std::string>& params = {});

  const OperatorInfo* getOperatorInfo(const std::string& name) const;

  std::vector<std::string> getRegisteredOperators() const;

  void executeOperator(const std::string& name, const std::unordered_map<std::string, std::string>& params = {});

  void loadOperatorFromFile(const std::string& filepath);

  // New methods for Engine integration
  void registerOperator(const std::string& name, const std::string& filepath);
  std::shared_ptr<Op> getOperator(const std::string& name);
  void setClusterInfo(const std::unordered_map<std::string, std::string>& cluster_info);

  void clear();

private:
  std::unordered_map<std::string, OperatorInfo> operators_;
  std::unordered_map<std::string, std::string> cluster_info_;

  void initializeBuiltInOperators();
};

}