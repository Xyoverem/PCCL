#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <base/registry.h>

namespace engine_c {

enum class OpType {
  COPY,
  REDUCE,
  NOTIFY,
  GET_NOTIFIED,
  ALLREDUCE,
  ALLGATHER,
  BROADCAST,
  REDUCE_SCATTER
};

enum class ReduceOp {
  SUM,
  AVG,
  MAX,
  MIN
};

class Value {
public:
  Value(DataType dtype, const std::vector<int>& shape, int device_id);
  Value(const Value& other);
  Value& operator=(const Value& other);

  DataType getDataType() const { return dtype_; }
  const std::vector<int>& getShape() const { return shape_; }
  int getDeviceId() const { return device_id_; }
  size_t getSize() const;

  void setDataType(DataType dtype) { dtype_ = dtype; }
  void setShape(const std::vector<int>& shape) { shape_ = shape; }
  void setDeviceId(int device_id) { device_id_ = device_id; }

private:
  DataType dtype_;
  std::vector<int> shape_;
  int device_id_;
};

class Op {
public:
  Op(OpType op_type, const std::string& name);
  virtual ~Op() = default;

  OpType getOpType() const { return op_type_; }
  const std::string& getName() const { return name_; }
  int getOpId() const { return op_id_; }

  void addInput(std::shared_ptr<Value> input);
  void addOutput(std::shared_ptr<Value> output);
  void addDependency(int op_id);

  const std::vector<std::shared_ptr<Value>>& getInputs() const { return inputs_; }
  const std::vector<std::shared_ptr<Value>>& getOutputs() const { return outputs_; }
  const std::vector<int>& getDependencies() const { return dependencies_; }

  virtual void execute() = 0;

protected:
  OpType op_type_;
  std::string name_;
  int op_id_;
  std::vector<std::shared_ptr<Value>> inputs_;
  std::vector<std::shared_ptr<Value>> outputs_;
  std::vector<int> dependencies_;

  static int next_op_id_;
};

class CopyOp : public Op {
public:
  CopyOp(int src_device, int dst_device);
  void execute() override;

  int getSrcDevice() const { return src_device_; }
  int getDstDevice() const { return dst_device_; }

private:
  int src_device_;
  int dst_device_;
};

class ReduceOpImpl : public Op {
public:
  ReduceOpImpl(ReduceOp reduce_op, const std::vector<int>& participants);
  void execute() override;

  ReduceOp getReduceOp() const { return reduce_op_; }
  const std::vector<int>& getParticipants() const { return participants_; }

private:
  ReduceOp reduce_op_;
  std::vector<int> participants_;
};

class AllreduceOp : public Op {
public:
  AllreduceOp(ReduceOp reduce_op, const std::vector<int>& participants);
  void execute() override;

  ReduceOp getReduceOp() const { return reduce_op_; }
  const std::vector<int>& getParticipants() const { return participants_; }

private:
  ReduceOp reduce_op_;
  std::vector<int> participants_;
};

class IRBuilder {
public:
  IRBuilder();
  ~IRBuilder() = default;

  int addOp(std::shared_ptr<Op> op);
  void addDependency(int src_op_id, int dst_op_id);
  void addValue(std::shared_ptr<Value> value);

  std::shared_ptr<Op> getOp(int op_id);
  std::shared_ptr<Value> getValue(int value_id);

  const std::unordered_map<int, std::shared_ptr<Op>>& getOps() const { return ops_; }
  const std::unordered_map<int, std::shared_ptr<Value>>& getValues() const { return values_; }

  std::shared_ptr<CopyOp> createCopyOp(int src_device, int dst_device);
  std::shared_ptr<ReduceOpImpl> createReduceOp(ReduceOp reduce_op, const std::vector<int>& participants);
  std::shared_ptr<AllreduceOp> createAllreduceOp(ReduceOp reduce_op, const std::vector<int>& participants);

  bool validateGraph() const;

private:
  std::unordered_map<int, std::shared_ptr<Op>> ops_;
  std::unordered_map<int, std::shared_ptr<Value>> values_;
  std::unordered_map<int, std::vector<int>> adjacency_list_;
  std::unordered_map<int, std::vector<int>> reverse_adjacency_list_;

  int next_value_id_;

  bool hasCycle() const;
  void dfsCycle(int op_id, std::unordered_set<int>& visited,
                std::unordered_set<int>& rec_stack) const;
};

class GraphExecutor {
public:
  GraphExecutor();
  ~GraphExecutor() = default;

  void setGraph(std::shared_ptr<IRBuilder> graph);
  void execute();

  void executeAsync();
  void waitCompletion();

  bool isCompleted() const { return completed_; }

  std::shared_ptr<IRBuilder> getGraph() const { return graph_; }

private:
  std::shared_ptr<IRBuilder> graph_;
  std::vector<int> execution_order_;
  bool completed_;

  void computeExecutionOrder();
  void executeOp(int op_id);

  std::unordered_set<int> completed_ops_;
  std::unordered_map<int, int> pending_dependencies_;
};

}