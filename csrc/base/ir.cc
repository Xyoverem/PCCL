#include <base/ir.h>
#include <algorithm>
#include <stack>
#include <queue>

namespace engine_c {

int Op::next_op_id_ = 0;

Value::Value(DataType dtype, const std::vector<int>& shape, int device_id)
    : dtype_(dtype), shape_(shape), device_id_(device_id) {}

Value::Value(const Value& other)
    : dtype_(other.dtype_), shape_(other.shape_), device_id_(other.device_id_) {}

Value& Value::operator=(const Value& other) {
  if (this != &other) {
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    device_id_ = other.device_id_;
  }
  return *this;
}

size_t Value::getSize() const {
  size_t size = 1;
  for (int dim : shape_) {
    size *= dim;
  }
  switch (dtype_) {
    case DataType::FLOAT32:
    case DataType::INT32:
      return size * 4;
    case DataType::FLOAT64:
    case DataType::INT64:
      return size * 8;
    case DataType::FLOAT16:
    case DataType::INT16:
      return size * 2;
    case DataType::INT8:
    case DataType::UINT8:
      return size;
    default:
      return size * 4;
  }
}

Op::Op(OpType op_type, const std::string& name)
    : op_type_(op_type), name_(name), op_id_(next_op_id_++) {}

void Op::addInput(std::shared_ptr<Value> input) {
  inputs_.push_back(input);
}

void Op::addOutput(std::shared_ptr<Value> output) {
  outputs_.push_back(output);
}

void Op::addDependency(int op_id) {
  dependencies_.push_back(op_id);
}

CopyOp::CopyOp(int src_device, int dst_device)
    : Op(OpType::COPY, "copy"), src_device_(src_device), dst_device_(dst_device) {}

void CopyOp::execute() {
}

ReduceOpImpl::ReduceOpImpl(ReduceOp reduce_op, const std::vector<int>& participants)
    : Op(OpType::REDUCE, "reduce"), reduce_op_(reduce_op), participants_(participants) {}

void ReduceOpImpl::execute() {
}

AllreduceOp::AllreduceOp(ReduceOp reduce_op, const std::vector<int>& participants)
    : Op(OpType::ALLREDUCE, "allreduce"), reduce_op_(reduce_op), participants_(participants) {}

void AllreduceOp::execute() {
}

IRBuilder::IRBuilder() : next_value_id_(0) {}

int IRBuilder::addOp(std::shared_ptr<Op> op) {
  int op_id = op->getOpId();
  ops_[op_id] = op;
  adjacency_list_[op_id] = {};
  reverse_adjacency_list_[op_id] = {};
  return op_id;
}

void IRBuilder::addDependency(int src_op_id, int dst_op_id) {
  adjacency_list_[src_op_id].push_back(dst_op_id);
  reverse_adjacency_list_[dst_op_id].push_back(src_op_id);

  auto dst_op = getOp(dst_op_id);
  if (dst_op) {
    dst_op->addDependency(src_op_id);
  }
}

void IRBuilder::addValue(std::shared_ptr<Value> value) {
  int value_id = next_value_id_++;
  values_[value_id] = value;
}

std::shared_ptr<Op> IRBuilder::getOp(int op_id) {
  auto it = ops_.find(op_id);
  return (it != ops_.end()) ? it->second : nullptr;
}

std::shared_ptr<Value> IRBuilder::getValue(int value_id) {
  auto it = values_.find(value_id);
  return (it != values_.end()) ? it->second : nullptr;
}

std::shared_ptr<CopyOp> IRBuilder::createCopyOp(int src_device, int dst_device) {
  auto op = std::make_shared<CopyOp>(src_device, dst_device);
  addOp(op);
  return op;
}

std::shared_ptr<ReduceOpImpl> IRBuilder::createReduceOp(ReduceOp reduce_op, const std::vector<int>& participants) {
  auto op = std::make_shared<ReduceOpImpl>(reduce_op, participants);
  addOp(op);
  return op;
}

std::shared_ptr<AllreduceOp> IRBuilder::createAllreduceOp(ReduceOp reduce_op, const std::vector<int>& participants) {
  auto op = std::make_shared<AllreduceOp>(reduce_op, participants);
  addOp(op);
  return op;
}

bool IRBuilder::validateGraph() const {
  return !hasCycle();
}

bool IRBuilder::hasCycle() const {
  std::unordered_set<int> visited;
  std::unordered_set<int> rec_stack;

  for (const auto& pair : ops_) {
    int op_id = pair.first;
    if (visited.find(op_id) == visited.end()) {
      if (dfsCycle(op_id, visited, rec_stack)) {
        return true;
      }
    }
  }
  return false;
}

bool IRBuilder::dfsCycle(int op_id, std::unordered_set<int>& visited,
                        std::unordered_set<int>& rec_stack) const {
  visited.insert(op_id);
  rec_stack.insert(op_id);

  auto it = adjacency_list_.find(op_id);
  if (it != adjacency_list_.end()) {
    for (int neighbor_id : it->second) {
      if (visited.find(neighbor_id) == visited.end()) {
        if (dfsCycle(neighbor_id, visited, rec_stack)) {
          return true;
        }
      } else if (rec_stack.find(neighbor_id) != rec_stack.end()) {
        return true;
      }
    }
  }

  rec_stack.erase(op_id);
  return false;
}

GraphExecutor::GraphExecutor() : completed_(false) {}

void GraphExecutor::setGraph(std::shared_ptr<IRBuilder> graph) {
  graph_ = graph;
  computeExecutionOrder();
  completed_ops_.clear();
  pending_dependencies_.clear();

  for (const auto& pair : graph_->getOps()) {
    int op_id = pair.first;
    pending_dependencies_[op_id] = pair.second->getDependencies().size();
  }
}

void GraphExecutor::execute() {
  if (!graph_) return;

  computeExecutionOrder();

  for (int op_id : execution_order_) {
    executeOp(op_id);
  }

  completed_ = true;
}

void GraphExecutor::executeAsync() {
}

void GraphExecutor::waitCompletion() {
  while (!isCompleted()) {
  }
}

void GraphExecutor::computeExecutionOrder() {
  if (!graph_) return;

  execution_order_.clear();
  std::unordered_set<int> visited;
  std::stack<int> stack;

  std::function<void(int)> topologicalSort = [&](int op_id) {
    if (visited.find(op_id) != visited.end()) return;

    visited.insert(op_id);

    auto adjacency_it = graph_->getOps().find(op_id);
    if (adjacency_it != graph_->getOps().end()) {
      for (int dep_id : adjacency_it->second->getDependencies()) {
        topologicalSort(dep_id);
      }
    }

    stack.push(op_id);
  };

  for (const auto& pair : graph_->getOps()) {
    topologicalSort(pair.first);
  }

  while (!stack.empty()) {
    execution_order_.push_back(stack.top());
    stack.pop();
  }
}

void GraphExecutor::executeOp(int op_id) {
  auto op = graph_->getOp(op_id);
  if (op) {
    op->execute();
    completed_ops_.insert(op_id);
  }
}

}