#include <base/operator.h>
#include <fstream>
#include <sstream>
#include <iostream>

namespace engine_c {

OperatorManager::OperatorManager() {
  initializeBuiltInOperators();
}

void OperatorManager::registerOperator(const OperatorInfo& info) {
  operators_[info.name] = info;
}

void OperatorManager::registerOperator(const std::string& name, OpType op_type, ExecutorType executor_type,
                                      std::function<std::shared_ptr<Op>(const std::unordered_map<std::string, std::string>&)> factory,
                                      const std::unordered_map<std::string, std::string>& default_params) {
  OperatorInfo info;
  info.name = name;
  info.op_type = op_type;
  info.executor_type = executor_type;
  info.factory = factory;
  info.default_params = default_params;

  operators_[name] = info;
}

bool OperatorManager::isOperatorRegistered(const std::string& name) const {
  return operators_.find(name) != operators_.end();
}

std::shared_ptr<Op> OperatorManager::createOperator(const std::string& name, const std::unordered_map<std::string, std::string>& params) {
  auto it = operators_.find(name);
  if (it == operators_.end()) {
    return nullptr;
  }

  const OperatorInfo& info = it->second;
  std::unordered_map<std::string, std::string> merged_params = info.default_params;
  for (const auto& param : params) {
    merged_params[param.first] = param.second;
  }

  return info.factory(merged_params);
}

const OperatorInfo* OperatorManager::getOperatorInfo(const std::string& name) const {
  auto it = operators_.find(name);
  if (it != operators_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::vector<std::string> OperatorManager::getRegisteredOperators() const {
  std::vector<std::string> names;
  names.reserve(operators_.size());

  for (const auto& pair : operators_) {
    names.push_back(pair.first);
  }

  return names;
}

void OperatorManager::executeOperator(const std::string& name, const std::unordered_map<std::string, std::string>& params) {
  auto op = createOperator(name, params);
  if (op) {
    op->execute();
  }
}

void OperatorManager::loadOperatorFromFile(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    return;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    std::string name, op_type_str, executor_type_str;

    if (!(iss >> name >> op_type_str >> executor_type_str)) continue;

    OpType op_type = OpType::COPY;
    ExecutorType executor_type = ExecutorType::CPU;

    if (op_type_str == "ALLREDUCE") {
      op_type = OpType::ALLREDUCE;
    } else if (op_type_str == "ALLGATHER") {
      op_type = OpType::ALLGATHER;
    } else if (op_type_str == "BROADCAST") {
      op_type = OpType::BROADCAST;
    } else if (op_type_str == "REDUCE") {
      op_type = OpType::REDUCE;
    }

    registerOperator(name, op_type, executor_type, nullptr);
  }
}

void OperatorManager::clear() {
  operators_.clear();
}

void OperatorManager::initializeBuiltInOperators() {
  registerOperator("copy", OpType::COPY, ExecutorType::CPU,
    [](const std::unordered_map<std::string, std::string>& params) -> std::shared_ptr<Op> {
      int src_device = 0;
      int dst_device = 1;

      auto src_it = params.find("src_device");
      if (src_it != params.end()) {
        src_device = std::stoi(src_it->second);
      }

      auto dst_it = params.find("dst_device");
      if (dst_it != params.end()) {
        dst_device = std::stoi(dst_it->second);
      }

      return std::make_shared<CopyOp>(src_device, dst_device);
    },
    {{"src_device", "0"}, {"dst_device", "1"}});

  registerOperator("allreduce", OpType::ALLREDUCE, ExecutorType::CPU,
    [](const std::unordered_map<std::string, std::string>& params) -> std::shared_ptr<Op> {
      ReduceOp reduce_op = ReduceOp::SUM;
      std::string participants_str = "0,1";

      auto reduce_it = params.find("reduce_op");
      if (reduce_it != params.end()) {
        if (reduce_it->second == "AVG") {
          reduce_op = ReduceOp::AVG;
        } else if (reduce_it->second == "MAX") {
          reduce_op = ReduceOp::MAX;
        } else if (reduce_it->second == "MIN") {
          reduce_op = ReduceOp::MIN;
        }
      }

      auto participants_it = params.find("participants");
      if (participants_it != params.end()) {
        participants_str = participants_it->second;
      }

      std::vector<int> participants;
      std::istringstream iss(participants_str);
      std::string token;
      while (std::getline(iss, token, ',')) {
        participants.push_back(std::stoi(token));
      }

      return std::make_shared<AllreduceOp>(reduce_op, participants);
    },
    {{"reduce_op", "SUM"}, {"participants", "0,1"}});

  registerOperator("reduce", OpType::REDUCE, ExecutorType::CPU,
    [](const std::unordered_map<std::string, std::string>& params) -> std::shared_ptr<Op> {
      ReduceOp reduce_op = ReduceOp::SUM;
      std::string participants_str = "0,1";

      auto reduce_it = params.find("reduce_op");
      if (reduce_it != params.end()) {
        if (reduce_it->second == "AVG") {
          reduce_op = ReduceOp::AVG;
        } else if (reduce_it->second == "MAX") {
          reduce_op = ReduceOp::MAX;
        } else if (reduce_it->second == "MIN") {
          reduce_op = ReduceOp::MIN;
        }
      }

      auto participants_it = params.find("participants");
      if (participants_it != params.end()) {
        participants_str = participants_it->second;
      }

      std::vector<int> participants;
      std::istringstream iss(participants_str);
      std::string token;
      while (std::getline(iss, token, ',')) {
        participants.push_back(std::stoi(token));
      }

      return std::make_shared<ReduceOpImpl>(reduce_op, participants);
    },
    {{"reduce_op", "SUM"}, {"participants", "0,1"}});
}

// New methods for Engine integration
void OperatorManager::registerOperator(const std::string& name, const std::string& filepath) {
  // Load operator configuration from file
  loadOperatorFromFile(filepath);

  // Verify the operator was loaded
  if (!isOperatorRegistered(name)) {
    throw std::runtime_error("Failed to load operator '" + name + "' from file: " + filepath);
  }
}

std::shared_ptr<Op> OperatorManager::getOperator(const std::string& name) {
  return createOperator(name);
}

void OperatorManager::setClusterInfo(const std::unordered_map<std::string, std::string>& cluster_info) {
  // Store cluster information for operator configuration
  // This can be used to modify operator behavior based on cluster topology
  cluster_info_ = cluster_info;
}

void OperatorManager::clear() {
  operators_.clear();
  cluster_info_.clear();
}

}