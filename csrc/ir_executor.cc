#include "ir_executor.h"
#include "base/engine.h"
#include "base/device.h"
#include "plugins/cuda_executor/executor.h"
#include "plugins/rdma_executor/executor.h"
#include "plugins/cpu_executor/executor.h"
#include <chrono>
#include <algorithm>
#include <stdexcept>

namespace pccl {

IRGraphExecutor::IRGraphExecutor() : is_ready_(false) {
    op_dispatch_["write"] = [this](const std::shared_ptr<IROperation>& op,
                                   const ExecutionContext& ctx) {
        return executeWriteOp(op, ctx);
    };
    op_dispatch_["reduce"] = [this](const std::shared_ptr<IROperation>& op,
                                    const ExecutionContext& ctx) {
        return executeReduceOp(op, ctx);
    };
    op_dispatch_["copy"] = [this](const std::shared_ptr<IROperation>& op,
                                  const ExecutionContext& ctx) {
        return executeCopyOp(op, ctx);
    };
    op_dispatch_["signal"] = [this](const std::shared_ptr<IROperation>& op,
                                    const ExecutionContext& ctx) {
        return executeSignalOp(op, ctx);
    };
    op_dispatch_["wait_signal"] = [this](const std::shared_ptr<IROperation>& op,
                                         const ExecutionContext& ctx) {
        return executeWaitSignalOp(op, ctx);
    };

    reset();
}

IRGraphExecutor::~IRGraphExecutor() {
    reset();
}

bool IRGraphExecutor::parseIRGraph(const std::string& json_graph) {
    reset();

    if (!parseJSON(json_graph)) {
        return false;
    }

    if (!validateDependencies()) {
        setError("Invalid operation dependencies");
        return false;
    }

    is_ready_ = true;
    return true;
}

bool IRGraphExecutor::executeGraph(const ExecutionContext& context) {
    if (!is_ready_) {
        setError("IR graph not parsed or ready for execution");
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    auto execution_order = getExecutionOrder();
    stats_.num_operations = operations_.size();
    stats_.num_values = values_.size();

    for (const auto& op : execution_order) {
        auto op_start = std::chrono::high_resolution_clock::now();

        if (!validateOperation(op)) {
            setError("Invalid operation: " + op->id);
            return false;
        }

        auto it = op_dispatch_.find(op->op_type);
        if (it == op_dispatch_.end()) {
            setError("Unsupported operation type: " + op->op_type);
            return false;
        }

        if (!it->second(op, context)) {
            setError("Failed to execute operation: " + op->id);
            return false;
        }

        auto op_end = std::chrono::high_resolution_clock::now();
        auto op_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            op_end - op_start);
        updateStats(op->op_type, op_duration.count() / 1000.0);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    stats_.execution_time_ms = total_duration.count() / 1000.0;
    stats_.success = true;

    return true;
}

bool IRGraphExecutor::executeGraphAsync(const ExecutionContext& context) {
    ExecutionContext async_context = context;
    async_context.async_execution = true;
    return executeGraph(async_context);
}

ExecutionStats IRGraphExecutor::getStatistics() const {
    return stats_;
}

void IRGraphExecutor::reset() {
    operations_.clear();
    values_.clear();
    graph_metadata_.clear();
    stats_ = ExecutionStats{};
    is_ready_ = false;
    last_error_.clear();
}

bool IRGraphExecutor::parseJSON(const std::string& json_str) {
    Json::Value root;
    Json::Reader reader;

    if (!reader.parse(json_str, root)) {
        setError("Failed to parse JSON: " + reader.getFormattedErrorMessages());
        return false;
    }

    if (root.isMember("metadata")) {
        graph_metadata_ = root["metadata"];
    }

    if (root.isMember("values")) {
        const Json::Value& values = root["values"];
        for (const auto& key : values.getMemberNames()) {
            auto value = convertJsonValue(values[key]);
            if (value) {
                values_[key] = value;
            } else {
                setError("Failed to convert value: " + key);
                return false;
            }
        }
    }

    if (root.isMember("operations")) {
        const Json::Value& operations = root["operations"];
        for (const auto& key : operations.getMemberNames()) {
            auto op = convertJsonOp(operations[key]);
            if (op) {
                op->id = key;
                operations_[key] = op;
            } else {
                setError("Failed to convert operation: " + key);
                return false;
            }
        }
    }

    return true;
}

std::shared_ptr<IRGraphExecutor::IROperation> IRGraphExecutor::convertJsonOp(
    const Json::Value& json_op) {
    auto op = std::make_shared<IROperation>();

    if (json_op.isMember("op_type")) {
        op->op_type = json_op["op_type"].asString();
    }

    if (json_op.isMember("inputs")) {
        const Json::Value& inputs = json_op["inputs"];
        if (inputs.isArray()) {
            for (const auto& input : inputs) {
                op->inputs.push_back(input.asString());
            }
        }
    }

    if (json_op.isMember("outputs")) {
        const Json::Value& outputs = json_op["outputs"];
        if (outputs.isArray()) {
            for (const auto& output : outputs) {
                op->outputs.push_back(output.asString());
            }
        }
    }

    if (json_op.isMember("attributes")) {
        const Json::Value& attributes = json_op["attributes"];
        for (const auto& key : attributes.getMemberNames()) {
            op->attributes[key] = attributes[key];
        }
    }

    if (json_op.isMember("metadata")) {
        const Json::Value& metadata = json_op["metadata"];
        for (const auto& key : metadata.getMemberNames()) {
            op->metadata[key] = metadata[key];
        }
    }

    return op;
}

std::shared_ptr<IRGraphExecutor::IRValue> IRGraphExecutor::convertJsonValue(
    const Json::Value& json_value) {
    auto value = std::make_shared<IRValue>();

    if (json_value.isMember("id")) {
        value->id = json_value["id"].asString();
    }

    if (json_value.isMember("dtype")) {
        value->dtype = json_value["dtype"].asString();
    }

    if (json_value.isMember("shape")) {
        const Json::Value& shape = json_value["shape"];
        if (shape.isArray()) {
            for (const auto& dim : shape) {
                value->shape.push_back(dim.asInt());
            }
        }
    }

    if (json_value.isMember("device_id")) {
        value->device_id = json_value["device_id"].asInt();
    }

    if (json_value.isMember("device_type")) {
        value->device_type = json_value["device_type"].asString();
    }

    if (json_value.isMember("metadata")) {
        const Json::Value& metadata = json_value["metadata"];
        for (const auto& key : metadata.getMemberNames()) {
            value->metadata[key] = metadata[key];
        }
    }

    return value;
}

bool IRGraphExecutor::dispatchHardwareOp(const std::shared_ptr<IROperation>& op,
                                        const ExecutionContext& context) {
    auto it = op_dispatch_.find(op->op_type);
    if (it == op_dispatch_.end()) {
        setError("Unknown operation type: " + op->op_type);
        return false;
    }
    return it->second(op, context);
}

bool IRGraphExecutor::executeWriteOp(const std::shared_ptr<IROperation>& op,
                                    const ExecutionContext& context) {
    if (op->inputs.size() != 1 || op->outputs.size() != 1) {
        setError("Write operation requires exactly 1 input and 1 output");
        return false;
    }

    if (context.device_type == "cuda") {
        auto device_id_it = op->attributes.find("device_id");
        int device_id = (device_id_it != op->attributes.end()) ?
                       std::stoi(device_id_it->second) : context.device_id;

        engine_c::CudaExecutor cuda_executor(device_id);
        cuda_executor.initialize();
    }

    return true;
}

bool IRGraphExecutor::executeReduceOp(const std::shared_ptr<IROperation>& op,
                                     const ExecutionContext& context) {
    if (op->inputs.size() < 2 || op->outputs.size() != 1) {
        setError("Reduce operation requires at least 2 inputs and 1 output");
        return false;
    }

    auto reduce_op_it = op->attributes.find("reduce_op");
    if (reduce_op_it == op->attributes.end()) {
        setError("Reduce operation missing 'reduce_op' attribute");
        return false;
    }

    if (context.device_type == "cuda") {
        auto device_id_it = op->attributes.find("device_id");
        int device_id = (device_id_it != op->attributes.end()) ?
                       std::stoi(device_id_it->second) : context.device_id;

        engine_c::CudaExecutor cuda_executor(device_id);
        cuda_executor.initialize();

        auto num_inputs_it = op->attributes.find("num_inputs");
        int num_inputs = (num_inputs_it != op->attributes.end()) ?
                       std::stoi(num_inputs_it->second) : op->inputs.size();

        stats_.num_operations++;
        stats_.operation_counts["reduce"]++;

        auto start_time = std::chrono::high_resolution_clock::now();

        cuda_executor.synchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        stats_.operation_times["reduce"] += duration.count() / 1000.0;
    }

    return true;
}

bool IRGraphExecutor::executeCopyOp(const std::shared_ptr<IROperation>& op,
                                   const ExecutionContext& context) {
    if (op->inputs.size() != 1 || op->outputs.size() != 1) {
        setError("Copy operation requires exactly 1 input and 1 output");
        return false;
    }

    if (context.device_type == "cuda") {
        auto src_device_id_it = op->attributes.find("src_device_id");
        auto dst_device_id_it = op->attributes.find("dst_device_id");

        int src_device_id = (src_device_id_it != op->attributes.end()) ?
                           std::stoi(src_device_id_it->second) : context.device_id;
        int dst_device_id = (dst_device_id_it != op->attributes.end()) ?
                           std::stoi(dst_device_id_it->second) : context.device_id;

        engine_c::CudaExecutor cuda_executor(src_device_id);
        cuda_executor.initialize();

        stats_.num_operations++;
        stats_.operation_counts["copy"]++;

        auto start_time = std::chrono::high_resolution_clock::now();

        if (src_device_id != dst_device_id) {
            cuda_executor.enableP2P(dst_device_id);
        }

        cuda_executor.synchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        stats_.operation_times["copy"] += duration.count() / 1000.0;
    }

    return true;
}

bool IRGraphExecutor::executeSignalOp(const std::shared_ptr<IROperation>& op,
                                     const ExecutionContext& context) {
    auto signal_id_it = op->attributes.find("signal_id");
    if (signal_id_it == op->attributes.end()) {
        setError("Signal operation missing 'signal_id' attribute");
        return false;
    }

    if (context.device_type == "cuda") {
        engine_c::CudaExecutor cuda_executor(context.device_id);
        cuda_executor.initialize();

        stats_.num_operations++;
        stats_.operation_counts["signal"]++;

        auto start_time = std::chrono::high_resolution_clock::now();

        cudaStream_t stream = cuda_executor.getCurrentStream();
        cudaEvent_t event = cuda_executor.getStreamManager().createEvent();
        cuda_executor.getStreamManager().recordEvent(event, stream);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        stats_.operation_times["signal"] += duration.count() / 1000.0;
    }

    return true;
}

bool IRGraphExecutor::executeWaitSignalOp(const std::shared_ptr<IROperation>& op,
                                        const ExecutionContext& context) {
    auto signal_id_it = op->attributes.find("signal_id");
    if (signal_id_it == op->attributes.end()) {
        setError("WaitSignal operation missing 'signal_id' attribute");
        return false;
    }

    if (context.device_type == "cuda") {
        engine_c::CudaExecutor cuda_executor(context.device_id);
        cuda_executor.initialize();

        stats_.num_operations++;
        stats_.operation_counts["wait_signal"]++;

        auto start_time = std::chrono::high_resolution_clock::now();

        cudaStream_t stream = cuda_executor.getCurrentStream();
        cudaEvent_t event = cuda_executor.getStreamManager().createEvent();
        cuda_executor.getStreamManager().waitForEvent(event, stream);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        stats_.operation_times["wait_signal"] += duration.count() / 1000.0;
    }

    return true;
}

bool IRGraphExecutor::validateOperation(const std::shared_ptr<IROperation>& op) {
    if (!op || op->id.empty() || op->op_type.empty()) {
        return false;
    }

    for (const auto& input_id : op->inputs) {
        if (values_.find(input_id) == values_.end()) {
            setError("Input value not found: " + input_id);
            return false;
        }
    }

    return true;
}

bool IRGraphExecutor::validateDependencies() {
    for (const auto& [op_id, op] : operations_) {
        for (const auto& input_id : op->inputs) {
            if (values_.find(input_id) == values_.end()) {
                bool found_producer = false;
                for (const auto& [producer_id, producer] : operations_) {
                    if (std::find(producer->outputs.begin(),
                                 producer->outputs.end(), input_id) != producer->outputs.end()) {
                        found_producer = true;
                        break;
                    }
                }
                if (!found_producer && input_id.substr(0, 4) != "ext_") {
                    setError("Unresolved input dependency: " + input_id);
                    return false;
                }
            }
        }
    }
    return true;
}

std::vector<std::shared_ptr<IRGraphExecutor::IROperation>> IRGraphExecutor::getExecutionOrder() {
    std::vector<std::shared_ptr<IROperation>> result;
    std::map<std::string, int> in_degree;
    std::map<std::string, std::vector<std::string>> graph;

    for (const auto& [op_id, op] : operations_) {
        in_degree[op_id] = 0;
    }

    for (const auto& [op_id, op] : operations_) {
        for (const auto& input_id : op->inputs) {
            for (const auto& [producer_id, producer] : operations_) {
                if (std::find(producer->outputs.begin(),
                             producer->outputs.end(), input_id) != producer->outputs.end()) {
                    graph[producer_id].push_back(op_id);
                    in_degree[op_id]++;
                }
            }
        }
    }

    std::queue<std::string> queue;
    for (const auto& [op_id, degree] : in_degree) {
        if (degree == 0) {
            queue.push(op_id);
        }
    }

    while (!queue.empty()) {
        std::string current = queue.front();
        queue.pop();
        result.push_back(operations_[current]);

        for (const auto& neighbor : graph[current]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push(neighbor);
            }
        }
    }

    if (result.size() != operations_.size()) {
        setError("Cycle detected in operation dependencies");
        return {};
    }

    return result;
}

void IRGraphExecutor::updateStats(const std::string& op_type, double time_ms) {
    stats_.operation_counts[op_type]++;
    stats_.operation_times[op_type] += time_ms;
}

void IRGraphExecutor::setError(const std::string& error) {
    last_error_ = error;
    stats_.error_message = error;
    stats_.success = false;
}

} // namespace pccl