#include "ir/json_parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace pccl {
namespace ir {

// IRValue implementation
IRValue::IRValue(const std::string& id,
                 const std::string& dtype,
                 const std::vector<int>& shape,
                 int device_id,
                 DeviceType device_type)
    : id(id), dtype(dtype), shape(shape), device_id(device_id), device_type(device_type) {}

nlohmann::json IRValue::to_json() const {
    nlohmann::json j;
    j["id"] = id;
    j["dtype"] = dtype;
    j["shape"] = shape;
    j["device_id"] = device_id;
    j["device_type"] = device_type_to_string(device_type);
    j["metadata"] = metadata;
    return j;
}

std::unique_ptr<IRValue> IRValue::from_json(const nlohmann::json& j) {
    auto value = std::make_unique<IRValue>(
        j["id"].get<std::string>(),
        j["dtype"].get<std::string>(),
        j["shape"].get<std::vector<int>>(),
        j["device_id"].get<int>(),
        parse_device_type(j["device_type"].get<std::string>())
    );

    if (j.contains("metadata") && j["metadata"].is_object()) {
        for (auto& [key, val] : j["metadata"].items()) {
            value->metadata[key] = val;
        }
    }

    return value;
}

// IROperation implementation
IROperation::IROperation(const std::string& id,
                         const std::string& op_type,
                         const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs)
    : id(id), op_type(op_type), inputs(inputs), outputs(outputs) {}

nlohmann::json IROperation::to_json() const {
    nlohmann::json j;
    j["id"] = id;
    j["op_type"] = op_type;
    j["inputs"] = inputs;
    j["outputs"] = outputs;
    j["attributes"] = attributes;
    j["metadata"] = metadata;
    return j;
}

std::unique_ptr<IROperation> IROperation::from_json(const nlohmann::json& j) {
    auto operation = std::make_unique<IROperation>(
        j["id"].get<std::string>(),
        j["op_type"].get<std::string>(),
        j["inputs"].get<std::vector<std::string>>(),
        j["outputs"].get<std::vector<std::string>>()
    );

    if (j.contains("attributes") && j["attributes"].is_object()) {
        for (auto& [key, val] : j["attributes"].items()) {
            operation->attributes[key] = val;
        }
    }

    if (j.contains("metadata") && j["metadata"].is_object()) {
        for (auto& [key, val] : j["metadata"].items()) {
            operation->metadata[key] = val;
        }
    }

    return operation;
}

// IRGraph implementation
void IRGraph::add_value(std::unique_ptr<IRValue> value) {
    values[value->id] = std::move(value);
}

void IRGraph::add_operation(std::unique_ptr<IROperation> operation) {
    operations[operation->id] = std::move(operation);
}

IRValue* IRGraph::get_value(const std::string& id) {
    auto it = values.find(id);
    return it != values.end() ? it->second.get() : nullptr;
}

IROperation* IRGraph::get_operation(const std::string& id) {
    auto it = operations.find(id);
    return it != operations.end() ? it->second.get() : nullptr;
}

nlohmann::json IRGraph::to_json() const {
    nlohmann::json j;
    j["ir_type"] = ir_type_to_string(ir_type);

    nlohmann::json values_json = nlohmann::json::object();
    for (const auto& [id, value] : values) {
        values_json[id] = value->to_json();
    }
    j["values"] = values_json;

    nlohmann::json operations_json = nlohmann::json::object();
    for (const auto& [id, operation] : operations) {
        operations_json[id] = operation->to_json();
    }
    j["operations"] = operations_json;

    j["metadata"] = metadata;
    return j;
}

std::unique_ptr<IRGraph> IRGraph::from_json(const nlohmann::json& j) {
    auto graph = std::make_unique<IRGraph>(parse_ir_type(j["ir_type"].get<std::string>()));

    // Parse values
    if (j.contains("values") && j["values"].is_object()) {
        for (auto& [id, value_json] : j["values"].items()) {
            graph->add_value(IRValue::from_json(value_json));
        }
    }

    // Parse operations
    if (j.contains("operations") && j["operations"].is_object()) {
        for (auto& [id, operation_json] : j["operations"].items()) {
            graph->add_operation(IROperation::from_json(operation_json));
        }
    }

    // Parse metadata
    if (j.contains("metadata") && j["metadata"].is_object()) {
        for (auto& [key, val] : j["metadata"].items()) {
            graph->metadata[key] = val;
        }
    }

    return graph;
}

// JSONParser implementation
std::unique_ptr<IRGraph> JSONParser::parse_graph(const std::string& json_str) {
    nlohmann::json j = nlohmann::json::parse(json_str);
    return IRGraph::from_json(j);
}

std::unique_ptr<IRGraph> JSONParser::parse_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    nlohmann::json j = nlohmann::json::parse(file);
    return IRGraph::from_json(j);
}

std::vector<std::string> JSONParser::validate_json(const std::string& json_str) {
    std::vector<std::string> errors;

    try {
        nlohmann::json j = nlohmann::json::parse(json_str);

        // Validate required fields
        std::vector<std::string> required_fields = {"ir_type", "values", "operations"};
        for (const auto& field : required_fields) {
            if (!j.contains(field)) {
                errors.push_back("Missing required field: " + field);
            }
        }

        // Validate IR type
        if (j.contains("ir_type")) {
            try {
                parse_ir_type(j["ir_type"].get<std::string>());
            } catch (const std::exception&) {
                errors.push_back("Invalid IR type: " + j["ir_type"].get<std::string>());
            }
        }

        // Validate values
        if (j.contains("values") && j["values"].is_object()) {
            for (auto& [vid, value_data] : j["values"].items()) {
                auto value_errors = validate_value_data(vid, value_data);
                errors.insert(errors.end(), value_errors.begin(), value_errors.end());
            }
        }

        // Validate operations
        if (j.contains("operations") && j["operations"].is_object()) {
            for (auto& [oid, operation_data] : j["operations"].items()) {
                auto op_errors = validate_operation_data(oid, operation_data);
                errors.insert(errors.end(), op_errors.begin(), op_errors.end());
            }
        }

    } catch (const nlohmann::json::parse_error& e) {
        errors.push_back("Invalid JSON: " + std::string(e.what()));
    }

    return errors;
}

std::string JSONParser::serialize_graph(const IRGraph& graph, int indent) {
    return graph.to_json().dump(indent);
}

void JSONParser::serialize_to_file(const IRGraph& graph, const std::string& filepath, int indent) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filepath);
    }
    file << graph.to_json().dump(indent);
}

std::unordered_map<std::string, nlohmann::json> JSONParser::get_statistics(const IRGraph& graph) {
    std::unordered_map<std::string, nlohmann::json> stats;

    stats["ir_type"] = ir_type_to_string(graph.ir_type);
    stats["num_values"] = static_cast<int>(graph.values.size());
    stats["num_operations"] = static_cast<int>(graph.operations.size());

    std::unordered_map<std::string, int> operation_types;
    std::unordered_map<std::string, int> device_distribution;
    int total_elements = 0;

    // Count operation types
    for (const auto& [id, operation] : graph.operations) {
        std::string op_type = operation->op_type;
        operation_types[op_type] = operation_types[op_type] + 1;
    }

    stats["operation_types"] = operation_types;

    // Count device distribution and total elements
    for (const auto& [id, value] : graph.values) {
        std::string device_type_str = device_type_to_string(value->device_type);
        device_distribution[device_type_str] = device_distribution[device_type_str] + 1;

        // Count total elements
        if (!value->shape.empty()) {
            int total = 1;
            for (int dim : value->shape) {
                total *= dim;
            }
            total_elements += total;
        }
    }

    stats["device_distribution"] = device_distribution;
    stats["total_elements"] = total_elements;

    return stats;
}

// Helper methods
IRType JSONParser::parse_ir_type(const std::string& type_str) {
    if (type_str == "collective") return IRType::COLLECTIVE;
    if (type_str == "primitive") return IRType::PRIMITIVE;
    if (type_str == "hardware") return IRType::HARDWARE;
    throw std::runtime_error("Unknown IR type: " + type_str);
}

DeviceType JSONParser::parse_device_type(const std::string& device_str) {
    if (device_str == "cpu") return DeviceType::CPU;
    if (device_str == "cuda") return DeviceType::CUDA;
    if (device_str == "rdma") return DeviceType::RDMA;
    if (device_str == "rocm") return DeviceType::ROCM;
    throw std::runtime_error("Unknown device type: " + device_str);
}

PrimitiveOpType JSONParser::parse_primitive_op_type(const std::string& op_str) {
    if (op_str == "write") return PrimitiveOpType::WRITE;
    if (op_str == "reduce") return PrimitiveOpType::REDUCE;
    if (op_str == "copy") return PrimitiveOpType::COPY;
    if (op_str == "signal") return PrimitiveOpType::SIGNAL;
    if (op_str == "wait_signal") return PrimitiveOpType::WAIT_SIGNAL;
    throw std::runtime_error("Unknown primitive operation type: " + op_str);
}

ReduceOpType JSONParser::parse_reduce_op_type(const std::string& op_str) {
    if (op_str == "sum") return ReduceOpType::SUM;
    if (op_str == "avg") return ReduceOpType::AVG;
    if (op_str == "max") return ReduceOpType::MAX;
    if (op_str == "min") return ReduceOpType::MIN;
    if (op_str == "product") return ReduceOpType::PRODUCT;
    throw std::runtime_error("Unknown reduce operation type: " + op_str);
}

std::vector<std::string> JSONParser::validate_value_data(const std::string& id, const nlohmann::json& data) {
    std::vector<std::string> errors;

    std::vector<std::string> required_fields = {"id", "dtype", "shape", "device_id", "device_type"};
    for (const auto& field : required_fields) {
        if (!data.contains(field)) {
            errors.push_back("Value '" + id + "': missing field '" + field + "'");
        }
    }

    // Validate device type
    if (data.contains("device_type")) {
        try {
            parse_device_type(data["device_type"].get<std::string>());
        } catch (const std::exception&) {
            errors.push_back("Value '" + id + "': invalid device type '" + data["device_type"].get<std::string>() + "'");
        }
    }

    return errors;
}

std::vector<std::string> JSONParser::validate_operation_data(const std::string& id, const nlohmann::json& data) {
    std::vector<std::string> errors;

    std::vector<std::string> required_fields = {"id", "op_type", "inputs", "outputs", "attributes"};
    for (const auto& field : required_fields) {
        if (!data.contains(field)) {
            errors.push_back("Operation '" + id + "': missing field '" + field + "'");
        }
    }

    return errors;
}

// Utility functions
std::string ir_type_to_string(IRType type) {
    switch (type) {
        case IRType::COLLECTIVE: return "collective";
        case IRType::PRIMITIVE: return "primitive";
        case IRType::HARDWARE: return "hardware";
        default: return "unknown";
    }
}

std::string device_type_to_string(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::RDMA: return "rdma";
        case DeviceType::ROCM: return "rocm";
        default: return "unknown";
    }
}

std::string primitive_op_type_to_string(PrimitiveOpType type) {
    switch (type) {
        case PrimitiveOpType::WRITE: return "write";
        case PrimitiveOpType::REDUCE: return "reduce";
        case PrimitiveOpType::COPY: return "copy";
        case PrimitiveOpType::SIGNAL: return "signal";
        case PrimitiveOpType::WAIT_SIGNAL: return "wait_signal";
        default: return "unknown";
    }
}

std::string reduce_op_type_to_string(ReduceOpType type) {
    switch (type) {
        case ReduceOpType::SUM: return "sum";
        case ReduceOpType::AVG: return "avg";
        case ReduceOpType::MAX: return "max";
        case ReduceOpType::MIN: return "min";
        case ReduceOpType::PRODUCT: return "product";
        default: return "unknown";
    }
}

} // namespace ir
} // namespace pccl