#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>
#include <functional>
#include "nlohmann/json.hpp"

namespace pccl {
namespace ir {

// Forward declarations
struct IRValue;
struct IROperation;
struct IRGraph;

enum class IRType {
    COLLECTIVE = "collective",
    PRIMITIVE = "primitive",
    HARDWARE = "hardware"
};

enum class PrimitiveOpType {
    WRITE = "write",
    REDUCE = "reduce",
    COPY = "copy",
    SIGNAL = "signal",
    WAIT_SIGNAL = "wait_signal"
};

enum class ReduceOpType {
    SUM = "sum",
    AVG = "avg",
    MAX = "max",
    MIN = "min",
    PRODUCT = "product"
};

enum class DeviceType {
    CPU = "cpu",
    CUDA = "cuda",
    RDMA = "rdma",
    ROCM = "rocm"
};

// IR value representing data
struct IRValue {
    std::string id;
    std::string dtype;
    std::vector<int> shape;
    int device_id;
    DeviceType device_type;
    std::unordered_map<std::string, nlohmann::json> metadata;

    IRValue(const std::string& id,
            const std::string& dtype,
            const std::vector<int>& shape,
            int device_id,
            DeviceType device_type);

    nlohmann::json to_json() const;
    static std::unique_ptr<IRValue> from_json(const nlohmann::json& j);
};

// IR operation
struct IROperation {
    std::string id;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, nlohmann::json> attributes;
    std::unordered_map<std::string, nlohmann::json> metadata;

    IROperation(const std::string& id,
                const std::string& op_type,
                const std::vector<std::string>& inputs,
                const std::vector<std::string>& outputs);

    nlohmann::json to_json() const;
    static std::unique_ptr<IROperation> from_json(const nlohmann::json& j);
};

// IR graph containing operations and values
struct IRGraph {
    IRType ir_type;
    std::unordered_map<std::string, std::unique_ptr<IRValue>> values;
    std::unordered_map<std::string, std::unique_ptr<IROperation>> operations;
    std::unordered_map<std::string, nlohmann::json> metadata;

    IRGraph(IRType type) : ir_type(type) {}

    void add_value(std::unique_ptr<IRValue> value);
    void add_operation(std::unique_ptr<IROperation> operation);
    IRValue* get_value(const std::string& id);
    IROperation* get_operation(const std::string& id);

    nlohmann::json to_json() const;
    static std::unique_ptr<IRGraph> from_json(const nlohmann::json& j);
};

// JSON parser for IR
class JSONParser {
public:
    JSONParser() = default;
    ~JSONParser() = default;

    // Parse JSON string to IR graph
    std::unique_ptr<IRGraph> parse_graph(const std::string& json_str);

    // Parse JSON file to IR graph
    std::unique_ptr<IRGraph> parse_from_file(const std::string& filepath);

    // Validate JSON format
    std::vector<std::string> validate_json(const std::string& json_str);

    // Serialize IR graph to JSON string
    std::string serialize_graph(const IRGraph& graph, int indent = 2);

    // Serialize IR graph to JSON file
    void serialize_to_file(const IRGraph& graph, const std::string& filepath, int indent = 2);

    // Get statistics about IR graph
    std::unordered_map<std::string, nlohmann::json> get_statistics(const IRGraph& graph);

private:
    // Helper methods for parsing
    IRType parse_ir_type(const std::string& type_str);
    DeviceType parse_device_type(const std::string& device_str);
    PrimitiveOpType parse_primitive_op_type(const std::string& op_str);
    ReduceOpType parse_reduce_op_type(const std::string& op_str);

    std::vector<std::string> validate_value_data(const std::string& id, const nlohmann::json& data);
    std::vector<std::string> validate_operation_data(const std::string& id, const nlohmann::json& data);
};

// Utility functions
std::string ir_type_to_string(IRType type);
std::string device_type_to_string(DeviceType type);
std::string primitive_op_type_to_string(PrimitiveOpType type);
std::string reduce_op_type_to_string(ReduceOpType type);

} // namespace ir
} // namespace pccl