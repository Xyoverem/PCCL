#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <functional>
#include "json/json.h"
#include "base/operator.h"
#include "base/device.h"
#include "base/value.h"

namespace pccl {

struct ExecutionContext {
    int device_id = 0;
    std::string device_type = "cpu";
    std::map<std::string, std::shared_ptr<Value>> inputs;
    std::map<std::string, std::shared_ptr<Value>> outputs;
    bool async_execution = false;
    int timeout_ms = 30000;
};

struct ExecutionStats {
    int num_operations = 0;
    int num_values = 0;
    double execution_time_ms = 0.0;
    std::map<std::string, int> operation_counts;
    std::map<std::string, double> operation_times;
    bool success = false;
    std::string error_message;
};

class IRGraphExecutor {
public:
    IRGraphExecutor();
    ~IRGraphExecutor();

    bool parseIRGraph(const std::string& json_graph);
    bool executeGraph(const ExecutionContext& context);
    bool executeGraphAsync(const ExecutionContext& context);

    ExecutionStats getStatistics() const;
    void reset();

    bool isReady() const { return is_ready_; }
    std::string getLastError() const { return last_error_; }

private:
    struct IROperation {
        std::string id;
        std::string op_type;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::map<std::string, Json::Value> attributes;
        std::map<std::string, Json::Value> metadata;
    };

    struct IRValue {
        std::string id;
        std::string dtype;
        std::vector<int> shape;
        int device_id;
        std::string device_type;
        std::map<std::string, Json::Value> metadata;
    };

    bool parseJSON(const std::string& json_str);
    std::shared_ptr<IROperation> convertJsonOp(const Json::Value& json_op);
    std::shared_ptr<IRValue> convertJsonValue(const Json::Value& json_value);

    bool dispatchHardwareOp(const std::shared_ptr<IROperation>& op,
                           const ExecutionContext& context);
    bool executeWriteOp(const std::shared_ptr<IROperation>& op,
                       const ExecutionContext& context);
    bool executeReduceOp(const std::shared_ptr<IROperation>& op,
                        const ExecutionContext& context);
    bool executeCopyOp(const std::shared_ptr<IROperation>& op,
                      const ExecutionContext& context);
    bool executeSignalOp(const std::shared_ptr<IROperation>& op,
                        const ExecutionContext& context);
    bool executeWaitSignalOp(const std::shared_ptr<IROperation>& op,
                           const ExecutionContext& context);

    bool validateOperation(const std::shared_ptr<IROperation>& op);
    bool validateDependencies();
    std::vector<std::shared_ptr<IROperation>> getExecutionOrder();

    void updateStats(const std::string& op_type, double time_ms);
    void setError(const std::string& error);

    std::map<std::string, std::shared_ptr<IROperation>> operations_;
    std::map<std::string, std::shared_ptr<IRValue>> values_;
    std::map<std::string, Json::Value> graph_metadata_;

    ExecutionStats stats_;
    bool is_ready_;
    std::string last_error_;

    std::map<std::string, std::function<bool(const std::shared_ptr<IROperation>&,
                                           const ExecutionContext&)>> op_dispatch_;
};

} // namespace pccl