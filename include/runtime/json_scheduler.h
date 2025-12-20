#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <future>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "ir/json_parser.h"

namespace pccl {
namespace runtime {

// Forward declarations
class ExecutionEngine;
class DeviceCoordinator;
class HardwareExecutor;

enum class ExecutionStatus {
    PENDING,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELLED
};

struct ExecutionTask {
    std::string task_id;
    std::string operation_id;
    std::unique_ptr<IROperation> operation;
    ExecutionStatus status;
    std::string error_message;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

    ExecutionTask(const std::string& tid, const std::string& oid,
                  std::unique_ptr<IROperation> op);
};

class JSONScheduler {
public:
    JSONScheduler();
    ~JSONScheduler();

    // Load and execute IR graph from JSON
    bool load_graph_from_json(const std::string& json_str);
    bool load_graph_from_file(const std::string& filepath);

    // Execute the loaded graph
    bool execute_graph_async();
    bool execute_graph_sync();

    // Get execution status
    ExecutionStatus get_execution_status() const;
    std::vector<ExecutionTask> get_execution_tasks() const;

    // Configuration
    void set_num_workers(size_t num_workers);
    void set_timeout_ms(int timeout_ms);
    void enable_profiling(bool enable);

    // Callbacks
    void set_task_completed_callback(std::function<void(const ExecutionTask&)> callback);
    void set_graph_completed_callback(std::function<void(bool)> callback);

    // Statistics
    std::unordered_map<std::string, int> get_execution_statistics() const;
    double get_execution_time_ms() const;

private:
    // Core execution methods
    void schedule_operations();
    bool execute_operation(ExecutionTask& task);
    void worker_thread();

    // Dependency management
    std::vector<std::string> get_ready_operations();
    void mark_operation_completed(const std::string& operation_id);
    bool are_dependencies_satisfied(const std::string& operation_id) const;

    // Task management
    std::string create_task_id(const std::string& operation_id);
    ExecutionTask* find_task(const std::string& task_id);

    // Hardware executor management
    std::unique_ptr<HardwareExecutor> get_executor_for_device(DeviceType device_type);

private:
    // Graph data
    std::unique_ptr<IRGraph> ir_graph_;
    std::unordered_map<std::string, std::unique_ptr<ExecutionTask>> tasks_;
    std::unordered_map<std::string, std::vector<std::string>> dependencies_;
    std::unordered_map<std::string, std::vector<std::string>> reverse_dependencies_;

    // Execution state
    ExecutionStatus status_;
    std::string error_message_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;

    // Threading
    size_t num_workers_;
    std::vector<std::thread> worker_threads_;
    std::queue<std::string> ready_queue_;
    std::queue<std::string> completed_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable completion_cv_;
    std::atomic<bool> shutdown_;

    // Configuration
    int timeout_ms_;
    bool profiling_enabled_;

    // Callbacks
    std::function<void(const ExecutionTask&)> task_completed_callback_;
    std::function<void(bool)> graph_completed_callback_;

    // Statistics
    std::unordered_map<std::string, int> stats_;
    mutable std::mutex stats_mutex_;
};

// Hardware executor interface
class HardwareExecutor {
public:
    virtual ~HardwareExecutor() = default;

    virtual bool execute_operation(const IROperation& operation) = 0;
    virtual DeviceType get_device_type() const = 0;
    virtual bool is_available() const = 0;
    virtual std::string get_name() const = 0;
};

// CUDA hardware executor
class CUDAHardwareExecutor : public HardwareExecutor {
public:
    CUDAHardwareExecutor(int device_id = 0);
    ~CUDAHardwareExecutor() override;

    bool execute_operation(const IROperation& operation) override;
    DeviceType get_device_type() const override { return DeviceType::CUDA; }
    bool is_available() const override;
    std::string get_name() const override { return "CUDA Executor"; }

private:
    int device_id_;
    bool initialized_;

    bool execute_write_op(const IROperation& operation);
    bool execute_reduce_op(const IROperation& operation);
    bool execute_copy_op(const IROperation& operation);
    bool execute_signal_op(const IROperation& operation);
    bool execute_wait_signal_op(const IROperation& operation);
};

// CPU hardware executor
class CPUHardwareExecutor : public HardwareExecutor {
public:
    CPUHardwareExecutor();
    ~CPUHardwareExecutor() override;

    bool execute_operation(const IROperation& operation) override;
    DeviceType get_device_type() const override { return DeviceType::CPU; }
    bool is_available() const override { return true; }
    std::string get_name() const override { return "CPU Executor"; }

private:
    bool execute_write_op(const IROperation& operation);
    bool execute_reduce_op(const IROperation& operation);
    bool execute_copy_op(const IROperation& operation);
    bool execute_signal_op(const IROperation& operation);
    bool execute_wait_signal_op(const IROperation& operation);
};

// RDMA hardware executor
class RDMAHardwareExecutor : public HardwareExecutor {
public:
    RDMAHardwareExecutor(int device_id = 0);
    ~RDMAHardwareExecutor() override;

    bool execute_operation(const IROperation& operation) override;
    DeviceType get_device_type() const override { return DeviceType::RDMA; }
    bool is_available() const override;
    std::string get_name() const override { return "RDMA Executor"; }

private:
    int device_id_;
    bool initialized_;
};

// Factory for creating hardware executors
class HardwareExecutorFactory {
public:
    static std::unique_ptr<HardwareExecutor> create(DeviceType device_type, int device_id = 0);
    static std::vector<std::unique_ptr<HardwareExecutor>> create_all_available();
};

} // namespace runtime
} // namespace pccl