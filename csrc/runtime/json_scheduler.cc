#include "runtime/json_scheduler.h"
#include <fstream>
#include <algorithm>
#include <chrono>

namespace pccl {
namespace runtime {

// ExecutionTask implementation
ExecutionTask::ExecutionTask(const std::string& tid, const std::string& oid,
                           std::unique_ptr<IROperation> op)
    : task_id(tid), operation_id(oid), operation(std::move(op)),
      status(ExecutionStatus::PENDING) {
}

// JSONScheduler implementation
JSONScheduler::JSONScheduler()
    : status_(ExecutionStatus::PENDING),
      num_workers_(4),
      timeout_ms_(30000),
      profiling_enabled_(false),
      shutdown_(false) {
}

JSONScheduler::~JSONScheduler() {
    shutdown();
}

bool JSONScheduler::load_graph_from_json(const std::string& json_str) {
    ir::JSONParser parser;
    ir_graph_ = parser.parse_graph(json_str);

    if (!ir_graph_) {
        error_message_ = "Failed to parse JSON graph";
        return false;
    }

    // Build dependency graph
    for (const auto& [op_id, operation] : ir_graph_->operations) {
        dependencies_[op_id] = operation->inputs;
        for (const auto& input_id : operation->inputs) {
            reverse_dependencies_[input_id].push_back(op_id);
        }

        // Create execution task
        auto task = std::make_unique<ExecutionTask>(
            create_task_id(op_id), op_id,
            std::unique_ptr<IROperation>(new IROperation(*operation))
        );
        tasks_[task->task_id] = std::move(task);
    }

    return true;
}

bool JSONScheduler::load_graph_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        error_message_ = "Cannot open file: " + filepath;
        return false;
    }

    std::string json_str((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    return load_graph_from_json(json_str);
}

bool JSONScheduler::execute_graph_async() {
    if (!ir_graph_) {
        error_message_ = "No graph loaded";
        return false;
    }

    status_ = ExecutionStatus::RUNNING;
    start_time_ = std::chrono::high_resolution_clock::now();

    // Start worker threads
    for (size_t i = 0; i < num_workers_; ++i) {
        worker_threads_.emplace_back(&JSONScheduler::worker_thread, this);
    }

    // Schedule operations
    schedule_operations();

    return true;
}

bool JSONScheduler::execute_graph_sync() {
    if (!execute_graph_async()) {
        return false;
    }

    // Wait for completion
    std::unique_lock<std::mutex> lock(queue_mutex_);
    completion_cv_.wait(lock, [this] {
        return shutdown_ || status_ == ExecutionStatus::COMPLETED ||
               status_ == ExecutionStatus::FAILED;
    });

    return status_ == ExecutionStatus::COMPLETED;
}

ExecutionStatus JSONScheduler::get_execution_status() const {
    return status_;
}

std::vector<ExecutionTask> JSONScheduler::get_execution_tasks() const {
    std::vector<ExecutionTask> tasks;
    for (const auto& [id, task] : tasks_) {
        tasks.push_back(*task);
    }
    return tasks;
}

void JSONScheduler::set_num_workers(size_t num_workers) {
    num_workers_ = std::max(size_t(1), num_workers);
}

void JSONScheduler::set_timeout_ms(int timeout_ms) {
    timeout_ms_ = timeout_ms;
}

void JSONScheduler::enable_profiling(bool enable) {
    profiling_enabled_ = enable;
}

void JSONScheduler::set_task_completed_callback(
    std::function<void(const ExecutionTask&)> callback) {
    task_completed_callback_ = callback;
}

void JSONScheduler::set_graph_completed_callback(
    std::function<void(bool)> callback) {
    graph_completed_callback_ = callback;
}

std::unordered_map<std::string, int> JSONScheduler::get_execution_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

double JSONScheduler::get_execution_time_ms() const {
    if (start_time_ == std::chrono::high_resolution_clock::time_point{}) {
        return 0.0;
    }

    auto end_time = (status_ == ExecutionStatus::COMPLETED ||
                    status_ == ExecutionStatus::FAILED) ? end_time_ :
                    std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time_);
    return duration.count();
}

void JSONScheduler::schedule_operations() {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    auto ready_ops = get_ready_operations();
    for (const auto& op_id : ready_ops) {
        ready_queue_.push(op_id);
    }

    queue_cv_.notify_all();
}

bool JSONScheduler::execute_operation(ExecutionTask& task) {
    if (!task.operation) {
        task.status = ExecutionStatus::FAILED;
        task.error_message = "No operation to execute";
        return false;
    }

    task.start_time = std::chrono::high_resolution_clock::now();
    task.status = ExecutionStatus::RUNNING;

    try {
        // Get hardware executor for the operation
        DeviceType device_type = DeviceType::CPU;  // Default

        // Try to determine device type from operation attributes
        if (task.operation->attributes.contains("device_type")) {
            device_type = parse_device_type(
                task.operation->attributes["device_type"].get<std::string>()
            );
        }

        auto executor = get_executor_for_device(device_type);
        if (!executor || !executor->is_available()) {
            task.status = ExecutionStatus::FAILED;
            task.error_message = "Hardware executor not available";
            return false;
        }

        // Execute the operation
        bool success = executor->execute_operation(*task.operation);

        task.end_time = std::chrono::high_resolution_clock::now();
        task.status = success ? ExecutionStatus::COMPLETED : ExecutionStatus::FAILED;

        if (!success) {
            task.error_message = "Operation execution failed";
        }

        return success;

    } catch (const std::exception& e) {
        task.end_time = std::chrono::high_resolution_clock::now();
        task.status = ExecutionStatus::FAILED;
        task.error_message = std::string("Exception: ") + e.what();
        return false;
    }
}

void JSONScheduler::worker_thread() {
    while (!shutdown_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // Wait for a task or shutdown
        queue_cv_.wait(lock, [this] {
            return !ready_queue_.empty() || shutdown_;
        });

        if (shutdown_) {
            break;
        }

        if (ready_queue_.empty()) {
            continue;
        }

        std::string task_id = ready_queue_.front();
        ready_queue_.pop();
        lock.unlock();

        // Execute the task
        ExecutionTask* task = find_task(task_id);
        if (task) {
            bool success = execute_operation(*task);

            // Mark operation as completed
            mark_operation_completed(task->operation_id);

            // Add to completed queue
            {
                std::lock_guard<std::mutex> completed_lock(queue_mutex_);
                completed_queue_.push(task_id);
            }

            // Call callback if set
            if (task_completed_callback_) {
                task_completed_callback_(*task);
            }

            // Update statistics
            {
                std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                std::string op_type = task->operation->op_type;
                stats_[op_type + "_executed"]++;
                if (success) {
                    stats_[op_type + "_success"]++;
                } else {
                    stats_[op_type + "_failed"]++;
                }
            }
        }

        lock.lock();
    }
}

std::vector<std::string> JSONScheduler::get_ready_operations() {
    std::vector<std::string> ready_ops;

    for (const auto& [op_id, task] : tasks_) {
        if (task->status == ExecutionStatus::PENDING &&
            are_dependencies_satisfied(op_id)) {
            ready_ops.push_back(task->task_id);
        }
    }

    return ready_ops;
}

void JSONScheduler::mark_operation_completed(const std::string& operation_id) {
    // Check if all operations are completed
    bool all_completed = true;
    for (const auto& [op_id, task] : tasks_) {
        if (task->status == ExecutionStatus::PENDING ||
            task->status == ExecutionStatus::RUNNING) {
            all_completed = false;
            break;
        }
    }

    if (all_completed) {
        status_ = ExecutionStatus::COMPLETED;
        end_time_ = std::chrono::high_resolution_clock::now();

        if (graph_completed_callback_) {
            graph_completed_callback_(true);
        }
    }

    // Schedule newly ready operations
    auto ready_ops = get_ready_operations();
    for (const auto& task_id : ready_ops) {
        ready_queue_.push(task_id);
    }
    queue_cv_.notify_all();
}

bool JSONScheduler::are_dependencies_satisfied(const std::string& operation_id) const {
    auto it = dependencies_.find(operation_id);
    if (it == dependencies_.end()) {
        return true;  // No dependencies
    }

    for (const auto& input_id : it->second) {
        // Check if the input is produced by another operation
        auto dep_it = reverse_dependencies_.find(input_id);
        if (dep_it != reverse_dependencies_.end()) {
            // Check if all producer operations are completed
            for (const auto& producer_id : dep_it->second) {
                auto task_it = tasks_.find(producer_id);
                if (task_it != tasks_.end() &&
                    task_it->second->status != ExecutionStatus::COMPLETED) {
                    return false;
                }
            }
        }
    }

    return true;
}

std::string JSONScheduler::create_task_id(const std::string& operation_id) {
    return "task_" + operation_id;
}

ExecutionTask* JSONScheduler::find_task(const std::string& task_id) {
    auto it = tasks_.find(task_id);
    return it != tasks_.end() ? it->second.get() : nullptr;
}

std::unique_ptr<HardwareExecutor> JSONScheduler::get_executor_for_device(DeviceType device_type) {
    // For now, create a simple executor
    // In a real implementation, this would manage a pool of executors
    switch (device_type) {
        case DeviceType::CUDA:
            return std::make_unique<CUDAHardwareExecutor>();
        case DeviceType::RDMA:
            return std::make_unique<RDMAHardwareExecutor>();
        case DeviceType::CPU:
        default:
            return std::make_unique<CPUHardwareExecutor>();
    }
}

void JSONScheduler::shutdown() {
    shutdown_ = true;
    queue_cv_.notify_all();

    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    worker_threads_.clear();
}

// HardwareExecutorFactory implementation
std::unique_ptr<HardwareExecutor> HardwareExecutorFactory::create(DeviceType device_type, int device_id) {
    switch (device_type) {
        case DeviceType::CUDA:
            return std::make_unique<CUDAHardwareExecutor>(device_id);
        case DeviceType::RDMA:
            return std::make_unique<RDMAHardwareExecutor>(device_id);
        case DeviceType::CPU:
        default:
            return std::make_unique<CPUHardwareExecutor>();
    }
}

std::vector<std::unique_ptr<HardwareExecutor>> HardwareExecutorFactory::create_all_available() {
    std::vector<std::unique_ptr<HardwareExecutor>> executors;

    // Always have CPU executor
    executors.push_back(std::make_unique<CPUHardwareExecutor>());

    // Try CUDA executors
    try {
        for (int i = 0; i < 8; ++i) {  // Check up to 8 CUDA devices
            auto executor = std::make_unique<CUDAHardwareExecutor>(i);
            if (executor->is_available()) {
                executors.push_back(std::move(executor));
            } else {
                break;  // Assume devices are contiguous
            }
        }
    } catch (...) {
        // CUDA not available
    }

    return executors;
}

} // namespace runtime
} // namespace pccl