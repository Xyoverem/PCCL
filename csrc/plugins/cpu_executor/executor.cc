#include <plugins/cpu_executor/executor.h>
#include <plugins/cpu_executor/device.h>
#include <cstring>
#include <algorithm>
#include <thread>
#include <chrono>
#include <random>

namespace engine_c {

struct TMP_EXECUTOR_STATIC_INITIALIZER {
  TMP_EXECUTOR_STATIC_INITIALIZER() {
    auto executor = TypeRegistry::registerExecutorType("cpu");
    regExec(executor, std::make_shared<DeviceExecutor>(new CpuExecutor()));
  }
};

static TMP_EXECUTOR_STATIC_INITIALIZER _____executor_tmp;

CpuExecutor::CpuExecutor() : initialized_(false), device_id_(0) {}

CpuExecutor::CpuExecutor(int device_id) : initialized_(false), device_id_(device_id) {}

CpuExecutor::~CpuExecutor() {
  shutdown();
}

void CpuExecutor::initialize() {
  if (initialized_) {
    return;
  }

  cpu_device_ = std::make_unique<CpuDevice>();
  initialized_ = true;
}

void CpuExecutor::shutdown() {
  if (!initialized_) {
    return;
  }

  cpu_device_.reset();
  initialized_ = false;
}

void* CpuExecutor::allocate(size_t size) {
  if (!initialized_) {
    return nullptr;
  }

  return cpu_device_->allocate(size);
}

void CpuExecutor::free(void* ptr) {
  if (!initialized_) {
    return;
  }

  cpu_device_->deallocate(ptr);
}

void CpuExecutor::copy(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  std::memcpy(dst, src, size);
}

void CpuExecutor::copyFromHost(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  std::memcpy(dst, src, size);
}

void CpuExecutor::copyToHost(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  std::memcpy(dst, src, size);
}

void CpuExecutor::execute(const void* kernel, const void* args, size_t arg_size,
                        dim3 grid, dim3 block, size_t shared_mem) {
  if (!initialized_) {
    return;
  }

  auto cpu_kernel = reinterpret_cast<CpuKernelFunc>(const_cast<void*>(kernel));
  if (cpu_kernel) {
    for (int gz = 0; gz < grid.z; gz++) {
      for (int gy = 0; gy < grid.y; gy++) {
        for (int gx = 0; gx < grid.x; gx++) {
          for (int bz = 0; bz < block.z; bz++) {
            for (int by = 0; by < block.y; by++) {
              for (int bx = 0; bx < block.x; bx++) {
                CpuKernelParams params = {
                  args,
                  static_cast<int>(gx),
                  static_cast<int>(gy),
                  static_cast<int>(gz),
                  static_cast<int>(bx),
                  static_cast<int>(by),
                  static_cast<int>(bz),
                  static_cast<int>(grid.x),
                  static_cast<int>(grid.y),
                  static_cast<int>(grid.z),
                  static_cast<int>(block.x),
                  static_cast<int>(block.y),
                  static_cast<int>(block.z)
                };
                cpu_kernel(&params);
              }
            }
          }
        }
      }
    }
  }
}

void CpuExecutor::synchronize() {
  if (!initialized_) {
    return;
  }

  std::this_thread::yield();
}

double CpuExecutor::measureKernelTime(const void* kernel, const void* args, size_t arg_size,
                                     dim3 grid, dim3 block, size_t shared_mem,
                                     int iterations) {
  if (!initialized_) {
    return 0.0;
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    execute(kernel, args, arg_size, grid, block, shared_mem);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  return static_cast<double>(duration.count()) / iterations / 1000.0;
}

bool CpuExecutor::supportsP2P() const {
  return false;
}

void CpuExecutor::enableP2P(int other_device_id) {
}

void* CpuExecutor::getDevicePointer() const {
  return nullptr;
}

int CpuExecutor::getDeviceId() const {
  return device_id_;
}

void CpuExecutor::setDevice(int device_id) {
  device_id_ = device_id;
}

size_t CpuExecutor::getTotalMemory() const {
  return 0;
}

size_t CpuExecutor::getAvailableMemory() const {
  return 0;
}

CpuMemoryManager::CpuMemoryManager() : allocated_bytes_(0) {}

CpuMemoryManager::~CpuMemoryManager() {
  clear();
}

void* CpuMemoryManager::allocate(size_t size) {
  if (size == 0) return nullptr;

  std::lock_guard<std::mutex> lock(mutex_);

  void* ptr = std::malloc(size);
  if (ptr) {
    allocated_bytes_ += size;
    allocations_[ptr] = size;
  }

  return ptr;
}

void CpuMemoryManager::free(void* ptr) {
  if (!ptr) return;

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(ptr);
  if (it != allocations_.end()) {
    allocated_bytes_ -= it->second;
    allocations_.erase(it);
    std::free(ptr);
  }
}

size_t CpuMemoryManager::getAllocatedBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocated_bytes_;
}

size_t CpuMemoryManager::getAllocationCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocations_.size();
}

void CpuMemoryManager::clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& pair : allocations_) {
    std::free(pair.first);
  }

  allocations_.clear();
  allocated_bytes_ = 0;
}

CpuThreadPool::CpuThreadPool(size_t num_threads) : stop_(false) {
  for (size_t i = 0; i < num_threads; ++i) {
    workers_.emplace_back([this] {
      for (;;) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(queue_mutex_);
          condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

          if (stop_ && tasks_.empty()) {
            return;
          }

          task = std::move(tasks_.front());
          tasks_.pop();
        }

        task();
      }
    });
  }
}

CpuThreadPool::~CpuThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }

  condition_.notify_all();

  for (std::thread& worker : workers_) {
    worker.join();
  }
}

void CpuThreadPool::submit(std::function<void()> task) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    if (stop_) {
      return;
    }

    tasks_.emplace(task);
  }

  condition_.notify_one();
}

size_t CpuThreadPool::getThreadCount() const {
  return workers_.size();
}

size_t CpuThreadPool::getQueueSize() const {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  return tasks_.size();
}

CpuKernelRegistry::CpuKernelRegistry() {}

CpuKernelRegistry::~CpuKernelRegistry() {}

void CpuKernelRegistry::registerKernel(const std::string& name, CpuKernelFunc kernel) {
  std::lock_guard<std::mutex> lock(mutex_);
  kernels_[name] = kernel;
}

CpuKernelFunc CpuKernelRegistry::getKernel(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = kernels_.find(name);
  return (it != kernels_.end()) ? it->second : nullptr;
}

bool CpuKernelRegistry::hasKernel(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  return kernels_.find(name) != kernels_.end();
}

std::vector<std::string> CpuKernelRegistry::getKernelNames() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> names;
  names.reserve(kernels_.size());

  for (const auto& pair : kernels_) {
    names.push_back(pair.first);
  }

  return names;
}

void CpuKernelRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  kernels_.clear();
}

}