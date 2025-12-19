#pragma once

#include <base/device.h>
#include <base/executor.h>
#include <base/registry.h>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <functional>
#include <vector>

namespace engine_c {

struct CpuKernelParams {
  const void* args;
  int grid_x, grid_y, grid_z;
  int block_x, block_y, block_z;
  int grid_dim_x, grid_dim_y, grid_dim_z;
  int block_dim_x, block_dim_y, block_dim_z;
};

typedef void (*CpuKernelFunc)(CpuKernelParams* params);

class CpuMemoryManager {
public:
  CpuMemoryManager();
  ~CpuMemoryManager();

  void* allocate(size_t size);
  void free(void* ptr);

  size_t getAllocatedBytes() const;
  size_t getAllocationCount() const;
  void clear();

private:
  mutable std::mutex mutex_;
  std::unordered_map<void*, size_t> allocations_;
  size_t allocated_bytes_;
};

class CpuThreadPool {
public:
  explicit CpuThreadPool(size_t num_threads = std::thread::hardware_concurrency());
  ~CpuThreadPool();

  template<class F>
  void submit(F&& f) {
    submit(std::function<void()>(std::forward<F>(f)));
  }

  void submit(std::function<void()> task);
  size_t getThreadCount() const;
  size_t getQueueSize() const;

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};

class CpuKernelRegistry {
public:
  CpuKernelRegistry();
  ~CpuKernelRegistry();

  void registerKernel(const std::string& name, CpuKernelFunc kernel);
  CpuKernelFunc getKernel(const std::string& name);
  bool hasKernel(const std::string& name);
  std::vector<std::string> getKernelNames() const;
  void clear();

private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, CpuKernelFunc> kernels_;
};

class CpuExecutor : public DeviceExecutor {
public:
  CpuExecutor();
  explicit CpuExecutor(int device_id);
  ~CpuExecutor() override;

  void initialize() override;
  void shutdown() override;

  void* allocate(size_t size) override;
  void free(void* ptr) override;

  void copy(void* dst, const void* src, size_t size) override;
  void copyFromHost(void* dst, const void* src, size_t size) override;
  void copyToHost(void* dst, const void* src, size_t size) override;

  void execute(const void* kernel, const void* args, size_t arg_size,
              dim3 grid, dim3 block, size_t shared_mem = 0) override;

  void synchronize() override;

  double measureKernelTime(const void* kernel, const void* args, size_t arg_size,
                           dim3 grid, dim3 block, size_t shared_mem = 0,
                           int iterations = 100) override;

  bool supportsP2P() const override;
  void enableP2P(int other_device_id) override;

  void* getDevicePointer() const override;
  int getDeviceId() const override;

  void setDevice(int device_id);
  size_t getTotalMemory() const;
  size_t getAvailableMemory() const;

  CpuMemoryManager* getMemoryManager() { return &memory_manager_; }
  CpuThreadPool* getThreadPool() { return &thread_pool_; }
  CpuKernelRegistry* getKernelRegistry() { return &kernel_registry_; }

private:
  bool initialized_;
  int device_id_;
  std::unique_ptr<CpuDevice> cpu_device_;
  CpuMemoryManager memory_manager_;
  CpuThreadPool thread_pool_;
  CpuKernelRegistry kernel_registry_;
};

}