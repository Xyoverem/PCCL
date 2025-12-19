#pragma once

#include <base/device.h>
#include <base/executor.h>
#include <base/registry.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <nvrtc.h>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <string>

namespace engine_c {

class CudaStreamManager {
public:
  CudaStreamManager();
  ~CudaStreamManager();

  void initialize(int device_id);

  cudaStream_t getStream(int stream_id = 0);
  cudaStream_t createStream();
  void destroyStream(cudaStream_t stream);
  void synchronizeStream(int stream_id = 0);

  cudaEvent_t createEvent();
  void recordEvent(cudaEvent_t event, cudaStream_t stream = 0);
  void waitForEvent(cudaEvent_t event, cudaStream_t stream = 0);

private:
  int device_id_;
  cudaStream_t default_stream_;
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> events_;
  mutable std::mutex mutex_;

  void checkCudaError(cudaError_t error, const std::string& operation) const;
};

class CudaMemoryManager {
public:
  CudaMemoryManager();
  ~CudaMemoryManager();

  void initialize(int device_id);

  void* allocate(size_t size);
  void free(void* ptr);

  size_t getAllocatedBytes() const;
  size_t getAllocationCount() const;
  void clear();

private:
  int device_id_;
  std::unordered_map<void*, size_t> allocations_;
  size_t allocated_bytes_;
  mutable std::mutex mutex_;

  void checkCudaError(cudaError_t error, const std::string& operation) const;
};

class CudaKernelManager {
public:
  CudaKernelManager();
  ~CudaKernelManager();

  void initialize(int device_id);

  CUfunction getFunction(const std::string& ptx, const std::string& function_name);

private:
  int device_id_;
  std::unordered_map<std::string, CUmodule> modules_;
  mutable std::mutex mutex_;

  void checkCuError(CUresult error, const std::string& operation) const;
};

class CudaBlasManager {
public:
  CudaBlasManager();
  ~CudaBlasManager();

  void initialize(int device_id);

  cublasHandle_t getHandle();
  void setStream(cudaStream_t stream);
  void synchronize();

private:
  int device_id_;
  cublasHandle_t blas_handle_;

  void checkCublasError(cublasStatus_t status, const std::string& operation) const;
};

class CudaExecutor : public DeviceExecutor {
public:
  CudaExecutor();
  explicit CudaExecutor(int device_id);
  ~CudaExecutor() override;

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

  cudaStream_t getCurrentStream() const;
  void setCurrentStream(cudaStream_t stream);

  size_t getTotalMemory() const;
  size_t getAvailableMemory() const;

  CudaStreamManager* getStreamManager();
  CudaMemoryManager* getMemoryManager();
  CudaKernelManager* getKernelManager();
  CudaBlasManager* getBlasManager();

private:
  bool initialized_;
  int device_id_;
  cudaStream_t current_stream_;

  std::unique_ptr<CudaDevice> cuda_device_;
  std::unique_ptr<CudaStreamManager> stream_manager_;
  std::unique_ptr<CudaMemoryManager> memory_manager_;
  std::unique_ptr<CudaKernelManager> kernel_manager_;
  std::unique_ptr<CudaBlasManager> blas_manager_;

  void checkCudaError(cudaError_t error, const std::string& operation) const;
};

}