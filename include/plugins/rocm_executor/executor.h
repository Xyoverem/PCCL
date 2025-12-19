#pragma once

#include <base/device.h>
#include <base/registry.h>
#include "device.h"
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <rocblas/rocblas.h>
#include <rocrand/rocrand.h>
#include <vector>
#include <memory>
#include <string>

namespace engine_c {

enum class ROCmKernelType {
  REDUCE_SUM,
  REDUCE_MAX,
  REDUCE_MIN,
  REDUCE_AVG,
  REDUCE_CUSTOM,
  ALLREDUCE_RING,
  ALLREDUCE_TREE,
  BROADCAST,
  GATHER,
  SCATTER,
  COPY,
  CUSTOM
};

class ROCmKernel {
public:
  ROCmKernel(const std::string& name, ROCmKernelType type, ROCmDevice* device);
  virtual ~ROCmKernel() = default;

  virtual void launch(const void* args, dim3 grid, dim3 block,
                     hipStream_t stream = 0) = 0;

  const std::string& getName() const { return name_; }
  ROCmKernelType getType() const { return type_; }
  ROCmDevice* getDevice() const { return device_; }

protected:
  std::string name_;
  ROCmKernelType type_;
  ROCmDevice* device_;

  void checkHipError(hipError_t error, const std::string& operation) const;
};

class ROCmReduceKernel : public ROCmKernel {
public:
  ROCmReduceKernel(ROCmDevice* device, ROCmKernelType type = ROCmKernelType::REDUCE_SUM);

  void launch(const void* args, dim3 grid, dim3 block,
             hipStream_t stream = 0) override;

  void setBlockSize(int block_size) { block_size_ = block_size; }
  void setGridSize(int grid_size) { grid_size_ = grid_size; }

private:
  int block_size_;
  int grid_size_;
  hipFunction_t reduce_function_;

  void loadReduceFunction();
  void launchSumKernel(const void* args, dim3 grid, dim3 block, hipStream_t stream);
  void launchMaxKernel(const void* args, dim3 grid, dim3 block, hipStream_t stream);
  void launchMinKernel(const void* args, dim3 grid, dim3 block, hipStream_t stream);
};

class ROCmAllreduceKernel : public ROCmKernel {
public:
  ROCmAllreduceKernel(ROCmDevice* device);

  void launch(const void* args, dim3 grid, dim3 block,
             hipStream_t stream = 0) override;

  void setParticipants(const std::vector<int>& participants);
  void setAlgorithm(const std::string& algorithm);
  void setDataType(DataType dtype);

private:
  std::vector<int> participants_;
  std::string algorithm_;
  DataType data_type_;

  hipFunction_t ring_function_;
  hipFunction_t tree_function_;
  hipFunction_t rabenseifner_function_;

  void loadAllreduceFunctions();
  void launchRingAllreduce(const void* args, dim3 grid, dim3 block, hipStream_t stream);
  void launchTreeAllreduce(const void* args, dim3 grid, dim3 block, hipStream_t stream);
  void launchRabenseifnerAllreduce(const void* args, dim3 grid, dim3 block, hipStream_t stream);
};

class ROCmCopyKernel : public ROCmKernel {
public:
  ROCmCopyKernel(ROCmDevice* device);

  void launch(const void* args, dim3 grid, dim3 block,
             hipStream_t stream = 0) override;

private:
  hipFunction_t copy_function_;
  void loadCopyFunction();
};

class ROCmCommKernel : public ROCmKernel {
public:
  ROCmCommKernel(ROCmDevice* device, ROCmKernelType type);

  void launch(const void* args, dim3 grid, dim3 block,
             hipStream_t stream = 0) override;

  void setSourceDevice(int src_device) { src_device_ = src_device; }
  void setDestinationDevice(int dst_device) { dst_device_ = dst_device; }
  void setTag(int tag) { tag_ = tag; }

private:
  int src_device_;
  int dst_device_;
  int tag_;

  hipFunction_t send_function_;
  hipFunction_t recv_function_;

  void loadCommFunctions();
};

class ROCmKernelRegistry {
public:
  ROCmKernelRegistry();
  ~ROCmKernelRegistry();

  void registerKernel(std::unique_ptr<ROCmKernel> kernel);
  ROCmKernel* getKernel(const std::string& name);
  std::vector<std::string> getKernelNames() const;

  ROCmReduceKernel* getReduceKernel(ROCmKernelType type);
  ROCmAllreduceKernel* getAllreduceKernel();
  ROCmCopyKernel* getCopyKernel();
  ROCmCommKernel* getSendKernel();
  ROCmCommKernel* getRecvKernel();

private:
  std::unordered_map<std::string, std::unique_ptr<ROCmKernel>> kernels_;
  ROCmDevice* device_;

  void initializeBuiltinKernels();
};

class ROCmStreamManager {
public:
  ROCmStreamManager(ROCmDevice* device);
  ~ROCmStreamManager();

  hipStream_t getStream(int stream_id = 0);
  hipStream_t createStream();
  void destroyStream(hipStream_t stream);
  void synchronizeStream(int stream_id = 0);
  void synchronizeAllStreams();

  hipEvent_t createEvent();
  void destroyEvent(hipEvent_t event);
  void recordEvent(hipEvent_t event, hipStream_t stream = 0);
  void waitForEvent(hipEvent_t event, hipStream_t stream = 0);

  bool isStreamBusy(int stream_id = 0);

private:
  ROCmDevice* device_;
  std::vector<hipStream_t> streams_;
  std::vector<hipEvent_t> events_;
  std::mutex stream_mutex_;

  hipStream_t default_stream_;
};

class ROCmBLASManager {
public:
  ROCmBLASManager(ROCmDevice* device);
  ~ROCmBLASManager();

  rocblas_handle_t getHandle() const { return blas_handle_; }

  void setStream(hipStream_t stream);
  void synchronize();

  void gemm(rocblas_operation transa, rocblas_operation transb,
           int m, int n, int k,
           const float* alpha, const float* A, int lda,
           const float* B, int ldb,
           const float* beta, float* C, int ldc);

  void gemmStridedBatched(rocblas_operation transa, rocblas_operation transb,
                        int m, int n, int k,
                        const float* alpha, const float* A, int lda, long long strideA,
                        const float* B, int ldb, long long strideB,
                        const float* beta, float* C, int ldc, long long strideC,
                        int batch_count);

private:
  ROCmDevice* device_;
  rocblas_handle_t blas_handle_;

  void checkRocblasError(rocblas_status status, const std::string& operation) const;
};

class ROCmExecutor : public DeviceExecutor {
public:
  ROCmExecutor();
  ROCmExecutor(int device_id);
  ~ROCmExecutor() override;

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

  ROCmDevice* getROCmDevice() const { return hip_device_.get(); }
  ROCmKernelRegistry* getKernelRegistry() { return kernel_registry_.get(); }
  ROCmStreamManager* getStreamManager() { return stream_manager_.get(); }
  ROCmBLASManager* getBLASManager() { return blas_manager_.get(); }

  hipStream_t getCurrentStream() const;
  void setCurrentStream(hipStream_t stream);

private:
  int device_id_;
  bool initialized_;

  std::unique_ptr<ROCmDevice> hip_device_;
  std::unique_ptr<ROCmDeviceManager> device_manager_;
  std::unique_ptr<ROCmKernelRegistry> kernel_registry_;
  std::unique_ptr<ROCmStreamManager> stream_manager_;
  std::unique_ptr<ROCmBLASManager> blas_manager_;

  hipStream_t current_stream_;

  void initializeComponents();
  void shutdownComponents();
  void checkDeviceAvailable();
};

}