#pragma once

#include <base/device.h>
#include <base/registry.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hsa/hsa.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace engine_c {

class ROCmMemoryManager {
public:
  ROCmMemoryManager();
  ~ROCmMemoryManager();

  void* allocate(size_t size, int hip_device = 0);
  void free(void* ptr, int hip_device = 0);

  void* allocateUnified(size_t size);
  void freeUnified(void* ptr);

  void* allocatePinned(size_t size);
  void freePinned(void* ptr);

  void copyToDevice(void* dst, const void* src, size_t size, int hip_device = 0);
  void copyFromDevice(void* dst, const void* src, size_t size, int hip_device = 0);
  void copyDeviceToDevice(void* dst, const void* src, size_t size,
                          int src_device, int dst_device);

  bool canAccessPeer(int src_device, int dst_device) const;
  void enablePeerAccess(int src_device, int dst_device);

private:
  std::unordered_map<int, std::vector<void*>> device_allocations_;
  std::vector<void*> unified_allocations_;
  std::vector<void*> pinned_allocations_;

  void checkHipError(hipError_t error, const std::string& operation) const;
  void checkHsaError(hsa_status_t status, const std::string& operation) const;
};

class ROCmKernelManager {
public:
  ROCmKernelManager();
  ~ROCmKernelManager();

  hipModule_t loadModule(const std::string& file_path);
  hipFunction_t getFunction(hipModule_t module, const std::string& function_name);

  void launchKernel(hipFunction_t function, const dim3& grid, const dim3& block,
                   void** args, size_t shared_mem = 0, hipStream_t stream = 0);

  hipStream_t createStream();
  void destroyStream(hipStream_t stream);
  void synchronizeStream(hipStream_t stream);
  void recordEvent(hipEvent_t event, hipStream_t stream);
  void waitForEvent(hipEvent_t event);

private:
  std::vector<hipModule_t> modules_;
  std::vector<hipStream_t> streams_;

  void checkHipError(hipError_t error, const std::string& operation) const;
};

class ROCmDevice {
public:
  ROCmDevice();
  ROCmDevice(int device_id);
  ~ROCmDevice();

  void initialize(int device_id);
  void shutdown();

  int getDeviceId() const { return device_id_; }
  const std::string& getDeviceName() const { return device_name_; }
  const std::string& getGcnArch() const { return gcn_arch_; }

  size_t getTotalMemory() const { return total_memory_; }
  size_t getAvailableMemory() const;

  int getComputeUnits() const { return compute_units_; }
  int getMaxThreadsPerBlock() const { return max_threads_per_block_; }
  int getMaxThreadsPerSM() const { return max_threads_per_sm_; }
  int getWarpSize() const { return warp_size_; }

  float getMemoryClockRate() const { return memory_clock_rate_; }
  float getMemoryBandwidth() const { return memory_bandwidth_; }
  float getComputeClockRate() const { return compute_clock_rate_; }

  void setDevice() const;
  void synchronize() const;

  hipStream_t getDefaultStream() const { return default_stream_; }

  ROCmMemoryManager* getMemoryManager() { return memory_manager_.get(); }
  ROCmKernelManager* getKernelManager() { return kernel_manager_.get(); }

  bool supportsUnifiedAddressing() const { return supports_unified_addressing_; }
  bool supportsManagedMemory() const { return supports_managed_memory_; }
  bool supportsP2P() const { return supports_p2p_; }

  void enableP2P(int other_device);

private:
  int device_id_;
  bool initialized_;

  std::string device_name_;
  std::string gcn_arch_;

  size_t total_memory_;
  int compute_units_;
  int max_threads_per_block_;
  int max_threads_per_sm_;
  int warp_size_;

  float memory_clock_rate_;
  float memory_bandwidth_;
  float compute_clock_rate_;

  bool supports_unified_addressing_;
  bool supports_managed_memory_;
  bool supports_p2p_;

  hipDeviceProp_t device_props_;
  hipStream_t default_stream_;

  std::unique_ptr<ROCmMemoryManager> memory_manager_;
  std::unique_ptr<ROCmKernelManager> kernel_manager_;

  void queryDeviceProperties();
  void calculateMemoryBandwidth();
  void checkHipError(hipError_t error, const std::string& operation) const;
};

class ROCmDeviceManager {
public:
  ROCmDeviceManager();
  ~ROCmDeviceManager();

  void initialize();
  void shutdown();

  int getDeviceCount() const { return device_count_; }
  bool isInitialized() const { return initialized_; }

  ROCmDevice* getDevice(int device_id);
  int getDefaultDevice() const { return default_device_; }
  void setDefaultDevice(int device_id);

  std::vector<int> getDeviceList() const;
  std::string getDeviceSummary() const;

  bool canAccessPeer(int src_device, int dst_device) const;
  void enableP2PBetweenAllDevices();

private:
  bool initialized_;
  int device_count_;
  int default_device_;

  std::vector<std::unique_ptr<ROCmDevice>> devices_;

  void initializeDevices();
  void shutdownDevices();
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

private:
  int device_id_;
  bool initialized_;

  std::unique_ptr<ROCmDevice> hip_device_;
  std::unique_ptr<ROCmDeviceManager> device_manager_;

  void checkDeviceAvailable();
};

class ROCmTopologyBuilder {
public:
  static void buildP2PTopology(std::vector<DeviceInfo>& devices,
                               std::vector<LinkInfo>& links);

private:
  static bool isP2PSupported(int device1, int device2);
  static float estimateP2PBandwidth(int device1, int device2);
  static float estimateP2PLatency(int device1, int device2);
};

}