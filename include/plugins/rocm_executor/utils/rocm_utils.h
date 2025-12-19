#pragma once

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hsa/hsa.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace engine_c {

class ROCmUtils {
public:
  static bool initialize();
  static void shutdown();

  static bool isROCmAvailable();
  static int getDeviceCount();

  static std::string getDeviceName(int device_id);
  static std::string getGCNArchitecture(int device_id);
  static std::vector<std::string> getDeviceExtensions(int device_id);

  static size_t getDeviceMemory(int device_id);
  static size_t getDeviceAvailableMemory(int device_id);

  static int getComputeUnits(int device_id);
  static int getMaxThreadsPerBlock(int device_id);
  static int getMaxThreadsPerSM(int device_id);
  static int getWarpSize(int device_id);

  static float getMemoryClockRate(int device_id);
  static float getComputeClockRate(int device_id);
  static float getMemoryBandwidth(int device_id);

  static bool supportsUnifiedAddressing(int device_id);
  static bool supportsManagedMemory(int device_id);
  static bool supportsP2P(int device_id, int other_device_id);
  static bool supportsCooperativeLaunch(int device_id);

  static void enablePeerAccess(int device_id, int peer_device_id);
  static void disablePeerAccess(int device_id, int peer_device_id);

  static hipError_t checkError(hipError_t error, const std::string& operation);
  static hiprtcResult checkRTCError(hiprtcResult error, const std::string& operation);
  static hsa_status_t checkHSAError(hsa_status_t status, const std::string& operation);

  static void setDevice(int device_id);
  static void synchronizeDevice(int device_id);
  static void resetDevice(int device_id);

  static hipStream_t createStream(hipStreamFlags flags = hipStreamNonBlocking);
  static void destroyStream(hipStream_t stream);
  static void streamSynchronize(hipStream_t stream);
  static void streamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData);

  static hipEvent_t createEvent(hipEventFlags flags = hipEventDefault);
  static void destroyEvent(hipEvent_t event);
  static void recordEvent(hipEvent_t event, hipStream_t stream = 0);
  static void waitForEvent(hipEvent_t event, hipStream_t stream = 0);
  static float getEventElapsedTime(hipEvent_t start, hipEvent_t stop);

  static void* allocateMemory(size_t size, int device_id = -1);
  static void freeMemory(void* ptr);
  static void* allocateUnifiedMemory(size_t size);
  static void freeUnifiedMemory(void* ptr);
  static void* allocatePinnedMemory(size_t size);
  static void freePinnedMemory(void* ptr);

  static void copyMemoryHtoD(void* dst, const void* src, size_t size, int device_id = -1);
  static void copyMemoryDtoH(void* dst, const void* src, size_t size, int device_id = -1);
  static void copyMemoryDtoD(void* dst, const void* src, size_t size, int src_device_id = -1, int dst_device_id = -1);

  static bool compileKernel(const std::string& source, const std::string& kernel_name, hipFunction_t* function);
  static bool compileKernelPTX(const std::string& ptx_source, const std::string& kernel_name, hipFunction_t* function);

  static std::string getROCmVersion();
  static std::string getHIPVersion();
  static std::string getROCBLASVersion();
  static std::string getrocRANDVersion();

  static bool isAmdArchitecture(const std::string& arch);
  static bool isGCNArch(const std::string& arch);
  static int getArchComputeCapability(const std::string& arch);

private:
  static bool initialized_;
  static std::unordered_map<int, hipDeviceProp_t> device_properties_;
  static std::mutex properties_mutex_;

  static void initializeDeviceProperties();
  static hipDeviceProp_t& getDeviceProperties(int device_id);
};

class ROCmMemoryPool {
public:
  ROCmMemoryPool(int device_id, size_t initial_size = 1024 * 1024 * 1024);
  ~ROCmMemoryPool();

  void* allocate(size_t size);
  void free(void* ptr);

  size_t getTotalSize() const { return total_size_; }
  size_t getUsedSize() const { return used_size_; }
  size_t getFreeSize() const { return total_size_ - used_size_; }

  void reset();
  void trim();

private:
  int device_id_;
  size_t total_size_;
  size_t used_size_;
  char* pool_memory_;

  struct Block {
    void* ptr;
    size_t size;
    bool free;
    Block* next;
  };

  Block* blocks_;
  std::mutex pool_mutex_;

  void coalesceFreeBlocks();
  Block* findFreeBlock(size_t size);
};

class ROCmKernelCache {
public:
  ROCmKernelCache();
  ~ROCmKernelCache();

  hipFunction_t getKernel(const std::string& kernel_name, const std::string& source);
  void clearCache();

  size_t getCacheSize() const { return cache_.size(); }

private:
  struct CacheEntry {
    hipModule_t module;
    hipFunction_t function;
    std::string source;
  };

  std::unordered_map<std::string, CacheEntry> cache_;
  std::mutex cache_mutex_;
};

class ROCmProfiler {
public:
  ROCmProfiler();
  ~ROCmProfiler();

  void startProfile();
  void stopProfile();
  void resetProfile();

  void recordKernelLaunch(const std::string& kernel_name, dim3 grid, dim3 block, size_t shared_mem);
  void recordMemoryTransfer(size_t size, bool is_host_to_device);

  struct KernelStats {
    std::string name;
    int launches;
    double total_time;
    double min_time;
    double max_time;
    size_t total_shared_mem;
  };

  struct MemoryStats {
    size_t total_transfers;
    size_t total_bytes;
    size_t host_to_device_bytes;
    size_t device_to_host_bytes;
    size_t device_to_device_bytes;
  };

  std::vector<KernelStats> getKernelStats() const;
  MemoryStats getMemoryStats() const;

  void printProfileSummary() const;

private:
  bool profiling_;
  std::unordered_map<std::string, KernelStats> kernel_stats_;
  MemoryStats memory_stats_;

  std::mutex stats_mutex_;
};

class ROCmTopology {
public:
  static std::vector<int> getDeviceList();
  static std::vector<std::vector<bool>> getP2PMatrix();
  static std::vector<std::vector<float>> getBandwidthMatrix();
  static std::vector<std::vector<float>> getLatencyMatrix();

  static bool devicesInSameNode(int device1, int device2);
  static float estimateNVLinkBandwidth(int device1, int device2);
  static float estimatePCIEBandwidth(int device1, int device2);

  static std::string getInterconnectType(int device1, int device2);

private:
  static bool detectNVLink(int device1, int device2);
  static bool detectPCIE(int device1, int device2);
};

class ROCmTimer {
public:
  ROCmTimer();
  ~ROCmTimer();

  void start();
  void stop();
  void reset();

  double getElapsedSeconds() const;
  double getElapsedMilliseconds() const;

private:
  hipEvent_t start_event_;
  hipEvent_t stop_event_;
  hipStream_t stream_;
  bool started_;
};

class ROCmStreamPool {
public:
  ROCmStreamPool(int device_id, size_t initial_size = 4);
  ~ROCmStreamPool();

  hipStream_t acquireStream();
  void releaseStream(hipStream_t stream);

  size_t getPoolSize() const;
  size_t getAvailableStreams() const;

private:
  int device_id_;
  std::vector<hipStream_t> streams_;
  std::vector<bool> available_;
  std::mutex pool_mutex_;
};

class ROCmEventPool {
public:
  ROCmEventPool(size_t initial_size = 4);
  ~ROCmEventPool();

  hipEvent_t acquireEvent();
  void releaseEvent(hipEvent_t event);

  size_t getPoolSize() const;
  size_t getAvailableEvents() const;

private:
  std::vector<hipEvent_t> events_;
  std::vector<bool> available_;
  std::mutex pool_mutex_;
};

inline bool ROCmUtils::isROCmAvailable() {
  int count;
  hipError_t error = hipGetDeviceCount(&count);
  return error == hipSuccess && count > 0;
}

inline hipStream_t ROCmUtils::createStream(hipStreamFlags flags) {
  hipStream_t stream;
  hipError_t error = hipStreamCreateWithFlags(&stream, flags);
  checkError(error, "hipStreamCreateWithFlags");
  return stream;
}

inline void ROCmUtils::destroyStream(hipStream_t stream) {
  hipError_t error = hipStreamDestroy(stream);
  checkError(error, "hipStreamDestroy");
}

inline hipEvent_t ROCmUtils::createEvent(hipEventFlags flags) {
  hipEvent_t event;
  hipError_t error = hipEventCreateWithFlags(&event, flags);
  checkError(error, "hipEventCreateWithFlags");
  return event;
}

inline void ROCmUtils::destroyEvent(hipEvent_t event) {
  hipError_t error = hipEventDestroy(event);
  checkError(error, "hipEventDestroy");
}

}