#pragma once

#include <base/device.h>
#include <base/executor.h>
#include <base/registry.h>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <string>

namespace engine_c {

class RdmaMemoryManager {
public:
  RdmaMemoryManager();
  ~RdmaMemoryManager();

  void initialize(int device_id);

  void* allocate(size_t size);
  void free(void* ptr);

  std::string registerBuffer(void* addr, size_t size);
  bool unregisterBuffer(void* addr);

  size_t getAllocatedBytes() const;
  size_t getAllocationCount() const;
  void clear();

private:
  bool initialized_;
  int device_id_;
  std::unordered_map<void*, std::pair<size_t, std::unique_ptr<std::string>>> allocations_;
  size_t allocated_bytes_;
  mutable std::mutex mutex_;
};

class RdmaConnectionManager {
public:
  RdmaConnectionManager();
  ~RdmaConnectionManager();

  void initialize(int device_id);

  std::string createConnection();
  void connectToPeer(const std::string& peer_handle);
  void disconnectFromPeer(const std::string& peer_handle);

  bool isConnected(const std::string& peer_handle) const;
  std::vector<std::string> getConnectedPeers() const;

  void enableP2P(int other_device_id);
  void clear();

  size_t getConnectionCount() const;

private:
  bool initialized_;
  int device_id_;
  std::unordered_set<std::string> connections_;
  mutable std::mutex mutex_;
};

class RdmaExecutor : public DeviceExecutor {
public:
  RdmaExecutor();
  explicit RdmaExecutor(int device_id);
  ~RdmaExecutor() override;

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

  RdmaDevice* getRdmaDevice();
  RdmaMemoryManager* getMemoryManager();
  RdmaConnectionManager* getConnectionManager();

private:
  bool initialized_;
  int device_id_;

  std::unique_ptr<RdmaDevice> rdma_device_;
  std::unique_ptr<RdmaMemoryManager> memory_manager_;
  std::unique_ptr<RdmaConnectionManager> connection_manager_;
};

}