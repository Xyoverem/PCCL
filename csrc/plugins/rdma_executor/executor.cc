#include <plugins/rdma_executor/executor.h>
#include <plugins/rdma_executor/device.h>
#include <plugins/rdma_executor/utils/rdma_utils.h>
#include <cstring>
#include <thread>
#include <chrono>

namespace engine_c {

struct TMP_RDMA_EXECUTOR_STATIC_INITIALIZER {
  TMP_RDMA_EXECUTOR_STATIC_INITIALIZER() {
    auto executor = TypeRegistry::registerExecutorType("rdma");
    regExec(executor, std::make_shared<DeviceExecutor>(new RdmaExecutor()));
  }
};

static TMP_RDMA_EXECUTOR_STATIC_INITIALIZER _____rdma_executor_tmp;

RdmaExecutor::RdmaExecutor() : initialized_(false), device_id_(0) {}

RdmaExecutor::RdmaExecutor(int device_id) : initialized_(false), device_id_(device_id) {}

RdmaExecutor::~RdmaExecutor() {
  shutdown();
}

void RdmaExecutor::initialize() {
  if (initialized_) {
    return;
  }

  rdma_device_ = std::make_unique<RdmaDevice>();
  memory_manager_ = std::make_unique<RdmaMemoryManager>();
  connection_manager_ = std::make_unique<RdmaConnectionManager>();

  if (!rdma_device_->remoteCommAvailable()) {
    return;
  }

  memory_manager_->initialize(device_id_);
  connection_manager_->initialize(device_id_);

  initialized_ = true;
}

void RdmaExecutor::shutdown() {
  if (!initialized_) {
    return;
  }

  connection_manager_.reset();
  memory_manager_.reset();
  rdma_device_.reset();

  initialized_ = false;
}

void* RdmaExecutor::allocate(size_t size) {
  if (!initialized_) {
    return nullptr;
  }

  return memory_manager_->allocate(size);
}

void RdmaExecutor::free(void* ptr) {
  if (!initialized_) {
    return;
  }

  memory_manager_->free(ptr);
}

void RdmaExecutor::copy(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  std::memcpy(dst, src, size);
}

void RdmaExecutor::copyFromHost(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  std::memcpy(dst, src, size);
}

void RdmaExecutor::copyToHost(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  std::memcpy(dst, src, size);
}

void RdmaExecutor::execute(const void* kernel, const void* args, size_t arg_size,
                          dim3 grid, dim3 block, size_t shared_mem) {
  if (!initialized_) {
    return;
  }
}

void RdmaExecutor::synchronize() {
  if (!initialized_) {
    return;
  }

  std::this_thread::yield();
}

double RdmaExecutor::measureKernelTime(const void* kernel, const void* args, size_t arg_size,
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

bool RdmaExecutor::supportsP2P() const {
  return initialized_ && rdma_device_->remoteCommAvailable();
}

void RdmaExecutor::enableP2P(int other_device_id) {
  if (!initialized_) {
    return;
  }

  connection_manager_->enableP2P(other_device_id);
}

void* RdmaExecutor::getDevicePointer() const {
  return nullptr;
}

int RdmaExecutor::getDeviceId() const {
  return device_id_;
}

RdmaDevice* RdmaExecutor::getRdmaDevice() {
  return rdma_device_.get();
}

RdmaMemoryManager* RdmaExecutor::getMemoryManager() {
  return memory_manager_.get();
}

RdmaConnectionManager* RdmaExecutor::getConnectionManager() {
  return connection_manager_.get();
}

RdmaMemoryManager::RdmaMemoryManager() : initialized_(false), allocated_bytes_(0) {}

RdmaMemoryManager::~RdmaMemoryManager() {
  clear();
}

void RdmaMemoryManager::initialize(int device_id) {
  device_id_ = device_id;
  initialized_ = true;
}

void* RdmaMemoryManager::allocate(size_t size) {
  if (!initialized_ || size == 0) {
    return nullptr;
  }

  void* ptr = std::aligned_alloc(64, size);
  if (!ptr) {
    return nullptr;
  }

  std::memset(ptr, 0, size);

  std::lock_guard<std::mutex> lock(mutex_);
  allocated_bytes_ += size;
  allocations_[ptr] = {size, nullptr};

  return ptr;
}

void RdmaMemoryManager::free(void* ptr) {
  if (!ptr) return;

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(ptr);
  if (it != allocations_.end()) {
    allocated_bytes_ -= it->second.size;
    allocations_.erase(it);
    std::free(ptr);
  }
}

std::string RdmaMemoryManager::registerBuffer(void* addr, size_t size) {
  if (!initialized_ || !addr) {
    return "";
  }

  RdmaDevice rdma_device;
  if (!rdma_device.remoteCommAvailable()) {
    return "";
  }

  rdma_device.activate();

  std::string handle = rdma_device.registerBuffer(addr, size);

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(addr);
  if (it != allocations_.end()) {
    it->second.handle = std::make_unique<std::string>(handle);
  }

  return handle;
}

bool RdmaMemoryManager::unregisterBuffer(void* addr) {
  if (!initialized_ || !addr) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(addr);
  if (it != allocations_.end()) {
    it->second.handle.reset();
    return true;
  }

  return false;
}

size_t RdmaMemoryManager::getAllocatedBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocated_bytes_;
}

size_t RdmaMemoryManager::getAllocationCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocations_.size();
}

void RdmaMemoryManager::clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& pair : allocations_) {
    std::free(pair.first);
  }

  allocations_.clear();
  allocated_bytes_ = 0;
}

RdmaConnectionManager::RdmaConnectionManager() : initialized_(false), device_id_(0) {}

RdmaConnectionManager::~RdmaConnectionManager() {
  clear();
}

void RdmaConnectionManager::initialize(int device_id) {
  device_id_ = device_id;
  initialized_ = true;
}

std::string RdmaConnectionManager::createConnection() {
  if (!initialized_) {
    return "";
  }

  RdmaDevice rdma_device;
  if (!rdma_device.remoteCommAvailable()) {
    return "";
  }

  return rdma_device.registerLocal();
}

void RdmaConnectionManager::connectToPeer(const std::string& peer_handle) {
  if (!initialized_ || peer_handle.empty()) {
    return;
  }

  RdmaDevice rdma_device;
  rdma_device.connect(peer_handle);

  std::lock_guard<std::mutex> lock(mutex_);
  connections_.insert(peer_handle);
}

void RdmaConnectionManager::disconnectFromPeer(const std::string& peer_handle) {
  if (!initialized_ || peer_handle.empty()) {
    return;
  }

  RdmaDevice rdma_device;
  rdma_device.disconnect(peer_handle);

  std::lock_guard<std::mutex> lock(mutex_);
  connections_.erase(peer_handle);
}

bool RdmaConnectionManager::isConnected(const std::string& peer_handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return connections_.find(peer_handle) != connections_.end();
}

std::vector<std::string> RdmaConnectionManager::getConnectedPeers() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return std::vector<std::string>(connections_.begin(), connections_.end());
}

void RdmaConnectionManager::enableP2P(int other_device_id) {
  if (!initialized_) {
    return;
  }
}

void RdmaConnectionManager::clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& peer_handle : connections_) {
    RdmaDevice rdma_device;
    rdma_device.disconnect(peer_handle);
  }

  connections_.clear();
}

size_t RdmaConnectionManager::getConnectionCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return connections_.size();
}

}