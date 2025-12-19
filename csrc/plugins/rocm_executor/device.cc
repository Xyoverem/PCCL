#include <plugins/rocm_executor/device.h>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace engine_c {

ROCmMemoryManager::ROCmMemoryManager() {}

ROCmMemoryManager::~ROCmMemoryManager() {
  for (auto& pair : device_allocations_) {
    for (void* ptr : pair.second) {
      hipFree(ptr);
    }
  }

  for (void* ptr : unified_allocations_) {
    hipFree(ptr);
  }

  for (void* ptr : pinned_allocations_) {
    hipHostFree(ptr);
  }
}

void* ROCmMemoryManager::allocate(size_t size, int hip_device) {
  hipSetDevice(hip_device);

  void* ptr;
  hipError_t error = hipMalloc(&ptr, size);
  checkHipError(error, "hipMalloc");

  device_allocations_[hip_device].push_back(ptr);
  return ptr;
}

void ROCmMemoryManager::free(void* ptr, int hip_device) {
  auto it = device_allocations_.find(hip_device);
  if (it != device_allocations_.end()) {
    auto vec_it = std::find(it->second.begin(), it->second.end(), ptr);
    if (vec_it != it->second.end()) {
      hipFree(ptr);
      it->second.erase(vec_it);
      return;
    }
  }

  hipFree(ptr);
}

void* ROCmMemoryManager::allocateUnified(size_t size) {
  void* ptr;
  hipError_t error = hipMallocManaged(&ptr, size);
  checkHipError(error, "hipMallocManaged");

  unified_allocations_.push_back(ptr);
  return ptr;
}

void ROCmMemoryManager::freeUnified(void* ptr) {
  auto it = std::find(unified_allocations_.begin(), unified_allocations_.end(), ptr);
  if (it != unified_allocations_.end()) {
    hipFree(ptr);
    unified_allocations_.erase(it);
  } else {
    hipFree(ptr);
  }
}

void* ROCmMemoryManager::allocatePinned(size_t size) {
  void* ptr;
  hipError_t error = hipHostMalloc(&ptr, size);
  checkHipError(error, "hipHostMalloc");

  pinned_allocations_.push_back(ptr);
  return ptr;
}

void ROCmMemoryManager::freePinned(void* ptr) {
  auto it = std::find(pinned_allocations_.begin(), pinned_allocations_.end(), ptr);
  if (it != pinned_allocations_.end()) {
    hipHostFree(ptr);
    pinned_allocations_.erase(it);
  } else {
    hipHostFree(ptr);
  }
}

void ROCmMemoryManager::copyToDevice(void* dst, const void* src, size_t size, int hip_device) {
  hipSetDevice(hip_device);
  hipError_t error = hipMemcpyHtoD(dst, src, size);
  checkHipError(error, "hipMemcpyHtoD");
}

void ROCmMemoryManager::copyFromDevice(void* dst, const void* src, size_t size, int hip_device) {
  hipSetDevice(hip_device);
  hipError_t error = hipMemcpyDtoH(dst, src, size);
  checkHipError(error, "hipMemcpyDtoH");
}

void ROCmMemoryManager::copyDeviceToDevice(void* dst, const void* src, size_t size,
                                          int src_device, int dst_device) {
  hipError_t error;
  if (src_device == dst_device) {
    hipSetDevice(src_device);
    error = hipMemcpyDtoD(dst, src, size);
  } else {
    error = hipMemcpyDtoDAsync(dst, src, size, 0);
  }
  checkHipError(error, "hipMemcpyDtoD");
}

bool ROCmMemoryManager::canAccessPeer(int src_device, int dst_device) const {
  int can_access;
  hipError_t error = hipDeviceCanAccessPeer(&can_access, src_device, dst_device);
  checkHipError(error, "hipDeviceCanAccessPeer");
  return can_access != 0;
}

void ROCmMemoryManager::enablePeerAccess(int src_device, int dst_device) {
  hipError_t error = hipDeviceEnablePeerAccess(dst_device);
  if (error != hipErrorPeerAccessAlreadyEnabled) {
    checkHipError(error, "hipDeviceEnablePeerAccess");
  }
}

void ROCmMemoryManager::checkHipError(hipError_t error, const std::string& operation) const {
  if (error != hipSuccess) {
    std::cerr << "ROCm Error in " << operation << ": " << hipGetErrorString(error) << std::endl;
  }
}

void ROCmMemoryManager::checkHsaError(hsa_status_t status, const std::string& operation) const {
  if (status != HSA_STATUS_SUCCESS) {
    std::cerr << "HSA Error in " << operation << ": " << static_cast<int>(status) << std::endl;
  }
}

ROCmKernelManager::ROCmKernelManager() {}

ROCmKernelManager::~ROCmKernelManager() {
  for (hipModule_t module : modules_) {
    hipModuleUnload(module);
  }

  for (hipStream_t stream : streams_) {
    hipStreamDestroy(stream);
  }
}

hipModule_t ROCmKernelManager::loadModule(const std::string& file_path) {
  hipModule_t module;
  hipError_t error = hipModuleLoad(&module, file_path.c_str());
  checkHipError(error, "hipModuleLoad");

  modules_.push_back(module);
  return module;
}

hipFunction_t ROCmKernelManager::getFunction(hipModule_t module, const std::string& function_name) {
  hipFunction_t function;
  hipError_t error = hipModuleGetFunction(&function, module, function_name.c_str());
  checkHipError(error, "hipModuleGetFunction");

  return function;
}

void ROCmKernelManager::launchKernel(hipFunction_t function, const dim3& grid, const dim3& block,
                                    void** args, size_t shared_mem, hipStream_t stream) {
  hipError_t error = hipModuleLaunchKernel(function, grid.x, grid.y, grid.z,
                                           block.x, block.y, block.z,
                                           shared_mem, stream, args);
  checkHipError(error, "hipModuleLaunchKernel");
}

hipStream_t ROCmKernelManager::createStream() {
  hipStream_t stream;
  hipError_t error = hipStreamCreate(&stream);
  checkHipError(error, "hipStreamCreate");

  streams_.push_back(stream);
  return stream;
}

void ROCmKernelManager::destroyStream(hipStream_t stream) {
  auto it = std::find(streams_.begin(), streams_.end(), stream);
  if (it != streams_.end()) {
    hipStreamDestroy(stream);
    streams_.erase(it);
  } else {
    hipStreamDestroy(stream);
  }
}

void ROCmKernelManager::synchronizeStream(hipStream_t stream) {
  hipError_t error = hipStreamSynchronize(stream);
  checkHipError(error, "hipStreamSynchronize");
}

void ROCmKernelManager::recordEvent(hipEvent_t event, hipStream_t stream) {
  hipError_t error = hipEventRecord(event, stream);
  checkHipError(error, "hipEventRecord");
}

void ROCmKernelManager::waitForEvent(hipEvent_t event) {
  hipError_t error = hipEventSynchronize(event);
  checkHipError(error, "hipEventSynchronize");
}

void ROCmKernelManager::checkHipError(hipError_t error, const std::string& operation) const {
  if (error != hipSuccess) {
    std::cerr << "ROCm Error in " << operation << ": " << hipGetErrorString(error) << std::endl;
  }
}

ROCmDevice::ROCmDevice() : device_id_(-1), initialized_(false), total_memory_(0),
                          compute_units_(0), max_threads_per_block_(0), max_threads_per_sm_(0),
                          warp_size_(0), memory_clock_rate_(0.0f), memory_bandwidth_(0.0f),
                          compute_clock_rate_(0.0f), supports_unified_addressing_(false),
                          supports_managed_memory_(false), supports_p2p_(false),
                          default_stream_(0) {}

ROCmDevice::ROCmDevice(int device_id) : device_id_(device_id), initialized_(false),
                                      total_memory_(0), compute_units_(0),
                                      max_threads_per_block_(0), max_threads_per_sm_(0),
                                      warp_size_(0), memory_clock_rate_(0.0f),
                                      memory_bandwidth_(0.0f), compute_clock_rate_(0.0f),
                                      supports_unified_addressing_(false),
                                      supports_managed_memory_(false),
                                      supports_p2p_(false), default_stream_(0) {}

ROCmDevice::~ROCmDevice() {
  shutdown();
}

void ROCmDevice::initialize(int device_id) {
  if (initialized_) {
    return;
  }

  device_id_ = device_id;

  hipError_t error = hipSetDevice(device_id_);
  checkHipError(error, "hipSetDevice");

  queryDeviceProperties();
  calculateMemoryBandwidth();

  error = hipStreamCreateWithFlags(&default_stream_, hipStreamNonBlocking);
  checkHipError(error, "hipStreamCreateWithFlags");

  memory_manager_ = std::make_unique<ROCmMemoryManager>();
  kernel_manager_ = std::make_unique<ROCmKernelManager>();

  initialized_ = true;
}

void ROCmDevice::shutdown() {
  if (!initialized_) {
    return;
  }

  if (default_stream_ != 0) {
    hipStreamDestroy(default_stream_);
    default_stream_ = 0;
  }

  memory_manager_.reset();
  kernel_manager_.reset();

  initialized_ = false;
}

size_t ROCmDevice::getAvailableMemory() const {
  size_t free, total;
  hipError_t error = hipMemGetInfo(&free, &total);
  checkHipError(error, "hipMemGetInfo");
  return free;
}

void ROCmDevice::setDevice() const {
  hipError_t error = hipSetDevice(device_id_);
  checkHipError(error, "hipSetDevice");
}

void ROCmDevice::synchronize() const {
  hipError_t error = hipDeviceSynchronize();
  checkHipError(error, "hipDeviceSynchronize");
}

void ROCmDevice::enableP2P(int other_device) {
  if (memory_manager_) {
    memory_manager_->enablePeerAccess(device_id_, other_device);
    memory_manager_->enablePeerAccess(other_device, device_id_);
  }
}

void ROCmDevice::queryDeviceProperties() {
  hipError_t error = hipGetDeviceProperties(&device_props_, device_id_);
  checkHipError(error, "hipGetDeviceProperties");

  device_name_ = device_props_.name;
  gcn_arch_ = device_props_.gcnArch;
  total_memory_ = device_props_.totalGlobalMem;
  compute_units_ = device_props_.multiProcessorCount;
  max_threads_per_block_ = device_props_.maxThreadsPerBlock;
  warp_size_ = device_props_.warpSize;

  memory_clock_rate_ = device_props_.memoryClockRate / 1000.0f;
  compute_clock_rate_ = device_props_.clockRate / 1000.0f;

  supports_unified_addressing_ = device_props_.unifiedAddressing;
  supports_managed_memory_ = device_props_.managedMemory;
  supports_p2p_ = device_props_.canMapHostMemory;

  max_threads_per_sm_ = device_props_.maxThreadsPerMultiProcessor;
}

void ROCmDevice::calculateMemoryBandwidth() {
  memory_bandwidth_ = (memory_clock_rate_ * 1000.0f * 256.0f) / (1024.0f * 1024.0f * 1024.0f);
}

void ROCmDevice::checkHipError(hipError_t error, const std::string& operation) const {
  if (error != hipSuccess) {
    std::cerr << "ROCm Device Error in " << operation << ": " << hipGetErrorString(error) << std::endl;
  }
}

ROCmDeviceManager::ROCmDeviceManager() : initialized_(false), device_count_(0), default_device_(0) {}

ROCmDeviceManager::~ROCmDeviceManager() {
  shutdown();
}

void ROCmDeviceManager::initialize() {
  if (initialized_) {
    return;
  }

  hipError_t error = hipGetDeviceCount(&device_count_);
  checkHipError(error, "hipGetDeviceCount");

  if (device_count_ == 0) {
    std::cerr << "No ROCm devices found" << std::endl;
    return;
  }

  initializeDevices();
  initialized_ = true;
}

void ROCmDeviceManager::shutdown() {
  if (!initialized_) {
    return;
  }

  shutdownDevices();
  initialized_ = false;
}

ROCmDevice* ROCmDeviceManager::getDevice(int device_id) {
  if (device_id < 0 || device_id >= device_count_) {
    return nullptr;
  }

  if (!devices_[device_id]) {
    devices_[device_id] = std::make_unique<ROCmDevice>(device_id);
    devices_[device_id]->initialize(device_id);
  }

  return devices_[device_id].get();
}

void ROCmDeviceManager::setDefaultDevice(int device_id) {
  if (device_id >= 0 && device_id < device_count_) {
    default_device_ = device_id;
    hipError_t error = hipSetDevice(device_id);
    checkHipError(error, "hipSetDevice");
  }
}

std::vector<int> ROCmDeviceManager::getDeviceList() const {
  std::vector<int> devices;
  for (int i = 0; i < device_count_; ++i) {
    devices.push_back(i);
  }
  return devices;
}

std::string ROCmDeviceManager::getDeviceSummary() const {
  std::string summary = "ROCm Devices:\n";
  summary += "Device Count: " + std::to_string(device_count_) + "\n";
  summary += "Default Device: " + std::to_string(default_device_) + "\n";

  for (int i = 0; i < device_count_; ++i) {
    if (devices_[i]) {
      summary += "Device " + std::to_string(i) + ": " + devices_[i]->getDeviceName() + "\n";
      summary += "  Memory: " + std::to_string(devices_[i]->getTotalMemory() / (1024*1024)) + " MB\n";
      summary += "  Compute Units: " + std::to_string(devices_[i]->getComputeUnits()) + "\n";
      summary += "  GCN Arch: " + devices_[i]->getGcnArch() + "\n";
    }
  }

  return summary;
}

bool ROCmDeviceManager::canAccessPeer(int src_device, int dst_device) const {
  int can_access;
  hipError_t error = hipDeviceCanAccessPeer(&can_access, src_device, dst_device);
  checkHipError(error, "hipDeviceCanAccessPeer");
  return can_access != 0;
}

void ROCmDeviceManager::enableP2PBetweenAllDevices() {
  for (int i = 0; i < device_count_; ++i) {
    for (int j = i + 1; j < device_count_; ++j) {
      if (canAccessPeer(i, j)) {
        hipError_t error = hipDeviceEnablePeerAccess(j);
        if (error != hipErrorPeerAccessAlreadyEnabled) {
          checkHipError(error, "hipDeviceEnablePeerAccess");
        }
      }
    }
  }
}

void ROCmDeviceManager::initializeDevices() {
  devices_.resize(device_count_);

  for (int i = 0; i < device_count_; ++i) {
    devices_[i] = std::make_unique<ROCmDevice>(i);
    devices_[i]->initialize(i);
  }
}

void ROCmDeviceManager::shutdownDevices() {
  for (auto& device : devices_) {
    device.reset();
  }
}

void ROCmDeviceManager::checkHipError(hipError_t error, const std::string& operation) const {
  if (error != hipSuccess) {
    std::cerr << "ROCm Device Manager Error in " << operation << ": " << hipGetErrorString(error) << std::endl;
  }
}

void ROCmTopologyBuilder::buildP2PTopology(std::vector<DeviceInfo>& devices,
                                          std::vector<LinkInfo>& links) {
  ROCmDeviceManager manager;
  manager.initialize();

  std::vector<int> device_list = manager.getDeviceList();
  devices.clear();
  links.clear();

  for (int device_id : device_list) {
    ROCmDevice* device = manager.getDevice(device_id);
    if (device) {
      DeviceInfo info;
      info.device_id = device_id;
      info.device_type = DeviceType::CUDA;
      info.device_name = device->getDeviceName();
      info.memory_bandwidth = device->getMemoryBandwidth();
      info.compute_capability = 8.0f;
      info.memory_size = device->getTotalMemory();
      info.numa_node = device_id % 2;
      devices.push_back(info);
    }
  }

  for (int i = 0; i < device_list.size(); ++i) {
    for (int j = i + 1; j < device_list.size(); ++j) {
      if (manager.canAccessPeer(i, j)) {
        float bandwidth = estimateP2PBandwidth(i, j);
        float latency = estimateP2PLatency(i, j);

        LinkInfo link1, link2;
        link1.src_device = i;
        link1.dst_device = j;
        link1.interconnect_type = InterconnectType::NVLINK;
        link1.bandwidth = bandwidth;
        link1.latency = latency;
        link1.bidirectional = true;

        link2 = link1;
        link2.src_device = j;
        link2.dst_device = i;

        links.push_back(link1);
      }
    }
  }
}

bool ROCmTopologyBuilder::isP2PSupported(int device1, int device2) {
  ROCmDeviceManager manager;
  manager.initialize();
  return manager.canAccessPeer(device1, device2);
}

float ROCmTopologyBuilder::estimateP2PBandwidth(int device1, int device2) {
  ROCmDeviceManager manager;
  manager.initialize();

  ROCmDevice* dev1 = manager.getDevice(device1);
  ROCmDevice* dev2 = manager.getDevice(device2);

  if (dev1 && dev2) {
    float avg_bandwidth = (dev1->getMemoryBandwidth() + dev2->getMemoryBandwidth()) / 2.0f;
    return avg_bandwidth * 0.8f;
  }

  return 25.0f;
}

float ROCmTopologyBuilder::estimateP2PLatency(int device1, int device2) {
  return 0.5f;
}

}