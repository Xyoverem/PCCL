#include <plugins/rocm_executor/utils/rocm_utils.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <cstring>
#include <algorithm>

namespace engine_c {

bool ROCmUtils::initialized_ = false;
std::unordered_map<int, hipDeviceProp_t> ROCmUtils::device_properties_;
std::mutex ROCmUtils::properties_mutex_;

bool ROCmUtils::initialize() {
  if (initialized_) {
    return true;
  }

  hipError_t error = hipInit(0);
  if (error != hipSuccess) {
    std::cerr << "Failed to initialize HIP: " << hipGetErrorString(error) << std::endl;
    return false;
  }

  int device_count;
  error = hipGetDeviceCount(&device_count);
  if (error != hipSuccess || device_count == 0) {
    std::cerr << "No ROCm devices found" << std::endl;
    return false;
  }

  initializeDeviceProperties();
  initialized_ = true;
  return true;
}

void ROCmUtils::shutdown() {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::mutex> lock(properties_mutex_);
  device_properties_.clear();
  initialized_ = false;

  hipError_t error = hipShutdown();
  checkError(error, "hipShutdown");
}

bool ROCmUtils::isROCmAvailable() {
  return isAvailable() && initialized_;
}

int ROCmUtils::getDeviceCount() {
  if (!initialized_) {
    return 0;
  }

  int count;
  hipError_t error = hipGetDeviceCount(&count);
  checkError(error, "hipGetDeviceCount");
  return count;
}

std::string ROCmUtils::getDeviceName(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return std::string(props.name);
}

std::string ROCmUtils::getGCNArchitecture(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return std::string(props.gcnArch);
}

std::vector<std::string> ROCmUtils::getDeviceExtensions(int device_id) {
  std::vector<std::string> extensions;

  hipDeviceProp_t& props = getDeviceProperties(device_id);

  extensions.push_back("cl_khr_fp16");
  extensions.push_back("cl_khr_fp64");
  extensions.push_back("cl_khr_int64_base_atomics");
  extensions.push_back("cl_khr_int64_extended_atomics");

  if (props.canMapHostMemory) {
    extensions.push_back("cl_khr_pci_bus_host");
  }

  return extensions;
}

size_t ROCmUtils::getDeviceMemory(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return props.totalGlobalMem;
}

size_t ROCmUtils::getDeviceAvailableMemory(int device_id) {
  size_t free, total;
  hipError_t error = hipMemGetInfo(&free, &total);
  checkError(error, "hipMemGetInfo");
  return free;
}

int ROCmUtils::getComputeUnits(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return props.multiProcessorCount;
}

int ROCmUtils::getMaxThreadsPerBlock(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return props.maxThreadsPerBlock;
}

int ROCmUtils::getMaxThreadsPerSM(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return props.maxThreadsPerMultiProcessor;
}

int ROCmUtils::getWarpSize(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return props.warpSize;
}

float ROCmUtils::getMemoryClockRate(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return props.memoryClockRate / 1000.0f;
}

float ROCmUtils::getComputeClockRate(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return props.clockRate / 1000.0f;
}

float ROCmUtils::getMemoryBandwidth(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  float memory_clock = props.memoryClockRate / 1000.0f;
  return (memory_clock * 1000.0f * 256.0f) / (1024.0f * 1024.0f * 1024.0f);
}

bool ROCmUtils::supportsUnifiedAddressing(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return props.unifiedAddressing;
}

bool ROCmUtils::supportsManagedMemory(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return props.managedMemory;
}

bool ROCmUtils::supportsP2P(int device_id, int other_device_id) {
  int can_access;
  hipError_t error = hipDeviceCanAccessPeer(&can_access, device_id, other_device_id);
  checkError(error, "hipDeviceCanAccessPeer");
  return can_access != 0;
}

bool ROCmUtils::supportsCooperativeLaunch(int device_id) {
  hipDeviceProp_t& props = getDeviceProperties(device_id);
  return props.cooperativeLaunch;
}

void ROCmUtils::enablePeerAccess(int device_id, int peer_device_id) {
  hipSetDevice(device_id);
  hipError_t error = hipDeviceEnablePeerAccess(peer_device_id);
  if (error != hipErrorPeerAccessAlreadyEnabled) {
    checkError(error, "hipDeviceEnablePeerAccess");
  }
}

void ROCmUtils::disablePeerAccess(int device_id, int peer_device_id) {
  hipSetDevice(device_id);
  hipError_t error = hipDeviceDisablePeerAccess(peer_device_id);
  checkError(error, "hipDeviceDisablePeerAccess");
}

hipError_t ROCmUtils::checkError(hipError_t error, const std::string& operation) {
  if (error != hipSuccess) {
    std::cerr << "HIP Error in " << operation << ": " << hipGetErrorString(error) << std::endl;
  }
  return error;
}

hiprtcResult ROCmUtils::checkRTCError(hiprtcResult error, const std::string& operation) {
  if (error != HIPRTC_SUCCESS) {
    std::cerr << "HIPRTC Error in " << operation << ": " << error << std::endl;
  }
  return error;
}

hsa_status_t ROCmUtils::checkHSAError(hsa_status_t status, const std::string& operation) {
  if (status != HSA_STATUS_SUCCESS) {
    std::cerr << "HSA Error in " << operation << ": " << static_cast<int>(status) << std::endl;
  }
  return status;
}

void ROCmUtils::setDevice(int device_id) {
  hipError_t error = hipSetDevice(device_id);
  checkError(error, "hipSetDevice");
}

void ROCmUtils::synchronizeDevice(int device_id) {
  hipError_t error = hipDeviceSynchronize();
  checkError(error, "hipDeviceSynchronize");
}

void ROCmUtils::resetDevice(int device_id) {
  hipError_t error = hipDeviceReset();
  checkError(error, "hipDeviceReset");
}

hipStream_t ROCmUtils::createStream(hipStreamFlags flags) {
  hipStream_t stream;
  hipError_t error = hipStreamCreateWithFlags(&stream, flags);
  checkError(error, "hipStreamCreateWithFlags");
  return stream;
}

void ROCmUtils::destroyStream(hipStream_t stream) {
  hipError_t error = hipStreamDestroy(stream);
  checkError(error, "hipStreamDestroy");
}

void ROCmUtils::streamSynchronize(hipStream_t stream) {
  hipError_t error = hipStreamSynchronize(stream);
  checkError(error, "hipStreamSynchronize");
}

void ROCmUtils::streamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData) {
  hipError_t error = hipStreamAddCallback(stream, callback, userData);
  checkError(error, "hipStreamAddCallback");
}

hipEvent_t ROCmUtils::createEvent(hipEventFlags flags) {
  hipEvent_t event;
  hipError_t error = hipEventCreateWithFlags(&event, flags);
  checkError(error, "hipEventCreateWithFlags");
  return event;
}

void ROCmUtils::destroyEvent(hipEvent_t event) {
  hipError_t error = hipEventDestroy(event);
  checkError(error, "hipEventDestroy");
}

void ROCmUtils::recordEvent(hipEvent_t event, hipStream_t stream) {
  hipError_t error = hipEventRecord(event, stream);
  checkError(error, "hipEventRecord");
}

void ROCmUtils::waitForEvent(hipEvent_t event, hipStream_t stream) {
  hipError_t error = hipStreamWaitEvent(stream, event);
  checkError(error, "hipStreamWaitEvent");
}

float ROCmUtils::getEventElapsedTime(hipEvent_t start, hipEvent_t stop) {
  float milliseconds;
  hipError_t error = hipEventElapsedTime(&milliseconds, start, stop);
  checkError(error, "hipEventElapsedTime");
  return milliseconds;
}

void* ROCmUtils::allocateMemory(size_t size, int device_id) {
  if (device_id >= 0) {
    hipSetDevice(device_id);
  }

  void* ptr;
  hipError_t error = hipMalloc(&ptr, size);
  checkError(error, "hipMalloc");
  return ptr;
}

void ROCmUtils::freeMemory(void* ptr) {
  hipError_t error = hipFree(ptr);
  checkError(error, "hipFree");
}

void* ROCmUtils::allocateUnifiedMemory(size_t size) {
  void* ptr;
  hipError_t error = hipMallocManaged(&ptr, size);
  checkError(error, "hipMallocManaged");
  return ptr;
}

void ROCmUtils::freeUnifiedMemory(void* ptr) {
  hipError_t error = hipFree(ptr);
  checkError(error, "hipFree");
}

void* ROCmUtils::allocatePinnedMemory(size_t size) {
  void* ptr;
  hipError_t error = hipHostMalloc(&ptr, size);
  checkError(error, "hipHostMalloc");
  return ptr;
}

void ROCmUtils::freePinnedMemory(void* ptr) {
  hipError_t error = hipHostFree(ptr);
  checkError(error, "hipHostFree");
}

void ROCmUtils::copyMemoryHtoD(void* dst, const void* src, size_t size, int device_id) {
  if (device_id >= 0) {
    hipSetDevice(device_id);
  }

  hipError_t error = hipMemcpyHtoD(dst, src, size);
  checkError(error, "hipMemcpyHtoD");
}

void ROCmUtils::copyMemoryDtoH(void* dst, const void* src, size_t size, int device_id) {
  if (device_id >= 0) {
    hipSetDevice(device_id);
  }

  hipError_t error = hipMemcpyDtoH(dst, src, size);
  checkError(error, "hipMemcpyDtoH");
}

void ROCmUtils::copyMemoryDtoD(void* dst, const void* src, size_t size, int src_device_id, int dst_device_id) {
  if (src_device_id >= 0 && dst_device_id >= 0 && src_device_id != dst_device_id) {
    hipSetDevice(src_device_id);
  }

  hipError_t error = hipMemcpyDtoD(dst, src, size);
  checkError(error, "hipMemcpyDtoD");
}

bool ROCmUtils::compileKernel(const std::string& source, const std::string& kernel_name, hipFunction_t* function) {
  hiprtcProgram program;
  hiprtcResult result = hiprtcCreateProgram(&program, source.c_str(), "kernel.cu", nullptr, nullptr);
  if (result != HIPRTC_SUCCESS) {
    return false;
  }

  hiprtcResult compile_result = hiprtcCompileProgram(program, 0, nullptr, nullptr);
  if (compile_result != HIPRTC_SUCCESS) {
    size_t log_size;
    hiprtcGetProgramLogSize(program, &log_size);
    if (log_size > 0) {
      std::string log(log_size, ' ');
      hiprtcGetProgramLog(program, &log[0]);
      std::cerr << "Kernel compilation failed: " << log << std::endl;
    }
    hiprtcDestroyProgram(&program);
    return false;
  }

  hipModule_t module;
  hiprtcGetCode(program, nullptr);
  hipError_t error = hipModuleLoadData(&module, program, nullptr);
  if (error != hipSuccess) {
    hiprtcDestroyProgram(&program);
    return false;
  }

  error = hipModuleGetFunction(function, module, kernel_name.c_str());
  if (error != hipSuccess) {
    hipModuleUnload(module);
    hiprtcDestroyProgram(&program);
    return false;
  }

  hiprtcDestroyProgram(&program);
  return true;
}

bool ROCmUtils::compileKernelPTX(const std::string& ptx_source, const std::string& kernel_name, hipFunction_t* function) {
  hipModule_t module;
  hipError_t error = hipModuleLoadData(&module, ptx_source.c_str(), "kernel.ptx");
  if (error != hipSuccess) {
    return false;
  }

  error = hipModuleGetFunction(function, module, kernel_name.c_str());
  if (error != hipSuccess) {
    hipModuleUnload(module);
    return false;
  }

  return true;
}

std::string ROCmUtils::getROCmVersion() {
  int version;
  hipError_t error = hipRuntimeGetVersion(&version);
  if (error == hipSuccess) {
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
  }
  return "unknown";
}

std::string ROCmUtils::getHIPVersion() {
  int version;
  hipError_t error = hipRuntimeGetVersion(&version);
  if (error == hipSuccess) {
    return std::to_string(version);
  }
  return "unknown";
}

std::string ROCmUtils::getROCBLASVersion() {
  char version[256];
  rocblas_status_t status = rocblas_get_version_string(version, sizeof(version));
  if (status == rocblas_status_success) {
    return std::string(version);
  }
  return "unknown";
}

std::string ROCmUtils::getrocRANDVersion() {
  uint64_t version;
  rocrand_status_t status = rocrand_get_version(&version);
  if (status == ROCRAND_STATUS_SUCCESS) {
    return std::to_string(version);
  }
  return "unknown";
}

bool ROCmUtils::isAmdArchitecture(const std::string& arch) {
  return arch.find("gfx") == 0;
}

bool ROCmUtils::isGCNArch(const std::string& arch) {
  return arch.find("gfx") == 0 && arch.length() >= 5;
}

int ROCmUtils::getArchComputeCapability(const std::string& arch) {
  if (arch.length() < 5 || arch.substr(0, 3) != "gfx") {
    return 0;
  }

  try {
    int gfx_version = std::stoi(arch.substr(3, 2));
    return gfx_version;
  } catch (const std::exception&) {
    return 0;
  }
}

void ROCmUtils::initializeDeviceProperties() {
  int device_count;
  hipError_t error = hipGetDeviceCount(&device_count);
  if (error != hipSuccess) {
    return;
  }

  std::lock_guard<std::mutex> lock(properties_mutex_);
  for (int i = 0; i < device_count; ++i) {
    hipDeviceProp_t props;
    error = hipGetDeviceProperties(&props, i);
    if (error == hipSuccess) {
      device_properties_[i] = props;
    }
  }
}

hipDeviceProp_t& ROCmUtils::getDeviceProperties(int device_id) {
  std::lock_guard<std::mutex> lock(properties_mutex_);
  return device_properties_[device_id];
}

}