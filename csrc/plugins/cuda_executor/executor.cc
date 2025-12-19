#include <plugins/cuda_executor/executor.h>
#include <plugins/cuda_executor/device.h>
#include <plugins/cuda_executor/utils/multimem.h>
#include <plugins/cuda_executor/utils/symmetric_memory.h>
#include <plugins/cuda_executor/utils/vector_datatype.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <nvrtc.h>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <string>

namespace engine_c {

struct TMP_CUDA_EXECUTOR_STATIC_INITIALIZER {
  TMP_CUDA_EXECUTOR_STATIC_INITIALIZER() {
    auto executor = TypeRegistry::registerExecutorType("cuda");
    regExec(executor, std::make_shared<DeviceExecutor>(new CudaExecutor()));
  }
};

static TMP_CUDA_EXECUTOR_STATIC_INITIALIZER _____cuda_executor_tmp;

CudaExecutor::CudaExecutor() : initialized_(false), device_id_0), current_stream_(0) {}

CudaExecutor::CudaExecutor(int device_id) : initialized_(false), device_id_(device_id), current_stream_(0) {}

CudaExecutor::~CudaExecutor() {
  shutdown();
}

void CudaExecutor::initialize() {
  if (initialized_) {
    return;
  }

  checkCudaError(cudaSetDevice(device_id_), "cudaSetDevice");

  cuda_device_ = std::make_unique<CudaDevice>();
  stream_manager_ = std::make_unique<CudaStreamManager>();
  memory_manager_ = std::make_unique<CudaMemoryManager>();
  kernel_manager_ = std::make_unique<CudaKernelManager>();
  blas_manager_ = std::make_unique<CudaBlasManager>();

  stream_manager_->initialize(device_id_);
  memory_manager_->initialize(device_id_);
  kernel_manager_->initialize(device_id_);
  blas_manager_->initialize(device_id_);

  initialized_ = true;
}

void CudaExecutor::shutdown() {
  if (!initialized_) {
    return;
  }

  if (current_stream_ != 0) {
    cudaStreamDestroy(current_stream_);
    current_stream_ = 0;
  }

  blas_manager_.reset();
  kernel_manager_.reset();
  memory_manager_.reset();
  stream_manager_.reset();
  cuda_device_.reset();

  initialized_ = false;
}

void* CudaExecutor::allocate(size_t size) {
  if (!initialized_) {
    return nullptr;
  }

  return memory_manager_->allocate(size);
}

void CudaExecutor::free(void* ptr) {
  if (!initialized_) {
    return;
  }

  memory_manager_->free(ptr);
}

void CudaExecutor::copy(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  checkCudaError(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, current_stream_),
                "cudaMemcpyAsync device-to-device");
}

void CudaExecutor::copyFromHost(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  checkCudaError(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, current_stream_),
                "cudaMemcpyAsync host-to-device");
}

void CudaExecutor::copyToHost(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  checkCudaError(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, current_stream_),
                "cudaMemcpyAsync device-to-host");
}

void CudaExecutor::execute(const void* kernel, const void* args, size_t arg_size,
                          dim3 grid, dim3 block, size_t shared_mem) {
  if (!initialized_) {
    return;
  }

  void** kernel_args = const_cast<void**>(reinterpret_cast<const void**>(args));
  checkCudaError(cudaLaunchKernel(reinterpret_cast<const void*>(kernel),
                                 grid, block, kernel_args, shared_mem, current_stream_),
                "cudaLaunchKernel");
}

void CudaExecutor::synchronize() {
  if (!initialized_) {
    return;
  }

  checkCudaError(cudaStreamSynchronize(current_stream_), "cudaStreamSynchronize");
}

double CudaExecutor::measureKernelTime(const void* kernel, const void* args, size_t arg_size,
                                       dim3 grid, dim3 block, size_t shared_mem,
                                       int iterations) {
  if (!initialized_) {
    return 0.0;
  }

  cudaEvent_t start, stop;
  checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
  checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

  checkCudaError(cudaEventRecord(start, current_stream_), "cudaEventRecord start");

  for (int i = 0; i < iterations; ++i) {
    execute(kernel, args, arg_size, grid, block, shared_mem);
  }

  checkCudaError(cudaEventRecord(stop, current_stream_), "cudaEventRecord stop");
  checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

  float milliseconds = 0;
  checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return static_cast<double>(milliseconds) / iterations;
}

bool CudaExecutor::supportsP2P() const {
  if (!initialized_) {
    return false;
  }

  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    return false;
  }

  for (int i = 0; i < device_count; ++i) {
    if (i != device_id_) {
      int can_access;
      error = cudaDeviceCanAccessPeer(&can_access, device_id_, i);
      if (error == cudaSuccess && can_access) {
        return true;
      }
    }
  }

  return false;
}

void CudaExecutor::enableP2P(int other_device_id) {
  if (!initialized_) {
    return;
  }

  int can_access;
  cudaError_t error = cudaDeviceCanAccessPeer(&can_access, device_id_, other_device_id);
  if (error == cudaSuccess && can_access) {
    error = cudaDeviceEnablePeerAccess(other_device_id, 0);
    if (error != cudaErrorPeerAccessAlreadyEnabled) {
      checkCudaError(error, "cudaDeviceEnablePeerAccess");
    }
  }
}

void* CudaExecutor::getDevicePointer() const {
  if (!initialized_) {
    return nullptr;
  }

  return reinterpret_cast<void*>(device_id_);
}

int CudaExecutor::getDeviceId() const {
  return device_id_;
}

cudaStream_t CudaExecutor::getCurrentStream() const {
  return current_stream_;
}

void CudaExecutor::setCurrentStream(cudaStream_t stream) {
  current_stream_ = stream;
}

size_t CudaExecutor::getTotalMemory() const {
  if (!initialized_) {
    return 0;
  }

  size_t free, total;
  checkCudaError(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
  return total;
}

size_t CudaExecutor::getAvailableMemory() const {
  if (!initialized_) {
    return 0;
  }

  size_t free, total;
  checkCudaError(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
  return free;
}

CudaStreamManager* CudaExecutor::getStreamManager() {
  return stream_manager_.get();
}

CudaMemoryManager* CudaExecutor::getMemoryManager() {
  return memory_manager_.get();
}

CudaKernelManager* CudaExecutor::getKernelManager() {
  return kernel_manager_.get();
}

CudaBlasManager* CudaExecutor::getBlasManager() {
  return blas_manager_.get();
}

void CudaExecutor::checkCudaError(cudaError_t error, const std::string& operation) const {
  if (error != cudaSuccess) {
    std::cerr << "CUDA Error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
  }
}

CudaStreamManager::CudaStreamManager() : device_id_(0), default_stream_(0) {}

CudaStreamManager::~CudaStreamManager() {
  for (auto stream : streams_) {
    cudaStreamDestroy(stream);
  }

  for (auto event : events_) {
    cudaEventDestroy(event);
  }

  if (default_stream_ != 0) {
    cudaStreamDestroy(default_stream_);
  }
}

void CudaStreamManager::initialize(int device_id) {
  device_id_ = device_id;
  checkCudaError(cudaSetDevice(device_id), "cudaSetDevice");
  checkCudaError(cudaStreamCreateWithFlags(&default_stream_, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags");
}

cudaStream_t CudaStreamManager::getStream(int stream_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (stream_id < 0) {
    return default_stream_;
  }

  if (stream_id < static_cast<int>(streams_.size())) {
    return streams_[stream_id];
  }

  return createStream();
}

cudaStream_t CudaStreamManager::createStream() {
  cudaStream_t stream;
  checkCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags");

  std::lock_guard<std::mutex> lock(mutex_);
  streams_.push_back(stream);
  return stream;
}

void CudaStreamManager::destroyStream(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = std::find(streams_.begin(), streams_.end(), stream);
  if (it != streams_.end()) {
    cudaStreamDestroy(stream);
    streams_.erase(it);
  }
}

void CudaStreamManager::synchronizeStream(int stream_id) {
  cudaStream_t stream = getStream(stream_id);
  checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

cudaEvent_t CudaStreamManager::createEvent() {
  cudaEvent_t event;
  checkCudaError(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                "cudaEventCreateWithFlags");

  std::lock_guard<std::mutex> lock(mutex_);
  events_.push_back(event);
  return event;
}

void CudaStreamManager::recordEvent(cudaEvent_t event, cudaStream_t stream) {
  checkCudaError(cudaEventRecord(event, stream), "cudaEventRecord");
}

void CudaStreamManager::waitForEvent(cudaEvent_t event, cudaStream_t stream) {
  checkCudaError(cudaStreamWaitEvent(stream, event), "cudaStreamWaitEvent");
}

void CudaStreamManager::checkCudaError(cudaError_t error, const std::string& operation) const {
  if (error != cudaSuccess) {
    std::cerr << "CUDA Stream Error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
  }
}

CudaMemoryManager::CudaMemoryManager() : device_id_(0), allocated_bytes_(0) {}

CudaMemoryManager::~CudaMemoryManager() {
  clear();
}

void CudaMemoryManager::initialize(int device_id) {
  device_id_ = device_id;
  checkCudaError(cudaSetDevice(device_id), "cudaSetDevice");
}

void* CudaMemoryManager::allocate(size_t size) {
  if (size == 0) return nullptr;

  void* ptr;
  checkCudaError(cudaMalloc(&ptr, size), "cudaMalloc");

  std::lock_guard<std::mutex> lock(mutex_);
  allocated_bytes_ += size;
  allocations_[ptr] = size;

  return ptr;
}

void CudaMemoryManager::free(void* ptr) {
  if (!ptr) return;

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(ptr);
  if (it != allocations_.end()) {
    allocated_bytes_ -= it->second;
    allocations_.erase(it);
    checkCudaError(cudaFree(ptr), "cudaFree");
  }
}

size_t CudaMemoryManager::getAllocatedBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocated_bytes_;
}

size_t CudaMemoryManager::getAllocationCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocations_.size();
}

void CudaMemoryManager::clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& pair : allocations_) {
    checkCudaError(cudaFree(pair.first), "cudaFree");
  }

  allocations_.clear();
  allocated_bytes_ = 0;
}

void CudaMemoryManager::checkCudaError(cudaError_t error, const std::string& operation) const {
  if (error != cudaSuccess) {
    std::cerr << "CUDA Memory Error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
  }
}

CudaKernelManager::CudaKernelManager() : device_id_(0) {}

CudaKernelManager::~CudaKernelManager() {
  for (auto& pair : modules_) {
    cuModuleUnload(pair.second);
  }
}

void CudaKernelManager::initialize(int device_id) {
  device_id_ = device_id;
  checkCuError(cuInit(0), "cuInit");
}

CUfunction CudaKernelManager::getFunction(const std::string& ptx, const std::string& function_name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = modules_.find(ptx);
  CUmodule module;

  if (it == modules_.end()) {
    checkCuError(cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr),
                 "cuModuleLoadDataEx");
    modules_[ptx] = module;
  } else {
    module = it->second;
  }

  CUfunction function;
  checkCuError(cuModuleGetFunction(&function, module, function_name.c_str()),
               "cuModuleGetFunction");

  return function;
}

void CudaKernelManager::checkCuError(CUresult error, const std::string& operation) const {
  if (error != CUDA_SUCCESS) {
    const char* error_string;
    cuGetErrorString(error, &error_string);
    std::cerr << "CUDA Driver Error in " << operation << ": " << error_string << std::endl;
  }
}

CudaBlasManager::CudaBlasManager() : device_id_(0), blas_handle_(nullptr) {}

CudaBlasManager::~CudaBlasManager() {
  if (blas_handle_) {
    cublasDestroy(blas_handle_);
  }
}

void CudaBlasManager::initialize(int device_id) {
  device_id_ = device_id;
  checkCublasError(cublasCreate(&blas_handle_), "cublasCreate");
  checkCudaError(cublasSetStream(blas_handle_, 0), "cublasSetStream");
}

cublasHandle_t CudaBlasManager::getHandle() {
  return blas_handle_;
}

void CudaBlasManager::setStream(cudaStream_t stream) {
  checkCudaError(cublasSetStream(blas_handle_, stream), "cublasSetStream");
}

void CudaBlasManager::synchronize() {
  checkCudaError(cublasStreamSynchronize(blas_handle_), "cublasStreamSynchronize");
}

void CudaBlasManager::checkCublasError(cublasStatus_t status, const std::string& operation) const {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS Error in " << operation << ": " << static_cast<int>(status) << std::endl;
  }
}

}