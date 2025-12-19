#include <plugins/rocm_executor/executor.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <mutex>

namespace engine_c {

ROCmKernel::ROCmKernel(const std::string& name, ROCmKernelType type, ROCmDevice* device)
    : name_(name), type_(type), device_(device) {}

void ROCmKernel::checkHipError(hipError_t error, const std::string& operation) const {
  if (error != hipSuccess) {
    std::cerr << "ROCm Kernel Error in " << operation << ": " << hipGetErrorString(error) << std::endl;
  }
}

ROCmReduceKernel::ROCmReduceKernel(ROCmDevice* device, ROCmKernelType type)
    : ROCmKernel("reduce", type, device), block_size_(256), grid_size_(1) {
  loadReduceFunction();
}

void ROCmReduceKernel::launch(const void* args, dim3 grid, dim3 block, hipStream_t stream) {
  switch (type_) {
    case ROCmKernelType::REDUCE_SUM:
      launchSumKernel(args, grid, block, stream);
      break;
    case ROCmKernelType::REDUCE_MAX:
      launchMaxKernel(args, grid, block, stream);
      break;
    case ROCmKernelType::REDUCE_MIN:
      launchMinKernel(args, grid, block, stream);
      break;
    default:
      break;
  }
}

void ROCmReduceKernel::loadReduceFunction() {
  std::string function_name;
  switch (type_) {
    case ROCmKernelType::REDUCE_SUM:
      function_name = "reduce_sum_kernel";
      break;
    case ROCmKernelType::REDUCE_MAX:
      function_name = "reduce_max_kernel";
      break;
    case ROCmKernelType::REDUCE_MIN:
      function_name = "reduce_min_kernel";
      break;
    default:
      return;
  }

  const char* kernel_code = R"(
extern "C" __global__
void reduce_sum_kernel(const float* input, float* output, int size) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (i < size) ? input[i] : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

extern "C" __global__
void reduce_max_kernel(const float* input, float* output, int size) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (i < size) ? input[i] : -FLT_MAX;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

extern "C" __global__
void reduce_min_kernel(const float* input, float* output, int size) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (i < size) ? input[i] : FLT_MAX;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}
)";

  hiprtcProgram program;
  hiprtcResult result = hiprtcCreateProgram(&program, kernel_code, "reduce_kernels.cu", nullptr, nullptr);
  if (result != HIPRTC_SUCCESS) {
    std::cerr << "Failed to create HIPRTC program" << std::endl;
    return;
  }

  result = hiprtcCompileProgram(program, 0, nullptr, nullptr);
  if (result != HIPRTC_SUCCESS) {
    size_t log_size;
    hiprtcGetProgramLogSize(program, &log_size);
    std::string log(log_size, ' ');
    hiprtcGetProgramLog(program, &log[0]);
    std::cerr << "Compilation failed: " << log << std::endl;
    hiprtcDestroyProgram(&program);
    return;
  }

  hipModule_t module;
  hiprtcGetCode(program, nullptr);
  result = hipModuleLoadData(&module, program, nullptr);
  if (result != HIP_SUCCESS) {
    std::cerr << "Failed to load HIP module" << std::endl;
  }

  hiprtcDestroyProgram(&program);

  hipError_t error = hipModuleGetFunction(&reduce_function_, module, function_name.c_str());
  if (error != hipSuccess) {
    std::cerr << "Failed to get function: " << function_name << std::endl;
  }
}

void ROCmReduceKernel::launchSumKernel(const void* args, dim3 grid, dim3 block, hipStream_t stream) {
  struct KernelArgs {
    const float* input;
    float* output;
    int size;
  };

  const KernelArgs* kernel_args = static_cast<const KernelArgs*>(args);
  void* kernel_params[] = {
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->input)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->output)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->size))
  };

  size_t shared_mem = block.x * sizeof(float);
  hipError_t error = hipModuleLaunchKernel(reduce_function_, grid.x, grid.y, grid.z,
                                             block.x, block.y, block.z,
                                             shared_mem, stream, kernel_params);
  checkHipError(error, "hipModuleLaunchKernel (reduce_sum)");
}

void ROCmReduceKernel::launchMaxKernel(const void* args, dim3 grid, dim3 block, hipStream_t stream) {
  struct KernelArgs {
    const float* input;
    float* output;
    int size;
  };

  const KernelArgs* kernel_args = static_cast<const KernelArgs*>(args);
  void* kernel_params[] = {
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->input)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->output)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->size))
  };

  size_t shared_mem = block.x * sizeof(float);
  hipError_t error = hipModuleLaunchKernel(reduce_function_, grid.x, grid.y, grid.z,
                                             block.x, block.y, block.z,
                                             shared_mem, stream, kernel_params);
  checkHipError(error, "hipModuleLaunchKernel (reduce_max)");
}

void ROCmReduceKernel::launchMinKernel(const void* args, dim3 grid, dim3 block, hipStream_t stream) {
  struct KernelArgs {
    const float* input;
    float* output;
    int size;
  };

  const KernelArgs* kernel_args = static_cast<const KernelArgs*>(args);
  void* kernel_params[] = {
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->input)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->output)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->size))
  };

  size_t shared_mem = block.x * sizeof(float);
  hipError_t error = hipModuleLaunchKernel(reduce_function_, grid.x, grid.y, grid.z,
                                             block.x, block.y, block.z,
                                             shared_mem, stream, kernel_params);
  checkHipError(error, "hipModuleLaunchKernel (reduce_min)");
}

ROCmAllreduceKernel::ROCmAllreduceKernel(ROCmDevice* device)
    : ROCmKernel("allreduce", ROCmKernelType::ALLREDUCE_RING, device) {
  loadAllreduceFunctions();
}

void ROCmAllreduceKernel::launch(const void* args, dim3 grid, dim3 block, hipStream_t stream) {
  if (algorithm_ == "ring") {
    launchRingAllreduce(args, grid, block, stream);
  } else if (algorithm_ == "tree") {
    launchTreeAllreduce(args, grid, block, stream);
  } else if (algorithm_ == "rabenseifner") {
    launchRabenseifnerAllreduce(args, grid, block, stream);
  }
}

void ROCmAllreduceKernel::setParticipants(const std::vector<int>& participants) {
  participants_ = participants;
}

void ROCmAllreduceKernel::setAlgorithm(const std::string& algorithm) {
  algorithm_ = algorithm;
}

void ROCmAllreduceKernel::setDataType(DataType dtype) {
  data_type_ = dtype;
}

void ROCmAllreduceKernel::loadAllreduceFunctions() {
  const char* kernel_code = R"(
extern "C" __global__
void ring_allreduce_kernel(float* buffer, int size, int rank, int nranks) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int block_size = blockDim.x;
  int chunk_size = size / nranks;

  for (int step = 0; step < nranks - 1; step++) {
    int send_to = (rank + 1) % nranks;
    int recv_from = (rank - 1 + nranks) % nranks;

    int send_offset = ((rank - step - 1 + nranks) % nranks) * chunk_size;
    int recv_offset = ((rank - step - 2 + nranks) % nranks) * chunk_size;

    int global_idx = send_offset + tid;
    if (global_idx < size) {
      sdata[tid] = buffer[global_idx];
    } else {
      sdata[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
      if (tid < s) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }

    if (tid == 0) {
      int target_idx = recv_offset;
      if (target_idx < size) {
        buffer[target_idx] = sdata[0];
      }
    }
    __syncthreads();
  }
}

extern "C" __global__
void tree_allreduce_kernel(float* buffer, int size, int rank, int nranks, int parent) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int block_size = blockDim.x;

  int global_idx = tid;
  if (global_idx < size) {
    sdata[tid] = buffer[global_idx];
  } else {
    sdata[tid] = 0.0f;
  }
  __syncthreads();

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0 && parent >= 0) {
    for (int i = 0; i < size; i += block_size) {
      buffer[i] = sdata[0];
    }
  }
  __syncthreads();
}
)";

  hiprtcProgram program;
  hiprtcResult result = hiprtcCreateProgram(&program, kernel_code, "allreduce_kernels.cu", nullptr, nullptr);
  if (result != HIPRTC_SUCCESS) {
    std::cerr << "Failed to create HIPRTC program for allreduce" << std::endl;
    return;
  }

  result = hiprtcCompileProgram(program, 0, nullptr, nullptr);
  if (result != HIPRTC_SUCCESS) {
    size_t log_size;
    hiprtcGetProgramLogSize(program, &log_size);
    std::string log(log_size, ' ');
    hiprtcGetProgramLog(program, &log[0]);
    std::cerr << "Allreduce compilation failed: " << log << std::endl;
    hiprtcDestroyProgram(&program);
    return;
  }

  hipModule_t module;
  hiprtcGetCode(program, nullptr);
  result = hipModuleLoadData(&module, program, nullptr);
  if (result != HIP_SUCCESS) {
    std::cerr << "Failed to load allreduce HIP module" << std::endl;
  }

  hiprtcDestroyProgram(&program);

  hipError_t error = hipModuleGetFunction(&ring_function_, module, "ring_allreduce_kernel");
  if (error != hipSuccess) {
    std::cerr << "Failed to get ring allreduce function" << std::endl;
  }

  error = hipModuleGetFunction(&tree_function_, module, "tree_allreduce_kernel");
  if (error != hipSuccess) {
    std::cerr << "Failed to get tree allreduce function" << std::endl;
  }

  error = hipModuleGetFunction(&rabenseifner_function_, module, "tree_allreduce_kernel");
  if (error != hipSuccess) {
    std::cerr << "Failed to get rabenseifner allreduce function" << std::endl;
  }
}

void ROCmAllreduceKernel::launchRingAllreduce(const void* args, dim3 grid, dim3 block, hipStream_t stream) {
  struct KernelArgs {
    float* buffer;
    int size;
    int rank;
    int nranks;
  };

  const KernelArgs* kernel_args = static_cast<const KernelArgs*>(args);
  void* kernel_params[] = {
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->buffer)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->size)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->rank)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->nranks))
  };

  size_t shared_mem = block.x * sizeof(float);
  hipError_t error = hipModuleLaunchKernel(ring_function_, grid.x, grid.y, grid.z,
                                             block.x, block.y, block.z,
                                             shared_mem, stream, kernel_params);
  checkHipError(error, "hipModuleLaunchKernel (ring_allreduce)");
}

void ROCmAllreduceKernel::launchTreeAllreduce(const void* args, dim3 grid, dim3 block, hipStream_t stream) {
  struct KernelArgs {
    float* buffer;
    int size;
    int rank;
    int nranks;
    int parent;
  };

  const KernelArgs* kernel_args = static_cast<const KernelArgs*>(args);
  void* kernel_params[] = {
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->buffer)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->size)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->rank)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->nranks)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->parent))
  };

  size_t shared_mem = block.x * sizeof(float);
  hipError_t error = hipModuleLaunchKernel(tree_function_, grid.x, grid.y, grid.z,
                                             block.x, block.y, block.z,
                                             shared_mem, stream, kernel_params);
  checkHipError(error, "hipModuleLaunchKernel (tree_allreduce)");
}

void ROCmAllreduceKernel::launchRabenseifnerAllreduce(const void* args, dim3 grid, dim3 block, hipStream_t stream) {
  launchTreeAllreduce(args, grid, block, stream);
}

ROCmCopyKernel::ROCmCopyKernel(ROCmDevice* device)
    : ROCmKernel("copy", ROCmKernelType::COPY, device) {
  loadCopyFunction();
}

void ROCmCopyKernel::launch(const void* args, dim3 grid, dim3 block, hipStream_t stream) {
  struct KernelArgs {
    const void* src;
    void* dst;
    size_t size;
  };

  const KernelArgs* kernel_args = static_cast<const KernelArgs*>(args);
  void* kernel_params[] = {
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->src)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->dst)),
    const_cast<void**>(reinterpret_cast<const void*>(&kernel_args->size))
  };

  hipError_t error = hipModuleLaunchKernel(copy_function_, grid.x, grid.y, grid.z,
                                             block.x, block.y, block.z,
                                             0, stream, kernel_params);
  checkHipError(error, "hipModuleLaunchKernel (copy)");
}

void ROCmCopyKernel::loadCopyFunction() {
  const char* kernel_code = R"(
extern "C" __global__
void copy_kernel(const void* src, void* dst, size_t size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const char* src_char = static_cast<const char*>(src);
  char* dst_char = static_cast<char*>(dst);

  if (tid < size) {
    dst_char[tid] = src_char[tid];
  }
}
)";

  hiprtcProgram program;
  hiprtcResult result = hiprtcCreateProgram(&program, kernel_code, "copy_kernels.cu", nullptr, nullptr);
  if (result != HIPRTC_SUCCESS) {
    std::cerr << "Failed to create HIPRTC program for copy" << std::endl;
    return;
  }

  result = hiprtcCompileProgram(program, 0, nullptr, nullptr);
  if (result != HIPRTC_SUCCESS) {
    std::cerr << "Copy kernel compilation failed" << std::endl;
    hiprtcDestroyProgram(&program);
    return;
  }

  hipModule_t module;
  hiprtcGetCode(program, nullptr);
  result = hipModuleLoadData(&module, program, nullptr);
  if (result != HIP_SUCCESS) {
    std::cerr << "Failed to load copy HIP module" << std::endl;
  }

  hiprtcDestroyProgram(&program);

  hipError_t error = hipModuleGetFunction(&copy_function_, module, "copy_kernel");
  if (error != hipSuccess) {
    std::cerr << "Failed to get copy function" << std::endl;
  }
}

ROCmCommKernel::ROCmCommKernel(ROCmDevice* device, ROCmKernelType type)
    : ROCmKernel("comm", type, device), src_device_(0), dst_device_(1), tag_(0) {
  loadCommFunctions();
}

void ROCmCommKernel::launch(const void* args, dim3 grid, dim3 block, hipStream_t stream) {
  if (type_ == ROCmKernelType::SEND) {
  } else if (type_ == ROCmKernelType::RECV) {
  }
}

void ROCmCommKernel::loadCommFunctions() {
  const char* kernel_code = R"(
extern "C" __global__
void send_kernel(const void* data, size_t size, int dst_device, int tag) {
}

extern "C" __global__
void recv_kernel(void* data, size_t size, int src_device, int tag) {
}
)";

  hiprtcProgram program;
  hiprtcResult result = hiprtcCreateProgram(&program, kernel_code, "comm_kernels.cu", nullptr, nullptr);
  if (result != HIPRTC_SUCCESS) {
    return;
  }

  hiprtcCompileProgram(program, 0, nullptr, nullptr);
  hiprtcGetCode(program, nullptr);

  hipModule_t module;
  hipModuleLoadData(&module, program, nullptr);
  hiprtcDestroyProgram(&program);

  hipError_t error = hipModuleGetFunction(&send_function_, module, "send_kernel");
  error = hipModuleGetFunction(&recv_function_, module, "recv_kernel");
}

ROCmKernelRegistry::ROCmKernelRegistry() : device_(nullptr) {}

ROCmKernelRegistry::~ROCmKernelRegistry() {
  kernels_.clear();
}

void ROCmKernelRegistry::registerKernel(std::unique_ptr<ROCmKernel> kernel) {
  kernels_[kernel->getName()] = std::move(kernel);
}

ROCmKernel* ROCmKernelRegistry::getKernel(const std::string& name) {
  auto it = kernels_.find(name);
  return (it != kernels_.end()) ? it->second.get() : nullptr;
}

std::vector<std::string> ROCmKernelRegistry::getKernelNames() const {
  std::vector<std::string> names;
  for (const auto& pair : kernels_) {
    names.push_back(pair.first);
  }
  return names;
}

ROCmReduceKernel* ROCmKernelRegistry::getReduceKernel(ROCmKernelType type) {
  std::string name = "reduce_" + std::to_string(static_cast<int>(type));
  return static_cast<ROCmReduceKernel*>(getKernel(name));
}

ROCmAllreduceKernel* ROCmKernelRegistry::getAllreduceKernel() {
  return static_cast<ROCmAllreduceKernel*>(getKernel("allreduce"));
}

ROCmCopyKernel* ROCmKernelRegistry::getCopyKernel() {
  return static_cast<ROCmCopyKernel*>(getKernel("copy"));
}

ROCmCommKernel* ROCmKernelRegistry::getSendKernel() {
  return static_cast<ROCmCommKernel*>(getKernel("send"));
}

ROCmCommKernel* ROCmKernelRegistry::getRecvKernel() {
  return static_cast<ROCmCommKernel*>(getKernel("recv"));
}

void ROCmKernelRegistry::initializeBuiltinKernels() {
  registerKernel(std::make_unique<ROCmReduceKernel>(device_, ROCmKernelType::REDUCE_SUM));
  registerKernel(std::make_unique<ROCmAllreduceKernel>(device_));
  registerKernel(std::make_unique<ROCmCopyKernel>(device_));
  registerKernel(std::make_unique<ROCmCommKernel>(device_, ROCmKernelType::SEND));
  registerKernel(std::make_unique<ROCmCommKernel>(device_, ROCmKernelType::RECV));
}

ROCmStreamManager::ROCmStreamManager(ROCmDevice* device)
    : device_(device), default_stream_(0) {
  hipError_t error = hipStreamCreateWithFlags(&default_stream_, hipStreamNonBlocking);
  checkHipError(error, "hipStreamCreateWithFlags");
}

ROCmStreamManager::~ROCmStreamManager() {
  for (hipStream_t stream : streams_) {
    hipStreamDestroy(stream);
  }

  for (hipEvent_t event : events_) {
    hipEventDestroy(event);
  }

  if (default_stream_ != 0) {
    hipStreamDestroy(default_stream_);
  }
}

hipStream_t ROCmStreamManager::getStream(int stream_id) {
  if (stream_id < 0) {
    return default_stream_;
  }

  std::lock_guard<std::mutex> lock(stream_mutex_);
  if (stream_id < static_cast<int>(streams_.size())) {
    return streams_[stream_id];
  }

  return createStream();
}

hipStream_t ROCmStreamManager::createStream() {
  hipStream_t stream;
  hipError_t error = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  checkHipError(error, "hipStreamCreateWithFlags");

  std::lock_guard<std::mutex> lock(stream_mutex_);
  streams_.push_back(stream);
  return stream;
}

void ROCmStreamManager::destroyStream(hipStream_t stream) {
  auto it = std::find(streams_.begin(), streams_.end(), stream);
  if (it != streams_.end()) {
    hipStreamDestroy(stream);
    streams_.erase(it);
  }
}

void ROCmStreamManager::synchronizeStream(int stream_id) {
  hipStream_t stream = getStream(stream_id);
  hipError_t error = hipStreamSynchronize(stream);
  checkHipError(error, "hipStreamSynchronize");
}

void ROCmStreamManager::synchronizeAllStreams() {
  std::lock_guard<std::mutex> lock(stream_mutex_);
  for (hipStream_t stream : streams_) {
    hipError_t error = hipStreamSynchronize(stream);
    checkHipError(error, "hipStreamSynchronize");
  }

  hipError_t error = hipStreamSynchronize(default_stream_);
  checkHipError(error, "hipStreamSynchronize");
}

hipEvent_t ROCmStreamManager::createEvent() {
  hipEvent_t event;
  hipError_t error = hipEventCreateWithFlags(&event, hipEventDisableTiming);
  checkHipError(error, "hipEventCreateWithFlags");

  std::lock_guard<std::mutex> lock(stream_mutex_);
  events_.push_back(event);
  return event;
}

void ROCmStreamManager::destroyEvent(hipEvent_t event) {
  auto it = std::find(events_.begin(), events_.end(), event);
  if (it != events_.end()) {
    hipEventDestroy(event);
    events_.erase(it);
  }
}

void ROCmStreamManager::recordEvent(hipEvent_t event, hipStream_t stream) {
  hipError_t error = hipEventRecord(event, stream);
  checkHipError(error, "hipEventRecord");
}

void ROCmStreamManager::waitForEvent(hipEvent_t event, hipStream_t stream) {
  hipError_t error = hipStreamWaitEvent(stream, event);
  checkHipError(error, "hipStreamWaitEvent");
}

bool ROCmStreamManager::isStreamBusy(int stream_id) {
  hipStream_t stream = getStream(stream_id);
  hipError_t error = hipStreamQuery(stream);
  return error == hipErrorNotReady;
}

void ROCmStreamManager::checkHipError(hipError_t error, const std::string& operation) const {
  if (error != hipSuccess) {
    std::cerr << "ROCm Stream Error in " << operation << ": " << hipGetErrorString(error) << std::endl;
  }
}

ROCmBLASManager::ROCmBLASManager(ROCmDevice* device) : device_(device) {
  rocblas_status_t status = rocblas_create_handle(&blas_handle_);
  if (status != rocblas_status_success) {
    std::cerr << "Failed to create rocblas handle" << std::endl;
  }
}

ROCmBLASManager::~ROCmBLASManager() {
  if (blas_handle_) {
    rocblas_destroy_handle(blas_handle_);
  }
}

void ROCmBLASManager::setStream(hipStream_t stream) {
  rocblas_status_t status = rocblas_set_stream(blas_handle_, stream);
  checkRocblasError(status, "rocblas_set_stream");
}

void ROCmBLASManager::synchronize() {
  rocblas_status_t status = rocblas_stream_synchronize(blas_handle_);
  checkRocblasError(status, "rocblas_stream_synchronize");
}

void ROCmBLASManager::gemm(rocblas_operation transa, rocblas_operation transb,
                         int m, int n, int k,
                         const float* alpha, const float* A, int lda,
                         const float* B, int ldb,
                         const float* beta, float* C, int ldc) {
  rocblas_status_t status = rocblas_sgemm(blas_handle_, transa, transb, m, n, k,
                                           alpha, A, lda, B, ldb, beta, C, ldc);
  checkRocblasError(status, "rocblas_sgemm");
}

void ROCmBLASManager::gemmStridedBatched(rocblas_operation transa, rocblas_operation transb,
                                       int m, int n, int k,
                                       const float* alpha, const float* A, int lda, long long strideA,
                                       const float* B, int ldb, long long strideB,
                                       const float* beta, float* C, int ldc, long long strideC,
                                       int batch_count) {
  rocblas_status_t status = rocblas_sgemm_strided_batched(blas_handle_, transa, transb,
                                                         m, n, k,
                                                         alpha, A, lda, strideA,
                                                         B, ldb, strideB,
                                                         beta, C, ldc, strideC,
                                                         batch_count);
  checkRocblasError(status, "rocblas_sgemm_strided_batched");
}

void ROCmBLASManager::checkRocblasError(rocblas_status status, const std::string& operation) const {
  if (status != rocblas_status_success) {
    std::cerr << "rocblas Error in " << operation << ": " << static_cast<int>(status) << std::endl;
  }
}

ROCmExecutor::ROCmExecutor() : device_id_(-1), initialized_(false), current_stream_(0) {}

ROCmExecutor::ROCmExecutor(int device_id) : device_id_(device_id), initialized_(false), current_stream_(0) {}

ROCmExecutor::~ROCmExecutor() {
  shutdown();
}

void ROCmExecutor::initialize() {
  if (initialized_) {
    return;
  }

  checkDeviceAvailable();

  device_manager_ = std::make_unique<ROCmDeviceManager>();
  device_manager_->initialize();

  hip_device_ = std::make_unique<ROCmDevice>();
  hip_device_->initialize(device_id_);

  initializeComponents();

  initialized_ = true;
}

void ROCmExecutor::shutdown() {
  if (!initialized_) {
    return;
  }

  shutdownComponents();

  hip_device_.reset();
  device_manager_.reset();

  initialized_ = false;
}

void* ROCmExecutor::allocate(size_t size) {
  if (!initialized_) {
    return nullptr;
  }

  return hip_device_->getMemoryManager()->allocate(size, device_id_);
}

void ROCmExecutor::free(void* ptr) {
  if (!initialized_) {
    return;
  }

  hip_device_->getMemoryManager()->free(ptr, device_id_);
}

void ROCmExecutor::copy(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  hip_device_->getMemoryManager()->copyDeviceToDevice(dst, src, size, device_id_, device_id_);
}

void ROCmExecutor::copyFromHost(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  hip_device_->getMemoryManager()->copyToDevice(dst, src, size, device_id_);
}

void ROCmExecutor::copyToHost(void* dst, const void* src, size_t size) {
  if (!initialized_) {
    return;
  }

  hip_device_->getMemoryManager()->copyFromDevice(dst, src, size, device_id_);
}

void ROCmExecutor::execute(const void* kernel, const void* args, size_t arg_size,
                          dim3 grid, dim3 block, size_t shared_mem) {
  if (!initialized_) {
    return;
  }

  hipStream_t stream = current_stream_;
  dim3 adjusted_block = block;
  dim3 adjusted_grid = grid;

  hipLaunchKernel_t hip_kernel = reinterpret_cast<hipLaunchKernel_t>(const_cast<void*>(kernel));
  hipError_t error = hipLaunchKernel(hip_kernel, adjusted_grid, adjusted_block, shared_mem, stream, const_cast<void*>(args));
  checkHipError(error, "hipLaunchKernel");
}

void ROCmExecutor::synchronize() {
  if (!initialized_) {
    return;
  }

  hip_device_->synchronize();
}

double ROCmExecutor::measureKernelTime(const void* kernel, const void* args, size_t arg_size,
                                     dim3 grid, dim3 block, size_t shared_mem,
                                     int iterations) {
  if (!initialized_) {
    return 0.0;
  }

  hipEvent_t start, stop;
  hipError_t error = hipEventCreate(&start);
  checkHipError(error, "hipEventCreate (start)");
  error = hipEventCreate(&stop);
  checkHipError(error, "hipEventCreate (stop)");

  hipEventRecord(start, current_stream_);

  for (int i = 0; i < iterations; ++i) {
    execute(kernel, args, arg_size, grid, block, shared_mem);
  }

  hipEventRecord(stop, current_stream_);
  hipEventSynchronize(stop);

  float milliseconds = 0;
  error = hipEventElapsedTime(&milliseconds, start, stop);
  checkHipError(error, "hipEventElapsedTime");

  hipEventDestroy(start);
  hipEventDestroy(stop);

  return static_cast<double>(milliseconds) / iterations;
}

bool ROCmExecutor::supportsP2P() const {
  return hip_device_ ? hip_device_->supportsP2P() : false;
}

void ROCmExecutor::enableP2P(int other_device_id) {
  if (!initialized_) {
    return;
  }

  hip_device_->enableP2P(other_device_id);
}

void* ROCmExecutor::getDevicePointer() const {
  return reinterpret_cast<void*>(device_id_);
}

int ROCmExecutor::getDeviceId() const {
  return device_id_;
}

hipStream_t ROCmExecutor::getCurrentStream() const {
  return current_stream_;
}

void ROCmExecutor::setCurrentStream(hipStream_t stream) {
  current_stream_ = stream;
}

void ROCmExecutor::initializeComponents() {
  kernel_registry_ = std::make_unique<ROCmKernelRegistry>();
  stream_manager_ = std::make_unique<ROCmStreamManager>(hip_device_.get());
  blas_manager_ = std::make_unique<ROCmBLASManager>(hip_device_.get());
}

void ROCmExecutor::shutdownComponents() {
  blas_manager_.reset();
  stream_manager_.reset();
  kernel_registry_.reset();
}

void ROCmExecutor::checkDeviceAvailable() {
  hipError_t error = hipGetDeviceCount(&device_id_);
  if (error != hipSuccess || device_id_ < 0) {
    std::cerr << "No ROCm devices available" << std::endl;
  }
}

void ROCmExecutor::checkHipError(hipError_t error, const std::string& operation) const {
  if (error != hipSuccess) {
    std::cerr << "ROCm Executor Error in " << operation << ": " << hipGetErrorString(error) << std::endl;
  }
}

}