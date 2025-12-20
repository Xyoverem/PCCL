#include <plugins/cuda_executor/device.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <common/serialize.h>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <string>
#include <nlohmann/json.hpp>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace engine_c {

struct TMP_STATIC_INITIALIZER {
  TMP_STATIC_INITIALIZER() {
    auto device = TypeRegistry::registerDeviceType("cuda");
    regDev(device, std::make_shared<DeviceBase>(new CudaDevice));
  }
};

static TMP_STATIC_INITIALIZER _____tmp;

struct MemInfo {
  long size;
  CUmemGenericAllocationHandle alloc_handle;
  CUdeviceptr ptr;
  char shareable_handle[64];

  std::string serialize() {
    nlohmann::json j;
    j["size"] = size;
    j["ptr"] = ptr;
    j["shareable"] = utils::serialize(shareable_handle,
                                             sizeof(shareable_handle));
    return j.dump();
  }

  static MemInfo deserialize(const std::string& mem_info) {
    auto j = nlohmann::json::parse(mem_info);
    MemInfo r;
    r.size = j["size"];
    r.ptr = j["ptr"];
    std::string shareable_handle_str = j["shareable"];
    utils::deserialize(&r.shareable_handle, shareable_handle_str);
    return r;
  }
};

static std::unordered_map<void*, MemInfo> mem_map;
static std::mutex mem_mutex;

bool CudaDevice::allocatorAvailable() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return error == cudaSuccess && device_count > 0;
}

void* CudaDevice::allocate(long nbytes) {
  static constexpr long granularity = 2 * 1024 * 1024;
  if (nbytes <= 0) return nullptr;

  int device_index = c10::cuda::getCurrentCUDAStream().device_index();

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_index;

  size_t aligned_size = ((nbytes + granularity - 1) / granularity) * granularity;

  CUmemGenericAllocationHandle alloc_handle;
  CUresult result = cuMemCreate(&alloc_handle, aligned_size, &prop, 0);
  if (result != CUDA_SUCCESS) {
    return nullptr;
  }

  CUdeviceptr dev_ptr;
  result = cuMemAddressReserve(&dev_ptr, aligned_size, granularity, 0, 0);
  if (result != CUDA_SUCCESS) {
    cuMemRelease(alloc_handle);
    return nullptr;
  }

  result = cuMemMap(dev_ptr, aligned_size, 0, alloc_handle, 0);
  if (result != CUDA_SUCCESS) {
    cuMemAddressFree(dev_ptr, aligned_size);
    cuMemRelease(alloc_handle);
    return nullptr;
  }

  CUmemAccessDesc access_desc = {};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = device_index;
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  result = cuMemSetAccess(dev_ptr, aligned_size, &access_desc, 1);
  if (result != CUDA_SUCCESS) {
    cuMemUnmap(dev_ptr, aligned_size);
    cuMemAddressFree(dev_ptr, aligned_size);
    cuMemRelease(alloc_handle);
    return nullptr;
  }

  char shareable_handle[64];

  result = cuMemExportToShareableHandle(
      reinterpret_cast<void*>(shareable_handle),
      alloc_handle,
      CU_MEM_HANDLE_TYPE_FABRIC,
      0
  );

  if (result != CUDA_SUCCESS) {
    cuMemUnmap(dev_ptr, aligned_size);
    cuMemAddressFree(dev_ptr, aligned_size);
    cuMemRelease(alloc_handle);
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mem_mutex);

  mem_map[reinterpret_cast<void*>(dev_ptr)] = {nbytes,
                                               alloc_handle,
                                               dev_ptr};

  memcpy(mem_map[reinterpret_cast<void*>(dev_ptr)].shareable_handle,
         shareable_handle,
         sizeof(shareable_handle));

  return reinterpret_cast<void*>(dev_ptr);
}

void CudaDevice::deallocate(void *ptr) {
  if (!ptr) return;
  std::lock_guard<std::mutex> lock(mem_mutex);
  auto it = mem_map.find(ptr);
  if (it != mem_map.end()) {
    cuMemUnmap(it->second.ptr, it->second.size);
    cuMemAddressFree(it->second.ptr, it->second.size);
    cuMemRelease(it->second.alloc_handle);
    mem_map.erase(it);
  }
}

bool CudaDevice::IPCAvailable() {
  return true;
}

std::string CudaDevice::allocateIpcBuffer(void **addr, long size) {
  *addr = allocate(size);
  MemInfo &mem = mem_map[*addr];
  return mem.serialize();
}

long CudaDevice::mapBuffer(std::string &shareable_handle, void **addr) {
  static constexpr long granularity = 2 * 1024 * 1024l;
  MemInfo mem_info = MemInfo::deserialize(shareable_handle);
  CUdeviceptr ptr;

  auto &shareable = mem_info.shareable_handle;
  CUresult result = cuMemImportFromShareableHandle(&mem_info.alloc_handle,
                                                   shareable,
                                                   CU_MEM_HANDLE_TYPE_FABRIC);
  if (result != CUDA_SUCCESS) {
    *addr = nullptr;
    return 0;
  }

  mem_info.size = (mem_info.size + granularity - 1) / granularity * granularity;

  result = cuMemAddressReserve(&ptr, mem_info.size, granularity, 0, 0);
  if (result != CUDA_SUCCESS) {
    cuMemRelease(mem_info.alloc_handle);
    return 0l;
  }

  result = cuMemMap(ptr,
                    mem_info.size,
                    0,
                    mem_info.alloc_handle,
                    0);

  if (result != CUDA_SUCCESS) {
    cuMemRelease(mem_info.alloc_handle);
    cuMemAddressFree(ptr, mem_info.size);
    return 0l;
  }

  mem_info.ptr = ptr;

  std::lock_guard<std::mutex> lock(mem_mutex);
  mem_map[reinterpret_cast<void *>(ptr)] = mem_info;
  return mem_info.size;
}

}