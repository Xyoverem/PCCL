#pragma once

#include <common.h>
#include <common/socket_utils.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstddef>
#include <thread>
#include <chrono>
#include <cstdlib>

namespace engine_c::cuda {

class NvlsManager {
   public:
    NvlsManager() = default;
    ~NvlsManager() = default;

    void initialize(int self_rank, int world_size, size_t buffer_size) {
        if (initialized_) return;
        initialized_ = true;

        // NVLS is opt-in: the nvls_allreduce_kernel is not yet competitive
        // with the fused kernel, so require explicit PCCL_NVLS_ENABLE=1 to
        // activate.  The old PCCL_DISABLE_NVLS env-var is still honoured as
        // an explicit disable even when PCCL_NVLS_ENABLE is set.
        bool nvls_disabled = (std::getenv("PCCL_DISABLE_NVLS") != nullptr);
        bool nvls_enabled  = (std::getenv("PCCL_NVLS_ENABLE") != nullptr);
        if (nvls_disabled || !nvls_enabled || world_size <= 1) return;

        int mc_supported = 0;
        CUdevice cu_dev;
        int device;
        cudaGetDevice(&device);
        if (cuDeviceGet(&cu_dev, device) == CUDA_SUCCESS) {
            cuDeviceGetAttribute(&mc_supported,
                CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cu_dev);
        }
        if (!mc_supported) return;

        PCCL_LOG_INFO("NVLS: multicast supported, setting up for {} GPUs", world_size);

        bool ok = true;

        CUmulticastObjectProp prop = {};
        prop.numDevices = static_cast<unsigned int>(world_size);
        prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        prop.flags = 0;

        size_t gran = 0;
        if (cuMulticastGetGranularity(&gran, &prop,
                CU_MULTICAST_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) {
            PCCL_LOG_WARN("NVLS: cuMulticastGetGranularity failed");
            return;
        }

        size_t aligned_size = ((buffer_size + gran - 1) / gran) * gran;
        aligned_size = ((aligned_size + 256 + gran - 1) / gran) * gran;
        buffer_size_ = aligned_size;
        barrier_offset_ = aligned_size - 256;

        CUmemGenericAllocationHandle mc_handle = 0;

        if (self_rank == 0) {
            prop.size = aligned_size;
            CUresult st = cuMulticastCreate(&mc_handle, &prop);
            if (st != CUDA_SUCCESS) {
                PCCL_LOG_WARN("NVLS: cuMulticastCreate failed: {}", (int)st);
                ok = false;
            } else {
                int mc_fd = -1;
                st = cuMemExportToShareableHandle(
                    reinterpret_cast<void*>(&mc_fd), mc_handle,
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
                if (st != CUDA_SUCCESS) {
                    PCCL_LOG_WARN("NVLS: export MC handle failed: {}", (int)st);
                    ok = false;
                } else {
                    common::SocketInstance::add_fd("nvls_mc", mc_fd);
                    PCCL_LOG_INFO("NVLS rank 0: MC object created, fd={}", mc_fd);
                }
            }
        } else {
            int mc_fd = -1;
            for (int retry = 0; retry < 100 && mc_fd < 0; retry++) {
                mc_fd = common::SocketInstance::get_remote_fd(0, "nvls_mc");
                if (mc_fd < 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
            if (mc_fd < 0) {
                PCCL_LOG_WARN("NVLS rank {}: failed to get MC fd from rank 0", self_rank);
                ok = false;
            } else {
                CUresult st = cuMemImportFromShareableHandle(
                    &mc_handle,
                    reinterpret_cast<void*>(static_cast<uintptr_t>(mc_fd)),
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
                if (st != CUDA_SUCCESS) {
                    PCCL_LOG_WARN("NVLS rank {}: import MC handle failed: {}", self_rank, (int)st);
                    ok = false;
                } else {
                    PCCL_LOG_INFO("NVLS rank {}: imported MC handle", self_rank);
                }
            }
        }

        if (!ok) {
            PCCL_LOG_WARN("NVLS setup failed, falling back to fused/DAG kernel");
            return;
        }

        CUmemAllocationProp alloc_prop = {};
        alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        alloc_prop.location.id = device;
        alloc_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        CUmemGenericAllocationHandle phys_handle;
        CUresult st = cuMemCreate(&phys_handle, aligned_size, &alloc_prop, 0);
        if (st != CUDA_SUCCESS) {
            PCCL_LOG_WARN("NVLS: phys alloc failed: {}", (int)st);
            ok = false;
        }

        if (ok) {
            st = cuMulticastAddDevice(mc_handle, cu_dev);
            if (st != CUDA_SUCCESS) {
                PCCL_LOG_WARN("NVLS: addDevice failed: {}", (int)st);
                ok = false;
            }
        }

        if (ok) {
            st = cuMulticastBindMem(mc_handle, 0, phys_handle, 0, aligned_size, 0);
            if (st != CUDA_SUCCESS) {
                PCCL_LOG_WARN("NVLS: bindMem failed: {}", (int)st);
                ok = false;
            }
        }

        if (ok) {
            CUdeviceptr phys_va = 0;
            st = cuMemAddressReserve(&phys_va, aligned_size, gran, 0, 0);
            if (st == CUDA_SUCCESS) {
                st = cuMemMap(phys_va, aligned_size, 0, phys_handle, 0);
            }
            if (st == CUDA_SUCCESS) {
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                st = cuMemSetAccess(phys_va, aligned_size, &access, 1);
            }
            if (st != CUDA_SUCCESS) {
                PCCL_LOG_WARN("NVLS: phys VA mapping failed: {}", (int)st);
                ok = false;
            }

            CUdeviceptr mc_va = 0;
            if (ok) {
                st = cuMemAddressReserve(&mc_va, aligned_size, gran, 0, 0);
                if (st == CUDA_SUCCESS) {
                    st = cuMemMap(mc_va, aligned_size, 0, mc_handle, 0);
                }
                if (st == CUDA_SUCCESS) {
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = device;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    st = cuMemSetAccess(mc_va, aligned_size, &access, 1);
                }
                if (st != CUDA_SUCCESS) {
                    PCCL_LOG_WARN("NVLS: MC VA mapping failed: {}", (int)st);
                    ok = false;
                }
            }

            if (ok) {
                available_ = true;
                mc_va_ = reinterpret_cast<void*>(mc_va);
                phys_va_ = reinterpret_cast<void*>(phys_va);
                PCCL_LOG_INFO("NVLS rank {}: ready, mc_va={:#x}, phys_va={:#x}, size={}",
                              self_rank, mc_va, phys_va, aligned_size);
                return;
            }
        }

        PCCL_LOG_WARN("NVLS setup failed, falling back to fused/DAG kernel");
    }

    bool available() const { return available_; }
    void* mc_va() const { return mc_va_; }
    void* phys_va() const { return phys_va_; }
    size_t barrier_offset() const { return barrier_offset_; }
    size_t buffer_size() const { return buffer_size_; }

   private:
    bool initialized_ = false;
    bool available_ = false;
    void* mc_va_ = nullptr;
    void* phys_va_ = nullptr;
    size_t barrier_offset_ = 0;
    size_t buffer_size_ = 0;
};

}  // namespace engine_c::cuda
