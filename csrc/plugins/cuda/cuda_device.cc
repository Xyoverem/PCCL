#include <plugins/base.h>
#include <plugins/registry.h>
#include <engine/workspace.h>
#include <engine/fused_step.h>
#include <common.h>
#include <fmt/format.h>
#include <unordered_map>
#include <mutex>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <unistd.h>
#include <cstdint>
#include <stdexcept>
#include "kernel/proxy_trigger.h"
#include "kernel/primitive_config.h"
#include <nvtx3/nvtx3.hpp>
#include "./cuda_defs.h"
#include "./cuda_executor.h"
#include "./tma_manager.h"
#include "./nvls_manager.h"

// Channel implementations
#include "channels/sm_channel.h"
#include "channels/tma_channel.h"
#include "channels/ce_channel.h"
#include "channels/multimem_channel.h"
#include "channels/rdma_channel.h"

namespace engine_c::cuda {

struct MemInfo
{
    long size;
    CUmemGenericAllocationHandle alloc_handle;
    CUdeviceptr ptr;
    int rank;
    std::string entry;
    int file_desc;
    int root_pid;

    std::string serialize() const
    {
        nlohmann::json j;
        j["size"] = size;
        j["root_pid"] = root_pid;
        j["entry"] = entry;
        j["rank"] = rank;
        return j.dump();
    }

    static MemInfo deserialize(const std::string& mem_info)
    {
        auto j = nlohmann::json::parse(mem_info);
        MemInfo r;
        r.size = j["size"];
        r.root_pid = j["root_pid"];
        r.entry = j["entry"];
        r.rank = j["rank"];
        return r;
    }
};

static std::string device_name = "Cuda";

class CudaDevice : public DeviceBase
{
   public:
    CudaDevice() {
        // Initialize channel objects
        channels_list_ = {
            &sm_channel_, &tma_channel_, &ce_channel_,
            &multimem_channel_, &rdma_channel_
        };
    }
    virtual ~CudaDevice()
    {
        std::lock_guard<std::mutex> lock(mem_mutex);
        for (auto& [ptr, mem_info] : mem_map) {
            if (getpid() == mem_info.root_pid) {
                close(mem_info.file_desc);
            }
            cuMemUnmap(mem_info.ptr, mem_info.size);
            cuMemAddressFree(mem_info.ptr, mem_info.size);
            cuMemRelease(mem_info.alloc_handle);
        }
    }

    const std::string& deviceName() const override {
        return device_name;
    }

    bool allocatorAvailable() override
    {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        return device_count > 0;
    }

    void* allocate(long nbytes) override
    {
        static constexpr long granularity = 2 * 1024 * 1024;
        if (nbytes <= 0) return nullptr;

        int device_index = c10::cuda::current_device();

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_index;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        size_t aligned_size = ((nbytes + granularity - 1) / granularity) * granularity;

        CUmemGenericAllocationHandle alloc_handle;
        CUDA_DRIVER_CHECK(cuMemCreate(&alloc_handle, aligned_size, &prop, 0));

        CUdeviceptr dev_ptr;
        CUDA_DRIVER_CHECK(cuMemAddressReserve(&dev_ptr, aligned_size, granularity, 0, 0));
        CUDA_DRIVER_CHECK(cuMemMap(dev_ptr, aligned_size, 0, alloc_handle, 0));

        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_index;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CUDA_DRIVER_CHECK(cuMemSetAccess(dev_ptr, aligned_size, &access_desc, 1));

        int file_desc = -1;
        CUDA_DRIVER_CHECK(
            cuMemExportToShareableHandle(reinterpret_cast<void*>(&file_desc), alloc_handle,
                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

        std::lock_guard<std::mutex> lock(mem_mutex);
        MemInfo info;
        info.size = aligned_size;
        info.alloc_handle = alloc_handle;
        info.ptr = dev_ptr;
        info.file_desc = file_desc;
        info.root_pid = getpid();
        info.entry = "cuda_vmm_segment";
        info.rank = std::stoi(common::Environs::getEnv("RANK").data());
        common::SocketInstance::add_fd("cuda_vmm_segment", file_desc);
        mem_map[reinterpret_cast<void*>(dev_ptr)] = info;

        return reinterpret_cast<void*>(dev_ptr);
    }

    void deallocate(void* ptr) override
    {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mem_mutex);
        auto it = mem_map.find(ptr);
        if (it != mem_map.end()) {
            if (getpid() == it->second.root_pid) {
                close(it->second.file_desc);
            }
            CUDA_DRIVER_CHECK(cuMemUnmap(it->second.ptr, it->second.size));
            CUDA_DRIVER_CHECK(cuMemAddressFree(it->second.ptr, it->second.size));
            CUDA_DRIVER_CHECK(cuMemRelease(it->second.alloc_handle));
            mem_map.erase(it);
        }
    }

    bool IpcAvailable() override
    {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        return device_count > 0;
    }

    std::string allocateBuffer(void** addr, long size) override
    {
        *addr = allocate(size);
        TORCH_CHECK(*addr != nullptr);
        std::lock_guard<std::mutex> lock(mem_mutex);
        auto it = mem_map.find(*addr);
        return it->second.serialize();
    }

    long mapBuffer(std::string& shareable_handle, void** addr) override
    {
        static constexpr long granularity = 2 * 1024 * 1024;
        MemInfo mem_info = MemInfo::deserialize(shareable_handle);
        int remote_fd = common::SocketInstance::get_remote_fd(mem_info.rank, mem_info.entry);
        TORCH_CHECK(remote_fd != -1);
        mem_info.file_desc = remote_fd;
        CUdeviceptr ptr;

        CUDA_DRIVER_CHECK(cuMemImportFromShareableHandle(
            &mem_info.alloc_handle,
            reinterpret_cast<void*>(static_cast<uintptr_t>(mem_info.file_desc)),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

        mem_info.size = (mem_info.size + granularity - 1) / granularity * granularity;
        CUDA_DRIVER_CHECK(cuMemAddressReserve(&ptr, mem_info.size, granularity, 0, 0));
        CUDA_DRIVER_CHECK(cuMemMap(ptr, mem_info.size, 0, mem_info.alloc_handle, 0));

        int device_index = c10::cuda::current_device();

        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_index;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CUDA_DRIVER_CHECK(cuMemSetAccess(ptr, mem_info.size, &access_desc, 1));

        mem_info.ptr = ptr;
        std::lock_guard<std::mutex> lock(mem_mutex);
        mem_map[reinterpret_cast<void*>(ptr)] = mem_info;
        *addr = reinterpret_cast<void*>(ptr);
        return mem_info.size;
    }

    void deallocateBuffer(void* addr) override { deallocate(addr); }

    void memcpy_sync(void* src, void* dst, long size, DeviceType src_type,
                     DeviceType dst_type) override
    {
        cudaMemcpyKind kind = getCudaMemcpyKind(src_type, dst_type);
        CUDA_CHECK(cudaMemcpy(dst, src, size, kind));
    }

    void memcpy_async(void* src, void* dst, long size, DeviceType src_type,
                      DeviceType dst_type, void* stream) override
    {
        cudaMemcpyKind kind = getCudaMemcpyKind(src_type, dst_type);
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, kind, static_cast<cudaStream_t>(stream)));
    }

    void memset_sync(void* dst, unsigned char val, long size, DeviceType src_type,
                     DeviceType dst_type) override
    {
        CUDA_CHECK(cudaMemset(dst, val, size));
    }

    void memset_async(void* dst, unsigned char val, long size, void* stream) override
    {
        CUDA_CHECK(cudaMemsetAsync(dst, val, size, static_cast<cudaStream_t>(stream)));
    }

    void* getStream() override
    {
        return static_cast<void*>(c10::cuda::getCurrentCUDAStream());
    }

    void streamSync(void* stream) override
    {
        CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    }

    bool ExecutorAvailable() override { return true; }

    // ---- Channel-based dispatch (NEW) ----
    std::vector<Channel*> channels() override {
        return channels_list_;
    }

    // parse() delegates to channels via DevicePlugin::parse() default impl.
    // Also handles legacy bare "notify"/"wait_notify"/"noop" primitives.
    ProxyTrigger parse(nlohmann::json &op_info, nlohmann::json &tensor_info) override {
        std::string primitive = op_info["primitive"];

        // Legacy bare names -> delegate to sm channel with prefixed name
        if (primitive == "notify") {
            op_info["primitive"] = "sm.notify";
            return sm_channel_.parse(op_info, tensor_info);
        }
        if (primitive == "wait_notify") {
            op_info["primitive"] = "sm.wait_notify";
            return sm_channel_.parse(op_info, tensor_info);
        }
        if (primitive == "noop") {
            ProxyTrigger t;
            t.raw[0] = 0;
            t.raw[1] = 0;
            return t;
        }

        // Delegate to DevicePlugin::parse() which routes via channel name
        return DevicePlugin::parse(op_info, tensor_info);
    }

    void execute(Workspace* workspace) override
    {
        static constexpr int TMA_SMEM_BYTES = TMA_SMEM_TOTAL;
        static const int cfg_num_blocks = [] {
            const char* env = std::getenv("PCCL_NUM_BLOCKS");
            return env ? std::atoi(env) : 20;
        }();
        static const int cfg_nvls_blocks = [] {
            const char* env = std::getenv("PCCL_NVLS_NUM_BLOCKS");
            return env ? std::atoi(env) : 108;
        }();
        static const bool exclusive_sm = (std::getenv("PCCL_EXCLUSIVE_SM") != nullptr);

        DeviceWorkspace* host_ws = workspace->dev_workspace_a;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        bool use_tma = PCCL_HAS_TMA_HOST && host_ws->has_tma_ops;
        int smem = use_tma ? TMA_SMEM_BYTES : 0;
        if (exclusive_sm && smem < TMA_SMEM_BYTES)
            smem = TMA_SMEM_BYTES;

        if (host_ws->use_nvls_ && host_ws->nvls_mc_va_) {
            { nvtx3::scoped_range nvtx_mk{"nvls_kernel"};
            launch_nvls_kernel(cfg_nvls_blocks, stream, workspace->dev_workspace_b);
            }
        } else if (host_ws->use_fused_ && host_ws->fused_desc_) {
            { nvtx3::scoped_range nvtx_mk{"fused_kernel"};
            launch_fused_kernel(cfg_num_blocks, stream, workspace->dev_workspace_b,
                                host_ws->fused_desc_, smem);
            }
        } else {
            int queue_capacity = host_ws->ring_buffers_[0].meta_a_->capacity;
            { nvtx3::scoped_range nvtx_mk{"executor_kernel"};
            launch_cuda_kernel(cfg_num_blocks, stream, workspace->dev_workspace_b, smem,
                               host_ws->total_primitives, queue_capacity);
            }
        }
    }

    void setupTma(
        const std::map<int, std::map<DeviceType, std::tuple<void*, void*>>>& remote_buffers,
        void* self_cuda_buf, void* output_buf, int elem_size, int self_rank,
        void* stream, DeviceWorkspace* host_workspace) override
    {
        tma_manager_.setup(remote_buffers, TypeRegistry::getTypeId("Cuda"),
                           self_cuda_buf, output_buf, elem_size, self_rank,
                           static_cast<cudaStream_t>(stream), host_workspace);
    }

    int tmaCachedElemSize() const { return tma_manager_.cached_elem_size(); }
    bool tmaInitialized() const { return tma_manager_.device_desc() != nullptr; }

    void initNvls(int self_rank, int world_size, size_t buffer_size) override {
        nvls_manager_.initialize(self_rank, world_size, buffer_size);
    }

    bool nvlsAvailable() const override { return nvls_manager_.available(); }
    void* nvlsMcVa() const override { return nvls_manager_.mc_va(); }
    void* nvlsPhysVa() const override { return nvls_manager_.phys_va(); }
    size_t nvlsBarrierOffset() const override { return nvls_manager_.barrier_offset(); }

    void prepareNvls(DeviceWorkspace* ws, bool is_allreduce,
                     int scalar_type, long numel) override {
        bool use = nvls_manager_.available() &&
                   is_allreduce &&
                   (scalar_type == 15) && // at::kBFloat16 == 15
                   (numel > 0);
        ws->use_nvls_ = use;
        if (use) {
            ws->nvls_total_elems_ = static_cast<int>(numel);
        }
    }

    FusedStepDescriptor* uploadFusedDescriptor(
        const FusedStepDescriptor& host_desc, void* stream) override {
        FusedStepDescriptor* dev_ptr = nullptr;
        cudaMalloc(&dev_ptr, sizeof(FusedStepDescriptor));
        cudaMemcpyAsync(dev_ptr, &host_desc,
                       sizeof(FusedStepDescriptor), cudaMemcpyHostToDevice,
                       static_cast<cudaStream_t>(stream));
        fused_device_ptrs_.push_back(dev_ptr);
        return dev_ptr;
    }

    NvlsManager& nvlsManager() { return nvls_manager_; }

   private:
    std::unordered_map<void*, MemInfo> mem_map;
    std::mutex mem_mutex;
    TmaManager tma_manager_;
    NvlsManager nvls_manager_;
    std::vector<FusedStepDescriptor*> fused_device_ptrs_;

    // Channel instances (owned by this device)
    SmChannel sm_channel_;
    TmaChannel tma_channel_;
    CeChannel ce_channel_;
    MultiMemChannel multimem_channel_;
    RdmaChannel rdma_channel_;
    std::vector<Channel*> channels_list_;

    cudaMemcpyKind getCudaMemcpyKind(DeviceType src_type, DeviceType dst_type)
    {
        std::string src_name = TypeRegistry::getTypeName(src_type);
        std::string dst_name = TypeRegistry::getTypeName(dst_type);

        if (src_name == "Cuda" && dst_name == "Cuda") return cudaMemcpyDeviceToDevice;
        else if (src_name == "Host" && dst_name == "Host") return cudaMemcpyHostToHost;
        else if (src_name == "Host" && dst_name == "Cuda") return cudaMemcpyHostToDevice;
        else if (src_name == "Cuda" && dst_name == "Host") return cudaMemcpyDeviceToHost;
        else return cudaMemcpyDefault;
    }
};

struct TMP_STATIC_INITIALIZER
{
    TMP_STATIC_INITIALIZER()
    {
        auto device = TypeRegistry::registerDeviceType(device_name);
        regDev(device, std::make_shared<CudaDevice>());
    }
};

static TMP_STATIC_INITIALIZER _____tmp;

}  // namespace engine_c::cuda
