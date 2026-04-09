#pragma once

#include "tma_descriptor.h"
#include <plugins/registry.h>
#include <engine/workspace.h>
#include <cuda_runtime.h>
#include <map>
#include <tuple>

namespace engine_c::cuda {

class TmaManager {
   public:
    TmaManager() = default;
    ~TmaManager();

    TmaManager(const TmaManager&) = delete;
    TmaManager& operator=(const TmaManager&) = delete;

    void setup(const std::map<int, std::map<DeviceType, std::tuple<void*, void*>>>& remote_buffers,
               DeviceType cuda_device, void* self_cuda_buf, void* output_buf,
               int elem_size, int self_rank, cudaStream_t stream,
               DeviceWorkspace* host_workspace);

    TmaDescriptors* device_desc() const { return device_desc_; }
    int cached_elem_size() const { return cached_elem_size_; }

   private:
    TmaDescriptors* host_desc_ = nullptr;
    TmaDescriptors* device_desc_ = nullptr;
    int cached_elem_size_ = 0;
    void* cached_self_addr_ = nullptr;
    void* cached_output_addr_ = nullptr;
    std::map<int, void*> cached_peer_addrs_;
};

}  // namespace engine_c::cuda
