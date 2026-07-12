#include "tma_manager.h"
#include <engine/engine.h>
#include <common/logging.h>
#include <cstring>

#define CUDA_CHECK(call) \
    do { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            PCCL_LOG_ERROR("CUDA error: {} ({})", (int)error, cudaGetErrorString(error)); \
        } \
    } while (0)

namespace engine_c::cuda {

#if PCCL_HAS_TMA_HOST

TmaManager::~TmaManager() {
    if (device_desc_) {
        cudaFree(device_desc_);
        device_desc_ = nullptr;
    }
    if (host_desc_) {
        delete host_desc_;
        host_desc_ = nullptr;
    }
}

void TmaManager::setup(
        const std::map<int, std::map<DeviceType, std::tuple<void*, void*>>>& remote_buffers,
        DeviceType cuda_device, void* self_cuda_buf, void* output_buf,
        int elem_size, int self_rank, cudaStream_t stream,
        DeviceWorkspace* host_workspace) {
    bool full_rebuild = (elem_size != cached_elem_size_) || (device_desc_ == nullptr);
    bool self_changed = (self_cuda_buf != cached_self_addr_);
    bool output_changed = (output_buf != cached_output_addr_);

    bool any_peer_changed = false;
    for (auto& [rank, dev_map] : remote_buffers) {
        if (rank >= TMA_MAX_PEERS) continue;
        auto it = dev_map.find(cuda_device);
        if (it != dev_map.end()) {
            void* peer_buf = std::get<0>(it->second);
            auto cached = cached_peer_addrs_.find(rank);
            if (cached == cached_peer_addrs_.end() || cached->second != peer_buf) {
                any_peer_changed = true;
                break;
            }
        }
    }

    if (!full_rebuild && !self_changed && !any_peer_changed && !output_changed) {
        return;
    }

    // Fast path: only output buffer changed — update just output_desc
    if (!full_rebuild && !self_changed && !any_peer_changed && output_changed) {
        int tile_outer = tma_tile_outer_for_elem_size(elem_size);
        if (output_buf) {
            host_desc_->output_desc = create_buffer_tma_desc(
                output_buf, BufferSize, elem_size, tile_outer);
            host_desc_->output_valid = true;
        } else {
            host_desc_->output_valid = false;
        }
        cached_output_addr_ = output_buf;
        CUDA_CHECK(cudaMemcpyAsync(device_desc_, host_desc_,
                                   sizeof(TmaDescriptors), cudaMemcpyHostToDevice, stream));
        host_workspace->tma_desc = device_desc_;
        return;
    }

    int tile_outer = tma_tile_outer_for_elem_size(elem_size);

    if (!host_desc_) {
        host_desc_ = new TmaDescriptors();
    }
    if (!device_desc_) {
        CUDA_CHECK(cudaMalloc(&device_desc_, sizeof(TmaDescriptors)));
    }

    PCCL_LOG_INFO("Setting up TMA descriptors: elem_size={}, tile={}x{}",
                  elem_size, TMA_TILE_INNER, tile_outer);

    std::memset(host_desc_, 0, sizeof(TmaDescriptors));

    host_desc_->self_desc = create_buffer_tma_desc(
        self_cuda_buf, BufferSize, elem_size, tile_outer);
    cached_self_addr_ = self_cuda_buf;

    int num_peers = 0;
    cached_peer_addrs_.clear();
    for (auto& [rank, dev_map] : remote_buffers) {
        if (rank >= TMA_MAX_PEERS) continue;
        auto it = dev_map.find(cuda_device);
        if (it != dev_map.end()) {
            void* peer_buf = std::get<0>(it->second);
            host_desc_->peer_desc[rank] = create_buffer_tma_desc(
                peer_buf, BufferSize, elem_size, tile_outer);
            cached_peer_addrs_[rank] = peer_buf;
            if (rank >= num_peers) num_peers = rank + 1;
        }
    }

    host_desc_->peer_desc[self_rank] = create_buffer_tma_desc(
        self_cuda_buf, BufferSize, elem_size, tile_outer);
    cached_peer_addrs_[self_rank] = self_cuda_buf;
    if (self_rank >= num_peers) num_peers = self_rank + 1;

    host_desc_->elem_size = elem_size;
    host_desc_->tile_inner = TMA_TILE_INNER;
    host_desc_->tile_outer = tile_outer;
    host_desc_->num_peers = num_peers;
    host_desc_->valid = true;

    // Create output descriptor if output buffer is available
    if (output_buf) {
        host_desc_->output_desc = create_buffer_tma_desc(
            output_buf, BufferSize, elem_size, tile_outer);
        host_desc_->output_valid = true;
    } else {
        host_desc_->output_valid = false;
    }
    cached_output_addr_ = output_buf;

    CUDA_CHECK(cudaMemcpyAsync(device_desc_, host_desc_,
                               sizeof(TmaDescriptors), cudaMemcpyHostToDevice, stream));

    host_workspace->tma_desc = device_desc_;

    cached_elem_size_ = elem_size;
    PCCL_LOG_INFO("TMA descriptors ready: {} peers, tile={}x{}, elem_size={}",
                  num_peers, TMA_TILE_INNER, tile_outer, elem_size);
}

#else

TmaManager::~TmaManager() = default;

void TmaManager::setup(
        const std::map<int, std::map<DeviceType, std::tuple<void*, void*>>>&,
        DeviceType, void*, void*, int, int, cudaStream_t,
        DeviceWorkspace* host_workspace) {
    host_workspace->tma_desc = nullptr;
}

#endif

}  // namespace engine_c::cuda
