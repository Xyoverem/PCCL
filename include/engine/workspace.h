#pragma once

#include <plugins/registry.h>
#include <engine/ring_buffer.h>
#include <engine/fused_step.h>
#include <engine/primitive.h>
#include <engine/host_proxy.h>
#include <plugins/host/ce_proxy.h>

#include <unordered_map>
#include <atomic>

namespace engine_c {

struct Chunk
{
    void* buffer;
    long size;
};

struct ChunkState
{
    int current_chunk;
    int completed_chunks;
    int total_chunks;
    int total_elems;
    int chunk_elems;
};

static constexpr int MAX_CHUNK_SIZE_BYTES = pccl::config::MAX_CHUNK_SIZE_BYTES;
static constexpr int CHUNK_SIZE_BYTES = MAX_CHUNK_SIZE_BYTES;
static constexpr int MAX_CHANNELS = pccl::config::MAX_CHANNELS;

struct alignas(128) ChannelState
{
    ChunkState chunk_state;
    ProxyTrigger active_trigger;
    int active_op_idx;
    int active_valid;
    int completed_primitives;
    int total_primitives;
};

struct DeviceWorkspace
{
    // === Per-call section: H2D copied every iteration ===
    void* input_buffer_;
    size_t io_copy_bytes_;
    void* tma_desc;
    HostProxyState* host_proxy;
    CeProxyState* ce_proxy;
    int total_primitives = 0;
    int completed_primitives = 0;
    void **peer_addr[2];
    void **peer_signals[2];
    bool has_tma_ops;
    int num_channels;
    int grid_barrier_count_ = 0;
    void* output_buffer_;
    FusedStepDescriptor* fused_desc_;
    bool use_fused_;
    char pad_fused_[3];
    int fused_ch_barrier_[MAX_FUSED_CHANNELS];

    // NVLS (multicast) fields
    void* nvls_mc_va_;              // multicast virtual address
    void* nvls_phys_va_;            // local physical VA (mapped via MC)
    size_t nvls_barrier_offset_;    // offset to barrier region in MC buffer
    int nvls_world_size_;
    int nvls_self_rank_;
    int nvls_total_elems_;          // elements in current allreduce (set per call)
    bool use_nvls_;
    char pad_nvls_[3];
    int nvls_grid_counters_[3];     // intra-GPU grid barrier counters (reset per call)

    // === Static section: H2D copied only on first call ===
    RingBuffer ring_buffers_[4];
    GraphBuffer graph_buffer_;

    void *self_addr[2];
    void *self_signals[2];

    void **peer_b_addr[2];
    void **peer_b_signals[2];

    void* runtime_chunk_[2];
    void* runtime_signals_[2];

    ChannelState channels[MAX_CHANNELS];
    RingBuffer channel_ready_queues[MAX_CHANNELS];

    static constexpr size_t PERCALL_COPY_SIZE = 208;
};

static_assert(offsetof(DeviceWorkspace, ring_buffers_) >= DeviceWorkspace::PERCALL_COPY_SIZE,
              "PERCALL_COPY_SIZE exceeds the per-call section boundary");

struct Workspace
{
    DeviceWorkspace* dev_workspace_a;
    DeviceWorkspace* dev_workspace_b;
    DeviceType device_a_;
    DeviceType device_b_;

    Workspace() : dev_workspace_a(nullptr), dev_workspace_b(nullptr) {}
};

class BufferManager
{
   public:
    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;

    static BufferManager& getInstance();

    static void registerDevice(DeviceType device_type, void* buffer, void* signals);
    static void* getSignals(DeviceType device_type);
    static void* getBuffer(DeviceType device_type);

   private:
    BufferManager() = default;
    std::unordered_map<DeviceType, void*> buffers_;
    std::unordered_map<DeviceType, void*> signals_;
};

}  // namespace engine_c
