#include <engine/memory_layout.h>
#include <engine/engine.h>
#include <cstring>
#include <stdexcept>

namespace engine_c {

void MemoryLayout::initialize(void* host_buffer, void* cuda_buffer,
                               void* host_signals, void* cuda_signals,
                               Workspace* workspace) {
    if (BufferSize < WORKSPACE_REGION_SIZE) {
        throw std::runtime_error("PCCL buffer is smaller than the workspace region");
    }

    char* host_start = (char*)host_buffer + (BufferSize - WORKSPACE_REGION_SIZE);
    char* cuda_start = (char*)cuda_buffer + (BufferSize - WORKSPACE_REGION_SIZE);

    workspace->dev_workspace_a = reinterpret_cast<DeviceWorkspace*>(host_start);
    workspace->dev_workspace_b = reinterpret_cast<DeviceWorkspace*>(cuda_start);

    host_start += DEV_WORKSPACE_SIZE;
    cuda_start += DEV_WORKSPACE_SIZE;

    DeviceWorkspace* ws = workspace->dev_workspace_a;
    std::memset(ws, 0, sizeof(DeviceWorkspace));

    ws->peer_addr[0] = reinterpret_cast<void**>(host_start);
    ws->peer_addr[1] = reinterpret_cast<void**>(host_start + PEER_SLOT_SIZE);
    ws->peer_b_addr[0] = reinterpret_cast<void**>(cuda_start);
    ws->peer_b_addr[1] = reinterpret_cast<void**>(cuda_start + PEER_SLOT_SIZE);
    ws->peer_signals[0] = reinterpret_cast<void**>(host_start + PEER_SLOT_SIZE * 2);
    ws->peer_signals[1] = reinterpret_cast<void**>(host_start + PEER_SLOT_SIZE * 3);
    ws->peer_b_signals[0] = reinterpret_cast<void**>(cuda_start + PEER_SLOT_SIZE * 2);
    ws->peer_b_signals[1] = reinterpret_cast<void**>(cuda_start + PEER_SLOT_SIZE * 3);

    ws->self_addr[0] = host_buffer;
    ws->self_addr[1] = cuda_buffer;
    ws->self_signals[0] = host_signals;
    ws->self_signals[1] = cuda_signals;

    host_start += PEER_SLOT_SIZE * 4;
    cuda_start += PEER_SLOT_SIZE * 4;

    ws->runtime_chunk_[0] = host_buffer;
    ws->runtime_chunk_[1] = cuda_buffer;
    ws->runtime_signals_[0] = host_signals;
    ws->runtime_signals_[1] = cuda_signals;

    ws->graph_buffer_.meta[0] = (PrimitiveMeta*)(host_start);
    ws->graph_buffer_.meta[1] = (PrimitiveMeta*)(cuda_start);
    ws->graph_buffer_.working_meta[0] = (WorkingMeta*)(host_start + GRAPH_META_SIZE);
    ws->graph_buffer_.working_meta[1] = (WorkingMeta*)(cuda_start + GRAPH_META_SIZE);
    ws->graph_buffer_.primitives[0] = (void*)(host_start + GRAPH_META_SIZE + WORKING_META_SIZE);
    ws->graph_buffer_.primitives[1] = (void*)(cuda_start + GRAPH_META_SIZE + WORKING_META_SIZE);

    host_start += GRAPH_META_SIZE + WORKING_META_SIZE + PRIMITIVES_SIZE;
    cuda_start += GRAPH_META_SIZE + WORKING_META_SIZE + PRIMITIVES_SIZE;

    auto initQueueMeta = [](QueueMeta* meta, long capacity) {
        meta->capacity = capacity;
        meta->head = 0;
        meta->tail = 0;
        meta->size = 0;
    };

    long queue_capacity = QUEUE_DATA_SIZE / sizeof(ProxyTrigger);

    // Ring buffer 0 (Host -> CUDA)
    ws->ring_buffers_[0].buffer_a_ = (ProxyTrigger*)(host_start);
    ws->ring_buffers_[0].meta_a_ = (QueueMeta*)(host_start + QUEUE_DATA_SIZE);
    initQueueMeta(ws->ring_buffers_[0].meta_a_, queue_capacity);
    host_start += QUEUE_TOTAL_SIZE;

    // Ring buffer 1 (CUDA -> Host)
    ws->ring_buffers_[1].buffer_b_ = (ProxyTrigger*)(host_start);
    ws->ring_buffers_[1].meta_b_ = (QueueMeta*)(host_start + QUEUE_DATA_SIZE);
    initQueueMeta(ws->ring_buffers_[1].meta_b_, queue_capacity);
    host_start += QUEUE_TOTAL_SIZE;

    // Ring buffer 2 (Host -> Host), both sides
    ws->ring_buffers_[2].buffer_a_ = (ProxyTrigger*)(host_start);
    ws->ring_buffers_[2].meta_a_ = (QueueMeta*)(host_start + QUEUE_DATA_SIZE);
    initQueueMeta(ws->ring_buffers_[2].meta_a_, queue_capacity);
    host_start += QUEUE_TOTAL_SIZE;

    ws->ring_buffers_[2].buffer_b_ = (ProxyTrigger*)(host_start);
    ws->ring_buffers_[2].meta_b_ = (QueueMeta*)(host_start + QUEUE_DATA_SIZE);
    initQueueMeta(ws->ring_buffers_[2].meta_b_, queue_capacity);

    // Ring buffer 0 CUDA side
    ws->ring_buffers_[0].buffer_b_ = (ProxyTrigger*)(cuda_start);
    ws->ring_buffers_[0].meta_b_ = (QueueMeta*)(cuda_start + QUEUE_DATA_SIZE);
    cuda_start += QUEUE_TOTAL_SIZE;

    // Ring buffer 1 CUDA side
    ws->ring_buffers_[1].buffer_a_ = (ProxyTrigger*)(cuda_start);
    ws->ring_buffers_[1].meta_a_ = (QueueMeta*)(cuda_start + QUEUE_DATA_SIZE);
    cuda_start += QUEUE_TOTAL_SIZE;

    // Ring buffer 3 (CUDA -> CUDA), both sides
    ws->ring_buffers_[3].buffer_a_ = (ProxyTrigger*)(cuda_start);
    ws->ring_buffers_[3].meta_a_ = (QueueMeta*)(cuda_start + QUEUE_DATA_SIZE);
    cuda_start += QUEUE_TOTAL_SIZE;

    ws->ring_buffers_[3].buffer_b_ = (ProxyTrigger*)(cuda_start);
    ws->ring_buffers_[3].meta_b_ = (QueueMeta*)(cuda_start + QUEUE_DATA_SIZE);
    cuda_start += QUEUE_TOTAL_SIZE;

    // Per-channel ready queues (CUDA side only)
    for (int c = 0; c < MAX_CHANNELS; c++) {
        ws->channel_ready_queues[c].buffer_b_ = (ProxyTrigger*)(cuda_start);
        ws->channel_ready_queues[c].meta_b_ = (QueueMeta*)(cuda_start + CHANNEL_QUEUE_DATA_SIZE);
        ws->channel_ready_queues[c].buffer_a_ = nullptr;
        ws->channel_ready_queues[c].meta_a_ = nullptr;
        cuda_start += CHANNEL_QUEUE_TOTAL_SIZE;
    }

    // Fused step descriptor (host + CUDA side)
    ws->fused_desc_ = nullptr;
    ws->use_fused_ = false;
    std::memset(ws->fused_ch_barrier_, 0, sizeof(ws->fused_ch_barrier_));

    // NVLS fields
    ws->nvls_mc_va_ = nullptr;
    ws->nvls_phys_va_ = nullptr;
    ws->nvls_barrier_offset_ = 0;
    ws->nvls_world_size_ = 0;
    ws->nvls_self_rank_ = 0;
    ws->nvls_total_elems_ = 0;
    ws->use_nvls_ = false;
    std::memset(ws->nvls_grid_counters_, 0, sizeof(ws->nvls_grid_counters_));

    // Host-side fused descriptor stored right after channel queues in host memory
    // Will be populated by GraphBuilder::buildFusedDescriptor if applicable

    ws->num_channels = 1;
    ws->total_primitives = -1;
}

}  // namespace engine_c
