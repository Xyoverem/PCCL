#include "cuda_executor.h"
#include "kernel/proxy_trigger.h"
#include "kernel/primitive_config.h"
#include "kernel/compute.h"
#include "kernel/tma_helpers.h"
#include "kernel/multimem_helpers.h"
#include "tma_descriptor.h"

#include <common/config.h>
#include <engine/workspace.h>
#include <engine/ring_buffer.h>
#include <engine/primitive.h>
#include <engine/host_proxy.h>
#include <plugins/host/ce_proxy.h>
#include <engine/fused_step.h>

#include <cuda_runtime.h>
#include <cuda.h>

namespace engine_c::cuda {

// Atomic ring buffer helpers
__device__ bool dequeueFromRingBuffer(RingBuffer* rb, ProxyTrigger* out) {
    QueueMeta* meta = rb->meta_b_;
    while (true) {
        int head = *(volatile int*)&meta->head;
        int tail = *(volatile int*)&meta->tail;
        if (head >= tail) return false;
        int old_head = atomicCAS(&meta->head, head, head + 1);
        if (old_head == head) {
            int slot = head % meta->capacity;
            *out = rb->buffer_b_[slot];
            return true;
        }
    }
}

__device__ void enqueueToRingBuffer(RingBuffer* rb, const ProxyTrigger* item) {
    QueueMeta* meta = rb->meta_b_;
    int slot_idx = atomicAdd(&meta->tail, 1);
    int slot = slot_idx % meta->capacity;
    rb->buffer_b_[slot] = *item;
    __threadfence();
}

// Primitive dispatch helpers
__device__ int dtypeElemSize(DataType dtype) {
    switch (dtype) {
        case DataType::F32:      return 4;
        case DataType::F16:      return 2;
        case DataType::BF16:     return 2;
        case DataType::FP8_E4M3: return 1;
        case DataType::FP8_E5M2: return 1;
        default:                 return 4;
    }
}

__device__ int dtypeVecN(DataType dtype) { return 16 / dtypeElemSize(dtype); }

__device__ void executeReduce(DeviceWorkspace* ws, const op* trigger, DataType dtype,
                               int chunk_offset, int chunk_count) {
    int peer_rank = trigger->op_handle.peer_rank;
    int offset_0  = trigger->op_handle.offset_0;
    int offset_1  = trigger->op_handle.offset_1;
    int offset_2  = trigger->op_handle.offset_2;

    char** peers = (char**)ws->peer_addr[1];
    char* peer_buf = peers[peer_rank];
    char* self_buf = (char*)ws->self_addr[1];
    int elem_size = dtypeElemSize(dtype);

    int actual_off0 = offset_0 + chunk_offset;
    int actual_off1 = offset_1 + chunk_offset;
    int actual_off2 = offset_2 + chunk_offset;

    void* a = (void*)(peer_buf + (long)actual_off0 * elem_size);
    void* b = (void*)(self_buf + (long)actual_off1 * elem_size);
    void* c = (void*)(self_buf + (long)actual_off2 * elem_size);

    vector_add(dtype, a, b, c, chunk_count);
}

__device__ void executeCopy(DeviceWorkspace* ws, const op* trigger, DataType dtype,
                             int chunk_offset, int chunk_count) {
    int peer_rank = trigger->op_handle.peer_rank;
    int offset_0  = trigger->op_handle.offset_0;
    int offset_1  = trigger->op_handle.offset_1;

    char** peers = (char**)ws->peer_addr[1];
    char* peer_buf = peers[peer_rank];
    char* dst_buf = (char*)ws->self_addr[1];
    int elem_size = dtypeElemSize(dtype);

    int actual_src = offset_0 + chunk_offset;
    int actual_dst = offset_1 + chunk_offset;

    void* src = (void*)(peer_buf + (long)actual_src * elem_size);
    void* dst = (void*)(dst_buf + (long)actual_dst * elem_size);

    vector_copy(dtype, src, dst, chunk_count);
}

__device__ void executeTmaCopy(DeviceWorkspace* ws, const op* trigger, DataType dtype,
                                int chunk_offset, int chunk_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (ws->tma_desc == nullptr) goto sm_fallback;
    {
        TmaDescriptors* tma = reinterpret_cast<TmaDescriptors*>(ws->tma_desc);
        if (!tma->valid) goto sm_fallback;
        int peer_rank  = trigger->op_handle.peer_rank;
        int offset_0   = trigger->op_handle.offset_0;
        int offset_1   = trigger->op_handle.offset_1;
        int elem_size  = tma->elem_size;
        int tile_inner = tma->tile_inner;
        int tile_outer = tma->tile_outer;
        int tile_elems = tile_inner * tile_outer;
        int tile_bytes = tile_elems * elem_size;

        extern __shared__ char smem_raw_unaligned[];
        char* smem_raw = reinterpret_cast<char*>(
            (reinterpret_cast<uintptr_t>(smem_raw_unaligned) + 127u) & ~uintptr_t(127));
        char* buf[2] = { smem_raw, smem_raw + tile_bytes };
        int mbar_offset = (2 * tile_bytes + 7) & ~7;
        uint64_t* mbar[2] = {
            reinterpret_cast<uint64_t*>(smem_raw + mbar_offset),
            reinterpret_cast<uint64_t*>(smem_raw + mbar_offset + 8)
        };

        int src_outer_start = (offset_0 + chunk_offset) / tile_inner;
        int dst_outer_start = (offset_1 + chunk_offset) / tile_inner;
        int num_subtiles = chunk_count / tile_elems;
        int tail_elems = chunk_count - num_subtiles * tile_elems;

        CUtensorMap* peer_tmap = &tma->peer_desc[peer_rank];
        CUtensorMap* self_tmap = &tma->self_desc;
        bool has_output = tma->output_valid;
        CUtensorMap* output_tmap = has_output ? &tma->output_desc : nullptr;

        if (threadIdx.x == 0) {
            tma::fence_tensormap_acquire_cta(peer_tmap);
            tma::fence_tensormap_acquire_cta(self_tmap);
            if (has_output) tma::fence_tensormap_acquire_cta(output_tmap);
            tma::mbarrier_init_local(mbar[0], 1);
            tma::mbarrier_init_local(mbar[1], 1);
        }
        __syncthreads();

        if (num_subtiles > 0 && threadIdx.x == 0) {
            tma::mbarrier_arrive_expect_tx(mbar[0], static_cast<uint32_t>(tile_bytes));
            tma::tma_load_2d(peer_tmap, mbar[0],
                             reinterpret_cast<void*>(buf[0]),
                             0, src_outer_start);
        }

        for (int st = 0; st < num_subtiles; st++) {
            int phase = st & 1;
            int next_phase = 1 - phase;

            if (st + 1 < num_subtiles && threadIdx.x == 0) {
                int next_src_crd1 = src_outer_start + (st + 1) * tile_outer;
                tma::mbarrier_arrive_expect_tx(mbar[next_phase],
                                               static_cast<uint32_t>(tile_bytes));
                tma::tma_load_2d(peer_tmap, mbar[next_phase],
                                 reinterpret_cast<void*>(buf[next_phase]),
                                 0, next_src_crd1);
            }

            tma::mbarrier_wait_parity(mbar[phase], (st >> 1) & 1);
            __syncthreads();

            if (threadIdx.x == 0) {
                tma::cp_async_bulk_wait_all();
                tma::fence_proxy_async_shared_cta();

                int dst_crd1 = dst_outer_start + st * tile_outer;
                tma::tma_store_2d(self_tmap,
                                  reinterpret_cast<const void*>(buf[phase]),
                                  0, dst_crd1);
                if (has_output) {
                    tma::tma_store_2d(output_tmap,
                                      reinterpret_cast<const void*>(buf[phase]),
                                      0, dst_crd1);
                }
                tma::cp_async_bulk_commit();
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            tma::cp_async_bulk_wait_all();
        }
        __syncthreads();

        if (tail_elems > 0) {
            char** peers = (char**)ws->peer_addr[1];
            char* peer_buf = peers[peer_rank];
            char* dst_buf = (char*)ws->self_addr[1];
            int tail_start = num_subtiles * tile_elems;
            int actual_src = offset_0 + chunk_offset + tail_start;
            int actual_dst = offset_1 + chunk_offset + tail_start;
            void* src = (void*)(peer_buf + (long)actual_src * elem_size);
            void* dst = (void*)(dst_buf + (long)actual_dst * elem_size);
            tma_vector_copy(dtype, src, dst, tail_elems);
            // Also copy tail to output buffer
            if (has_output && ws->output_buffer_) {
                void* out_dst = (void*)((char*)ws->output_buffer_ + (long)actual_dst * elem_size);
                tma_vector_copy(dtype, src, out_dst, tail_elems);
            }
        }
        return;
    }
    sm_fallback:
#endif
    {
        int peer_rank = trigger->op_handle.peer_rank;
        int offset_0  = trigger->op_handle.offset_0;
        int offset_1  = trigger->op_handle.offset_1;

        char** peers = (char**)ws->peer_addr[1];
        char* peer_buf = peers[peer_rank];
        char* dst_buf = (char*)ws->self_addr[1];
        int elem_size = dtypeElemSize(dtype);
        int actual_src = offset_0 + chunk_offset;
        int actual_dst = offset_1 + chunk_offset;

        void* src = (void*)(peer_buf + (long)actual_src * elem_size);
        void* dst = (void*)(dst_buf + (long)actual_dst * elem_size);

        tma_vector_copy(dtype, src, dst, chunk_count);
        // SM fallback: also copy to output buffer
        if (ws->output_buffer_) {
            void* out_dst = (void*)((char*)ws->output_buffer_ + (long)actual_dst * elem_size);
            tma_vector_copy(dtype, src, out_dst, chunk_count);
        }
    }
}

__device__ void executeTmaReduce(DeviceWorkspace* ws, const op* trigger, DataType dtype,
                                  int chunk_offset, int chunk_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (ws->tma_desc == nullptr) goto sm_fallback;
    {
        TmaDescriptors* tma = reinterpret_cast<TmaDescriptors*>(ws->tma_desc);
        if (!tma->valid) goto sm_fallback;
        int peer_rank  = trigger->op_handle.peer_rank;
        int offset_0   = trigger->op_handle.offset_0;
        int offset_1   = trigger->op_handle.offset_1;
        int offset_2   = trigger->op_handle.offset_2;
        int elem_size  = tma->elem_size;
        int tile_inner = tma->tile_inner;
        int tile_outer = tma->tile_outer;
        int tile_elems = tile_inner * tile_outer;
        int tile_bytes = tile_elems * elem_size;

        extern __shared__ char smem_raw_unaligned[];
        char* smem_raw = reinterpret_cast<char*>(
            (reinterpret_cast<uintptr_t>(smem_raw_unaligned) + 127u) & ~uintptr_t(127));
        // Double-buffered: 2 tile buffers (in-place compute) + 2 mbarriers
        char* buf[2] = { smem_raw, smem_raw + tile_bytes };
        int mbar_offset = (2 * tile_bytes + 7) & ~7;
        uint64_t* mbar[2] = {
            reinterpret_cast<uint64_t*>(smem_raw + mbar_offset),
            reinterpret_cast<uint64_t*>(smem_raw + mbar_offset + 8)
        };

        char* self_buf = (char*)ws->self_addr[1];

        int remote_outer_start = (offset_0 + chunk_offset) / tile_inner;
        int local_elem_start   = offset_1 + chunk_offset;
        int dst_outer_start    = (offset_2 + chunk_offset) / tile_inner;
        int num_subtiles       = chunk_count / tile_elems;
        int tail_elems         = chunk_count - num_subtiles * tile_elems;

        CUtensorMap* peer_tmap = &tma->peer_desc[peer_rank];
        CUtensorMap* self_tmap = &tma->self_desc;

        bool use_hw_reduce = (offset_1 == offset_2);

        if (threadIdx.x == 0) {
            tma::fence_tensormap_acquire_cta(peer_tmap);
            tma::fence_tensormap_acquire_cta(self_tmap);
            tma::mbarrier_init_local(mbar[0], 1);
            tma::mbarrier_init_local(mbar[1], 1);
        }
        __syncthreads();

        // Prologue: start loading first tile into buf[0]
        if (num_subtiles > 0 && threadIdx.x == 0) {
            tma::mbarrier_arrive_expect_tx(mbar[0], static_cast<uint32_t>(tile_bytes));
            tma::tma_load_2d(peer_tmap, mbar[0],
                             reinterpret_cast<void*>(buf[0]),
                             0, remote_outer_start);
        }

        if (use_hw_reduce) {
            for (int st = 0; st < num_subtiles; st++) {
                int phase = st & 1;
                int next_phase = 1 - phase;

                if (st + 1 < num_subtiles && threadIdx.x == 0) {
                    int next_remote_crd1 = remote_outer_start + (st + 1) * tile_outer;
                    tma::mbarrier_arrive_expect_tx(mbar[next_phase],
                                                   static_cast<uint32_t>(tile_bytes));
                    tma::tma_load_2d(peer_tmap, mbar[next_phase],
                                     reinterpret_cast<void*>(buf[next_phase]),
                                     0, next_remote_crd1);
                }

                tma::mbarrier_wait_parity(mbar[phase], (st >> 1) & 1);

                if (threadIdx.x == 0) {
                    tma::cp_async_bulk_wait_all_read();
                    tma::fence_proxy_async_shared_cta();
                    int dst_crd1 = dst_outer_start + st * tile_outer;
                    tma::tma_reduce_add_2d(self_tmap,
                                           reinterpret_cast<const void*>(buf[phase]),
                                           0, dst_crd1);
                    tma::cp_async_bulk_commit();
                }
            }

            if (threadIdx.x == 0) {
                tma::cp_async_bulk_wait_all();
            }
            __syncthreads();
        } else {
            // SM reduce path: load peer → add with self_buf in SM → store back
            for (int st = 0; st < num_subtiles; st++) {
                int phase = st & 1;
                int next_phase = 1 - phase;
                int local_offset_elems = local_elem_start + st * tile_elems;

                if (st + 1 < num_subtiles && threadIdx.x == 0) {
                    int next_remote_crd1 = remote_outer_start + (st + 1) * tile_outer;
                    tma::mbarrier_arrive_expect_tx(mbar[next_phase],
                                                   static_cast<uint32_t>(tile_bytes));
                    tma::tma_load_2d(peer_tmap, mbar[next_phase],
                                     reinterpret_cast<void*>(buf[next_phase]),
                                     0, next_remote_crd1);
                }

                tma::mbarrier_wait_parity(mbar[phase], (st >> 1) & 1);
                __syncthreads();

                if (threadIdx.x == 0) {
                    tma::cp_async_bulk_wait_all();
                }
                __syncthreads();

                {
                    void* a_ptr = (void*)buf[phase];
                    void* b_ptr = (void*)(self_buf + (long)local_offset_elems * elem_size);
                    smem_vector_add(dtype, a_ptr, b_ptr, a_ptr, tile_elems);
                }
                __syncthreads();

                tma::fence_proxy_async_shared_cta();

                if (threadIdx.x == 0) {
                    int dst_crd1 = dst_outer_start + st * tile_outer;
                    tma::tma_store_2d(self_tmap,
                                      reinterpret_cast<const void*>(buf[phase]),
                                      0, dst_crd1);
                    tma::cp_async_bulk_commit();
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                tma::cp_async_bulk_wait_all();
            }
            __syncthreads();
        }

        if (tail_elems > 0) {
            char** peers = (char**)ws->peer_addr[1];
            char* peer_buf = peers[peer_rank];
            int tail_start = num_subtiles * tile_elems;
            int actual_off0 = offset_0 + chunk_offset + tail_start;
            int actual_off1 = offset_1 + chunk_offset + tail_start;
            int actual_off2 = offset_2 + chunk_offset + tail_start;
            void* a = (void*)(peer_buf + (long)actual_off0 * elem_size);
            void* b = (void*)(self_buf + (long)actual_off1 * elem_size);
            void* c = (void*)(self_buf + (long)actual_off2 * elem_size);
            tma_vector_add(dtype, a, b, c, tail_elems);
        }
        return;
    }
    sm_fallback:
#endif
    {
        int peer_rank  = trigger->op_handle.peer_rank;
        int offset_0   = trigger->op_handle.offset_0;
        int offset_1   = trigger->op_handle.offset_1;
        int offset_2   = trigger->op_handle.offset_2;

        char** peers = (char**)ws->peer_addr[1];
        char* peer_buf = peers[peer_rank];
        char* self_buf = (char*)ws->self_addr[1];
        int elem_size = dtypeElemSize(dtype);

        int actual_off0 = offset_0 + chunk_offset;
        int actual_off1 = offset_1 + chunk_offset;
        int actual_off2 = offset_2 + chunk_offset;

        void* a = (void*)(peer_buf + (long)actual_off0 * elem_size);
        void* b = (void*)(self_buf + (long)actual_off1 * elem_size);
        void* c = (void*)(self_buf + (long)actual_off2 * elem_size);

        tma_vector_add(dtype, a, b, c, chunk_count);
    }
}

__device__ void executeNotify(DeviceWorkspace* ws, const op* trigger) {
    int target_rank = trigger->signal_handle.peer_rank;
    int signal_id   = trigger->signal_handle.offset;

    int** peer_sigs = (int**)ws->peer_signals[1];
    volatile int* target_signals = (volatile int*)peer_sigs[target_rank];

    atomicAdd((int*)&target_signals[signal_id], 1);
    __threadfence_system();
}

__device__ void executeWaitNotify(DeviceWorkspace* ws, const op* trigger) {
    int signal_id = trigger->signal_handle.offset;

    volatile int* self_sigs = (volatile int*)ws->self_signals[1];

    int initial_val = self_sigs[signal_id];
    if (initial_val > 0) {
        atomicAdd((int*)&self_sigs[signal_id], -1);
        __threadfence_system();
        return;
    }

    while (self_sigs[signal_id] == 0) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }

    atomicAdd((int*)&self_sigs[signal_id], -1);
    __threadfence_system();
}

// Map primitive_type to DataType
__device__ DataType primitiveTypeToDataType(char ptype) {
    switch (ptype) {
        case cuda_reduce_f32:  case cuda_copy_f32:  case cuda_multimem_reduce_f32:
        case cuda_ce_copy_f32: case cuda_tma_copy_f32: case cuda_tma_reduce_f32:
            return DataType::F32;
        case cuda_reduce_f16:  case cuda_copy_f16:  case cuda_multimem_reduce_f16:
        case cuda_ce_copy_f16: case cuda_tma_copy_f16: case cuda_tma_reduce_f16:
            return DataType::F16;
        case cuda_reduce_bf16: case cuda_copy_bf16: case cuda_multimem_reduce_bf16:
        case cuda_ce_copy_bf16: case cuda_tma_copy_bf16: case cuda_tma_reduce_bf16:
            return DataType::BF16;
        case cuda_reduce_f8_e4m3: case cuda_copy_f8_e4m3: case cuda_multimem_reduce_f8_e4m3:
        case cuda_ce_copy_f8_e4m3: case cuda_tma_copy_f8_e4m3: case cuda_tma_reduce_f8_e4m3:
            return DataType::FP8_E4M3;
        case cuda_reduce_f8_e5m2: case cuda_copy_f8_e5m2: case cuda_multimem_reduce_f8_e5m2:
        case cuda_ce_copy_f8_e5m2: case cuda_tma_copy_f8_e5m2: case cuda_tma_reduce_f8_e5m2:
            return DataType::FP8_E5M2;
        default: return DataType::F32;
    }
}

__device__ bool isReduceType(char ptype) {
    return (ptype >= cuda_reduce_f32 && ptype <= cuda_reduce_f8_e5m2) ||
           (ptype >= cuda_multimem_reduce_f32 && ptype <= cuda_multimem_reduce_f8_e5m2) ||
           (ptype >= cuda_tma_reduce_f32 && ptype <= cuda_tma_reduce_f8_e5m2);
}

__device__ bool isCopyType(char ptype) {
    return (ptype >= cuda_copy_f32 && ptype <= cuda_copy_f8_e5m2) ||
           (ptype >= cuda_tma_copy_f32 && ptype <= cuda_tma_copy_f8_e5m2);
}

__device__ bool isCEType(char ptype) {
    return (ptype >= cuda_ce_copy_f32 && ptype <= cuda_ce_copy_f8_e5m2);
}

__device__ bool isTMAType(char ptype) {
    return (ptype >= cuda_tma_copy_f32 && ptype <= cuda_tma_copy_f8_e5m2) ||
           (ptype >= cuda_tma_reduce_f32 && ptype <= cuda_tma_reduce_f8_e5m2);
}

__device__ bool isMultimemReduceType(char ptype) {
    return ptype >= cuda_multimem_reduce_f32 && ptype <= cuda_multimem_reduce_f8_e5m2;
}

__device__ bool isMultimemStoreType(char ptype) {
    return ptype >= cuda_multimem_store_f32 && ptype <= cuda_multimem_store_f8_e5m2;
}

// Dependency propagation — enqueues to a specific channel's ready queue
__device__ void updateDependencies(DeviceWorkspace* ws, int op_idx, RingBuffer* ready_queue) {
    PrimitiveMeta* meta = &ws->graph_buffer_.meta[1][op_idx];
    WorkingMeta* working = ws->graph_buffer_.working_meta[1];
    ProxyTrigger* primitives = (ProxyTrigger*)ws->graph_buffer_.primitives[1];

    for (int i = 0; i < meta->num_next_ops_; i++) {
        int next_idx = meta->next_primitive_index[i];
        int old = atomicAdd(&working[next_idx].remaining_deps, -1);
        if (old == 1) {
            enqueueToRingBuffer(ready_queue, &primitives[next_idx]);
        }
    }
}

// Persistent kernel with multi-block chunked execution
//
// IMPORTANT: All control-flow decisions (return / continue) that gate
// __syncthreads() barriers MUST be uniform across every thread in the block.
// Global-memory reads (completed_primitives, active_valid, …) are therefore
// performed by thread 0 only, written to __shared__ memory, and broadcast
// via __syncthreads() so that every thread in the block takes the same path.
// Letting each thread do its own atomicAdd-read would race with writes from
// other blocks, causing intra-block divergence on __syncthreads() → deadlock
// (observable under -G where atomics are slow enough to widen the window).

__device__ constexpr int MAX_CHUNK_ELEMS = MAX_CHUNK_SIZE_BYTES / sizeof(float);
__device__ constexpr int MIN_CHUNK_ELEMS = pccl::config::MIN_CHUNK_ELEMS;

__global__ void cuda_executor_kernel(DeviceWorkspace* ws,
                                     int num_primitives, int queue_capacity) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // =========== Setup prologue (block 0 only, replaces separate setup_kernel) ===========

    if (bid == 0) {
        // Parallel within block: reset working_meta
        for (int i = tid; i < num_primitives; i += blockDim.x) {
            ws->graph_buffer_.working_meta[1][i].remaining_deps =
                ws->graph_buffer_.meta[1][i].num_dependencies_;
        }

        // Serial init on thread 0
        if (tid == 0) {
            ws->ring_buffers_[0].meta_b_->head = 0;
            ws->ring_buffers_[0].meta_b_->tail = 0;
            ws->ring_buffers_[1].meta_a_->head = 0;
            ws->ring_buffers_[1].meta_a_->tail = 0;
            ws->ring_buffers_[3].meta_a_->head = 0;
            ws->ring_buffers_[3].meta_a_->tail = 0;
            ws->ring_buffers_[3].meta_b_->head = 0;
            ws->ring_buffers_[3].meta_b_->tail = 0;
            ws->ring_buffers_[3].meta_b_->capacity = queue_capacity;

            int num_ch = ws->num_channels;
            for (int c = 0; c < num_ch; c++) {
                ws->channels[c].active_valid = 0;
                ws->channels[c].completed_primitives = 0;
                ws->channels[c].total_primitives = 0;
                ws->channels[c].chunk_state.current_chunk = 0;
                ws->channels[c].chunk_state.completed_chunks = 0;
                ws->channels[c].chunk_state.total_chunks = 0;
                ws->channels[c].chunk_state.total_elems = 0;
                ws->channels[c].chunk_state.chunk_elems = 0;

                ws->channel_ready_queues[c].meta_b_->head = 0;
                ws->channel_ready_queues[c].meta_b_->tail = 0;
            }

            for (int i = 0; i < num_primitives; i++) {
                PrimitiveMeta* meta = &ws->graph_buffer_.meta[1][i];
                if (meta->device_type == 1) {
                    ws->channels[meta->channel].total_primitives++;
                }
            }
        }

        __syncthreads();

        // Parallel within block: enqueue zero-dependency CUDA primitives
        for (int i = tid; i < num_primitives; i += blockDim.x) {
            PrimitiveMeta* meta = &ws->graph_buffer_.meta[1][i];
            if (meta->num_dependencies_ == 0 && meta->device_type == 1) {
                ProxyTrigger* primitives = (ProxyTrigger*)ws->graph_buffer_.primitives[1];
                RingBuffer* q = &ws->channel_ready_queues[meta->channel];
                int pos = atomicAdd(&q->meta_b_->tail, 1);
                int cap = q->meta_b_->capacity;
                q->buffer_b_[pos % cap] = primitives[i];
            }
        }

        __syncthreads();
        __threadfence();

        // Signal other blocks that setup is complete
        if (tid == 0) {
            atomicExch(&ws->grid_barrier_count_, 1);
        }
    }

    // =========== Main dispatch loop ===========

    // Non-block-0 blocks: wait for setup completion before entering dispatch
    if (bid != 0) {
        if (tid == 0) {
            while (*(volatile int*)&ws->grid_barrier_count_ == 0) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(32);
#endif
            }
        }
        __syncthreads();
    }

    // Block-partition across channels
    int num_ch = ws->num_channels;
    int blocks_per_ch = gridDim.x / num_ch;
    if (blocks_per_ch < 1) blocks_per_ch = 1;
    int my_channel = bid / blocks_per_ch;
    if (my_channel >= num_ch) my_channel = num_ch - 1;
    int local_bid = bid - my_channel * blocks_per_ch;

    ChannelState* ch = &ws->channels[my_channel];
    RingBuffer* my_queue = &ws->channel_ready_queues[my_channel];

    __shared__ int sh_chunk_offset;
    __shared__ int sh_chunk_count;
    __shared__ bool sh_is_last;
    __shared__ bool sh_got_work;
    __shared__ bool sh_should_exit;
    __shared__ int sh_active_valid;

    while (true) {
        if (tid == 0) {
            int completed = *(volatile int*)&ch->completed_primitives;
            sh_should_exit = (completed >= ch->total_primitives);
            sh_got_work = false;
            sh_is_last = false;
        }
        __syncthreads();
        if (sh_should_exit) break;

        if (local_bid == 0 && tid == 0) {
            if (*(volatile int*)&ch->active_valid == 0) {
                ProxyTrigger t;
                if (dequeueFromRingBuffer(my_queue, &t)) {
                    ch->active_trigger = t;
                    op trigger;
                    trigger.raw = t;
                    char ptype = trigger.op_handle.primitive_type;
                    ch->active_op_idx = trigger.op_handle.op_index;

                    if (ptype == cuda_notify || ptype == cuda_wait_notify ||
                        ptype == cuda_rdma_write || ptype == cuda_rdma_read ||
                        isCEType(ptype)) {
                        ch->chunk_state.current_chunk = 0;
                        ch->chunk_state.completed_chunks = 0;
                        ch->chunk_state.total_chunks = 0;
                        ch->chunk_state.total_elems = 0;
                        ch->chunk_state.chunk_elems = 0;
                        __threadfence();
                        atomicExch(&ch->active_valid, 1);
                    } else {
                        int total_elems = trigger.op_handle.size;

                        if (isTMAType(ptype)) {
                            // TMA: one partition per block, no chunk-grab loop
                            ch->chunk_state.current_chunk = 0;
                            ch->chunk_state.completed_chunks = 0;
                            ch->chunk_state.total_chunks = blocks_per_ch;
                            ch->chunk_state.total_elems = total_elems;
                            ch->chunk_state.chunk_elems = 0;
                        } else {
                            int min_chunks = blocks_per_ch * 2;
                            int chunk_elems = total_elems / min_chunks;
                            chunk_elems = chunk_elems & ~15;
                            if (chunk_elems < MIN_CHUNK_ELEMS) chunk_elems = MIN_CHUNK_ELEMS;
                            if (chunk_elems > MAX_CHUNK_ELEMS) chunk_elems = MAX_CHUNK_ELEMS;

                            int total_chunks = (total_elems + chunk_elems - 1) / chunk_elems;
                            if (total_chunks < 1) total_chunks = 1;

                            ch->chunk_state.current_chunk = 0;
                            ch->chunk_state.completed_chunks = 0;
                            ch->chunk_state.total_chunks = total_chunks;
                            ch->chunk_state.total_elems = total_elems;
                            ch->chunk_state.chunk_elems = chunk_elems;
                        }
                        __threadfence();
                        atomicExch(&ch->active_valid, 1);
                    }
                }
            }
        }

        if (tid == 0) {
            sh_active_valid = *(volatile int*)&ch->active_valid;
        }
        __syncthreads();

        if (sh_active_valid == 0) {
#if __CUDA_ARCH__ >= 700
            if (tid == 0) __nanosleep(64);
#endif
            continue;
        }

        op trigger;
        trigger.raw = ch->active_trigger;
        char ptype = trigger.op_handle.primitive_type;
        int total_elems = ch->chunk_state.total_elems;

        if (ptype == cuda_notify || ptype == cuda_wait_notify) {
            if (local_bid == 0 && tid == 0) {
                if (ptype == cuda_notify) {
                    executeNotify(ws, &trigger);
                } else {
                    executeWaitNotify(ws, &trigger);
                }
                int op_idx = ch->active_op_idx;
                updateDependencies(ws, op_idx, my_queue);
                atomicAdd(&ch->completed_primitives, 1);
                atomicExch(&ch->active_valid, 0);
            }
            __syncthreads();
            continue;
        }

        if (ptype == cuda_noop) {
            if (local_bid == 0 && tid == 0) {
                int op_idx = ch->active_op_idx;
                updateDependencies(ws, op_idx, my_queue);
                atomicAdd(&ch->completed_primitives, 1);
                atomicExch(&ch->active_valid, 0);
            }
            __syncthreads();
            continue;
        }

        if (ptype == cuda_rdma_write || ptype == cuda_rdma_read) {
            if (local_bid == 0 && tid == 0) {
                HostProxyState* proxy = ws->host_proxy;
                if (proxy) {
                    int slot = atomicAdd((int*)&proxy->cmd_tail, 1) % PROXY_QUEUE_CAPACITY;
                    proxy->cmd_queue[slot].type = (ptype == cuda_rdma_write)
                        ? RdmaCommand::WRITE : RdmaCommand::READ;
                    proxy->cmd_queue[slot].peer_rank = trigger.op_handle.peer_rank;
                    proxy->cmd_queue[slot].local_offset = (int64_t)trigger.op_handle.offset_0;
                    proxy->cmd_queue[slot].remote_offset = (int64_t)trigger.op_handle.offset_1;
                    proxy->cmd_queue[slot].size = (int64_t)trigger.op_handle.size;
                    proxy->cmd_queue[slot].completion_slot = slot;
                    __threadfence_system();
                    while (atomicAdd((int*)&proxy->completion_flags[slot], 0) == 0) {}
                    atomicExch((int*)&proxy->completion_flags[slot], 0);
                    __threadfence_system();
                }
                int op_idx = ch->active_op_idx;
                updateDependencies(ws, op_idx, my_queue);
                atomicAdd(&ch->completed_primitives, 1);
                atomicExch(&ch->active_valid, 0);
            }
            __syncthreads();
            continue;
        }

        if (isCEType(ptype)) {
            if (local_bid == 0 && tid == 0) {
                CeProxyState* proxy = ws->ce_proxy;
                if (proxy) {
                    DataType dtype = primitiveTypeToDataType(ptype);
                    int elem_size = dtypeElemSize(dtype);
                    char** peers = (char**)ws->peer_addr[1];
                    void* src = (void*)(peers[trigger.op_handle.peer_rank]
                                + (int64_t)trigger.op_handle.offset_0 * elem_size);
                    void* dst = (void*)((char*)ws->self_addr[1]
                                + (int64_t)trigger.op_handle.offset_1 * elem_size);
                    int64_t bytes = (int64_t)trigger.op_handle.size * elem_size;

                    int slot = atomicAdd((int*)&proxy->cmd_tail, 1) % CE_PROXY_QUEUE_CAPACITY;
                    proxy->cmd_queue[slot].type = CeCommand::COPY;
                    proxy->cmd_queue[slot].src = src;
                    proxy->cmd_queue[slot].dst = dst;
                    proxy->cmd_queue[slot].size_bytes = bytes;
                    proxy->cmd_queue[slot].completion_slot = slot;
                    __threadfence_system();
                    while (atomicAdd((int*)&proxy->completion_flags[slot], 0) == 0) {}
                    atomicExch((int*)&proxy->completion_flags[slot], 0);
                    __threadfence_system();
                }
                int op_idx = ch->active_op_idx;
                updateDependencies(ws, op_idx, my_queue);
                atomicAdd(&ch->completed_primitives, 1);
                atomicExch(&ch->active_valid, 0);
            }
            __syncthreads();
            continue;
        }

        if (isTMAType(ptype)) {
            // TMA path: static per-block tile partitioning
            // Each block claims one slot via atomicAdd.
            // Re-entry (after first claim) gets slot >= blocks_per_ch → skip.
            if (tid == 0) {
                int my_slot = atomicAdd(&ch->chunk_state.current_chunk, 1);
                if (my_slot < blocks_per_ch) {
                    sh_got_work = true;
                    int my_start = ((long)my_slot * total_elems) / blocks_per_ch;
                    int my_end   = ((long)(my_slot + 1) * total_elems) / blocks_per_ch;
                    my_start = my_start & ~127;
                    if (my_slot + 1 < blocks_per_ch) my_end = my_end & ~127;
                    sh_chunk_offset = my_start;
                    sh_chunk_count = my_end - my_start;
                }
            }
            __syncthreads();

            if (sh_got_work) {
                if (sh_chunk_count > 0) {
                    DataType dtype = primitiveTypeToDataType(ptype);
                    if (isReduceType(ptype)) executeTmaReduce(ws, &trigger, dtype, sh_chunk_offset, sh_chunk_count);
                    else executeTmaCopy(ws, &trigger, dtype, sh_chunk_offset, sh_chunk_count);
                }
                __threadfence_system();

                if (tid == 0) {
                    int done = atomicAdd(&ch->chunk_state.completed_chunks, 1) + 1;
                    if (done == blocks_per_ch) sh_is_last = true;
                }
            }
        } else if (isMultimemReduceType(ptype) || isMultimemStoreType(ptype)) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            if (tid == 0) {
                int chunk_elems = ch->chunk_state.chunk_elems;
                int my_chunk = atomicAdd(&ch->chunk_state.current_chunk, 1);
                int total_chunks = ch->chunk_state.total_chunks;
                if (my_chunk < total_chunks) {
                    sh_got_work = true;
                    sh_chunk_offset = my_chunk * chunk_elems;
                    int remaining = total_elems - sh_chunk_offset;
                    sh_chunk_count = (remaining > chunk_elems) ? chunk_elems : remaining;
                }
            }
            __syncthreads();

            if (sh_got_work) {
                char* mc   = (char*)ws->nvls_mc_va_;
                char* phys = (char*)ws->nvls_phys_va_;
                int elem_size = dtypeElemSize(primitiveTypeToDataType(ptype));
                int off0 = trigger.op_handle.offset_0 + sh_chunk_offset;
                int off1 = trigger.op_handle.offset_1 + sh_chunk_offset;

                constexpr int VEC = 8;
                if (isMultimemReduceType(ptype)) {
                    for (int i = tid * VEC; i < sh_chunk_count; i += blockDim.x * VEC) {
                        int elem_idx = off0 + i;
                        uint32_t d0, d1, d2, d3;
                        multimem_ld_reduce_add_v4_bf16(&d0, &d1, &d2, &d3,
                            (void*)(mc + (long)elem_idx * elem_size));
                        uint32_t* dst = (uint32_t*)(phys + (long)(off1 + i) * elem_size);
                        dst[0] = d0; dst[1] = d1; dst[2] = d2; dst[3] = d3;
                    }
                } else {
                    for (int i = tid * VEC; i < sh_chunk_count; i += blockDim.x * VEC) {
                        int elem_idx = off0 + i;
                        uint32_t* src = (uint32_t*)(phys + (long)elem_idx * elem_size);
                        multimem_st_v4_bf16(
                            (void*)(mc + (long)(off1 + i) * elem_size),
                            src[0], src[1], src[2], src[3]);
                    }
                }
                __threadfence_system();

                if (tid == 0) {
                    int done = atomicAdd(&ch->chunk_state.completed_chunks, 1) + 1;
                    if (done == ch->chunk_state.total_chunks) sh_is_last = true;
                }
            }
#endif
        } else {
            // SM path: existing chunk-grab loop
            if (tid == 0) {
                int chunk_elems = ch->chunk_state.chunk_elems;
                int my_chunk = atomicAdd(&ch->chunk_state.current_chunk, 1);
                int total_chunks = ch->chunk_state.total_chunks;
                if (my_chunk < total_chunks) {
                    sh_got_work = true;
                    sh_chunk_offset = my_chunk * chunk_elems;
                    int remaining = total_elems - sh_chunk_offset;
                    sh_chunk_count = (remaining > chunk_elems) ? chunk_elems : remaining;
                }
            }
            __syncthreads();

            if (sh_got_work) {
                DataType dtype = primitiveTypeToDataType(ptype);
                if (isReduceType(ptype)) {
                    executeReduce(ws, &trigger, dtype, sh_chunk_offset, sh_chunk_count);
                } else if (isCopyType(ptype)) {
                    executeCopy(ws, &trigger, dtype, sh_chunk_offset, sh_chunk_count);
                }

                __threadfence_system();

                if (tid == 0) {
                    int done = atomicAdd(&ch->chunk_state.completed_chunks, 1) + 1;
                    if (done == ch->chunk_state.total_chunks) sh_is_last = true;
                }
            }
        }
        __syncthreads();

        if (sh_is_last && tid == 0) {
            int op_idx = ch->active_op_idx;
            updateDependencies(ws, op_idx, my_queue);
            atomicAdd(&ch->completed_primitives, 1);
            atomicExch(&ch->active_valid, 0);
        }

        __syncthreads();
    }
}

// Host launch wrappers
void launch_cuda_kernel(int num_blocks, cudaStream_t stream,
                        DeviceWorkspace* workspace, int dynamic_smem_bytes,
                        int num_primitives, int queue_capacity) {
    int threads_per_block = 256;
    if (num_blocks > 64) num_blocks = 64;
    if (num_blocks < 4) num_blocks = 4;
    if (dynamic_smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(cuda_executor_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_bytes);
    }
    cuda_executor_kernel<<<num_blocks, threads_per_block, dynamic_smem_bytes, stream>>>(
        workspace, num_primitives, queue_capacity);
}

// =========================================================================
// Fused step executor — replaces ring buffer + DAG dispatch for linear chains
// =========================================================================

__device__ void fusedExecuteNotify(DeviceWorkspace* ws, int target_rank, int signal_id) {
    int** peer_sigs = (int**)ws->peer_signals[1];
    volatile int* target_signals = (volatile int*)peer_sigs[target_rank];
    atomicAdd((int*)&target_signals[signal_id], 1);
    __threadfence_system();
}

__device__ void fusedExecuteWait(DeviceWorkspace* ws, int signal_id) {
    volatile int* self_sigs = (volatile int*)ws->self_signals[1];
    int initial_val = self_sigs[signal_id];
    if (initial_val > 0) {
        atomicAdd((int*)&self_sigs[signal_id], -1);
        __threadfence_system();
        return;
    }
    while (self_sigs[signal_id] == 0) {
        #if __CUDA_ARCH__ >= 700
        __nanosleep(64);
        #endif
    }
    atomicAdd((int*)&self_sigs[signal_id], -1);
    __threadfence_system();
}

__global__ void fused_allreduce_kernel(DeviceWorkspace* ws,
                                        FusedStepDescriptor* desc) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    int num_ch = desc->num_channels;
    int blocks_per_ch = gridDim.x / num_ch;
    if (blocks_per_ch < 1) blocks_per_ch = 1;
    int my_channel = bid / blocks_per_ch;
    if (my_channel >= num_ch) my_channel = num_ch - 1;
    int local_bid = bid - my_channel * blocks_per_ch;

    int ch_start = desc->channel_offsets[my_channel];
    int ch_end = desc->channel_offsets[my_channel + 1];

    __shared__ FusedStep sh_step;

    int wait_count = 0;

    for (int si = ch_start; si < ch_end; si++) {
        if (tid == 0) sh_step = desc->steps[si];
        __syncthreads();

        if (sh_step.has_notify && local_bid == 0 && tid == 0)
            fusedExecuteNotify(ws, sh_step.notify_peer_rank, sh_step.notify_signal_id);

        if (sh_step.has_wait) {
            wait_count++;
            if (local_bid == 0 && tid == 0) {
                fusedExecuteWait(ws, sh_step.wait_signal_id);
                __threadfence();
                atomicAdd(&ws->fused_ch_barrier_[my_channel], 1);
            }
            if (local_bid != 0 && tid == 0) {
                int spin2 = 0;
                while (*(volatile int*)&ws->fused_ch_barrier_[my_channel] < wait_count) {
                    if (++spin2 > 4000000) break;
#if __CUDA_ARCH__ >= 700
                    __nanosleep(32);
#endif
                }
            }
            __syncthreads();
        }

        char ptype = sh_step.primitive_type;
        if (ptype == 0) continue;

        int total_elems = sh_step.size;
        DataType dtype = primitiveTypeToDataType(ptype);

        // Build trigger once per step (hoisted out of any inner loop)
        op trigger;
        trigger.op_handle.primitive_type = ptype;
        trigger.op_handle.peer_rank = sh_step.peer_rank;
        trigger.op_handle.offset_0 = sh_step.offset_0;
        trigger.op_handle.offset_1 = sh_step.offset_1;
        trigger.op_handle.offset_2 = sh_step.offset_2;
        trigger.op_handle.size = total_elems;

        if (isTMAType(ptype)) {
            // Work partitioning is deterministic from local_bid — no shared memory needed.
            if (local_bid < blocks_per_ch) {
                int my_start = ((long)local_bid * total_elems) / blocks_per_ch;
                int my_end   = ((long)(local_bid + 1) * total_elems) / blocks_per_ch;
                my_start = my_start & ~127;
                if (local_bid + 1 < blocks_per_ch) my_end = my_end & ~127;
                int chunk_count = my_end - my_start;
                if (chunk_count > 0) {
                    if (isReduceType(ptype))
                        executeTmaReduce(ws, &trigger, dtype, my_start, chunk_count);
                    else
                        executeTmaCopy(ws, &trigger, dtype, my_start, chunk_count);
                }
            }
        } else {
            int chunk_elems = MAX_CHUNK_ELEMS;
            if (chunk_elems > total_elems) chunk_elems = total_elems;
            int total_chunks = (total_elems + chunk_elems - 1) / chunk_elems;

            for (int ci = local_bid; ci < total_chunks; ci += blocks_per_ch) {
                int chunk_offset = ci * chunk_elems;
                int remaining = total_elems - chunk_offset;
                int count = (remaining > chunk_elems) ? chunk_elems : remaining;
                if (isReduceType(ptype))
                    executeReduce(ws, &trigger, dtype, chunk_offset, count);
                else if (isCopyType(ptype))
                    executeCopy(ws, &trigger, dtype, chunk_offset, count);
            }
        }

        __threadfence_system();
        __syncthreads();
    }
}

void launch_fused_kernel(int num_blocks, cudaStream_t stream,
                         DeviceWorkspace* workspace,
                         FusedStepDescriptor* device_desc,
                         int dynamic_smem_bytes) {
    int threads_per_block = 256;
    if (num_blocks > 64) num_blocks = 64;
    if (num_blocks < 4) num_blocks = 4;

    if (dynamic_smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(fused_allreduce_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_bytes);
    }
    fused_allreduce_kernel<<<num_blocks, threads_per_block, dynamic_smem_bytes, stream>>>(
        workspace, device_desc);
}

// =========================================================================
// NVLS allreduce kernel — 2-phase reduce-scatter + allgather via NVSwitch
// multicast. O(1) communication steps regardless of GPU count.
// =========================================================================

// Grid-wide + inter-GPU barrier using MC ld_reduce.
// Uses epoch-based counting to avoid stale-value issues across iterations.
// Phase 1: intra-GPU grid barrier — all blocks arrive, last block writes epoch
// Phase 2: inter-GPU barrier — all blocks poll MC for convergence
//
// The last block signals epoch readiness by setting the counter to nblocks+1
// AFTER __threadfence_system(), so non-last blocks only proceed once the epoch
// write is globally visible.
__device__ void nvls_barrier(DeviceWorkspace* ws, int barrier_id,
                              char* phys, char* mc, size_t barrier_offset,
                              int world_size) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __threadfence();

    int nblocks = gridDim.x;

    if (threadIdx.x == 0) {
        int arrived = atomicAdd(&ws->nvls_grid_counters_[barrier_id], 1) + 1;
        if (arrived == nblocks) {
            volatile uint32_t* my_slot = (volatile uint32_t*)(phys + barrier_offset + barrier_id * 4);
            *my_slot = *my_slot + 1;
            __threadfence_system();
            atomicExch(&ws->nvls_grid_counters_[barrier_id], nblocks + 1);
        }
    }

    if (threadIdx.x == 0) {
        while (*(volatile int*)&ws->nvls_grid_counters_[barrier_id] < nblocks + 1) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(16);
#endif
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        volatile uint32_t* my_slot = (volatile uint32_t*)(phys + barrier_offset + barrier_id * 4);
        uint32_t target = (*my_slot) * (uint32_t)world_size;
        volatile uint32_t* mc_slot = (volatile uint32_t*)(mc + barrier_offset + barrier_id * 4);
        while (multimem_ld_reduce_add_u32((void*)mc_slot) < target) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(32);
#endif
        }
    }
    __syncthreads();
#endif
}

__global__ void nvls_allreduce_kernel(DeviceWorkspace* ws) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nblocks = gridDim.x;
    const int nthreads = blockDim.x;

    char* mc   = (char*)ws->nvls_mc_va_;
    char* phys = (char*)ws->nvls_phys_va_;
    int world_size     = ws->nvls_world_size_;
    int self_rank      = ws->nvls_self_rank_;
    int total_elems    = ws->nvls_total_elems_;
    size_t barrier_off = ws->nvls_barrier_offset_;

    // 8 bf16 elements = 16 bytes per vector op
    constexpr int VEC = 8;

    // Chunk partitioning: each GPU owns one chunk
    int chunk_size = total_elems / world_size;
    int my_start = self_rank * chunk_size;
    int my_end   = (self_rank == world_size - 1) ? total_elems : (my_start + chunk_size);
    // Align to VEC boundary
    my_start = my_start & ~(VEC - 1);
    my_end   = my_end & ~(VEC - 1);
    int my_elems = my_end - my_start;

    // Block partitioning within this GPU's chunk
    int bstart = my_start + ((long)bid * my_elems) / nblocks;
    int bend   = my_start + ((long)(bid + 1) * my_elems) / nblocks;
    bstart = bstart & ~(VEC - 1);
    if (bid + 1 < nblocks) bend = bend & ~(VEC - 1);

    // ===== Barrier 0: ensure all GPUs' input data is in MC physical memory =====
    nvls_barrier(ws, 0, phys, mc, barrier_off, world_size);

    // ===== Phase 1: Reduce-scatter via multimem.ld_reduce =====
    // Read from mc_va (NVSwitch sums all GPUs), write to local phys_va
    for (int i = bstart + tid * VEC; i < bend; i += nthreads * VEC) {
        uint32_t d0, d1, d2, d3;
        multimem_ld_reduce_add_v4_bf16(&d0, &d1, &d2, &d3, (void*)(mc + (long)i * 2));
        uint32_t* dst = (uint32_t*)(phys + (long)i * 2);
        dst[0] = d0;
        dst[1] = d1;
        dst[2] = d2;
        dst[3] = d3;
    }

    // ===== Barrier 1: all GPUs done with reduce-scatter =====
    nvls_barrier(ws, 1, phys, mc, barrier_off, world_size);

    // ===== Phase 2: Allgather via multimem.st =====
    // Read local reduced chunk, write to mc_va (NVSwitch broadcasts to all GPUs)
    for (int i = bstart + tid * VEC; i < bend; i += nthreads * VEC) {
        uint32_t* src = (uint32_t*)(phys + (long)i * 2);
        multimem_st_v4_bf16((void*)(mc + (long)i * 2), src[0], src[1], src[2], src[3]);
    }

    // ===== Barrier 2: all GPUs done with allgather =====
    nvls_barrier(ws, 2, phys, mc, barrier_off, world_size);
#endif
}

void launch_nvls_kernel(int num_blocks, cudaStream_t stream,
                         DeviceWorkspace* workspace) {
    int threads_per_block = 512;
    if (num_blocks > 132) num_blocks = 132;
    if (num_blocks < 4) num_blocks = 4;
    nvls_allreduce_kernel<<<num_blocks, threads_per_block, 0, stream>>>(workspace);
}

}  // namespace engine_c::cuda
