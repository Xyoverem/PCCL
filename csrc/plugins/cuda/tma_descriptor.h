#pragma once

#include <cuda.h>
#include <stdexcept>
#include <cstdint>

// Host-side CUtensorMap utilities for TMA bulk copy.
// CUtensorMap first appeared in CUDA 12. Keep the SM fallback buildable with
// CUDA 11.x, where TMA primitives are transparently executed by the SM path.
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
#define PCCL_HAS_TMA_HOST 1
#else
#define PCCL_HAS_TMA_HOST 0
#endif

namespace engine_c::cuda {

static constexpr int TMA_TILE_INNER = 128;
static constexpr int TMA_MAX_PEERS = 8;
static constexpr int TMA_SMEM_TOTAL = 128 * 1024;
static constexpr int TMA_SMEM_OVERHEAD = 256;  // alignment (128) + mbarriers (16) + padding
static constexpr int TMA_SMEM_PER_BUF = (TMA_SMEM_TOTAL - TMA_SMEM_OVERHEAD) / 2;

#if PCCL_HAS_TMA_HOST

// Compute adaptive tile_outer based on shared memory budget per double-buffer slot.
// Returns clamped value in [1, 256] (CUtensorMap boxDim limit).
inline int tma_tile_outer_for_elem_size(int elem_size, int smem_per_buf = TMA_SMEM_PER_BUF) {
    int outer = smem_per_buf / (TMA_TILE_INNER * elem_size);
    if (outer > 256) outer = 256;
    if (outer < 1) outer = 1;
    return outer;
}

inline CUtensorMapDataType tma_dtype_from_elem_size(int elem_size) {
    switch (elem_size) {
        case 4: return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case 2: return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        case 1: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        default:
            throw std::runtime_error("Unsupported element size for TMA descriptor");
    }
}

// Creates a 2D CUtensorMap for a flat buffer. Models the buffer as:
//   dim-0 (inner) = TMA_TILE_INNER elements
//   dim-1 (outer) = buffer_size_bytes / (TMA_TILE_INNER * elem_size)
// Each TMA instruction moves TMA_TILE_INNER * tile_outer * elem_size bytes.
inline CUtensorMap create_buffer_tma_desc(void* buffer_ptr,
                                           long buffer_size_bytes,
                                           int elem_size,
                                           int tile_outer = 0) {
    if (tile_outer <= 0) tile_outer = tma_tile_outer_for_elem_size(elem_size);
    long total_elems = buffer_size_bytes / elem_size;
    long outer_dim = total_elems / TMA_TILE_INNER;
    if (outer_dim < 1) outer_dim = 1;

    CUtensorMap tensor_map{};

    const cuuint64_t globalDim[2] = {
        static_cast<cuuint64_t>(TMA_TILE_INNER),
        static_cast<cuuint64_t>(outer_dim)
    };

    // globalStride[0] is dim-1 stride in bytes (dim-0 stride is implicit)
    const cuuint64_t globalStride[1] = {
        static_cast<cuuint64_t>(TMA_TILE_INNER * elem_size)
    };

    const cuuint32_t boxDim[2] = {
        static_cast<cuuint32_t>(TMA_TILE_INNER),
        static_cast<cuuint32_t>(tile_outer)
    };

    const cuuint32_t elementStride[2] = {1, 1};

    CUresult st = cuTensorMapEncodeTiled(
        &tensor_map,
        tma_dtype_from_elem_size(elem_size),
        2,
        buffer_ptr,
        globalDim,
        globalStride,
        boxDim,
        elementStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (st != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(st, &err_str);
        throw std::runtime_error(
            std::string("cuTensorMapEncodeTiled failed: ") +
            (err_str ? err_str : "unknown error"));
    }

    return tensor_map;
}

#endif  // PCCL_HAS_TMA_HOST

}  // namespace engine_c::cuda

#include <engine/workspace.h>

namespace engine_c::cuda {

#if PCCL_HAS_TMA_HOST
struct TmaDescriptors {
    CUtensorMap self_desc;
    CUtensorMap output_desc;
    CUtensorMap peer_desc[TMA_MAX_PEERS];
    int elem_size;
    int tile_inner;
    int tile_outer;
    int num_peers;
    bool valid;
    bool output_valid;
};
#else
struct TmaDescriptors {};
#endif

}  // namespace engine_c::cuda
