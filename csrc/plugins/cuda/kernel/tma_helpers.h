#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// TMA inline PTX helpers for Hopper SM90+ (see TMA_GUIDE.md for details)

namespace engine_c::cuda::tma {

static constexpr int TMA_TILE_INNER = 128;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

__device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrier_init_local(uint64_t* bar,
                                                     uint32_t arrive_count) {
    uint32_t bar_smem = cast_smem_ptr_to_uint(bar);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(bar_smem), "r"(arrive_count)
        : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar,
                                                      uint32_t phase) {
    uint32_t bar_smem = cast_smem_ptr_to_uint(bar);
    uint32_t done;
    do {
        asm volatile(
            "{\n"
            ".reg .pred P1;\n"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%1], %2;\n"
            "selp.b32 %0, 1, 0, P1;\n"
            "}\n"
            : "=r"(done)
            : "r"(bar_smem), "r"(phase)
            : "memory");
    } while (!done);
}

// Required between shared stores and subsequent TMA store/reduce
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void fence_tensormap_release_cta() {
    asm volatile("fence.proxy.tensormap::generic.release.cta;" ::: "memory");
}

__device__ __forceinline__ void fence_tensormap_acquire_cta(
        const CUtensorMap* tmap_ptr) {
    uint64_t gmem_desc = reinterpret_cast<uint64_t>(tmap_ptr);
    asm volatile(
        "fence.proxy.tensormap::generic.acquire.cta [%0], 128;"
        :: "l"(gmem_desc)
        : "memory");
}

__device__ __forceinline__ void tensormap_replace_global_address_in_smem(
        CUtensorMap* smem_tmap,
        const void* new_addr) {
    uint32_t tmap_smem = cast_smem_ptr_to_uint(smem_tmap);
    uint64_t new_addr_u64 = reinterpret_cast<uint64_t>(new_addr);
    asm volatile(
        "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
        :: "r"(tmap_smem), "l"(new_addr_u64)
        : "memory");
}

// Re-arm an mbarrier for the next TMA load (sets expected tx bytes)
__device__ __forceinline__ void mbarrier_arrive_expect_tx(
        uint64_t* bar, uint32_t tx_count) {
    uint32_t bar_smem = cast_smem_ptr_to_uint(bar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(bar_smem), "r"(tx_count)
        : "memory");
}

// TMA Load 2D: global -> shared, tracked by mbarrier
__device__ __forceinline__ void tma_load_2d(
        const CUtensorMap* tmap,
        uint64_t* mbar,
        void* dst_smem,
        int32_t crd0,
        int32_t crd1) {
    uint32_t dst_smem_u32 = cast_smem_ptr_to_uint(dst_smem);
    uint32_t mbar_u32     = cast_smem_ptr_to_uint(mbar);
    uint64_t tmap_u64     = reinterpret_cast<uint64_t>(tmap);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.tile "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst_smem_u32),
           "l"(tmap_u64),
           "r"(crd0),
           "r"(crd1),
           "r"(mbar_u32)
        : "memory");
}

// TMA Store 2D: shared -> global, bulk_group
__device__ __forceinline__ void tma_store_2d(
        const CUtensorMap* tmap,
        const void* src_smem,
        int32_t crd0,
        int32_t crd1) {
    uint32_t src_smem_u32 = cast_smem_ptr_to_uint(src_smem);
    uint64_t tmap_u64     = reinterpret_cast<uint64_t>(tmap);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.tile "
        "[%0, {%1, %2}], [%3];"
        :: "l"(tmap_u64), "r"(crd0), "r"(crd1), "r"(src_smem_u32)
        : "memory");
}

// TMA Reduce Add 2D: shared -> global with atomic add, bulk_group
__device__ __forceinline__ void tma_reduce_add_2d(
        const CUtensorMap* tmap,
        const void* src_smem,
        int32_t crd0,
        int32_t crd1) {
    uint32_t src_smem_u32 = cast_smem_ptr_to_uint(src_smem);
    uint64_t tmap_u64     = reinterpret_cast<uint64_t>(tmap);
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.bulk_group.add.tile "
        "[%0, {%1, %2}], [%3];"
        :: "l"(tmap_u64), "r"(crd0), "r"(crd1), "r"(src_smem_u32)
        : "memory");
}

// Bulk-group commit / wait
__device__ __forceinline__ void cp_async_bulk_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void cp_async_bulk_wait_all() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void cp_async_bulk_wait_all_read() {
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

#endif  // __CUDA_ARCH__ >= 900

}  // namespace engine_c::cuda::tma
