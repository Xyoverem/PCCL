#pragma once

// NVLS multicast memory helpers (H100+ / sm_90a)
// These use PTX inline assembly for multimem load/store/reduce operations.

namespace engine_c::cuda {

// multimem.ld_reduce.add.u32: atomically load-add from multicast VA
__device__ __forceinline__
uint32_t multimem_ld_reduce_add_u32(void* addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t result;
    asm volatile(
        "multimem.ld_reduce.global.add.u32 %0, [%1];"
        : "=r"(result) : "l"(addr) : "memory");
    return result;
#else
    return *(volatile uint32_t*)addr;
#endif
}

// multimem.ld_reduce.add.v4.bf16: load-reduce 8 bf16 values (4x uint32_t = 16 bytes)
__device__ __forceinline__
void multimem_ld_reduce_add_v4_bf16(uint32_t* d0, uint32_t* d1,
                                     uint32_t* d2, uint32_t* d3,
                                     void* addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "multimem.ld_reduce.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(*d0), "=r"(*d1), "=r"(*d2), "=r"(*d3)
        : "l"(addr) : "memory");
#else
    uint32_t* src = (uint32_t*)addr;
    *d0 = src[0]; *d1 = src[1]; *d2 = src[2]; *d3 = src[3];
#endif
}

// multimem.st.v4.bf16: store 8 bf16 values (4x uint32_t = 16 bytes) to multicast VA
__device__ __forceinline__
void multimem_st_v4_bf16(void* addr, uint32_t d0, uint32_t d1,
                          uint32_t d2, uint32_t d3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "multimem.st.global.v4.bf16x2 [%0], {%1, %2, %3, %4};"
        :: "l"(addr), "r"(d0), "r"(d1), "r"(d2), "r"(d3) : "memory");
#else
    uint32_t* dst = (uint32_t*)addr;
    dst[0] = d0; dst[1] = d1; dst[2] = d2; dst[3] = d3;
#endif
}

}  // namespace engine_c::cuda
