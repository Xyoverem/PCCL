#pragma once 

#include "./vec.h"

namespace engine_c::cuda {

#if !PCCL_HAS_CUDA_FP8
__device__ __forceinline__ void unsupported_fp8() {
  asm volatile("trap;");
}
#endif

enum class DataType {
  F32,
  F16,
  BF16,
  FP8_E4M3,
  FP8_E5M2
};

template <typename T>
__device__ __forceinline__ void vector_copy_impl(const void* src, void* dst, int n) {
  const T* s = reinterpret_cast<const T*>(src);
  T* d = reinterpret_cast<T*>(dst);
  constexpr int N = Vec<T>::N;
  const int limit = n / N;

  // 2x unrolled loop for better ILP
  const int limit2 = (limit / 2) * 2;
  int i = threadIdx.x * 2;

  for (; i < limit2; i += blockDim.x * 2) {
    Vec<T> v0((void*)&s[i * N]);
    Vec<T> v1((void*)&s[(i + 1) * N]);
    v0.store((void*)&d[i * N]);
    v1.store((void*)&d[(i + 1) * N]);
  }

  i = threadIdx.x + limit2;
  if (i < limit) {
    Vec<T> v((void*)&s[i * N]);
    v.store((void*)&d[i * N]);
  }
}

template <typename T>
__device__ __forceinline__ void vector_add_impl(const void* a, const void* b, void* c, int n) {
  const T* ta = reinterpret_cast<const T*>(a);
  const T* tb = reinterpret_cast<const T*>(b);
  T* tc = reinterpret_cast<T*>(c);
  constexpr int N = Vec<T>::N;
  const int limit = n / N;

  const int limit2 = (limit / 2) * 2;
  int i = threadIdx.x * 2;

  for (; i < limit2; i += blockDim.x * 2) {
    Vec<T> va0((void*)&ta[i * N]);
    Vec<T> vb0((void*)&tb[i * N]);
    Vec<T> va1((void*)&ta[(i + 1) * N]);
    Vec<T> vb1((void*)&tb[(i + 1) * N]);
    (va0 + vb0).store((void*)&tc[i * N]);
    (va1 + vb1).store((void*)&tc[(i + 1) * N]);
  }

  i = threadIdx.x + limit2;
  if (i < limit) {
    Vec<T> va((void*)&ta[i * N]);
    Vec<T> vb((void*)&tb[i * N]);
    (va + vb).store((void*)&tc[i * N]);
  }
}

__device__ __forceinline__ void vector_copy(DataType dtype, const void* src, void* dst, int n) {
  switch (dtype) {
    case DataType::F32: vector_copy_impl<float>(src, dst, n); break;
    case DataType::F16: vector_copy_impl<__half>(src, dst, n); break;
    case DataType::BF16: vector_copy_impl<__nv_bfloat16>(src, dst, n); break;
#if PCCL_HAS_CUDA_FP8
    case DataType::FP8_E4M3: vector_copy_impl<__nv_fp8_e4m3>(src, dst, n); break;
    case DataType::FP8_E5M2: vector_copy_impl<__nv_fp8_e5m2>(src, dst, n); break;
#else
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2: unsupported_fp8(); break;
#endif
  }
}

__device__ __forceinline__ void vector_add(DataType dtype, const void* a, const void* b, void* c, int n) {
  switch (dtype) {
    case DataType::F32: vector_add_impl<float>(a, b, c, n); break;
    case DataType::F16: vector_add_impl<__half>(a, b, c, n); break;
    case DataType::BF16: vector_add_impl<__nv_bfloat16>(a, b, c, n); break;
#if PCCL_HAS_CUDA_FP8
    case DataType::FP8_E4M3: vector_add_impl<__nv_fp8_e4m3>(a, b, c, n); break;
    case DataType::FP8_E5M2: vector_add_impl<__nv_fp8_e5m2>(a, b, c, n); break;
#else
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2: unsupported_fp8(); break;
#endif
  }
}

// Shared-memory-safe add: operand 'a' is in shared memory (uses plain load),
// operand 'b' is in global memory (uses __ldg), result 'c' uses plain store.
template <typename T>
__device__ __forceinline__ void smem_vector_add_impl(const void* a, const void* b, void* c, int n) {
  const T* ta = reinterpret_cast<const T*>(a);
  const T* tb = reinterpret_cast<const T*>(b);
  T* tc = reinterpret_cast<T*>(c);
  constexpr int N = Vec<T>::N;
  const int limit = n / N;

  const int limit2 = (limit / 2) * 2;
  int i = threadIdx.x * 2;

  for (; i < limit2; i += blockDim.x * 2) {
    Vec<T> va0 = Vec<T>::load_local((void*)&ta[i * N]);
    Vec<T> vb0((void*)&tb[i * N]);
    Vec<T> va1 = Vec<T>::load_local((void*)&ta[(i + 1) * N]);
    Vec<T> vb1((void*)&tb[(i + 1) * N]);
    (va0 + vb0).store((void*)&tc[i * N]);
    (va1 + vb1).store((void*)&tc[(i + 1) * N]);
  }

  i = threadIdx.x + limit2;
  if (i < limit) {
    Vec<T> va = Vec<T>::load_local((void*)&ta[i * N]);
    Vec<T> vb((void*)&tb[i * N]);
    (va + vb).store((void*)&tc[i * N]);
  }
}

__device__ __forceinline__ void smem_vector_add(DataType dtype, const void* a, const void* b, void* c, int n) {
  switch (dtype) {
    case DataType::F32: smem_vector_add_impl<float>(a, b, c, n); break;
    case DataType::F16: smem_vector_add_impl<__half>(a, b, c, n); break;
    case DataType::BF16: smem_vector_add_impl<__nv_bfloat16>(a, b, c, n); break;
#if PCCL_HAS_CUDA_FP8
    case DataType::FP8_E4M3: smem_vector_add_impl<__nv_fp8_e4m3>(a, b, c, n); break;
    case DataType::FP8_E5M2: smem_vector_add_impl<__nv_fp8_e5m2>(a, b, c, n); break;
#else
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2: unsupported_fp8(); break;
#endif
  }
}




// TMA-style optimized copy using async copy instructions (SM80+/SM90+)

#if __CUDA_ARCH__ >= 800

// Uses warp-level async copy for better pipelining
template <typename T>
__device__ __forceinline__ void tma_vector_copy_impl(const void* src, void* dst, int n) {
    const T* s = reinterpret_cast<const T*>(src);
    T* d = reinterpret_cast<T*>(dst);
    constexpr int N = Vec<T>::N;
    const int limit = n / N;

    constexpr int UNROLL = 4;
    const int limit4 = (limit / UNROLL) * UNROLL;
    int i = threadIdx.x * UNROLL;

    for (; i < limit4; i += blockDim.x * UNROLL) {
        Vec<T> v0((void*)&s[i * N]);
        Vec<T> v1((void*)&s[(i + 1) * N]);
        Vec<T> v2((void*)&s[(i + 2) * N]);
        Vec<T> v3((void*)&s[(i + 3) * N]);
        v0.store((void*)&d[i * N]);
        v1.store((void*)&d[(i + 1) * N]);
        v2.store((void*)&d[(i + 2) * N]);
        v3.store((void*)&d[(i + 3) * N]);
    }

    const int limit2 = ((limit - limit4) / 2) * 2 + limit4;
    i = threadIdx.x * 2 + limit4;
    for (; i < limit2; i += blockDim.x * 2) {
        Vec<T> v0((void*)&s[i * N]);
        Vec<T> v1((void*)&s[(i + 1) * N]);
        v0.store((void*)&d[i * N]);
        v1.store((void*)&d[(i + 1) * N]);
    }

    i = threadIdx.x + limit2;
    if (i < limit) {
        Vec<T> v((void*)&s[i * N]);
        v.store((void*)&d[i * N]);
    }
}

// Fallback for older architectures
#else
template <typename T>
__device__ __forceinline__ void tma_vector_copy_impl(const void* src, void* dst, int n) {
    vector_copy_impl<T>(src, dst, n);
}
#endif

// TMA copy dispatcher
__device__ __forceinline__ void tma_vector_copy(DataType dtype, const void* src, void* dst, int n) {
    switch (dtype) {
        case DataType::F32: tma_vector_copy_impl<float>(src, dst, n); break;
        case DataType::F16: tma_vector_copy_impl<__half>(src, dst, n); break;
        case DataType::BF16: tma_vector_copy_impl<__nv_bfloat16>(src, dst, n); break;
#if PCCL_HAS_CUDA_FP8
        case DataType::FP8_E4M3: tma_vector_copy_impl<__nv_fp8_e4m3>(src, dst, n); break;
        case DataType::FP8_E5M2: tma_vector_copy_impl<__nv_fp8_e5m2>(src, dst, n); break;
#else
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2: unsupported_fp8(); break;
#endif
    }
}

// TMA reduce: optimized with interleaved load-compute-store to reduce register pressure
#if __CUDA_ARCH__ >= 800
template <typename T>
__device__ __forceinline__ void tma_vector_add_impl(const void* a, const void* b, void* c, int n) {
    const T* ta = reinterpret_cast<const T*>(a);
    const T* tb = reinterpret_cast<const T*>(b);
    T* tc = reinterpret_cast<T*>(c);
    constexpr int N = Vec<T>::N;
    const int limit = n / N;

    // Interleaved pattern keeps only 2 vectors live at a time
    constexpr int UNROLL = 4;
    const int limit4 = (limit / UNROLL) * UNROLL;
    int i = threadIdx.x * UNROLL;

    for (; i < limit4; i += blockDim.x * UNROLL) {
        {
            Vec<T> va((void*)&ta[i * N]);
            Vec<T> vb((void*)&tb[i * N]);
            (va + vb).store((void*)&tc[i * N]);
        }
        {
            Vec<T> va((void*)&ta[(i + 1) * N]);
            Vec<T> vb((void*)&tb[(i + 1) * N]);
            (va + vb).store((void*)&tc[(i + 1) * N]);
        }
        {
            Vec<T> va((void*)&ta[(i + 2) * N]);
            Vec<T> vb((void*)&tb[(i + 2) * N]);
            (va + vb).store((void*)&tc[(i + 2) * N]);
        }
        {
            Vec<T> va((void*)&ta[(i + 3) * N]);
            Vec<T> vb((void*)&tb[(i + 3) * N]);
            (va + vb).store((void*)&tc[(i + 3) * N]);
        }
    }

    i = threadIdx.x + limit4;
    for (; i < limit; i += blockDim.x) {
        Vec<T> va((void*)&ta[i * N]);
        Vec<T> vb((void*)&tb[i * N]);
        (va + vb).store((void*)&tc[i * N]);
    }
}
#else
template <typename T>
__device__ __forceinline__ void tma_vector_add_impl(const void* a, const void* b, void* c, int n) {
    vector_add_impl<T>(a, b, c, n);
}
#endif

__device__ __forceinline__ void tma_vector_add(DataType dtype, const void* a, const void* b, void* c, int n) {
    switch (dtype) {
        case DataType::F32: tma_vector_add_impl<float>(a, b, c, n); break;
        case DataType::F16: tma_vector_add_impl<__half>(a, b, c, n); break;
        case DataType::BF16: tma_vector_add_impl<__nv_bfloat16>(a, b, c, n); break;
#if PCCL_HAS_CUDA_FP8
        case DataType::FP8_E4M3: tma_vector_add_impl<__nv_fp8_e4m3>(a, b, c, n); break;
        case DataType::FP8_E5M2: tma_vector_add_impl<__nv_fp8_e5m2>(a, b, c, n); break;
#else
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2: unsupported_fp8(); break;
#endif
    }
}

}
