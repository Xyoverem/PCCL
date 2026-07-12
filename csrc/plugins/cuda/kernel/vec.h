#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define PCCL_HAS_CUDA_FP8 1
#else
#define PCCL_HAS_CUDA_FP8 0
#endif

namespace engine_c::cuda {

template <typename T>
struct alignas(16) Vec
{
    static constexpr int N = 16 / sizeof(T);
    T v[N];

    __device__ __forceinline__ Vec() {}

    __device__ __forceinline__ Vec(const void* ptr)
    {
        *reinterpret_cast<int4*>(v) = __ldg(reinterpret_cast<const int4*>(ptr));
    }

    static __device__ __forceinline__ Vec load_local(const void* ptr)
    {
        Vec r;
        *reinterpret_cast<int4*>(r.v) = *reinterpret_cast<const int4*>(ptr);
        return r;
    }

    __device__ __forceinline__ void store(void* ptr) const
    {
        *reinterpret_cast<int4*>(ptr) = *reinterpret_cast<const int4*>(v);
    }

    __device__ __forceinline__ Vec operator+(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < N; ++i)
            res.v[i] = v[i] + other.v[i];
        return res;
    }

    __device__ __forceinline__ Vec operator-(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < N; ++i)
            res.v[i] = v[i] - other.v[i];
        return res;
    }

    __device__ __forceinline__ Vec operator*(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < N; ++i)
            res.v[i] = v[i] * other.v[i];
        return res;
    }

    __device__ __forceinline__ Vec operator/(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < N; ++i)
            res.v[i] = v[i] / other.v[i];
        return res;
    }
};

template <>
struct alignas(16) Vec<float>
{
    static constexpr int N = 16 / sizeof(float);
    float4 v;

    __device__ __forceinline__ Vec() {}

    __device__ __forceinline__ Vec(const void* ptr)
    {
        v = __ldg(reinterpret_cast<const float4*>(ptr));
    }

    static __device__ __forceinline__ Vec load_local(const void* ptr)
    {
        Vec r;
        r.v = *reinterpret_cast<const float4*>(ptr);
        return r;
    }

    __device__ __forceinline__ void store(void* ptr) const
    {
        *reinterpret_cast<float4*>(ptr) = v;
    }

    __device__ __forceinline__ Vec operator+(const Vec& other) const
    {
        Vec res;
        res.v.x = v.x + other.v.x;
        res.v.y = v.y + other.v.y;
        res.v.z = v.z + other.v.z;
        res.v.w = v.w + other.v.w;
        return res;
    }

    __device__ __forceinline__ Vec operator-(const Vec& other) const
    {
        Vec res;
        res.v.x = v.x - other.v.x;
        res.v.y = v.y - other.v.y;
        res.v.z = v.z - other.v.z;
        res.v.w = v.w - other.v.w;
        return res;
    }

    __device__ __forceinline__ Vec operator*(const Vec& other) const
    {
        Vec res;
        res.v.x = v.x * other.v.x;
        res.v.y = v.y * other.v.y;
        res.v.z = v.z * other.v.z;
        res.v.w = v.w * other.v.w;
        return res;
    }

    __device__ __forceinline__ Vec operator/(const Vec& other) const
    {
        Vec res;
        res.v.x = v.x / other.v.x;
        res.v.y = v.y / other.v.y;
        res.v.z = v.z / other.v.z;
        res.v.w = v.w / other.v.w;
        return res;
    }
};

template <>
struct alignas(16) Vec<__half>
{
    static constexpr int N = 16 / sizeof(__half);
    __half2 v[4];

    __device__ __forceinline__ Vec() {}

    __device__ __forceinline__ Vec(const void* ptr)
    {
        *reinterpret_cast<int4*>(v) = __ldg(reinterpret_cast<const int4*>(ptr));
    }

    static __device__ __forceinline__ Vec load_local(const void* ptr)
    {
        Vec r;
        *reinterpret_cast<int4*>(r.v) = *reinterpret_cast<const int4*>(ptr);
        return r;
    }

    __device__ __forceinline__ void store(void* ptr) const
    {
        *reinterpret_cast<int4*>(ptr) = *reinterpret_cast<const int4*>(v);
    }

    __device__ __forceinline__ Vec operator+(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            res.v[i] = __hadd2(v[i], other.v[i]);
        return res;
    }

    __device__ __forceinline__ Vec operator-(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            res.v[i] = __hsub2(v[i], other.v[i]);
        return res;
    }

    __device__ __forceinline__ Vec operator*(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            res.v[i] = __hmul2(v[i], other.v[i]);
        return res;
    }

    __device__ __forceinline__ Vec operator/(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            res.v[i] = __h2div(v[i], other.v[i]);
        return res;
    }
};

template <>
struct alignas(16) Vec<__nv_bfloat16>
{
    static constexpr int N = 16 / sizeof(__nv_bfloat16);
    __nv_bfloat162 v[4];

    __device__ __forceinline__ Vec() {}

    __device__ __forceinline__ Vec(const void* ptr)
    {
        *reinterpret_cast<int4*>(v) = __ldg(reinterpret_cast<const int4*>(ptr));
    }

    static __device__ __forceinline__ Vec load_local(const void* ptr)
    {
        Vec r;
        *reinterpret_cast<int4*>(r.v) = *reinterpret_cast<const int4*>(ptr);
        return r;
    }

    __device__ __forceinline__ void store(void* ptr) const
    {
        *reinterpret_cast<int4*>(ptr) = *reinterpret_cast<const int4*>(v);
    }

    __device__ __forceinline__ Vec operator+(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            res.v[i] = __hadd2(v[i], other.v[i]);
        return res;
    }

    __device__ __forceinline__ Vec operator-(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            res.v[i] = __hsub2(v[i], other.v[i]);
        return res;
    }

    __device__ __forceinline__ Vec operator*(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            res.v[i] = __hmul2(v[i], other.v[i]);
        return res;
    }

    __device__ __forceinline__ Vec operator/(const Vec& other) const
    {
        Vec res;
        auto* a = reinterpret_cast<const __nv_bfloat16*>(v);
        auto* b = reinterpret_cast<const __nv_bfloat16*>(other.v);
        auto* r = reinterpret_cast<__nv_bfloat16*>(res.v);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            r[i] = __hdiv(a[i], b[i]);
        return res;
    }
};

#if PCCL_HAS_CUDA_FP8

template <>
struct alignas(16) Vec<__nv_fp8_e4m3>
{
    static constexpr int N = 16 / sizeof(__nv_fp8_e4m3);
    __nv_fp8x2_storage_t v[8];
    constexpr static __nv_fp8_interpretation_t fp8_type = __NV_E4M3;

    __device__ __forceinline__ Vec() {}

    __device__ __forceinline__ Vec(const void* ptr)
    {
        *reinterpret_cast<int4*>(v) = __ldg(reinterpret_cast<const int4*>(ptr));
    }

    static __device__ __forceinline__ Vec load_local(const void* ptr)
    {
        Vec r;
        *reinterpret_cast<int4*>(r.v) = *reinterpret_cast<const int4*>(ptr);
        return r;
    }

    __device__ __forceinline__ void store(void* ptr) const
    {
        *reinterpret_cast<int4*>(ptr) = *reinterpret_cast<const int4*>(v);
    }

    __device__ __forceinline__ Vec operator+(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half2 self_val = __nv_cvt_fp8x2_to_halfraw2(v[i], fp8_type);
            half2 other_val = __nv_cvt_fp8x2_to_halfraw2(other.v[i], fp8_type);
            res.v[i] = __nv_cvt_halfraw2_to_fp8x2(__hadd2(self_val, other_val), __NV_SATFINITE, fp8_type);
        }
        return res;
    }

    __device__ __forceinline__ Vec operator-(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half2 self_val = __nv_cvt_fp8x2_to_halfraw2(v[i], fp8_type);
            half2 other_val = __nv_cvt_fp8x2_to_halfraw2(other.v[i], fp8_type);
            res.v[i] = __nv_cvt_halfraw2_to_fp8x2(__hsub2(self_val, other_val), __NV_SATFINITE, fp8_type);
        }
        return res;
    }

    __device__ __forceinline__ Vec operator*(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half2 self_val = __nv_cvt_fp8x2_to_halfraw2(v[i], fp8_type);
            half2 other_val = __nv_cvt_fp8x2_to_halfraw2(other.v[i], fp8_type);
            res.v[i] = __nv_cvt_halfraw2_to_fp8x2(__hmul2(self_val, other_val), __NV_SATFINITE, fp8_type);
        }
        return res;
    }

    __device__ __forceinline__ Vec operator/(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half2 self_val = __nv_cvt_fp8x2_to_halfraw2(v[i], fp8_type);
            half2 other_val = __nv_cvt_fp8x2_to_halfraw2(other.v[i], fp8_type);
            res.v[i] = __nv_cvt_halfraw2_to_fp8x2(__h2div(self_val, other_val), __NV_SATFINITE, fp8_type);
        }
        return res;
    }
};

template <>
struct alignas(16) Vec<__nv_fp8_e5m2>
{
    static constexpr int N = 16 / sizeof(__nv_fp8_e5m2);
    __nv_fp8x2_storage_t v[8];
    constexpr static __nv_fp8_interpretation_t fp8_type = __NV_E5M2;

    __device__ __forceinline__ Vec() {}

    __device__ __forceinline__ Vec(const void* ptr)
    {
        *reinterpret_cast<int4*>(v) = __ldg(reinterpret_cast<const int4*>(ptr));
    }

    static __device__ __forceinline__ Vec load_local(const void* ptr)
    {
        Vec r;
        *reinterpret_cast<int4*>(r.v) = *reinterpret_cast<const int4*>(ptr);
        return r;
    }

    __device__ __forceinline__ void store(void* ptr) const
    {
        *reinterpret_cast<int4*>(ptr) = *reinterpret_cast<const int4*>(v);
    }

    __device__ __forceinline__ Vec operator+(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half2 self_val = __nv_cvt_fp8x2_to_halfraw2(v[i], fp8_type);
            half2 other_val = __nv_cvt_fp8x2_to_halfraw2(other.v[i], fp8_type);
            res.v[i] =
                __nv_cvt_halfraw2_to_fp8x2(__hadd2(self_val, other_val), __NV_SATFINITE, fp8_type);
        }
        return res;
    }

    __device__ __forceinline__ Vec operator-(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half2 self_val = __nv_cvt_fp8x2_to_halfraw2(v[i], fp8_type);
            half2 other_val = __nv_cvt_fp8x2_to_halfraw2(other.v[i], fp8_type);
            res.v[i] =
                __nv_cvt_halfraw2_to_fp8x2(__hsub2(self_val, other_val), __NV_SATFINITE, fp8_type);
        }
        return res;
    }

    __device__ __forceinline__ Vec operator*(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half2 self_val = __nv_cvt_fp8x2_to_halfraw2(v[i], fp8_type);
            half2 other_val = __nv_cvt_fp8x2_to_halfraw2(other.v[i], fp8_type);
            res.v[i] =
                __nv_cvt_halfraw2_to_fp8x2(__hmul2(self_val, other_val), __NV_SATFINITE, fp8_type);
        }
        return res;
    }

    __device__ __forceinline__ Vec operator/(const Vec& other) const
    {
        Vec res;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half2 self_val = __nv_cvt_fp8x2_to_halfraw2(v[i], fp8_type);
            half2 other_val = __nv_cvt_fp8x2_to_halfraw2(other.v[i], fp8_type);
            res.v[i] =
                __nv_cvt_halfraw2_to_fp8x2(__h2div(self_val, other_val), __NV_SATFINITE, fp8_type);
        }
        return res;
    }
};

#endif  // PCCL_HAS_CUDA_FP8

using f32x4 = Vec<float>;
using f16x8 = Vec<__half>;
using bf16x8 = Vec<__nv_bfloat16>;
#if PCCL_HAS_CUDA_FP8
using f8_e4m3x16 = Vec<__nv_fp8_e4m3>;
using f8_e5m2x16 = Vec<__nv_fp8_e5m2>;
#endif

}  // namespace engine_c::cuda
