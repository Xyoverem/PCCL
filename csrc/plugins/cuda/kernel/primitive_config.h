#pragma once

namespace engine_c::cuda {

using primitive_type = char;

constexpr primitive_type cuda_noop = 0;

constexpr primitive_type cuda_reduce_f32 = 1;
constexpr primitive_type cuda_reduce_f16 = 2;
constexpr primitive_type cuda_reduce_bf16 = 3;
constexpr primitive_type cuda_reduce_f8_e4m3 = 4;
constexpr primitive_type cuda_reduce_f8_e5m2 = 5;

constexpr primitive_type cuda_copy_f32 = 6;
constexpr primitive_type cuda_copy_f16 = 7;
constexpr primitive_type cuda_copy_bf16 = 8;
constexpr primitive_type cuda_copy_f8_e4m3 = 9;
constexpr primitive_type cuda_copy_f8_e5m2 = 10;

constexpr primitive_type cuda_multimem_reduce_f32 = 11;
constexpr primitive_type cuda_multimem_reduce_f16 = 12;
constexpr primitive_type cuda_multimem_reduce_bf16 = 13;
constexpr primitive_type cuda_multimem_reduce_f8_e4m3 = 14;
constexpr primitive_type cuda_multimem_reduce_f8_e5m2 = 15;

constexpr primitive_type cuda_multimem_store_f32 = 16;
constexpr primitive_type cuda_multimem_store_f16 = 17;
constexpr primitive_type cuda_multimem_store_bf16 = 18;
constexpr primitive_type cuda_multimem_store_f8_e4m3 = 19;
constexpr primitive_type cuda_multimem_store_f8_e5m2 = 20;

constexpr primitive_type cuda_wait_notify = 21;
constexpr primitive_type cuda_notify = 22;

constexpr primitive_type cuda_ce_copy_f32 = 23;
constexpr primitive_type cuda_ce_copy_f16 = 24;
constexpr primitive_type cuda_ce_copy_bf16 = 25;
constexpr primitive_type cuda_ce_copy_f8_e4m3 = 26;
constexpr primitive_type cuda_ce_copy_f8_e5m2 = 27;

constexpr primitive_type cuda_tma_copy_f32 = 28;
constexpr primitive_type cuda_tma_copy_f16 = 29;
constexpr primitive_type cuda_tma_copy_bf16 = 30;
constexpr primitive_type cuda_tma_copy_f8_e4m3 = 31;
constexpr primitive_type cuda_tma_copy_f8_e5m2 = 32;

constexpr primitive_type cuda_tma_reduce_f32 = 33;
constexpr primitive_type cuda_tma_reduce_f16 = 34;
constexpr primitive_type cuda_tma_reduce_bf16 = 35;
constexpr primitive_type cuda_tma_reduce_f8_e4m3 = 36;
constexpr primitive_type cuda_tma_reduce_f8_e5m2 = 37;

constexpr primitive_type cuda_rdma_write = 40;
constexpr primitive_type cuda_rdma_read  = 41;

}
