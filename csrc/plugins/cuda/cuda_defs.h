#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <c10/util/Exception.h>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        const cudaError_t error = call;                                                            \
        if (error != cudaSuccess) {                                                                \
            TORCH_CHECK(false, "CUDA error at ", __FILE__, ":", __LINE__, "\nError code: ", error, \
                        " (", cudaGetErrorString(error), ")");                                     \
        }                                                                                          \
    } while (0)

#define CUDA_DRIVER_CHECK(call)                                                  \
    do {                                                                         \
        const CUresult result = call;                                            \
        if (result != CUDA_SUCCESS) {                                            \
            const char* error_string;                                            \
            cuGetErrorString(result, &error_string);                             \
            TORCH_CHECK(false, "CUDA Driver error at ", __FILE__, ":", __LINE__, \
                        "\nError code: ", result, " (", error_string, ")");      \
        }                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                                        \
    do {                                                                                          \
        cublasStatus_t status = call;                                                             \
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublas error at ", __FILE__, ":", __LINE__, \
                    "\nError code: ", status);                                                    \
    } while (0)
