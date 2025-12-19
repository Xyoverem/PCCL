cmake_minimum_required(VERSION 3.12)

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocrand REQUIRED)

set(ROCM_SOURCES
    device.cc
    executor.cc
    utils/rocm_utils.cc
)

set(ROCM_HEADERS
    ${CMAKE_SOURCE_DIR}/include/plugins/rocm_executor/device.h
    ${CMAKE_SOURCE_DIR}/include/plugins/rocm_executor/executor.h
    ${CMAKE_SOURCE_DIR}/include/plugins/rocm_executor/utils/rocm_utils.h
)

add_library(pccl_rocm_executor SHARED ${ROCM_SOURCES})

target_include_directories(pccl_rocm_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${HIP_INCLUDE_DIRS}
    ${ROCBLAS_INCLUDE_DIRS}
    ${ROCRAND_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

target_link_libraries(pccl_rocm_executor PRIVATE
    ${HIP_LIBRARIES}
    ${ROCBLAS_LIBRARIES}
    ${ROCRAND_LIBRARIES}
    pccl_base
)

target_compile_definitions(pccl_rocm_executor PRIVATE
    PCCL_ROCM_SUPPORT=1
)

set_target_properties(pccl_rocm_executor PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_library(pccl_rocm_python_bindings MODULE python_bindings.cc)

target_include_directories(pccl_rocm_python_bindings PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/pybind11/include
    ${HIP_INCLUDE_DIRS}
    ${ROCBLAS_INCLUDE_DIRS}
    ${ROCRAND_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(pccl_rocm_python_bindings PRIVATE
    pccl_rocm_executor
    ${TORCH_LIBRARIES}
)

target_compile_definitions(pccl_rocm_python_bindings PRIVATE
    PCCL_ROCM_SUPPORT=1
)

set_target_properties(pccl_rocm_python_bindings PROPERTIES
    PREFIX ""
    OUTPUT_NAME "pccl_native_rocm"
    SUFFIX ${PYTHON_MODULE_EXTENSION}
)