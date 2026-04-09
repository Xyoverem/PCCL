#pragma once

#include <engine/workspace.h>
#include <engine/fused_step.h>
#include <cuda_runtime.h>

namespace engine_c::cuda {

void launch_cuda_kernel(int num_blocks, cudaStream_t stream,
                        DeviceWorkspace* workspace, int dynamic_smem_bytes,
                        int num_primitives, int queue_capacity);

void launch_fused_kernel(int num_blocks, cudaStream_t stream,
                         DeviceWorkspace* workspace,
                         FusedStepDescriptor* device_desc,
                         int dynamic_smem_bytes);

void launch_nvls_kernel(int num_blocks, cudaStream_t stream,
                         DeviceWorkspace* workspace);

}  // namespace engine_c::cuda
