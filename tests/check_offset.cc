#include <cstdio>
#include <cstddef>
#include "include/engine/workspace.h"
int main() {
    printf("sizeof(DeviceWorkspace) = %zu\n", sizeof(engine_c::DeviceWorkspace));
    printf("offsetof(ring_buffers_) = %zu\n", offsetof(engine_c::DeviceWorkspace, ring_buffers_));
    printf("offsetof(grid_barrier_count_) = %zu\n", offsetof(engine_c::DeviceWorkspace, grid_barrier_count_));
    printf("offsetof(fused_ch_barrier_) = %zu\n", offsetof(engine_c::DeviceWorkspace, fused_ch_barrier_));
    printf("offsetof(nvls_grid_counters_) = %zu\n", offsetof(engine_c::DeviceWorkspace, nvls_grid_counters_));
    printf("PERCALL_COPY_SIZE = %zu\n", engine_c::DeviceWorkspace::PERCALL_COPY_SIZE);
    return 0;
}
