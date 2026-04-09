#pragma once

#include <common/config.h>
#include <engine/workspace.h>
#include <engine/fused_step.h>
#include <cstring>

namespace engine_c {

struct MemoryLayout {
    static constexpr long WORKSPACE_REGION_SIZE    = pccl::config::WORKSPACE_REGION_SIZE;
    static constexpr long DEV_WORKSPACE_SIZE       = pccl::config::DEV_WORKSPACE_SIZE;
    static constexpr long PEER_SLOT_SIZE           = pccl::config::PEER_SLOT_SIZE;
    static constexpr long GRAPH_META_SIZE          = pccl::config::GRAPH_META_SIZE;
    static constexpr long PRIMITIVES_SIZE          = pccl::config::PRIMITIVES_SIZE;
    static constexpr long WORKING_META_SIZE        = pccl::config::WORKING_META_SIZE;
    static constexpr long QUEUE_DATA_SIZE          = pccl::config::QUEUE_DATA_SIZE;
    static constexpr long QUEUE_META_SIZE          = pccl::config::QUEUE_META_SIZE;
    static constexpr long QUEUE_TOTAL_SIZE         = pccl::config::QUEUE_TOTAL_SIZE;
    static constexpr long CHANNEL_QUEUE_DATA_SIZE  = pccl::config::CHANNEL_QUEUE_DATA_SIZE;
    static constexpr long CHANNEL_QUEUE_META_SIZE  = pccl::config::CHANNEL_QUEUE_META_SIZE;
    static constexpr long CHANNEL_QUEUE_TOTAL_SIZE = pccl::config::CHANNEL_QUEUE_TOTAL_SIZE;
    static constexpr long FUSED_DESC_SIZE          = sizeof(FusedStepDescriptor);

    static void initialize(void* host_buffer, void* cuda_buffer,
                           void* host_signals, void* cuda_signals,
                           Workspace* workspace);
};

}  // namespace engine_c
