#pragma once

#include <common/config.h>

namespace engine_c {

struct FusedStep {
    char primitive_type;
    char pad[3];
    int peer_rank;
    int offset_0;
    int offset_1;
    int offset_2;
    int size;
    int notify_signal_id;
    int notify_peer_rank;
    int wait_signal_id;
    int wait_peer_rank;
    bool has_notify;
    bool has_wait;
    bool pad2[2];
};

static constexpr int MAX_FUSED_STEPS = pccl::config::MAX_FUSED_STEPS;
static constexpr int MAX_FUSED_CHANNELS = pccl::config::MAX_FUSED_CHANNELS;

struct FusedStepDescriptor {
    FusedStep steps[MAX_FUSED_STEPS];
    int num_steps;
    int num_channels;
    int channel_offsets[MAX_FUSED_CHANNELS + 1];
    bool has_tma_ops;
    bool has_multimem_ops;
};

}  // namespace engine_c
