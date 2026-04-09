#pragma once

#include <engine/primitive.h>
#include <engine/ring_buffer.h>

namespace engine_c::cuda {

struct OpProxyTriggerCudaHandle {
    char primitive_type;
    int op_index;
    int peer_rank;
    int offset_0;
    int offset_1;
    int offset_2;
    int size;
};

struct SignalProxyTriggerCudaHandle {
    char primitive_type;
    int op_index;
    int peer_rank;
    int offset;
    int participants;
};

union op {
    ProxyTrigger raw;
    OpProxyTriggerCudaHandle op_handle;
    SignalProxyTriggerCudaHandle signal_handle;
};

}
