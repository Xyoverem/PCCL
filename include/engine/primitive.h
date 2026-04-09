#pragma once

#include <plugins/base.h>
#include <cstdint>

namespace engine_c {

struct PrimitiveMeta
{
    int op_seq_index_;
    int num_next_ops_;
    int num_dependencies_;
    int next_primitive_index[8];
    int device_type;
    int channel;
    uint64_t params[4];
};

struct WorkingMeta
{
    int remaining_deps;
};

struct GraphBuffer
{
    PrimitiveMeta* meta[2];
    WorkingMeta* working_meta[2];
    void* primitives[2];
};

}  // namespace engine_c
