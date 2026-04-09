#pragma once

#include <cstdint>
#include <plugins/registry.h>

namespace engine_c {

struct ProxyTrigger
{
    uint64_t raw[4];
};

struct QueueMeta
{
    int head;
    int tail;
    int capacity;
    int size;
};

struct RingBuffer
{
    ProxyTrigger* buffer_a_;
    QueueMeta* meta_a_;
    DeviceType device_a_;
    ProxyTrigger* buffer_b_;
    QueueMeta* meta_b_;
    DeviceType device_b_;
};

}  // namespace engine_c
