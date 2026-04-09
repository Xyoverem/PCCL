#pragma once

#include <cstdint>
#include <thread>
#include <atomic>
#include <common/config.h>

namespace engine_c {

static constexpr int CE_PROXY_QUEUE_CAPACITY = pccl::config::CE_PROXY_QUEUE_CAPACITY;

struct CeCommand {
    enum Type : int32_t { NOOP = 0, COPY = 1, SHUTDOWN = 255 };
    int32_t type;
    int32_t completion_slot;
    void* src;
    void* dst;
    int64_t size_bytes;
};

struct CeProxyState {
    CeCommand cmd_queue[CE_PROXY_QUEUE_CAPACITY];
    volatile int32_t cmd_tail;
    volatile int32_t cmd_head;
    volatile int32_t completion_flags[CE_PROXY_QUEUE_CAPACITY];
    volatile int32_t shutdown;
};

class CeProxy {
public:
    CeProxy() = default;
    ~CeProxy();

    void start();
    void stop();
    CeProxyState* state() const { return state_; }

private:
    std::thread thread_;
    CeProxyState* state_ = nullptr;
    void* ce_stream_ = nullptr;
    std::atomic<bool> running_{false};

    void poll_loop();
};

}  // namespace engine_c
