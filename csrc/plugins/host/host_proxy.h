#pragma once

#include <cstdint>
#include <common/config.h>

namespace engine_c {

struct RdmaCommand {
    enum Type : int32_t {
        NOOP = 0,
        WRITE = 1,
        READ = 2,
        WRITE_IMM = 3,
        SHUTDOWN = 255
    };
    int32_t type;
    int32_t peer_rank;
    int64_t local_offset;
    int64_t remote_offset;
    int64_t size;
    int32_t completion_slot;
    int32_t _pad;
};

static constexpr int PROXY_QUEUE_CAPACITY = pccl::config::PROXY_QUEUE_CAPACITY;

struct HostProxyState {
    RdmaCommand cmd_queue[PROXY_QUEUE_CAPACITY];
    volatile int32_t cmd_tail;
    volatile int32_t cmd_head;
    volatile int32_t completion_flags[PROXY_QUEUE_CAPACITY];
    volatile int32_t shutdown;
};

}  // namespace engine_c

#ifdef PCCL_RDMA_ENABLED

#include <infiniband/verbs.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace engine_c {

struct RdmaPeerInfo {
    struct ibv_qp* qp;
    uint32_t remote_rkey;
    uint64_t remote_addr;
};

struct RdmaTransport {
    struct ibv_cq* cq;
    struct ibv_mr* local_mr;
    void* local_buffer;
    std::unordered_map<int, RdmaPeerInfo> peers;
};

class HostProxy {
public:
    HostProxy() = default;
    ~HostProxy();

    void start(RdmaTransport transport);
    void stop();
    HostProxyState* state() const { return state_; }
    RdmaTransport& transport() { return transport_; }

    void addPeer(int rank, RdmaPeerInfo info);
    void removePeer(int rank);

private:
    std::thread thread_;
    HostProxyState* state_ = nullptr;
    RdmaTransport transport_;
    std::atomic<bool> running_{false};
    std::mutex peers_mutex_;

    void poll_loop();
};

}  // namespace engine_c

#endif  // PCCL_RDMA_ENABLED
