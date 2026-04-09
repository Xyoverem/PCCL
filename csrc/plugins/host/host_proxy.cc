#include "host_proxy.h"

#ifdef PCCL_RDMA_ENABLED

#include <cuda_runtime.h>
#include <cstring>
#include <mutex>
#include <common/logging.h>
#include <immintrin.h>

namespace engine_c {

HostProxy::~HostProxy()
{
    stop();
    if (state_) {
        cudaFreeHost(state_);
        state_ = nullptr;
    }
}

void HostProxy::start(RdmaTransport transport)
{
    transport_ = std::move(transport);

    void* raw = nullptr;
    cudaHostAlloc(&raw, sizeof(HostProxyState),
                  cudaHostAllocMapped | cudaHostAllocPortable);
    state_ = reinterpret_cast<HostProxyState*>(raw);
    std::memset(raw, 0, sizeof(HostProxyState));

    running_.store(true);
    thread_ = std::thread(&HostProxy::poll_loop, this);
    PCCL_LOG_INFO("HostProxy started");
}

void HostProxy::addPeer(int rank, RdmaPeerInfo info)
{
    std::lock_guard<std::mutex> lock(peers_mutex_);
    transport_.peers[rank] = info;
    PCCL_LOG_INFO("HostProxy: added peer rank {}", rank);
}

void HostProxy::removePeer(int rank)
{
    std::lock_guard<std::mutex> lock(peers_mutex_);
    transport_.peers.erase(rank);
}

void HostProxy::stop()
{
    if (!running_.load()) return;
    running_.store(false);
    if (state_) {
        const_cast<volatile int32_t&>(state_->shutdown) = 1;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    PCCL_LOG_INFO("HostProxy stopped");
}

void HostProxy::poll_loop()
{
    while (running_.load()) {
        int32_t head = state_->cmd_head;
        int32_t tail = __atomic_load_n(&state_->cmd_tail, __ATOMIC_ACQUIRE);

        if (head >= tail) {
            if (state_->shutdown) break;
            _mm_pause();
            continue;
        }

        int slot = head % PROXY_QUEUE_CAPACITY;
        const RdmaCommand& cmd = state_->cmd_queue[slot];

        if (cmd.type == RdmaCommand::SHUTDOWN) break;

        if (cmd.type == RdmaCommand::WRITE || cmd.type == RdmaCommand::READ) {
            std::lock_guard<std::mutex> lock(peers_mutex_);
            auto it = transport_.peers.find(cmd.peer_rank);
            if (it != transport_.peers.end() && transport_.local_mr) {
                const RdmaPeerInfo& peer = it->second;

                struct ibv_sge sge = {};
                sge.addr = reinterpret_cast<uint64_t>(transport_.local_buffer) + cmd.local_offset;
                sge.length = static_cast<uint32_t>(cmd.size);
                sge.lkey = transport_.local_mr->lkey;

                struct ibv_send_wr wr = {};
                wr.wr_id = static_cast<uint64_t>(slot);
                wr.sg_list = &sge;
                wr.num_sge = 1;
                wr.opcode = (cmd.type == RdmaCommand::WRITE)
                    ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
                wr.send_flags = IBV_SEND_SIGNALED;
                wr.wr.rdma.remote_addr = peer.remote_addr + cmd.remote_offset;
                wr.wr.rdma.rkey = peer.remote_rkey;

                struct ibv_send_wr* bad_wr = nullptr;
                int ret = ibv_post_send(peer.qp, &wr, &bad_wr);
                if (ret == 0) {
                    struct ibv_wc wc = {};
                    while (ibv_poll_cq(transport_.cq, 1, &wc) == 0) {
                        if (!running_.load()) break;
                    }
                }
            }
        }

        __atomic_store_n(
            const_cast<int32_t*>(&state_->completion_flags[cmd.completion_slot]),
            1, __ATOMIC_RELEASE);

        state_->cmd_head = head + 1;
    }
}

}  // namespace engine_c

#endif  // PCCL_RDMA_ENABLED
