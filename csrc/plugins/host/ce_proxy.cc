#include "ce_proxy.h"

#include <cuda_runtime.h>
#include <cstring>
#include <common/logging.h>
#include <immintrin.h>

namespace engine_c {

CeProxy::~CeProxy()
{
    stop();
    if (ce_stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(ce_stream_));
        ce_stream_ = nullptr;
    }
    if (state_) {
        cudaFreeHost(state_);
        state_ = nullptr;
    }
}

void CeProxy::start()
{
    if (running_.load()) return;

    void* raw = nullptr;
    cudaHostAlloc(&raw, sizeof(CeProxyState),
                  cudaHostAllocMapped | cudaHostAllocPortable);
    state_ = reinterpret_cast<CeProxyState*>(raw);
    std::memset(raw, 0, sizeof(CeProxyState));

    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    ce_stream_ = static_cast<void*>(stream);

    running_.store(true);
    thread_ = std::thread(&CeProxy::poll_loop, this);
    PCCL_LOG_INFO("CeProxy started");
}

void CeProxy::stop()
{
    if (!running_.load()) return;
    running_.store(false);
    if (state_) {
        const_cast<volatile int32_t&>(state_->shutdown) = 1;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    PCCL_LOG_INFO("CeProxy stopped");
}

void CeProxy::poll_loop()
{
    while (running_.load()) {
        int32_t head = state_->cmd_head;
        int32_t tail = __atomic_load_n(&state_->cmd_tail, __ATOMIC_ACQUIRE);

        if (head >= tail) {
            if (state_->shutdown) break;
            _mm_pause();
            continue;
        }

        int slot = head % CE_PROXY_QUEUE_CAPACITY;
        const CeCommand& cmd = state_->cmd_queue[slot];

        if (cmd.type == CeCommand::SHUTDOWN) break;

        if (cmd.type == CeCommand::COPY) {
            auto stream = static_cast<cudaStream_t>(ce_stream_);
            cudaMemcpyAsync(cmd.dst, cmd.src,
                            static_cast<size_t>(cmd.size_bytes),
                            cudaMemcpyDeviceToDevice, stream);
            cudaStreamSynchronize(stream);
        }

        __atomic_store_n(
            const_cast<int32_t*>(&state_->completion_flags[cmd.completion_slot]),
            1, __ATOMIC_RELEASE);

        state_->cmd_head = head + 1;
    }
}

}  // namespace engine_c
