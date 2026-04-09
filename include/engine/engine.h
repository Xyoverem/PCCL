#pragma once

#include <torch/extension.h>
#include <common/config.h>
#include <string>
#include <memory>

namespace engine_c {

static constexpr long DefaultBufferSize = pccl::config::DEFAULT_BUFFER_SIZE;
static constexpr long MinBufferSize = pccl::config::MIN_BUFFER_SIZE;
static constexpr long SignalSize = pccl::config::SIGNAL_SIZE;

inline long& getBufferSize() {
    static long buffer_size = DefaultBufferSize;
    return buffer_size;
}

#define BufferSize (engine_c::getBufferSize())

class Engine
{
   public:
    static Engine& getInstance();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    bool regOp(const std::string& name, const std::string& filename);
    void exeOp(const std::string& name, at::Tensor& input, at::Tensor& output);
    void exeOpAsync(const std::string& name, at::Tensor& input, at::Tensor& output);
    void syncOp(const std::string& name);
    void resetSignals(const std::string& name);

    std::string& exportEndpoint();
    void updateEndpoint(int rank, std::string& endpoint);

   private:
    Engine();
    ~Engine();

    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine_c
