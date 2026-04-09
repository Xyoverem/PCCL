#pragma once

#include <plugins/device.h>
#include <string>
#include <stdexcept>
#include <cstring>
#include <fmt/format.h>
#include "../kernel/proxy_trigger.h"
#include "../kernel/primitive_config.h"

namespace engine_c::cuda {

// ---------------------------------------------------------------------------
// RdmaChannel (CUDA-side): GPU-initiated RDMA operations (rdma.write, rdma.read)
//   These are parsed by the CUDA plugin since the GPU kernel dispatches them
//   to the host proxy for RDMA execution.
// ---------------------------------------------------------------------------
class RdmaChannel : public Channel {
   public:
    const std::string& name() const override {
        static const std::string n = "rdma";
        return n;
    }

    std::vector<std::string> supported_ops() const override {
        return {"write", "read"};
    }

    ProxyTrigger parse(nlohmann::json& op_info,
                       nlohmann::json& tensor_info) override {
        std::string primitive = op_info["primitive"];
        nlohmann::json params = op_info["params"];
        union op proxy_trigger;

        if (primitive == "rdma.write") {
            proxy_trigger.op_handle.primitive_type = cuda_rdma_write;
            proxy_trigger.op_handle.op_index = op_info["index"];
            proxy_trigger.op_handle.peer_rank = params["target_rank"];
            proxy_trigger.op_handle.offset_0 = params["src_offset"];
            proxy_trigger.op_handle.offset_1 = params["dst_offset"];
            proxy_trigger.op_handle.size = params["size"];
            return proxy_trigger.raw;
        }
        if (primitive == "rdma.read") {
            proxy_trigger.op_handle.primitive_type = cuda_rdma_read;
            proxy_trigger.op_handle.op_index = op_info["index"];
            proxy_trigger.op_handle.peer_rank = params["source_rank"];
            proxy_trigger.op_handle.offset_0 = params["src_offset"];
            proxy_trigger.op_handle.offset_1 = params["dst_offset"];
            proxy_trigger.op_handle.size = params["size"];
            return proxy_trigger.raw;
        }

        throw std::runtime_error(
            fmt::format("RdmaChannel: unknown primitive '{}'", primitive));
    }
};

}  // namespace engine_c::cuda
