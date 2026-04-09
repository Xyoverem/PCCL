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
// MultiMemChannel: NVLS multicast operations (multimem.reduce, multimem.store)
// ---------------------------------------------------------------------------
class MultiMemChannel : public Channel {
   public:
    const std::string& name() const override {
        static const std::string n = "multimem";
        return n;
    }

    std::vector<std::string> supported_ops() const override {
        return {"reduce", "store"};
    }

    ProxyTrigger parse(nlohmann::json& op_info,
                       nlohmann::json& tensor_info) override {
        std::string primitive = op_info["primitive"];
        nlohmann::json params = op_info["params"];
        union op proxy_trigger;

        std::string dtype = tensor_info["dtype"];

        if (primitive == "multimem.reduce") {
            proxy_trigger.op_handle.primitive_type = lookupDtype(dtype, true);
            proxy_trigger.op_handle.op_index = op_info["index"];
            proxy_trigger.op_handle.peer_rank = params.value("source_rank", params.value("target_rank", 0));
            proxy_trigger.op_handle.offset_0 = params["remote_offset"];
            proxy_trigger.op_handle.offset_1 = params["src_offset"];
            proxy_trigger.op_handle.offset_2 = params["dst_offset"];
            proxy_trigger.op_handle.size = params["count"];
            return proxy_trigger.raw;
        }
        if (primitive == "multimem.store") {
            proxy_trigger.op_handle.primitive_type = lookupDtype(dtype, false);
            proxy_trigger.op_handle.op_index = op_info["index"];
            proxy_trigger.op_handle.peer_rank = params.value("source_rank", params.value("target_rank", 0));
            proxy_trigger.op_handle.offset_0 = params["src_offset"];
            proxy_trigger.op_handle.offset_1 = params["dst_offset"];
            proxy_trigger.op_handle.size = params["size"];
            return proxy_trigger.raw;
        }

        throw std::runtime_error(
            fmt::format("MultiMemChannel: unknown primitive '{}'", primitive));
    }

   private:
    static char lookupDtype(const std::string& dtype, bool is_reduce) {
        if (is_reduce) {
            if (dtype == "float32")     return cuda_multimem_reduce_f32;
            if (dtype == "float16")     return cuda_multimem_reduce_f16;
            if (dtype == "bfloat16")    return cuda_multimem_reduce_bf16;
            if (dtype == "float8_e4m3") return cuda_multimem_reduce_f8_e4m3;
            if (dtype == "float8_e5m2") return cuda_multimem_reduce_f8_e5m2;
        } else {
            if (dtype == "float32")     return cuda_multimem_store_f32;
            if (dtype == "float16")     return cuda_multimem_store_f16;
            if (dtype == "bfloat16")    return cuda_multimem_store_bf16;
            if (dtype == "float8_e4m3") return cuda_multimem_store_f8_e4m3;
            if (dtype == "float8_e5m2") return cuda_multimem_store_f8_e5m2;
        }
        throw std::runtime_error(fmt::format("MultiMemChannel: unsupported dtype '{}'", dtype));
    }
};

}  // namespace engine_c::cuda
