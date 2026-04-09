#pragma once

#include <plugins/device.h>
#include <string>
#include <stdexcept>
#include <fmt/format.h>
#include "../kernel/proxy_trigger.h"
#include "../kernel/primitive_config.h"

namespace engine_c::cuda {

// ---------------------------------------------------------------------------
// SmChannel: SM-based operations (sm.copy, sm.reduce, sm.notify, sm.wait_notify)
// ---------------------------------------------------------------------------
class SmChannel : public Channel {
   public:
    const std::string& name() const override {
        static const std::string n = "sm";
        return n;
    }

    std::vector<std::string> supported_ops() const override {
        return {"copy", "reduce", "notify", "wait_notify"};
    }

    ProxyTrigger parse(nlohmann::json& op_info,
                       nlohmann::json& tensor_info) override {
        std::string primitive = op_info["primitive"];
        nlohmann::json params = op_info["params"];
        union op proxy_trigger;

        if (primitive == "sm.reduce") {
            return parseDataOp(proxy_trigger, op_info, tensor_info, true,
                              cuda_reduce_f32, cuda_reduce_f16, cuda_reduce_bf16,
                              cuda_reduce_f8_e4m3, cuda_reduce_f8_e5m2);
        }
        if (primitive == "sm.copy") {
            return parseDataOp(proxy_trigger, op_info, tensor_info, false,
                              cuda_copy_f32, cuda_copy_f16, cuda_copy_bf16,
                              cuda_copy_f8_e4m3, cuda_copy_f8_e5m2);
        }
        if (primitive == "sm.notify" || primitive == "notify") {
            proxy_trigger.signal_handle.primitive_type = cuda_notify;
            proxy_trigger.signal_handle.op_index = op_info["index"];
            proxy_trigger.signal_handle.peer_rank = params["target_rank"];
            proxy_trigger.signal_handle.offset = params["signal_id"];
            proxy_trigger.signal_handle.participants = 0;
            return proxy_trigger.raw;
        }
        if (primitive == "sm.wait_notify" || primitive == "wait_notify") {
            proxy_trigger.signal_handle.primitive_type = cuda_wait_notify;
            proxy_trigger.signal_handle.op_index = op_info["index"];
            proxy_trigger.signal_handle.peer_rank = params["source_rank"];
            proxy_trigger.signal_handle.offset = params["signal_id"];
            proxy_trigger.signal_handle.participants = 0;
            return proxy_trigger.raw;
        }

        throw std::runtime_error(
            fmt::format("SmChannel: unknown primitive '{}'", primitive));
    }

   private:
    static char lookupDtype(const std::string& dtype,
                            char f32, char f16, char bf16, char f8e4m3, char f8e5m2) {
        if (dtype == "float32")     return f32;
        if (dtype == "float16")     return f16;
        if (dtype == "bfloat16")    return bf16;
        if (dtype == "float8_e4m3") return f8e4m3;
        if (dtype == "float8_e5m2") return f8e5m2;
        throw std::runtime_error(fmt::format("Unsupported dtype '{}'", dtype));
    }

    static ProxyTrigger parseDataOp(union op& proxy_trigger,
                                     nlohmann::json& op_info,
                                     nlohmann::json& tensor_info,
                                     bool is_reduce,
                                     char f32, char f16, char bf16,
                                     char f8e4m3, char f8e5m2) {
        nlohmann::json params = op_info["params"];
        std::string dtype = tensor_info["dtype"];
        proxy_trigger.op_handle.primitive_type = lookupDtype(dtype, f32, f16, bf16, f8e4m3, f8e5m2);
        proxy_trigger.op_handle.op_index = op_info["index"];
        proxy_trigger.op_handle.peer_rank = params.value("source_rank", params.value("target_rank", 0));
        if (is_reduce) {
            proxy_trigger.op_handle.offset_0 = params["remote_offset"];
            proxy_trigger.op_handle.offset_1 = params["src_offset"];
            proxy_trigger.op_handle.offset_2 = params["dst_offset"];
            proxy_trigger.op_handle.size = params["count"];
        } else {
            proxy_trigger.op_handle.offset_0 = params["src_offset"];
            proxy_trigger.op_handle.offset_1 = params["dst_offset"];
            proxy_trigger.op_handle.size = params["size"];
        }
        return proxy_trigger.raw;
    }
};

}  // namespace engine_c::cuda
