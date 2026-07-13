#include <engine/graph_builder.h>
#include <plugins/base.h>
#include <plugins/cuda/kernel/primitive_config.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>

namespace engine_c {

using json = nlohmann::json;

bool GraphBuilder::loadFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    json j;
    file >> j;

    if (!j.contains("operations")) {
        return false;
    }

    json tensor_info;
    if (j.contains("tensor_info")) {
        tensor_info = j["tensor_info"];
    }

    if (j.contains("collective_type")) {
        collective_type_ = j["collective_type"].get<std::string>();
    }

    // JSON v2 format: operations have "executor" field
    for (const auto& op : j["operations"]) {
        PrimitiveMeta meta;

        if (!op.contains("index") || !op.contains("executor") || !op.contains("primitive")) {
            return false;
        }

        meta.op_seq_index_ = op["index"];

        // Use channel-based dispatch: find channel from executor string
        // e.g. "cuda_sm" -> Cuda plugin, "sm" channel
        //      "host" -> Host plugin (no specific channel)
        //      "rdma" -> Cuda plugin, "rdma" channel
        std::string executor_str = op["executor"];
        DeviceType device_type;
        if (executor_str == "host") {
            device_type = TypeRegistry::getTypeId("Host");
            meta.device_type = 0;
        } else {
            // cuda_sm, cuda_tma, cuda_ce, rdma -> all use Cuda device
            device_type = TypeRegistry::getTypeId("Cuda");
            meta.device_type = 1;
        }

        // Parse dependencies
        if (op.contains("dependencies")) {
            meta.num_dependencies_ = op["dependencies"].size();
        } else {
            meta.num_dependencies_ = 0;
        }

        // Parse next ops (max 8)
        if (op.contains("next_ops")) {
            meta.num_next_ops_ = std::min((int)op["next_ops"].size(), 8);
            for (int i = 0; i < meta.num_next_ops_; i++) {
                meta.next_primitive_index[i] = op["next_ops"][i];
            }
        } else {
            meta.num_next_ops_ = 0;
        }

        // Parse channel (default 0 for backward compat)
        meta.channel = op.value("channel", 0);

        // Use device-specific parse to convert JSON to ProxyTrigger
        // The device's parse() method now delegates to the appropriate channel
        auto device = getDev(device_type);
        if (!device) {
            return false;
        }

        json op_copy = op;
        json tensor_info_copy = tensor_info;

        // Try channel-based dispatch first (new path)
        Channel* channel = findChannelByExecutor(executor_str);
        ProxyTrigger trigger;
        if (channel) {
            trigger = channel->parse(op_copy, tensor_info_copy);
        } else {
            // Fallback to device-level parse (handles legacy primitives)
            trigger = device->parse(op_copy, tensor_info_copy);
        }
        operators_.push_back(trigger);

        std::memcpy(meta.params, trigger.raw, sizeof(meta.params));
        metadata_.push_back(meta);
    }

    num_primitives_ = metadata_.size();
    return true;
}

bool GraphBuilder::build(DeviceWorkspace* workspace)
{
    if (num_primitives_ == 0) {
        return false;
    }

    workspace->total_primitives = num_primitives_;
    workspace->completed_primitives = 0;

    // Determine num_channels and count per-channel primitives
    int max_channel = 0;
    for (int i = 0; i < num_primitives_; i++) {
        if (metadata_[i].channel > max_channel) {
            max_channel = metadata_[i].channel;
        }
    }
    workspace->num_channels = max_channel + 1;

    // Scan primitive types to detect TMA operations (type IDs 28-37)
    workspace->has_tma_ops = false;
    for (int i = 0; i < num_primitives_; i++) {
        char ptype = reinterpret_cast<const char*>(&operators_[i])[0];
        if (ptype >= 28 && ptype <= 37) {
            workspace->has_tma_ops = true;
            break;
        }
    }

    std::memcpy(workspace->graph_buffer_.meta[0], metadata_.data(),
                sizeof(PrimitiveMeta) * num_primitives_);

    for (int i = 0; i < num_primitives_; i++) {
        workspace->graph_buffer_.working_meta[0][i].remaining_deps =
            workspace->graph_buffer_.meta[0][i].num_dependencies_;
    }

    std::memcpy(workspace->graph_buffer_.primitives[0], operators_.data(),
                sizeof(ProxyTrigger) * num_primitives_);

    return true;
}

bool GraphBuilder::requiresCeProxy() const
{
    for (const ProxyTrigger& trigger : operators_) {
        char primitive_type = reinterpret_cast<const char*>(&trigger)[0];
        if (primitive_type >= cuda::cuda_ce_copy_f32 &&
            primitive_type <= cuda::cuda_ce_copy_f8_e5m2) {
            return true;
        }
    }
    return false;
}

bool GraphBuilder::buildFusedDescriptor(FusedStepDescriptor* desc)
{
    if (num_primitives_ == 0) return false;

    std::memset(desc, 0, sizeof(FusedStepDescriptor));

    for (int i = 0; i < num_primitives_; i++) {
        if (metadata_[i].num_dependencies_ > 1 || metadata_[i].num_next_ops_ > 1)
            return false;
    }

    int max_channel = 0;
    for (int i = 0; i < num_primitives_; i++) {
        if (metadata_[i].channel > max_channel)
            max_channel = metadata_[i].channel;
    }
    int num_ch = max_channel + 1;
    desc->num_channels = num_ch;

    std::vector<std::vector<int>> channel_chains(num_ch);
    std::vector<int> chain_starts;

    for (int i = 0; i < num_primitives_; i++) {
        if (metadata_[i].num_dependencies_ == 0 && metadata_[i].device_type == 1)
            chain_starts.push_back(i);
    }

    for (int start : chain_starts) {
        int ch = metadata_[start].channel;
        int cur = start;
        while (cur >= 0) {
            channel_chains[ch].push_back(cur);
            if (metadata_[cur].num_next_ops_ > 0)
                cur = metadata_[cur].next_primitive_index[0];
            else
                cur = -1;
        }
    }

    int total_in_chains = 0;
    for (int c = 0; c < num_ch; c++)
        total_in_chains += channel_chains[c].size();

    int cuda_prims = 0;
    for (int i = 0; i < num_primitives_; i++)
        if (metadata_[i].device_type == 1) cuda_prims++;
    if (total_in_chains != cuda_prims) return false;

    static constexpr char NOTIFY = 22;
    static constexpr char WAIT_NOTIFY = 21;
    static constexpr char TMA_COPY_START = 28;
    static constexpr char TMA_COPY_END = 32;
    static constexpr char TMA_REDUCE_START = 33;
    static constexpr char TMA_REDUCE_END = 37;
    static constexpr char MULTIMEM_REDUCE_START = 11;
    static constexpr char MULTIMEM_REDUCE_END = 15;
    static constexpr char MULTIMEM_STORE_START = 16;
    static constexpr char MULTIMEM_STORE_END = 20;

    auto getPtype = [&](int idx) -> char {
        return reinterpret_cast<const char*>(&operators_[idx])[0];
    };

    auto isSignal = [](char pt) { return pt == NOTIFY || pt == WAIT_NOTIFY; };
    auto isTMA = [](char pt) { return (pt >= TMA_COPY_START && pt <= TMA_COPY_END) ||
                                       (pt >= TMA_REDUCE_START && pt <= TMA_REDUCE_END); };
    auto isMultimem = [](char pt) { return (pt >= MULTIMEM_REDUCE_START && pt <= MULTIMEM_REDUCE_END) ||
                                            (pt >= MULTIMEM_STORE_START && pt <= MULTIMEM_STORE_END); };

    desc->has_tma_ops = false;
    desc->has_multimem_ops = false;
    int step_idx = 0;

    for (int c = 0; c < num_ch; c++) {
        desc->channel_offsets[c] = step_idx;
        const auto& chain = channel_chains[c];
        int pos = 0;

        while (pos < (int)chain.size()) {
            if (step_idx >= MAX_FUSED_STEPS) return false;

            FusedStep& step = desc->steps[step_idx];
            std::memset(&step, 0, sizeof(FusedStep));

            char pt = getPtype(chain[pos]);

            if (pt == NOTIFY) {
                auto* sig = reinterpret_cast<const char*>(&operators_[chain[pos]]);
                step.has_notify = true;
                step.notify_peer_rank = *reinterpret_cast<const int*>(sig + 8);
                step.notify_signal_id = *reinterpret_cast<const int*>(sig + 12);
                pos++;

                if (pos < (int)chain.size() && getPtype(chain[pos]) == WAIT_NOTIFY) {
                    auto* wsig = reinterpret_cast<const char*>(&operators_[chain[pos]]);
                    step.has_wait = true;
                    step.wait_peer_rank = *reinterpret_cast<const int*>(wsig + 8);
                    step.wait_signal_id = *reinterpret_cast<const int*>(wsig + 12);
                    pos++;
                }

                if (pos < (int)chain.size() && !isSignal(getPtype(chain[pos]))) {
                    char dpt = getPtype(chain[pos]);
                    auto* raw = reinterpret_cast<const char*>(&operators_[chain[pos]]);
                    step.primitive_type = dpt;
                    step.peer_rank = *reinterpret_cast<const int*>(raw + 8);
                    step.offset_0 = *reinterpret_cast<const int*>(raw + 12);
                    step.offset_1 = *reinterpret_cast<const int*>(raw + 16);
                    step.offset_2 = *reinterpret_cast<const int*>(raw + 20);
                    step.size = *reinterpret_cast<const int*>(raw + 24);
                    if (isTMA(dpt)) desc->has_tma_ops = true;
                    if (isMultimem(dpt)) desc->has_multimem_ops = true;
                    pos++;
                }
            } else if (pt == WAIT_NOTIFY) {
                auto* wsig = reinterpret_cast<const char*>(&operators_[chain[pos]]);
                step.has_wait = true;
                step.wait_peer_rank = *reinterpret_cast<const int*>(wsig + 8);
                step.wait_signal_id = *reinterpret_cast<const int*>(wsig + 12);
                pos++;

                if (pos < (int)chain.size() && !isSignal(getPtype(chain[pos]))) {
                    char dpt = getPtype(chain[pos]);
                    auto* raw = reinterpret_cast<const char*>(&operators_[chain[pos]]);
                    step.primitive_type = dpt;
                    step.peer_rank = *reinterpret_cast<const int*>(raw + 8);
                    step.offset_0 = *reinterpret_cast<const int*>(raw + 12);
                    step.offset_1 = *reinterpret_cast<const int*>(raw + 16);
                    step.offset_2 = *reinterpret_cast<const int*>(raw + 20);
                    step.size = *reinterpret_cast<const int*>(raw + 24);
                    if (isTMA(dpt)) desc->has_tma_ops = true;
                    if (isMultimem(dpt)) desc->has_multimem_ops = true;
                    pos++;
                }

                if (pos < (int)chain.size() && getPtype(chain[pos]) == NOTIFY) {
                    auto* sig = reinterpret_cast<const char*>(&operators_[chain[pos]]);
                    step.has_notify = true;
                    step.notify_peer_rank = *reinterpret_cast<const int*>(sig + 8);
                    step.notify_signal_id = *reinterpret_cast<const int*>(sig + 12);
                    pos++;
                }
            } else {
                return false;
            }

            step_idx++;
        }
    }

    desc->channel_offsets[num_ch] = step_idx;
    desc->num_steps = step_idx;
    return true;
}

}  // namespace engine_c
