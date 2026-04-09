#include <plugins/device.h>
#include <stdexcept>
#include <unordered_map>
#include <mutex>
#include <fmt/format.h>

namespace engine_c {

// ---------------------------------------------------------------------------
// Plugin registry implementation
// ---------------------------------------------------------------------------
struct PluginRegistry {
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<DevicePlugin>> map;

    static PluginRegistry& instance() {
        static PluginRegistry reg;
        return reg;
    }
};

void registerPlugin(const std::string& name, std::shared_ptr<DevicePlugin> plugin)
{
    auto& reg = PluginRegistry::instance();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.map[name] = std::move(plugin);
}

DevicePlugin* getPlugin(const std::string& name)
{
    auto& reg = PluginRegistry::instance();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.map.find(name);
    if (it == reg.map.end()) return nullptr;
    return it->second.get();
}

std::shared_ptr<DevicePlugin> getPluginShared(const std::string& name)
{
    auto& reg = PluginRegistry::instance();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.map.find(name);
    if (it == reg.map.end()) return nullptr;
    return it->second;
}

std::vector<DevicePlugin*> allPlugins()
{
    auto& reg = PluginRegistry::instance();
    std::lock_guard<std::mutex> lock(reg.mutex);
    std::vector<DevicePlugin*> result;
    result.reserve(reg.map.size());
    for (auto& [name, plugin] : reg.map) {
        result.push_back(plugin.get());
    }
    return result;
}

// ---------------------------------------------------------------------------
// DevicePlugin::getChannel -- find a channel by name within this plugin
// ---------------------------------------------------------------------------
Channel* DevicePlugin::getChannel(const std::string& channel_name)
{
    for (auto* ch : channels()) {
        if (ch->name() == channel_name) return ch;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// DevicePlugin::parse -- default impl: extract channel from primitive name
//   e.g. "sm.copy" -> channel "sm", then delegate to channel->parse()
//   Subclasses can override for custom dispatch.
// ---------------------------------------------------------------------------
ProxyTrigger DevicePlugin::parse(nlohmann::json& op_info,
                                  nlohmann::json& tensor_info)
{
    std::string primitive = op_info["primitive"];
    auto dot = primitive.find('.');
    if (dot != std::string::npos) {
        std::string channel_name = primitive.substr(0, dot);
        Channel* ch = getChannel(channel_name);
        if (ch) return ch->parse(op_info, tensor_info);
    }
    throw std::runtime_error(
        fmt::format("DevicePlugin '{}': no channel found for primitive '{}'",
                     deviceName(), primitive));
}

// ---------------------------------------------------------------------------
// findChannel -- search all plugins for a channel by name
// ---------------------------------------------------------------------------
Channel* findChannel(const std::string& channel_name)
{
    auto& reg = PluginRegistry::instance();
    std::lock_guard<std::mutex> lock(reg.mutex);
    for (auto& [name, plugin] : reg.map) {
        Channel* ch = plugin->getChannel(channel_name);
        if (ch) return ch;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// findChannelByExecutor -- map executor string to channel
//   "cuda_sm"  -> plugin "Cuda", channel "sm"
//   "cuda_tma" -> plugin "Cuda", channel "tma"
//   "cuda_ce"  -> plugin "Cuda", channel "ce"
//   "rdma"     -> plugin "Host", channel "rdma"
//   "host"     -> plugin "Host" (no specific channel)
// ---------------------------------------------------------------------------
Channel* findChannelByExecutor(const std::string& executor_str)
{
    auto& reg = PluginRegistry::instance();
    std::lock_guard<std::mutex> lock(reg.mutex);

    auto underscore = executor_str.find('_');
    if (underscore != std::string::npos) {
        std::string plugin_prefix = executor_str.substr(0, underscore);
        std::string channel_suffix = executor_str.substr(underscore + 1);

        for (auto& [name, plugin] : reg.map) {
            std::string lower_name = name;
            std::transform(lower_name.begin(), lower_name.end(),
                          lower_name.begin(), ::tolower);
            if (lower_name == plugin_prefix) {
                Channel* ch = plugin->getChannel(channel_suffix);
                if (ch) return ch;
            }
        }
    }

    for (auto& [name, plugin] : reg.map) {
        Channel* ch = plugin->getChannel(executor_str);
        if (ch) return ch;
    }

    return nullptr;
}

}  // namespace engine_c
