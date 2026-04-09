#include "engine/endpoint.h"
#include <nlohmann/json.hpp>

namespace engine_c {

Endpoint::Endpoint() {}

std::string& Endpoint::serialize()
{
    Endpoint& instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.mutex_);
    nlohmann::json j;

    for (const auto& pair : instance.endpoint_infos_) {
        j[pair.first] = pair.second;
    }

    instance.endpoint_handle_ = j.dump();
    return instance.endpoint_handle_;
}

void Endpoint::add_info(const std::string& key, const std::string& value)
{
    Endpoint& instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.mutex_);
    instance.endpoint_infos_[key] = value;
}

}  // namespace engine_c
