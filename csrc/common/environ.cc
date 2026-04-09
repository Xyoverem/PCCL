#include <common.h>
#include <cstdlib>
#include <unistd.h>
#include <stdlib.h>

namespace engine_c::common {

Environs::Environs() {}

std::string_view Environs::_getEnv(const std::string& env)
{
    auto it = env_cache_.find(env);
    if (it != env_cache_.end()) {
        return it->second;
    }

    char* val = std::getenv(env.c_str());
    TORCH_CHECK(val != nullptr, "Environment ", env, " is not set.");
    auto insert_result = env_cache_.emplace(env, val);
    return insert_result.first->second;
}

}  // namespace engine_c::common
