#pragma once

#include <vector>
#include <map>
#include <string>
#include <string_view>

namespace engine_c::common {

class Environs
{

    Environs();
    std::string_view _getEnv(const std::string& env);

   public:
    static Environs& getInstance()
    {
        static Environs instance;
        return instance;
    }

    static const std::string_view getEnv(const std::string& env)
    {
        return getInstance()._getEnv(env);
    }

    static const std::string getEnvOrDefault(const std::string& env, const std::string& default_val)
    {
        char* val = std::getenv(env.c_str());
        return val != nullptr ? std::string(val) : default_val;
    }

    static const std::vector<std::string>& listOpt()
    {
        return getInstance().opts_;
    }

    static void registerOpt(std::string option)
    {
        getInstance().opts_.push_back(std::move(option));
    }

   private:
    std::map<std::string, std::string> env_cache_;
    std::vector<std::string> opts_;
};

}  // namespace engine_c::common
