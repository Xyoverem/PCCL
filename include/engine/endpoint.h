#pragma once

#include <string>
#include <map>
#include <mutex>

namespace engine_c {

class Endpoint
{
    Endpoint();
    static Endpoint& getInstance()
    {
        static Endpoint ins;
        return ins;
    }

   public:
    static std::string& serialize();
    static void add_info(const std::string& key, const std::string& value);

   private:
    std::mutex mutex_;
    std::string endpoint_handle_;
    std::map<std::string, std::string> endpoint_infos_;
};

}  // namespace engine_c
