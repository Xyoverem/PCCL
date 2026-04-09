#pragma once

#include <string>
#include <map>
#include <set>
#include <atomic>
#include <mutex>

namespace engine_c {

using GeneralType = int;
using PrimitiveType = GeneralType;
using DeviceType = GeneralType;

class TypeRegistry
{
   public:
    static PrimitiveType registerPrimitive(const std::string& name);
    static DeviceType registerDeviceType(const std::string& name);

    static void registerCompatibility(const std::string& device_name, const std::string& type_name);

    static std::string getTypeName(GeneralType type);
    static GeneralType getTypeId(const std::string& name);

    static const std::set<DeviceType>& getDeviceTypes();

    static const std::set<GeneralType>& getCompatibleTypes(DeviceType device_type);
    static const std::set<DeviceType>& getCompatibleDevices(GeneralType type);

   private:
    TypeRegistry() = default;
    ~TypeRegistry() = default;
    TypeRegistry(const TypeRegistry&) = delete;
    TypeRegistry& operator=(const TypeRegistry&) = delete;

    static TypeRegistry& getInstance()
    {
        static TypeRegistry instance;
        return instance;
    }

    PrimitiveType _registerPrimitiveType(const std::string& name);
    DeviceType _registerDeviceType(const std::string& name);
    void _registerCompatibility(DeviceType device_type, GeneralType target_type);
    std::string _getTypeName(GeneralType type) const;
    GeneralType _getTypeId(const std::string& name) const;

    mutable std::mutex mutex_;

    std::map<std::string, GeneralType> name_cache_;
    std::map<PrimitiveType, std::string> primitive_cache_;
    std::map<DeviceType, std::string> device_cache_;
    std::map<DeviceType, std::set<GeneralType>> compatibility_map_;
    std::map<GeneralType, std::set<DeviceType>> reverse_compatibility_map_;

    std::atomic<GeneralType> next_type_id_{0};
    std::set<DeviceType> device_type_set_;
};

}  // namespace engine_c
