#include <plugins/registry.h>

namespace engine_c {

PrimitiveType TypeRegistry::registerPrimitive(const std::string& name) { return getInstance()._registerPrimitiveType(name); }
DeviceType TypeRegistry::registerDeviceType(const std::string& name) { return getInstance()._registerDeviceType(name); }

void TypeRegistry::registerCompatibility(const std::string& device_name, const std::string& type_name)
{
    auto& instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.mutex_);
    DeviceType device_type = instance._getTypeId(device_name);
    GeneralType target_type = instance._getTypeId(type_name);
    instance._registerCompatibility(device_type, target_type);
}

std::string TypeRegistry::getTypeName(GeneralType type) { return getInstance()._getTypeName(type); }
GeneralType TypeRegistry::getTypeId(const std::string& name) { return getInstance()._getTypeId(name); }

const std::set<DeviceType>& TypeRegistry::getDeviceTypes()
{
    auto& instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.mutex_);
    return instance.device_type_set_;
}

const std::set<GeneralType>& TypeRegistry::getCompatibleTypes(DeviceType device_type)
{
    auto& instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.mutex_);
    return instance.compatibility_map_.at(device_type);
}

const std::set<DeviceType>& TypeRegistry::getCompatibleDevices(GeneralType type)
{
    auto& instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.mutex_);
    return instance.reverse_compatibility_map_.at(type);
}

PrimitiveType TypeRegistry::_registerPrimitiveType(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = name_cache_.find(name);
    if (it != name_cache_.end()) return it->second;
    PrimitiveType new_id = next_type_id_++;
    name_cache_[name] = new_id;
    primitive_cache_[new_id] = name;
    return new_id;
}

DeviceType TypeRegistry::_registerDeviceType(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = name_cache_.find(name);
    if (it != name_cache_.end()) return it->second;
    DeviceType new_id = next_type_id_++;
    name_cache_[name] = new_id;
    device_cache_[new_id] = name;
    device_type_set_.insert(new_id);
    compatibility_map_[new_id] = std::set<GeneralType>();
    return new_id;
}

void TypeRegistry::_registerCompatibility(DeviceType device_type, GeneralType target_type)
{
    compatibility_map_[device_type].insert(target_type);
    reverse_compatibility_map_[target_type].insert(device_type);
}

std::string TypeRegistry::_getTypeName(GeneralType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = primitive_cache_.find(type);
    if (it != primitive_cache_.end()) return it->second;
    auto dit = device_cache_.find(type);
    if (dit != device_cache_.end()) return dit->second;
    return "";
}

GeneralType TypeRegistry::_getTypeId(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = name_cache_.find(name);
    if (it != name_cache_.end()) return it->second;
    return -1;
}

}  // namespace engine_c
