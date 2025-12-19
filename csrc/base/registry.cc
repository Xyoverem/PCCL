#include <base/registry.h>

namespace engine_c {

PrimitiveType TypeRegistry::registerPrimitive(std::string_view name) {
  TypeRegistry& registry = getInstance();
  return registry._registerPrimitiveType(name);
}

DataType TypeRegistry::registerDataType(std::string_view name) {
  TypeRegistry& registry = getInstance();
  return registry._registerDataType(name);
}

ComputeType TypeRegistry::registerComputeType(std::string_view name) {
  TypeRegistry& registry = getInstance();
  return registry._registerComputeType(name);
}

DeviceType TypeRegistry::registerDeviceType(std::string_view name) {
  TypeRegistry& registry = getInstance();
  return registry._registerDeviceType(name);
}

ExecutorType TypeRegistry::registerExecutorType(std::string_view name) {
  TypeRegistry& registry = getInstance();
  return registry._registerExecutorType(name);
}

void TypeRegistry::registerCompatibility(std::string_view executor_name, std::string_view type_name) {
  TypeRegistry& registry = getInstance();
  registry._registerCompatibility(executor_name, type_name);
}

std::string_view TypeRegistry::getTypeName(GeneralType type) {
  TypeRegistry& registry = getInstance();
  return registry._getTypeName(type);
}

GeneralType TypeRegistry::getTypeId(std::string_view name) {
  TypeRegistry& registry = getInstance();
  return registry._getTypeId(name);
}

const std::set<DeviceType>& TypeRegistry::getDeviceTypes() {
  TypeRegistry& registry = getInstance();

  static std::set<DeviceType> device_types = {
    registry.getTypeId("CPU"),
    registry.getTypeId("CUDA"),
    registry.getTypeId("RDMA"),
    registry.getTypeId("ROCM")
  };

  return device_types;
}

const std::set<GeneralType>& TypeRegistry::getCompatibleTypes(ExecutorType executor_type) {
  TypeRegistry& registry = getInstance();
  return registry._getCompatibleTypes(executor_type);
}

const std::set<ExecutorType>& TypeRegistry::getCompatibleExecutors(GeneralType type) {
  TypeRegistry& registry = getInstance();
  return registry._getCompatibleExecutors(type);
}

void TypeRegistry::clear() {
  TypeRegistry& registry = getInstance();
  registry.clear();
}

TypeRegistry::TypeRegistry() : next_type_id_(0) {}

PrimitiveType TypeRegistry::_registerPrimitiveType(std::string_view name) {
  GeneralType type_id = next_type_id_++;

  std::lock_guard<std::mutex> lock(mutex_);
  primitive_cache_[type_id] = std::string(name);
  name_cache_[std::string(name)] = type_id;

  return type_id;
}

DataType TypeRegistry::_registerDataType(std::string_view name) {
  GeneralType type_id = next_type_id_++;

  std::lock_guard<std::mutex> lock(mutex_);
  data_cache_[type_id] = std::string(name);
  name_cache_[std::string(name)] = type_id;

  return type_id;
}

ComputeType TypeRegistry::_registerComputeType(std::string_view name) {
  GeneralType type_id = next_type_id_++;

  std::lock_guard<std::mutex> lock(mutex_);
  compute_cache_[type_id] = std::string(name);
  name_cache_[std::string(name)] = type_id;

  return type_id;
}

DeviceType TypeRegistry::_registerDeviceType(std::string_view name) {
  GeneralType type_id = next_type_id_++;

  std::lock_guard<std::mutex> lock(mutex_);
  device_cache_[type_id] = std::string(name);
  name_cache_[std::string(name)] = type_id;

  return type_id;
}

ExecutorType TypeRegistry::_registerExecutorType(std::string_view name) {
  GeneralType type_id = next_type_id_++;

  std::lock_guard<std::mutex> lock(mutex_);
  executor_cache_[type_id] = std::string(name);
  name_cache_[std::string(name)] = type_id;

  return type_id;
}

void TypeRegistry::_registerCompatibilityInternal(ExecutorType executor_type, GeneralType target_type) {
  compatibility_map_[executor_type].insert(target_type);
  reverse_compatibility_map_[target_type].insert(executor_type);
}

std::string_view TypeRegistry::_getTypeName(GeneralType type) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = primitive_cache_.find(type);
  if (it != primitive_cache_.end()) {
    return it->second;
  }

  it = data_cache_.find(type);
  if (it != data_cache_.end()) {
    return it->second;
  }

  it = compute_cache_.find(type);
  if (it != compute_cache_.end()) {
    return it->second;
  }

  it = device_cache_.find(type);
  if (it != device_cache_.end()) {
    return it->second;
  }

  it = executor_cache_.find(type);
  if (it != executor_cache_.end()) {
    return it->second;
  }

  return "";
}

GeneralType TypeRegistry::_getTypeId(std::string_view name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = name_cache_.find(std::string(name));
  if (it != name_cache_.end()) {
    return it->second;
  }
  return 0;
}

const std::set<GeneralType>& TypeRegistry::_getCompatibleTypes(ExecutorType executor_type) const {
  static std::set<GeneralType> empty_set;
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = compatibility_map_.find(executor_type);
  return (it != compatibility_map_.end()) ? it->second : empty_set;
}

const std::set<ExecutorType>& TypeRegistry::_getCompatibleExecutors(GeneralType type) const {
  static std::set<ExecutorType> empty_set;
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = reverse_compatibility_map_.find(type);
  return (it != reverse_compatibility_map_.end()) ? it->second : empty_set;
}

void TypeRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  name_cache_.clear();
  primitive_cache_.clear();
  data_cache_.clear();
  compute_cache_.clear();
  device_cache_.clear();
  executor_cache_.clear();
  compatibility_map_.clear();
  reverse_compatibility_map_.clear();
  next_type_id_ = 0;
}

}

