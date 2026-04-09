#include <plugins/base.h>
#include <plugins/registry.h>
#include <stdexcept>

namespace engine_c {

// ---------------------------------------------------------------------------
// getDev / regDev -- delegate to the string-keyed plugin registry
//   via TypeRegistry::getTypeName() for DeviceType -> name lookup.
// ---------------------------------------------------------------------------

std::shared_ptr<DeviceBase> getDev(DeviceType device_type) {
    std::string name = TypeRegistry::getTypeName(device_type);
    if (name.empty()) {
        throw std::runtime_error("Device type not registered in TypeRegistry");
    }
    auto plugin = getPluginShared(name);
    if (!plugin) {
        throw std::runtime_error("Device type not registered");
    }
    return plugin;
}

void regDev(DeviceType device_type, std::shared_ptr<DeviceBase> device) {
    std::string name = TypeRegistry::getTypeName(device_type);
    if (!name.empty()) {
        registerPlugin(name, device);
    }
}

}  // namespace engine_c
