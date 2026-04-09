#pragma once

#include <plugins/device.h>

namespace engine_c {

// ---------------------------------------------------------------------------
// DeviceBase: backward-compatible alias for DevicePlugin.
//   Existing plugins (CudaDevice, HostDevice) inherit from this.
//   New code should use DevicePlugin directly.
// ---------------------------------------------------------------------------
using DeviceBase = DevicePlugin;

// Legacy registry (delegates to new plugin registry internally)
std::shared_ptr<DeviceBase> getDev(DeviceType device_type);
void regDev(DeviceType device_type, std::shared_ptr<DeviceBase> device);

}  // namespace engine_c
