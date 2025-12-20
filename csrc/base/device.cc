#include "base/registry.h"
#include <base/device.h>
#include <stdexcept>
#include <string>
#include <map>

namespace engine_c {

bool AllocatorBase::allocatorAvailable() {
  return false;
}

void* AllocatorBase::allocate(long nbytes) {
  throw std::runtime_error("AllocatorBase::allocate() not implemented");
}

void AllocatorBase::deallocate(void* ptr) {
  throw std::runtime_error("AllocatorBase::deallocate() not implemented");
}

bool RemoteCommunicatorBase::remoteCommAvailable() {
  return false;;
}

std::string RemoteCommunicatorBase::activate() {
  throw std::runtime_error("RemoteCommunicatorBase::activate() not implemented");
}

std::string RemoteCommunicatorBase::registerBuffer(void* addr, long size) {
  throw std::runtime_error("RemoteCommunicatorBase::registerBuffer() not implemented");
}

std::string RemoteCommunicatorBase::registerLocal() {
  throw std::runtime_error("RemoteCommunicatorBase::registerLocal() not implemented");
}

void RemoteCommunicatorBase::connect(std::string handle) {
  throw std::runtime_error("RemoteCommunicatorBase::connect() not implemented");
}

void RemoteCommunicatorBase::disconnect(std::string handle) {
  throw std::runtime_error("RemoteCommunicatorBase::disconnect() not implemented");
}

bool IpcCommunicatorBase::IPCAvailable() {
  return false;
}

std::string IpcCommunicatorBase::registerBuffer(void* addr, long size) {
  throw std::runtime_error("IpcCommunicatorBase::registerBuffer() not implemented");
}

std::string IpcCommunicatorBase::allocateIpcBuffer(void** addr, long size) {
  throw std::runtime_error("IpcCommunicatorBase::allocateIpcBuffer() not implemented");
}

long IpcCommunicatorBase::mapBuffer(std::string& shareable_ShareableHandle, void** addr) {
  throw std::runtime_error("IpcCommunicatorBase::mapBuffer() not implemented");
}

std::shared_ptr<DeviceBase> getDev(DeviceType device_type) {
  throw std::runtime_error("getDev() not implemented for device type: " + std::to_string(device_type));
}

static std::map<DeviceType, std::shared_ptr<DeviceBase>> registered_devices_;

void regDev(DeviceType device_type, std::shared_ptr<DeviceBase> device) {
  registered_devices_[device_type] = device;
}

}
