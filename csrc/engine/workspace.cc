#include <engine/workspace.h>

namespace engine_c {

BufferManager& BufferManager::getInstance()
{
    static BufferManager instance;
    return instance;
}

void BufferManager::registerDevice(DeviceType device_type, void* buffer, void* signals)
{
    getInstance().buffers_[device_type] = buffer;
    getInstance().signals_[device_type] = signals;
}

void* BufferManager::getSignals(DeviceType device_type)
{
    auto& signals = getInstance().signals_;
    auto it = signals.find(device_type);
    if (it != signals.end()) {
        return it->second;
    }
    return nullptr;
}

void* BufferManager::getBuffer(DeviceType device_type)
{
    auto& buffers = getInstance().buffers_;
    auto it = buffers.find(device_type);
    if (it != buffers.end()) {
        return it->second;
    }
    return nullptr;
}

}  // namespace engine_c
