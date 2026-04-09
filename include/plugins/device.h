#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <nlohmann/json.hpp>
#include <engine/ring_buffer.h>
#include <plugins/registry.h>

namespace engine_c {

struct DeviceWorkspace;
struct Workspace;
struct HostProxyState;
struct CeProxyState;
struct FusedStepDescriptor;

// ---------------------------------------------------------------------------
// Channel: provides a set of operations for a specific execution backend.
//   e.g. SmChannel -> sm.copy, sm.reduce, sm.notify, sm.wait_notify
//        TmaChannel -> tma.copy, tma.reduce, tma.copy_notify, tma.reduce_notify
//        RdmaChannel -> rdma.write, rdma.read, rdma.notify, rdma.wait_notify
// ---------------------------------------------------------------------------
class Channel {
   public:
    virtual ~Channel() = default;

    // Channel name: "sm", "tma", "ce", "rdma", "multimem", ...
    virtual const std::string& name() const = 0;

    // List of operation names this channel supports (e.g. {"copy","reduce","notify","wait_notify"})
    virtual std::vector<std::string> supported_ops() const { return {}; }

    // Parse a JSON operation into a ProxyTrigger (per-channel op dispatch)
    virtual ProxyTrigger parse(nlohmann::json& op_info,
                               nlohmann::json& tensor_info) = 0;
};

// ---------------------------------------------------------------------------
// DevicePlugin: manages one device type (cuda, host, rocm, ...).
//   Combines lifecycle, memory, IPC, remote comm, and execution.
//   Method names kept compatible with legacy DeviceBase interface.
// ---------------------------------------------------------------------------
class DevicePlugin {
   public:
    virtual ~DevicePlugin() = default;

    // Device identifier: "Cuda", "Host", "Rocm", ...
    virtual const std::string& deviceName() const = 0;

    // ---- Lifecycle ----
    virtual bool allocatorAvailable() { return false; }

    // ---- Memory ----
    virtual void* allocate(long nbytes) { return nullptr; }
    virtual void deallocate(void*) {}

    // ---- IPC ----
    virtual bool IpcAvailable() { return false; }
    virtual std::string allocateBuffer(void** addr, long size) { return ""; }
    virtual long mapBuffer(std::string& shareable_handle, void** addr) { return 0; }
    virtual void deallocateBuffer(void* addr) {}

    // ---- Remote communication ----
    virtual bool remoteCommAvailable() { return false; }
    virtual std::string activate() { return ""; }
    virtual std::string regBuffer(void* addr, long size) { return ""; }
    virtual void unregBuffer(void* addr) {}
    virtual void regRemoteHandle(std::string& handle) {}
    virtual void connect(std::string handle) {}
    virtual void disconnect(std::string handle) {}

    // ---- Channels (NEW) ----
    virtual std::vector<Channel*> channels() { return {}; }
    Channel* getChannel(const std::string& channel_name);

    // ---- Execution ----
    virtual bool ExecutorAvailable() { return false; }

    // Legacy parse: dispatches to the appropriate channel.
    // Override in plugin OR let the default impl delegate to channels.
    virtual ProxyTrigger parse(nlohmann::json& op_info,
                               nlohmann::json& tensor_info);

    virtual void execute(Workspace* workspace) {}
    virtual void memcpy_sync(void* src, void* dst, long size,
                             DeviceType src_type, DeviceType dst_type) = 0;
    virtual void memcpy_async(void* src, void* dst, long size,
                              DeviceType src_type, DeviceType dst_type,
                              void* stream) {}
    virtual void memset_sync(void* dst, unsigned char val, long size,
                             DeviceType src_type, DeviceType dst_type) {}
    virtual void memset_async(void* dst, unsigned char val, long size,
                              void* stream) {}
    virtual void* getStream() { return nullptr; }
    virtual void streamSync(void* stream) {}

    // ---- NVLS (multicast) ----
    virtual void initNvls(int self_rank, int world_size, size_t buffer_size) {}
    virtual bool nvlsAvailable() const { return false; }
    virtual void* nvlsMcVa() const { return nullptr; }
    virtual void* nvlsPhysVa() const { return nullptr; }
    virtual size_t nvlsBarrierOffset() const { return 0; }
    virtual void prepareNvls(DeviceWorkspace* ws, bool is_allreduce,
                             int scalar_type, long numel) {}

    // ---- Fused descriptor ----
    virtual FusedStepDescriptor* uploadFusedDescriptor(
        const FusedStepDescriptor& host_desc, void* stream) { return nullptr; }

    // ---- Device-specific hooks ----
    virtual HostProxyState* getProxyState() { return nullptr; }
    virtual CeProxyState* getCeProxyState() { return nullptr; }
    virtual void setupTma(
        const std::map<int, std::map<DeviceType, std::tuple<void*, void*>>>& remote_buffers,
        void* self_cuda_buf, void* output_buf, int elem_size, int self_rank,
        void* stream, DeviceWorkspace* host_workspace) {}
};

// ---------------------------------------------------------------------------
// Plugin registry (new interface, coexists with getDev/regDev)
// ---------------------------------------------------------------------------
void registerPlugin(const std::string& name, std::shared_ptr<DevicePlugin> plugin);
DevicePlugin* getPlugin(const std::string& name);
std::shared_ptr<DevicePlugin> getPluginShared(const std::string& name);
std::vector<DevicePlugin*> allPlugins();

// Find a channel across all plugins by "channel_name" (e.g. "sm", "tma", "rdma")
Channel* findChannel(const std::string& channel_name);

// Find a channel from executor string (e.g. "cuda_sm" -> sm channel from cuda plugin)
Channel* findChannelByExecutor(const std::string& executor_str);

}  // namespace engine_c
