#include <engine/engine.h>
#include <engine/endpoint.h>
#include <engine/workspace.h>
#include <engine/memory_layout.h>
#include <engine/graph_builder.h>
#include <engine/primitive.h>
#include <engine/fused_step.h>
#include <plugins/registry.h>
#include <plugins/base.h>
#include <common.h>
#include <common/nvtx.h>
#include <common/serialize.h>
#include <common/socket_utils.h>
#include <fmt/format.h>
#include <string>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <set>
#include <memory>
#include <thread>

static_assert(offsetof(engine_c::DeviceWorkspace, ring_buffers_) <= engine_c::DeviceWorkspace::PERCALL_COPY_SIZE,
              "PERCALL_COPY_SIZE too small for per-call fields");
#include <vector>

namespace engine_c {

class Engine::Impl
{
   public:
    Impl() {}
    ~Impl() = default;

    void first_call_setup(Workspace* workspace, DeviceWorkspace* host_workspace);
    void prepare_and_launch(const std::string& name, at::Tensor& input, at::Tensor& output, bool sync);

   public:
    int rank_;
    std::string node_id;
    std::mutex engine_mutex_;
    std::map<int, std::string> endpoints_mappings_;
    std::map<int, std::map<std::string, std::string>> endpoint_attrs_;
    std::map<int, std::map<DeviceType, std::tuple<void*, void*>>> remote_buffers_;
    std::map<int, std::map<std::tuple<DeviceType, DeviceType>, std::tuple<void*, void*>>>
        remote_comm_buffer_;
    std::map<std::string, std::unique_ptr<GraphBuilder>> graphs_;
    std::map<std::string, std::unique_ptr<Workspace>> workspaces_;
    std::mutex workspace_mutex_;
    std::set<std::string> graph_cached_ops_;
    std::map<std::string, FusedStepDescriptor> fused_descriptors_;
    std::map<std::string, std::string> collective_type_;

    // Cached output-gap ranges for the partial D2D output copy optimisation.
    // Each gap is a contiguous element range [offset, offset+count) in the
    // working buffer that executeTmaCopy did NOT dual-write to the output
    // tensor (i.e. ranges produced only by reduce steps).
    struct ElemRange { long offset; long count; };
    std::map<std::string, std::vector<ElemRange>> output_gaps_;

    bool nvls_initialized_ = false;
};

static std::string read_boot_id() {
    std::ifstream f("/proc/sys/kernel/random/boot_id");
    std::string s;
    if (f.is_open()) std::getline(f, s);
    return s;
}

Engine& Engine::getInstance()
{
    static Engine instance;
    return instance;
}

Engine::~Engine() = default;

Engine::Engine()
{
    PCCL_LOG_INFO("Engine initializing...");

    impl_ = std::make_unique<Engine::Impl>();
    impl_->rank_ = std::stoi(common::Environs::getEnv("RANK").data());
    std::string node_id = read_boot_id();
    Endpoint::add_info("node_id", node_id);
    impl_->node_id = node_id;

    // Configure buffer size from environment variable if set
    auto buf_size_env = common::Environs::getEnvOrDefault("PCCL_BUFFER_SIZE", "");
    if (!buf_size_env.empty()) {
        long requested = std::stol(buf_size_env);
        if (requested >= MinBufferSize) {
            getBufferSize() = requested;
            PCCL_LOG_INFO("Buffer size set to {} bytes from PCCL_BUFFER_SIZE", requested);
        } else {
            PCCL_LOG_WARN("PCCL_BUFFER_SIZE={} too small, using minimum {}",
                          requested, MinBufferSize);
            getBufferSize() = MinBufferSize;
        }
    }

    common::SocketInstance::start(impl_->rank_);

    // Broadcast buffer size info for peers
    Endpoint::add_info("buffer_size", std::to_string(BufferSize));

    auto device_types = TypeRegistry::getDeviceTypes();

    std::vector<std::tuple<DeviceType, std::shared_ptr<DeviceBase>>> comm_devices;
    std::vector<std::tuple<DeviceType, void*, void*>> pccl_runtime_buffers;

    for (auto device_type = device_types.begin(); device_type != device_types.end();
         device_type++) {

        auto device = getDev(*device_type);
        std::string device_name = TypeRegistry::getTypeName(*device_type);

        if (device->remoteCommAvailable()) {
            auto remote_handle = device->activate();
            std::string remote_handle_entry = fmt::format("remote_handle.{}", device_name);
            Endpoint::add_info(remote_handle_entry, remote_handle);
            comm_devices.push_back(std::make_tuple(*device_type, device));
        }

        if (device->IpcAvailable()) {
            void *buffer, *signals;
            auto ipc_data_handle = device->allocateBuffer(&buffer, BufferSize + SignalSize);
            signals = reinterpret_cast<void*>(static_cast<char*>(buffer) + BufferSize);
            BufferManager::registerDevice(*device_type, buffer, signals);
            pccl_runtime_buffers.push_back(std::make_tuple(*device_type, buffer, signals));

            std::string ipc_data_entry = fmt::format("ipc_handle.{}", device_name);
            Endpoint::add_info(ipc_data_entry, ipc_data_handle);

            std::string mem_key = fmt::format("mem_buffer.{}", device_name);
            Endpoint::add_info(mem_key, common::serialize(&buffer, sizeof(buffer)));
        } else if (device->allocatorAvailable()) {
            void* buffer = device->allocate(BufferSize + SignalSize);
            void* signals = static_cast<void*>(static_cast<char*>(buffer) + BufferSize);
            BufferManager::registerDevice(*device_type, buffer, signals);
            pccl_runtime_buffers.push_back(std::make_tuple(*device_type, buffer, signals));

            std::string mem_key = fmt::format("mem_buffer.{}", device_name);
            Endpoint::add_info(mem_key, common::serialize(&buffer, sizeof(buffer)));
        }
    }

    for (auto& [comm_dev_type, comm_device] : comm_devices) {
        for (auto& [mem_dev_type, buffer, signals] : pccl_runtime_buffers) {
            std::string buffer_handle = comm_device->regBuffer(buffer, BufferSize + SignalSize);
            std::string comm_dev_name = TypeRegistry::getTypeName(comm_dev_type);
            std::string mem_dev_name = TypeRegistry::getTypeName(mem_dev_type);
            std::string buffer_entry =
                fmt::format("buffer_handle.{}.{}", comm_dev_name, mem_dev_name);
            Endpoint::add_info(buffer_entry, buffer_handle);
        }
    }

    PCCL_LOG_INFO("Engine initialized successfully, rank: {}", impl_->rank_);
}

bool Engine::regOp(const std::string& name, const std::string& filename)
{
    PCCL_LOG_INFO("Registering operation '{}' from file: {}", name, filename);

    auto graph_builder = std::make_unique<GraphBuilder>();
    if (!graph_builder->loadFromFile(filename)) {
        PCCL_LOG_ERROR("Failed to load graph from file: {}", filename);
        return false;
    }
    impl_->graphs_[name] = std::move(graph_builder);

    std::lock_guard<std::mutex> lock(impl_->workspace_mutex_);

    auto workspace = std::make_unique<Workspace>();
    auto device_types = TypeRegistry::getDeviceTypes();

    for (auto dt : device_types) {
        std::string dt_name = TypeRegistry::getTypeName(dt);
        if (dt_name == "Host") {
            workspace->device_a_ = dt;
        } else if (dt_name == "Cuda") {
            workspace->device_b_ = dt;
        }
    }

    void* host_buffer = BufferManager::getBuffer(workspace->device_a_);
    void* cuda_buffer = BufferManager::getBuffer(workspace->device_b_);
    void* host_signals = BufferManager::getSignals(workspace->device_a_);
    void* cuda_signals = BufferManager::getSignals(workspace->device_b_);

    MemoryLayout::initialize(host_buffer, cuda_buffer, host_signals, cuda_signals, workspace.get());

    impl_->workspaces_[name] = std::move(workspace);
    PCCL_LOG_INFO("Operation '{}' registered successfully", name);
    return true;
}

// Common setup logic for first execution
void Engine::Impl::first_call_setup(Workspace* workspace, DeviceWorkspace* host_workspace) {
    static constexpr long wksp_size = 4096;

    for (auto& rb : remote_buffers_) {
        host_workspace->peer_addr[0][rb.first] = std::get<0>(rb.second[workspace->device_a_]);
        host_workspace->peer_addr[1][rb.first] = std::get<0>(rb.second[workspace->device_b_]);
        host_workspace->peer_signals[0][rb.first] = std::get<1>(rb.second[workspace->device_a_]);
        host_workspace->peer_signals[1][rb.first] = std::get<1>(rb.second[workspace->device_b_]);
    }

    host_workspace->peer_addr[0][rank_] = BufferManager::getBuffer(workspace->device_a_);
    host_workspace->peer_addr[1][rank_] = BufferManager::getBuffer(workspace->device_b_);
    host_workspace->peer_signals[0][rank_] = BufferManager::getSignals(workspace->device_a_);
    host_workspace->peer_signals[1][rank_] = BufferManager::getSignals(workspace->device_b_);

    getDev(workspace->device_b_)->memcpy_sync(
        host_workspace->peer_addr[0], host_workspace->peer_b_addr[0],
        wksp_size * 4, workspace->device_a_, workspace->device_b_);

    QueueMeta init_qmeta[4];
    for (int i = 0; i < 4; i++) {
        init_qmeta[i].head = 0;
        init_qmeta[i].tail = 0;
        init_qmeta[i].capacity = host_workspace->ring_buffers_[0].meta_a_->capacity;
        init_qmeta[i].size = 0;
    }
    getDev(workspace->device_b_)->memcpy_sync(&init_qmeta[0], host_workspace->ring_buffers_[0].meta_b_,
        sizeof(QueueMeta), workspace->device_a_, workspace->device_b_);
    getDev(workspace->device_b_)->memcpy_sync(&init_qmeta[1], host_workspace->ring_buffers_[1].meta_a_,
        sizeof(QueueMeta), workspace->device_a_, workspace->device_b_);
    getDev(workspace->device_b_)->memcpy_sync(&init_qmeta[2], host_workspace->ring_buffers_[3].meta_a_,
        sizeof(QueueMeta), workspace->device_a_, workspace->device_b_);
    getDev(workspace->device_b_)->memcpy_sync(&init_qmeta[3], host_workspace->ring_buffers_[3].meta_b_,
        sizeof(QueueMeta), workspace->device_a_, workspace->device_b_);

    // Initialize per-channel ready queue metadata on CUDA
    long ch_queue_cap = MemoryLayout::CHANNEL_QUEUE_DATA_SIZE / sizeof(ProxyTrigger);
    for (int c = 0; c < MAX_CHANNELS; c++) {
        QueueMeta ch_init;
        ch_init.head = 0;
        ch_init.tail = 0;
        ch_init.capacity = static_cast<int>(ch_queue_cap);
        ch_init.size = 0;
        getDev(workspace->device_b_)->memcpy_sync(&ch_init, host_workspace->channel_ready_queues[c].meta_b_,
            sizeof(QueueMeta), workspace->device_a_, workspace->device_b_);
    }

    // Setup NVLS multicast via device plugin (one-time)
    if (!nvls_initialized_) {
        nvls_initialized_ = true;
        int world_size = static_cast<int>(remote_buffers_.size()) + 1;
        getDev(workspace->device_b_)->initNvls(rank_, world_size,
            static_cast<size_t>(BufferSize));
    }

    auto cuda_dev = getDev(workspace->device_b_);
    if (cuda_dev->nvlsAvailable()) {
        host_workspace->nvls_mc_va_ = cuda_dev->nvlsMcVa();
        host_workspace->nvls_phys_va_ = cuda_dev->nvlsPhysVa();
        host_workspace->nvls_barrier_offset_ = cuda_dev->nvlsBarrierOffset();
        host_workspace->nvls_world_size_ = static_cast<int>(remote_buffers_.size()) + 1;
        host_workspace->nvls_self_rank_ = rank_;
    }
}

void Engine::Impl::prepare_and_launch(const std::string& name,
                                at::Tensor& input, at::Tensor& output, bool sync) {
    auto it = graphs_.find(name);
    TORCH_CHECK(it != graphs_.end(), fmt::format("Unknown operation: {}", name));
    auto& graph_builder = it->second;

    std::lock_guard<std::mutex> lock(workspace_mutex_);

    auto ws_it = workspaces_.find(name);
    TORCH_CHECK(ws_it != workspaces_.end(),
                fmt::format("Workspace not found for operation: {}. Was regOp called?", name));

    Workspace* workspace = ws_it->second.get();
    DeviceWorkspace* host_workspace = workspace->dev_workspace_a;

    bool is_first_call = (host_workspace->total_primitives == -1);
    if (is_first_call) {
        first_call_setup(workspace, host_workspace);
    }

    bool graph_cached = graph_cached_ops_.count(name) > 0;

    // Eagerly resolve collective type from graph builder (needed before
    // first graph build to correctly set output_buffer_).
    if (collective_type_.count(name) == 0) {
        collective_type_[name] = graph_builder->collectiveType();
    }
    bool graph_is_allreduce = (collective_type_[name] == "allreduce");
    std::string coll_type = collective_type_[name];

    void** saved_peer_addr_0 = host_workspace->peer_addr[0];
    void** saved_peer_addr_1 = host_workspace->peer_addr[1];
    void** saved_peer_signals_0 = host_workspace->peer_signals[0];
    void** saved_peer_signals_1 = host_workspace->peer_signals[1];

    host_workspace->input_buffer_ = (void*)input.data_ptr();
    host_workspace->io_copy_bytes_ = input.nbytes();
    // Disable dual-write for alltoall: SM copies write to a scratch area
    // beyond the tensor, so the output descriptor must be null to prevent
    // the kernel from writing out-of-bounds into the output tensor.
    if (coll_type == "alltoall") {
        host_workspace->output_buffer_ = nullptr;
    } else {
        host_workspace->output_buffer_ = (void*)output.data_ptr();
    }

    auto cuda_dev = getDev(workspace->device_b_);
    void* stream = cuda_dev->getStream();

    // Determine if NVLS path should be used for this call (device plugin decides)
    cuda_dev->prepareNvls(host_workspace, graph_is_allreduce,
                          static_cast<int>(input.scalar_type()), input.numel());
    bool use_nvls_this_call = host_workspace->use_nvls_ && host_workspace->nvls_mc_va_;

    if (!graph_cached) {
        graph_builder->build(host_workspace);

        size_t meta_size = sizeof(PrimitiveMeta) * host_workspace->total_primitives;
        size_t prim_size = sizeof(ProxyTrigger) * host_workspace->total_primitives;

        cuda_dev->memcpy_async(host_workspace->graph_buffer_.meta[0],
                               host_workspace->graph_buffer_.meta[1],
                               meta_size, workspace->device_a_, workspace->device_b_, stream);
        cuda_dev->memcpy_async(host_workspace->graph_buffer_.primitives[0],
                               host_workspace->graph_buffer_.primitives[1],
                               prim_size, workspace->device_a_, workspace->device_b_, stream);

        // Process CPU entry triggers (CUDA-only graphs skip this loop)
        for (int i = 0; i < host_workspace->total_primitives; i++) {
            PrimitiveMeta* meta = &host_workspace->graph_buffer_.meta[0][i];
            if (meta->num_dependencies_ == 0 && meta->device_type == 0) {
                ProxyTrigger* ops = (ProxyTrigger*)host_workspace->graph_buffer_.primitives[0];
                QueueMeta* qmeta = host_workspace->ring_buffers_[2].meta_a_;
                ProxyTrigger* qbuf = host_workspace->ring_buffers_[2].buffer_a_;
                int next_tail = (qmeta->tail + 1) % qmeta->capacity;
                qbuf[qmeta->tail] = ops[i];
                qmeta->tail = next_tail;
            }
        }

        // Try to build fused step descriptor for linear-chain DAGs
        FusedStepDescriptor host_fused;
        bool fused_enabled = (std::getenv("PCCL_DISABLE_FUSED") == nullptr);
        if (fused_enabled && graph_builder->buildFusedDescriptor(&host_fused)) {
            fused_descriptors_[name] = host_fused;
            auto* dev_ptr = cuda_dev->uploadFusedDescriptor(fused_descriptors_[name], stream);
            host_workspace->fused_desc_ = dev_ptr;
            host_workspace->use_fused_ = true;
            PCCL_LOG_INFO("Fused step executor enabled for '{}': {} steps, {} channels",
                          name, host_fused.num_steps, host_fused.num_channels);

            // Pre-compute output gaps: ranges NOT covered by TMA-copy steps.
            // executeTmaCopy dual-writes copy-step ranges to the output
            // tensor, so only the complement needs a D2D copy afterwards.
            // TMA copy primitive types: 28..32.
            if (host_fused.has_tma_ops) {
                struct R { long s, e; };
                std::vector<R> covered;
                for (int si = 0; si < host_fused.num_steps; si++) {
                    char pt = host_fused.steps[si].primitive_type;
                    if (pt >= 28 && pt <= 32) {  // TMA copy types
                        long s = host_fused.steps[si].offset_1;
                        long e = s + host_fused.steps[si].size;
                        covered.push_back({s, e});
                    }
                }
                std::sort(covered.begin(), covered.end(),
                          [](const R& a, const R& b) { return a.s < b.s; });
                // Merge overlapping ranges
                std::vector<R> merged;
                for (auto& r : covered) {
                    if (!merged.empty() && r.s <= merged.back().e)
                        merged.back().e = std::max(merged.back().e, r.e);
                    else
                        merged.push_back(r);
                }
                // Gaps are the complement of merged ranges within
                // [0, total_elems).  total_elems isn't known until
                // runtime, so store the merged coverage instead and
                // compute gaps at call-time.  However, for any correct
                // allreduce the tensor is evenly partitioned and we can
                // derive gaps cheaply from the sorted coverage list.
                // Store gaps as ranges between merged intervals.
                // We use total=0 as sentinel; gaps are recomputed on
                // first call when total_elems is known.
                output_gaps_[name] = {};  // placeholder
                // We can already compute inter-interval gaps (all except
                // the possible trailing gap which depends on total_elems).
                auto& gaps = output_gaps_[name];
                long prev = 0;
                for (auto& r : merged) {
                    if (r.s > prev)
                        gaps.push_back({prev, r.s - prev});
                    prev = r.e;
                }
                // sentinel: mark that we need to check trailing gap at
                // runtime.  Store prev as a negative-count sentinel.
                gaps.push_back({prev, -1});  // -1 = trailing sentinel
                PCCL_LOG_INFO("Output-gap optimisation: {} gap(s) for '{}'",
                              gaps.size() - 1, name);
            }
        } else {
            host_workspace->fused_desc_ = nullptr;
            host_workspace->use_fused_ = false;
        }

        graph_cached_ops_.insert(name);
    }

    if (host_workspace->has_tma_ops) {
        NvtxRange nvtx_tma{"setupTma"};
        int input_elem_size = static_cast<int>(input.element_size());
        void* self_cuda_buf = BufferManager::getBuffer(workspace->device_b_);
        void* output_ptr = (coll_type == "alltoall") ? nullptr : (void*)output.data_ptr();
        cuda_dev->setupTma(
            remote_buffers_, self_cuda_buf, output_ptr,
            input_elem_size, rank_,
            stream, host_workspace);
    } else {
        host_workspace->tma_desc = nullptr;
    }

    // Reset HOST-SIDE ring buffer head/tail (needed every iteration)
    host_workspace->ring_buffers_[0].meta_a_->head = 0;
    host_workspace->ring_buffers_[0].meta_a_->tail = 0;
    host_workspace->ring_buffers_[1].meta_b_->head = 0;
    host_workspace->ring_buffers_[1].meta_b_->tail = 0;
    host_workspace->ring_buffers_[2].meta_a_->head = 0;
    host_workspace->ring_buffers_[2].meta_a_->tail = 0;
    host_workspace->ring_buffers_[2].meta_b_->head = 0;
    host_workspace->ring_buffers_[2].meta_b_->tail = 0;

    // Swap pointers for H2D copy
    host_workspace->peer_addr[0] = reinterpret_cast<void**>(host_workspace->peer_b_addr[0]);
    host_workspace->peer_addr[1] = reinterpret_cast<void**>(host_workspace->peer_b_addr[1]);
    host_workspace->peer_signals[0] = reinterpret_cast<void**>(host_workspace->peer_b_signals[0]);
    host_workspace->peer_signals[1] = reinterpret_cast<void**>(host_workspace->peer_b_signals[1]);

    // Inject host proxy state for RDMA ops (nullptr if no RDMA)
    auto host_dev = getDev(workspace->device_a_);
    host_workspace->host_proxy = host_dev->getProxyState();
    host_workspace->ce_proxy = host_dev->getCeProxyState();

    { NvtxRange nvtx_ws{"h2d_workspace"};
    size_t copy_size = is_first_call ? MemoryLayout::DEV_WORKSPACE_SIZE
                                     : DeviceWorkspace::PERCALL_COPY_SIZE;
    cuda_dev->memcpy_async(workspace->dev_workspace_a, workspace->dev_workspace_b,
                           copy_size,
                           workspace->device_a_, workspace->device_b_, stream);
    }

    // Restore host pointers
    host_workspace->peer_addr[0] = saved_peer_addr_0;
    host_workspace->peer_addr[1] = saved_peer_addr_1;
    host_workspace->peer_signals[0] = saved_peer_signals_0;
    host_workspace->peer_signals[1] = saved_peer_signals_1;

    { NvtxRange nvtx_in{"d2d_input_copy"};
    if (use_nvls_this_call) {
        cuda_dev->memcpy_async((void*)input.data_ptr(), cuda_dev->nvlsPhysVa(),
                               input.nbytes(), workspace->device_b_, workspace->device_b_, stream);
    } else if (coll_type == "allgather") {
        size_t chunk_bytes = input.nbytes();
        size_t dst_offset = static_cast<size_t>(rank_) * chunk_bytes;
        cuda_dev->memcpy_async(
            (void*)input.data_ptr(),
            (char*)host_workspace->self_addr[1] + dst_offset,
            chunk_bytes, workspace->device_b_, workspace->device_b_, stream);
    } else {
        cuda_dev->memcpy_async((void*)input.data_ptr(), host_workspace->self_addr[1],
                               input.nbytes(), workspace->device_b_, workspace->device_b_, stream);
    }
    }

    { NvtxRange nvtx_exe{"execute"};
    cuda_dev->execute(workspace);
    }

    { NvtxRange nvtx_out{"d2d_output_copy"};
    if (use_nvls_this_call) {
        cuda_dev->memcpy_async(cuda_dev->nvlsPhysVa(), (void*)output.data_ptr(),
                               output.nbytes(), workspace->device_b_, workspace->device_b_, stream);
    } else if (host_workspace->use_fused_ && graph_is_allreduce
               && output_gaps_.count(name)) {
        // ---- Partial-copy optimisation for fused allreduce ----
        // executeTmaCopy already dual-wrote the COPY-step ranges to the
        // output tensor.  Only the pre-computed gap ranges (produced
        // exclusively by reduce steps) need a D2D copy.
        int elem_sz = input.element_size();
        long total_elems = input.numel();
        auto& gaps = output_gaps_[name];
        for (auto& g : gaps) {
            long count = g.count;
            if (count == -1) {
                // trailing sentinel: fill to end of tensor
                count = total_elems - g.offset;
                if (count <= 0) continue;
            }
            size_t off = static_cast<size_t>(g.offset) * elem_sz;
            size_t len = static_cast<size_t>(count) * elem_sz;
            cuda_dev->memcpy_async(
                (char*)host_workspace->self_addr[1] + off,
                (char*)output.data_ptr() + off,
                len, workspace->device_b_, workspace->device_b_, stream);
        }
    } else if (coll_type == "reduce_scatter") {
        int world_size = static_cast<int>(remote_buffers_.size()) + 1;
        size_t chunk_bytes = input.nbytes() / world_size;
        size_t src_offset = static_cast<size_t>(rank_) * chunk_bytes;
        cuda_dev->memcpy_async(
            (char*)host_workspace->self_addr[1] + src_offset,
            (void*)output.data_ptr(),
            chunk_bytes, workspace->device_b_, workspace->device_b_, stream);
    } else if (coll_type == "alltoall") {
        // Assemble output from original area (own chunk) + scratch area.
        // Own chunk[rank] is at rank*chunk in the IPC buffer.
        // Each partner p's data is at tensor_bytes + p*chunk in the scratch area.
        int world_size = static_cast<int>(remote_buffers_.size()) + 1;
        size_t tensor_bytes = input.nbytes();
        size_t chunk_bytes = tensor_bytes / world_size;
        for (int p = 0; p < world_size; p++) {
            size_t src;
            if (p == rank_) {
                src = static_cast<size_t>(rank_) * chunk_bytes;
            } else {
                src = tensor_bytes + static_cast<size_t>(p) * chunk_bytes;
            }
            size_t dst = static_cast<size_t>(p) * chunk_bytes;
            cuda_dev->memcpy_async(
                (char*)host_workspace->self_addr[1] + src,
                (char*)output.data_ptr() + dst,
                chunk_bytes, workspace->device_b_, workspace->device_b_, stream);
        }
    } else {
        cuda_dev->memcpy_async(host_workspace->self_addr[1], (void*)output.data_ptr(),
                               output.nbytes(), workspace->device_b_, workspace->device_b_, stream);
    }
    }

    if (sync) {
        cuda_dev->streamSync(stream);
    }
}

void Engine::exeOp(const std::string& name, at::Tensor& input, at::Tensor& output)
{
    impl_->prepare_and_launch(name, input, output, true);
}

void Engine::exeOpAsync(const std::string& name, at::Tensor& input, at::Tensor& output)
{
    impl_->prepare_and_launch(name, input, output, false);
}

void Engine::syncOp(const std::string& name)
{
    std::lock_guard<std::mutex> lock(impl_->workspace_mutex_);
    auto ws_it = impl_->workspaces_.find(name);
    if (ws_it == impl_->workspaces_.end()) return;
    Workspace* workspace = ws_it->second.get();
    auto dev = getDev(workspace->device_b_);
    void* stream = dev->getStream();
    dev->streamSync(stream);
}

void Engine::resetSignals(const std::string& name)
{
    std::lock_guard<std::mutex> lock(impl_->workspace_mutex_);

    auto ws_it = impl_->workspaces_.find(name);
    if (ws_it == impl_->workspaces_.end()) return;

    Workspace* workspace = ws_it->second.get();
    DeviceWorkspace* host_workspace = workspace->dev_workspace_a;

    auto dev = getDev(workspace->device_b_);
    dev->memset_sync(
        host_workspace->self_signals[1], 0, SignalSize,
        workspace->device_b_, workspace->device_b_);
    std::memset(host_workspace->self_signals[0], 0, SignalSize);
    void* stream = dev->getStream();
    dev->streamSync(stream);
}

std::string& Engine::exportEndpoint()
{
    return Endpoint::serialize();
}

void Engine::updateEndpoint(int rank, std::string& endpoint)
{
    PCCL_LOG_INFO("Updating endpoint for rank: {}", rank);

    impl_->endpoints_mappings_[rank] = endpoint;
    impl_->endpoint_attrs_[rank] = nlohmann::json::parse(endpoint);

    auto& endpoint_attrs = impl_->endpoint_attrs_[rank];
    auto device_types = TypeRegistry::getDeviceTypes();
    std::map<DeviceType, std::string> device_type_names;

    for (auto device_type : device_types) {
        device_type_names[device_type] = TypeRegistry::getTypeName(device_type);
    }

    std::set<DeviceType> comm_devs;
    impl_->remote_buffers_[rank] = {};

    for (auto device_type : device_types) {
        auto device = getDev(device_type);
        std::string device_name = device_type_names[device_type];
        if (device->remoteCommAvailable()) {
            std::string remote_handle_key = fmt::format("remote_handle.{}", device_name);
            comm_devs.insert(device_type);
            if (endpoint_attrs.find(remote_handle_key) != endpoint_attrs.end()) {
                std::string remote_handle = endpoint_attrs[remote_handle_key];
                // Inject peer rank into the handle for devices that need it
                auto handle_json = nlohmann::json::parse(remote_handle);
                handle_json["rank"] = rank;
                std::string enriched_handle = handle_json.dump();
                device->connect(enriched_handle);
            }
        }

        if (impl_->node_id == endpoint_attrs["node_id"] and device->IpcAvailable()) {
            std::string ipc_handle_key = fmt::format("ipc_handle.{}", device_name);
            if (endpoint_attrs.find(ipc_handle_key) != endpoint_attrs.end()) {
                std::string ipc_handle = endpoint_attrs[ipc_handle_key];
                void* remote_buffer;
                TORCH_CHECK(device->mapBuffer(ipc_handle, &remote_buffer) >=
                            BufferSize + SignalSize);
                void* remote_signals =
                    reinterpret_cast<void*>(static_cast<char*>(remote_buffer) + BufferSize);
                impl_->remote_buffers_[rank][device_type] =
                    std::make_tuple(remote_buffer, remote_signals);
            }
        }
    }

    impl_->remote_comm_buffer_[rank] = {};
    for (auto comm_device_type : comm_devs) {
        auto comm_device = getDev(comm_device_type);
        std::string comm_device_name = device_type_names[comm_device_type];
        for (auto mem_device_type : device_types) {
            std::string mem_device_name = device_type_names[mem_device_type];
            std::string buffer_handle_key =
                fmt::format("buffer_handle.{}.{}", comm_device_name, mem_device_name);

            if (endpoint_attrs.find(buffer_handle_key) != endpoint_attrs.end()) {
                std::string remote_buffer_handle = endpoint_attrs[buffer_handle_key];
                // Inject peer rank into buffer handle for devices that need it
                auto buf_json = nlohmann::json::parse(remote_buffer_handle);
                buf_json["rank"] = rank;
                std::string enriched_buf_handle = buf_json.dump();
                comm_device->regRemoteHandle(enriched_buf_handle);
                std::tuple<DeviceType, DeviceType> remote_key =
                    std::make_tuple(comm_device_type, mem_device_type);
                void* remote_buffer = reinterpret_cast<void*>(
                    std::stol(endpoint_attrs[fmt::format("mem_buffer.{}", mem_device_name)]));
                void* remote_signals =
                    reinterpret_cast<void*>(static_cast<char*>(remote_buffer) + BufferSize);
                impl_->remote_comm_buffer_[rank][remote_key] =
                    std::make_tuple(remote_buffer, remote_signals);
            }
        }
    }

    PCCL_LOG_INFO("Endpoint updated for rank: {}", rank);
}

}  // namespace engine_c
