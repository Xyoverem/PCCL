#include <plugins/base.h>
#include <plugins/registry.h>
#include <engine/workspace.h>
#include <common.h>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <string>
#include <nlohmann/json.hpp>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <fmt/format.h>
#include <stdexcept>
#include <engine/workspace.h>

#ifdef PCCL_RDMA_ENABLED
#include <infiniband/verbs.h>
#include "host_proxy.h"
#endif
#include "ce_proxy.h"

namespace engine_c::host {

struct MemInfo
{
    std::string filename;
    long size;
    int fd;

    std::string serialize()
    {
        nlohmann::json j;
        j["filename"] = filename;
        j["size"] = size;
        return j.dump();
    }

    static MemInfo deserialize(const std::string& mem_info)
    {
        auto j = nlohmann::json::parse(mem_info);
        MemInfo r;
        r.filename = j["filename"];
        r.size = j["size"];
        r.fd = shm_open(r.filename.c_str(), O_RDWR, 0666);
        TORCH_CHECK(r.fd != -1);
        return r;
    }
};

static std::atomic<int> shm_counter{0};

static std::string generate_shm_name()
{
    int rank = std::stoi(common::Environs::getEnv("RANK").data());
    int id = shm_counter.fetch_add(1);
    return fmt::format("/pccl_shm_{}_{}", rank, id);
}

static std::string device_name = "Host";

class HostDevice : public DeviceBase
{
   public:
    HostDevice() = default;

    const std::string& deviceName() const override {
        return device_name;
    }
    virtual ~HostDevice()
    {
        ce_proxy_.stop();
        std::lock_guard<std::mutex> lock(mem_mutex);
#ifdef PCCL_RDMA_ENABLED
        if (host_proxy_) {
            host_proxy_->stop();
            host_proxy_.reset();
        }
        for (auto& [addr, mr] : mrs_) {
            ibv_dereg_mr(mr);
        }
        for (auto& [rank, qp] : qps_) {
            ibv_destroy_qp(qp);
        }
        if (cq_) ibv_destroy_cq(cq_);
        if (pd_) ibv_dealloc_pd(pd_);
        if (ib_ctx_) ibv_close_device(ib_ctx_);
#endif
        for (auto& [ptr, mem_info] : mem_map) {
            munmap(ptr, mem_info.size);
            if (mem_info.fd != -1) {
                close(mem_info.fd);
            }
            shm_unlink(mem_info.filename.c_str());
        }
        for (void* ptr : mem_ptrs) {
            std::free(ptr);
        }
    }

    bool allocatorAvailable() override { return true; }

    void* allocate(long nbytes) override
    {
        void* ptr = std::aligned_alloc(1024, nbytes);
        mem_ptrs.insert(ptr);
        std::memset(ptr, 0, nbytes);
        return ptr;
    }

    void deallocate(void* ptr) override
    {
        mem_ptrs.erase(ptr);
        std::free(ptr);
    }

    bool IpcAvailable() override { return true; }

    std::string allocateBuffer(void** addr, long size) override
    {
        TORCH_CHECK(size > 0);
        std::string filename = generate_shm_name();
        int shm_fd = shm_open(filename.c_str(), O_CREAT | O_RDWR, 0666);
        TORCH_CHECK(shm_fd != -1 and ftruncate(shm_fd, size) != -1);
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        TORCH_CHECK(ptr != MAP_FAILED);

        std::lock_guard<std::mutex> lock(mem_mutex);
        mem_map[ptr] = {filename, size, shm_fd};
        *addr = ptr;
        return mem_map[ptr].serialize();
    }

    long mapBuffer(std::string& shareable_handle, void** addr) override
    {
        MemInfo mem_info = MemInfo::deserialize(shareable_handle);
        void* ptr =
            mmap(nullptr, mem_info.size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_info.fd, 0);
        TORCH_CHECK(ptr != MAP_FAILED)
        *addr = ptr;
        std::lock_guard<std::mutex> lock(mem_mutex);
        mem_map[ptr] = mem_info;
        return mem_info.size;
    }

    void deallocateBuffer(void* addr) override
    {
        auto item = mem_map.find(addr);
        if (item == mem_map.end()) return;
        auto mem_info = item->second;
        munmap(addr, mem_info.size);
        if (mem_info.fd != -1) close(mem_info.fd);
        shm_unlink(mem_info.filename.c_str());
        mem_map.erase(addr);
    }

    void memcpy_sync(void* src, void* dst, long size, DeviceType src_type,
                     DeviceType dst_type) override
    {
        std::memcpy(dst, src, size);
    }

    void memset_sync(void* dst, unsigned char val, long size, DeviceType src_type,
                     DeviceType dst_type) override
    {
        std::memset(dst, val, size);
    }

    bool ExecutorAvailable() override { return false; }

    ProxyTrigger parse(nlohmann::json &op_info, nlohmann::json &tensor_info) override {
        ProxyTrigger t;
        std::memset(&t, 0, sizeof(t));
        return t;
    }

    void execute(Workspace* workspace) override {}

#ifdef PCCL_RDMA_ENABLED
    bool remoteCommAvailable() override
    {
        int num_devices = 0;
        struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
        if (dev_list) ibv_free_device_list(dev_list);
        return num_devices > 0;
    }

    std::string activate() override
    {
        int num_devices = 0;
        struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
        TORCH_CHECK(dev_list && num_devices > 0, "No RDMA devices found");

        ib_ctx_ = ibv_open_device(dev_list[0]);
        ibv_free_device_list(dev_list);
        TORCH_CHECK(ib_ctx_, "Failed to open RDMA device");

        pd_ = ibv_alloc_pd(ib_ctx_);
        TORCH_CHECK(pd_, "Failed to allocate protection domain");

        cq_ = ibv_create_cq(ib_ctx_, 256, nullptr, nullptr, 0);
        TORCH_CHECK(cq_, "Failed to create completion queue");

        ib_port_ = 1;
        struct ibv_port_attr port_attr;
        TORCH_CHECK(ibv_query_port(ib_ctx_, ib_port_, &port_attr) == 0,
                     "Failed to query IB port");
        lid_ = port_attr.lid;

        union ibv_gid gid;
        TORCH_CHECK(ibv_query_gid(ib_ctx_, ib_port_, 0, &gid) == 0,
                     "Failed to query GID");

        nlohmann::json j;
        j["lid"] = lid_;
        j["gid"] = std::string(reinterpret_cast<char*>(&gid), sizeof(gid));
        return j.dump();
    }

    std::string regBuffer(void* addr, long size) override
    {
        int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        struct ibv_mr* mr = ibv_reg_mr(pd_, addr, size, access);
        TORCH_CHECK(mr, fmt::format("ibv_reg_mr failed for addr={} size={}", addr, size));

        std::lock_guard<std::mutex> lock(mem_mutex);
        mrs_[addr] = mr;

        // Start the host proxy if not already running, using the first registered buffer
        if (!host_proxy_) {
            host_proxy_ = std::make_unique<HostProxy>();
            RdmaTransport transport;
            transport.cq = cq_;
            transport.local_mr = mr;
            transport.local_buffer = addr;
            host_proxy_->start(std::move(transport));
        }

        nlohmann::json j;
        j["rkey"] = mr->rkey;
        j["lkey"] = mr->lkey;
        j["addr"] = reinterpret_cast<uint64_t>(addr);
        j["size"] = size;
        return j.dump();
    }

    void unregBuffer(void* addr) override
    {
        std::lock_guard<std::mutex> lock(mem_mutex);
        auto it = mrs_.find(addr);
        if (it != mrs_.end()) {
            ibv_dereg_mr(it->second);
            mrs_.erase(it);
        }
    }

    void regRemoteHandle(std::string& handle) override
    {
        auto j = nlohmann::json::parse(handle);
        if (!j.contains("rank")) return;
        int rank = j["rank"];

        // Buffer handle (from regBuffer): contains rkey, addr
        if (j.contains("rkey") && j.contains("addr")) {
            remote_rkeys_[rank] = j["rkey"];
            remote_addrs_[rank] = j["addr"];

            // If QP exists and is in INIT, and we have peer connection info,
            // attempt full RTR/RTS transition now
            auto qp_it = qps_.find(rank);
            auto lid_it = peer_lids_.find(rank);
            auto gid_it = peer_gids_.find(rank);
            if (qp_it != qps_.end() && lid_it != peer_lids_.end() && gid_it != peer_gids_.end()) {
                // Rebuild handle with QP number for connect()
                nlohmann::json connect_j;
                connect_j["rank"] = rank;
                connect_j["lid"] = lid_it->second;
                connect_j["gid"] = std::string(reinterpret_cast<char*>(&gid_it->second),
                                                sizeof(union ibv_gid));
                connect_j["qp_num"] = qp_it->second->qp_num;
                std::string connect_handle = connect_j.dump();
                // Note: connect() will skip QP creation since it already exists
                // and will do RTR/RTS since qp_num is present
                // But we need PEER's qp_num, not ours. Skip for now.
                // Full QP exchange requires a 2nd round-trip protocol extension.
            }

            // If proxy is running and peer is fully connected, register peer
            if (host_proxy_ && qp_it != qps_.end()) {
                RdmaPeerInfo info;
                info.qp = qp_it->second;
                info.remote_rkey = j["rkey"];
                info.remote_addr = j["addr"];
                host_proxy_->addPeer(rank, info);
            }
        }
        // Connection handles ({lid, gid}) are processed in connect()
    }

    void connect(std::string handle) override
    {
        auto j = nlohmann::json::parse(handle);
        int peer_rank = j["rank"];
        uint16_t peer_lid = j["lid"];
        std::string gid_bytes = j["gid"];

        union ibv_gid peer_gid;
        std::memcpy(&peer_gid, gid_bytes.data(), sizeof(peer_gid));

        // Create QP for this peer if it doesn't exist yet
        if (qps_.find(peer_rank) == qps_.end()) {
            createQpForPeer(peer_rank);
        }
        struct ibv_qp* qp = qps_[peer_rank];

        // If peer included their QP number, do full RTR/RTS transition
        if (j.contains("qp_num")) {
            uint32_t peer_qpn = j["qp_num"];

            struct ibv_qp_attr attr = {};
            attr.qp_state = IBV_QPS_RTR;
            attr.path_mtu = IBV_MTU_4096;
            attr.dest_qp_num = peer_qpn;
            attr.rq_psn = 0;
            attr.max_dest_rd_atomic = 1;
            attr.min_rnr_timer = 12;
            attr.ah_attr.dlid = peer_lid;
            attr.ah_attr.sl = 0;
            attr.ah_attr.src_path_bits = 0;
            attr.ah_attr.port_num = ib_port_;
            attr.ah_attr.is_global = 1;
            attr.ah_attr.grh.dgid = peer_gid;
            attr.ah_attr.grh.sgid_index = 0;
            attr.ah_attr.grh.hop_limit = 64;

            int rtr_flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                            IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
            TORCH_CHECK(ibv_modify_qp(qp, &attr, rtr_flags) == 0,
                         "Failed to transition QP to RTR");

            std::memset(&attr, 0, sizeof(attr));
            attr.qp_state = IBV_QPS_RTS;
            attr.timeout = 14;
            attr.retry_cnt = 7;
            attr.rnr_retry = 7;
            attr.sq_psn = 0;
            attr.max_rd_atomic = 1;

            int rts_flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
            TORCH_CHECK(ibv_modify_qp(qp, &attr, rts_flags) == 0,
                         "Failed to transition QP to RTS");

            // Register peer with the host proxy for RDMA command dispatch
            if (host_proxy_) {
                auto rkey_it = remote_rkeys_.find(peer_rank);
                auto addr_it = remote_addrs_.find(peer_rank);
                if (rkey_it != remote_rkeys_.end() && addr_it != remote_addrs_.end()) {
                    RdmaPeerInfo info;
                    info.qp = qp;
                    info.remote_rkey = rkey_it->second;
                    info.remote_addr = addr_it->second;
                    host_proxy_->addPeer(peer_rank, info);
                }
            }
        }
        // Store peer connection info for later use
        peer_lids_[peer_rank] = peer_lid;
        peer_gids_[peer_rank] = peer_gid;

        PCCL_LOG_INFO("RDMA: connect() for peer rank {} (QP state: {})",
                       peer_rank, j.contains("qp_num") ? "RTS" : "INIT");
    }

    void disconnect(std::string handle) override
    {
        auto j = nlohmann::json::parse(handle);
        int peer_rank = j["rank"];
        auto it = qps_.find(peer_rank);
        if (it != qps_.end()) {
            struct ibv_qp_attr attr = {};
            attr.qp_state = IBV_QPS_RESET;
            ibv_modify_qp(it->second, &attr, IBV_QP_STATE);
        }
    }

    struct ibv_qp* createQpForPeer(int peer_rank)
    {
        struct ibv_qp_init_attr qp_init = {};
        qp_init.send_cq = cq_;
        qp_init.recv_cq = cq_;
        qp_init.qp_type = IBV_QPT_RC;
        qp_init.cap.max_send_wr = 128;
        qp_init.cap.max_recv_wr = 128;
        qp_init.cap.max_send_sge = 1;
        qp_init.cap.max_recv_sge = 1;

        struct ibv_qp* qp = ibv_create_qp(pd_, &qp_init);
        TORCH_CHECK(qp, "Failed to create QP");

        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = ib_port_;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;

        int init_flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        TORCH_CHECK(ibv_modify_qp(qp, &attr, init_flags) == 0,
                     "Failed to transition QP to INIT");

        qps_[peer_rank] = qp;
        return qp;
    }

    struct ibv_pd* getPd() const { return pd_; }
    struct ibv_cq* getCq() const { return cq_; }
    struct ibv_qp* getQp(int peer_rank) const
    {
        auto it = qps_.find(peer_rank);
        return it != qps_.end() ? it->second : nullptr;
    }
    struct ibv_mr* getMr(void* addr) const
    {
        auto it = mrs_.find(addr);
        return it != mrs_.end() ? it->second : nullptr;
    }
    uint32_t getRemoteRkey(int peer_rank) const
    {
        auto it = remote_rkeys_.find(peer_rank);
        return it != remote_rkeys_.end() ? it->second : 0;
    }
    uint64_t getRemoteAddr(int peer_rank) const
    {
        auto it = remote_addrs_.find(peer_rank);
        return it != remote_addrs_.end() ? it->second : 0;
    }

    HostProxyState* getProxyState() override
    {
        return host_proxy_ ? host_proxy_->state() : nullptr;
    }
#endif

    CeProxyState* getCeProxyState() override
    {
        if (!ce_proxy_.state()) {
            ce_proxy_.start();
        }
        return ce_proxy_.state();
    }

   private:
    std::map<void*, MemInfo> mem_map;
    std::unordered_set<void*> mem_ptrs;
    std::mutex mem_mutex;
    CeProxy ce_proxy_;

#ifdef PCCL_RDMA_ENABLED
    struct ibv_context* ib_ctx_ = nullptr;
    struct ibv_pd* pd_ = nullptr;
    struct ibv_cq* cq_ = nullptr;
    uint8_t ib_port_ = 1;
    uint16_t lid_ = 0;
    std::unordered_map<int, struct ibv_qp*> qps_;
    std::unordered_map<void*, struct ibv_mr*> mrs_;
    std::unordered_map<int, uint32_t> remote_rkeys_;
    std::unordered_map<int, uint64_t> remote_addrs_;
    std::unordered_map<int, uint16_t> peer_lids_;
    std::unordered_map<int, union ibv_gid> peer_gids_;
    std::unique_ptr<HostProxy> host_proxy_;
#endif
};

struct TMP_STATIC_INITIALIZER
{
    TMP_STATIC_INITIALIZER()
    {
        auto device = TypeRegistry::registerDeviceType(device_name);
        regDev(device, std::make_shared<HostDevice>());
    }
};

static TMP_STATIC_INITIALIZER _____tmp;

}  // namespace engine_c::host
