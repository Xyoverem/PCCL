#include <network/network.h>
#include <infiniband/verbs.h>
#include <cstring>
#include <memory>
#include <algorithm>

namespace engine_c {
namespace network {

class CompleteRdmaConnection : public RdmaConnection {
public:
  CompleteRdmaConnection();
  ~CompleteRdmaConnection() override;

  bool connect(const NetworkAddress& address) override;
  void disconnect() override;

  bool registerMemoryRegion(void* addr, size_t size, uint32_t& lkey, uint32_t& rkey) override;
  bool unregisterMemoryRegion(void* addr) override;

  bool postRDMAWrite(uint64_t remote_addr, uint32_t rkey, void* local_addr, size_t length);
  bool postRDMARead(uint64_t remote_addr, uint32_t rkey, void* local_addr, size_t length);
  bool postAtomicFetchAndAdd(uint64_t remote_addr, uint32_t rkey, uint64_t add_value, uint64_t* result);

  bool setupQueuePairs(int max_send_wr, int max_recv_wr, int max_send_sge, int max_recv_sge);
  bool modifyQueuePairToInit();
  bool modifyQueuePairToRTR(uint32_t remote_qpn, uint16_t dlid, uint8_t port_num);
  bool modifyQueuePairToRTS();

  int pollCompletionQueue(int max_entries, ibv_wc* wc);
  bool postReceive(void* addr, size_t length, uint32_t lkey, uint64_t wr_id);

  bool enableReliableConnected();
  bool enableReliableDatagram();
  bool enableExtendedTransport();

private:
  struct MemoryRegion {
    void* addr;
    size_t size;
    ibv_mr* mr;
    uint32_t lkey;
    uint32_t rkey;
  };

  struct QueuePairConfig {
    int max_send_wr;
    int max_recv_wr;
    int max_send_sge;
    int max_recv_sge;
    int max_inline_data;
  };

  bool initializeVerbsContext();
  bool createProtectionDomain();
  bool createCompletionQueues(int cqe_size);
  bool setupAddressVectors();

  QueuePairConfig qp_config_;
  std::unordered_map<void*, std::unique_ptr<MemoryRegion>> memory_regions_;

  ibv_device* device_;
  ibv_context* context_;
  ibv_pd* protection_domain_;
  ibv_cq* send_cq_;
  ibv_cq* recv_cq_;
  ibv_qp* queue_pair_;
  ibv_comp_channel* comp_channel_;

  ibv_port_attr port_attr_;
  ibv_device_attr device_attr_;
  ibv_gid gid_;

  uint32_t qp_num_;
  uint16_t lid_;
  uint8_t port_num_;
  uint8_t gid_index_;

  bool verbs_initialized_;
  bool qp_configured_;
};

CompleteRdmaConnection::CompleteRdmaConnection()
  : device_(nullptr), context_(nullptr), protection_domain_(nullptr),
    send_cq_(nullptr), recv_cq_(nullptr), queue_pair_(nullptr), comp_channel_(nullptr),
    qp_num_(0), lid_(0), port_num_(1), gid_index_(0),
    verbs_initialized_(false), qp_configured_(false) {

  qp_config_.max_send_wr = 1024;
  qp_config_.max_recv_wr = 1024;
  qp_config_.max_send_sge = 4;
  qp_config_.max_recv_sge = 4;
  qp_config_.max_inline_data = 64;
}

CompleteRdmaConnection::~CompleteRdmaConnection() {
  disconnect();
}

bool CompleteRdmaConnection::initializeVerbsContext() {
  struct ibv_device** device_list = ibv_get_device_list(nullptr);
  if (!device_list) {
    return false;
  }

  device_ = device_list[0];
  if (!device_) {
    ibv_free_device_list(device_list);
    return false;
  }

  context_ = ibv_open_device(device_);
  if (!context_) {
    ibv_free_device_list(device_list);
    return false;
  }

  ibv_free_device_list(device_list);

  if (ibv_query_device(context_, &device_attr_) != 0) {
    return false;
  }

  for (uint8_t port = 1; port <= device_attr_.phys_port_cnt; ++port) {
    if (ibv_query_port(context_, port, &port_attr_) == 0 &&
        port_attr_.state == IBV_PORT_ACTIVE) {
      port_num_ = port;
      break;
    }
  }

  if (port_num_ == 0) {
    return false;
  }

  if (ibv_query_gid(context_, port_num_, gid_index_, &gid_) != 0) {
    return false;
  }

  lid_ = port_attr_.lid;
  verbs_initialized_ = true;

  return true;
}

bool CompleteRdmaConnection::createProtectionDomain() {
  if (!context_) {
    return false;
  }

  protection_domain_ = ibv_alloc_pd(context_);
  return protection_domain_ != nullptr;
}

bool CompleteRdmaConnection::createCompletionQueues(int cqe_size) {
  if (!context_) {
    return false;
  }

  send_cq_ = ibv_create_cq(context_, cqe_size, nullptr, nullptr, 0);
  if (!send_cq_) {
    return false;
  }

  recv_cq_ = ibv_create_cq(context_, cqe_size, nullptr, nullptr, 0);
  if (!recv_cq_) {
    ibv_destroy_cq(send_cq_);
    send_cq_ = nullptr;
    return false;
  }

  comp_channel_ = ibv_create_comp_channel(context_);
  if (!comp_channel_) {
    ibv_destroy_cq(send_cq_);
    ibv_destroy_cq(recv_cq_);
    send_cq_ = nullptr;
    recv_cq_ = nullptr;
    return false;
  }

  ibv_req_notify_cq(send_cq_, 0);
  ibv_req_notify_cq(recv_cq_, 0);

  return true;
}

bool CompleteRdmaConnection::setupQueuePairs(int max_send_wr, int max_recv_wr, int max_send_sge, int max_recv_sge) {
  qp_config_.max_send_wr = max_send_wr;
  qp_config_.max_recv_wr = max_recv_wr;
  qp_config_.max_send_sge = max_send_sge;
  qp_config_.max_recv_sge = max_recv_sge;

  if (!protection_domain_ || !send_cq_ || !recv_cq_) {
    return false;
  }

  struct ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));

  qp_init_attr.send_cq = send_cq_;
  qp_init_attr.recv_cq = recv_cq_;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.cap.max_send_wr = qp_config_.max_send_wr;
  qp_init_attr.cap.max_recv_wr = qp_config_.max_recv_wr;
  qp_init_attr.cap.max_send_sge = qp_config_.max_send_sge;
  qp_init_attr.cap.max_recv_sge = qp_config_.max_recv_sge;
  qp_init_attr.cap.max_inline_data = qp_config_.max_inline_data;

  queue_pair_ = ibv_create_qp(protection_domain_, &qp_init_attr);
  if (!queue_pair_) {
    return false;
  }

  qp_num_ = queue_pair_->qp_num;
  qp_configured_ = true;

  return true;
}

bool CompleteRdmaConnection::modifyQueuePairToInit() {
  if (!queue_pair_) {
    return false;
  }

  struct ibv_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(qp_attr));

  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = port_num_;
  qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  int ret = ibv_modify_qp(queue_pair_, &qp_attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

  return ret == 0;
}

bool CompleteRdmaConnection::modifyQueuePairToRTR(uint32_t remote_qpn, uint16_t dlid, uint8_t port_num) {
  if (!queue_pair_) {
    return false;
  }

  struct ibv_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(qp_attr));

  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.path_mtu = IBV_MTU_4096;
  qp_attr.dest_qp_num = remote_qpn;
  qp_attr.rq_psn = 0;
  qp_attr.max_dest_rd_atomic = 16;
  qp_attr.min_rnr_timer = 12;
  qp_attr.ah_attr.is_global = 1;
  qp_attr.ah_attr.dlid = dlid;
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = port_num;
  qp_attr.ah_attr.grh.dgid = gid_;
  qp_attr.ah_attr.grh.flow_label = 0;
  qp_attr.ah_attr.grh.sgid_index = gid_index_;
  qp_attr.ah_attr.grh.hop_limit = 1;
  qp_attr.ah_attr.grh.traffic_class = 0;

  int ret = ibv_modify_qp(queue_pair_, &qp_attr,
                          IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                          IBV_QP_MIN_RNR_TIMER | IBV_QP_AV);

  return ret == 0;
}

bool CompleteRdmaConnection::modifyQueuePairToRTS() {
  if (!queue_pair_) {
    return false;
  }

  struct ibv_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(qp_attr));

  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.timeout = 14;
  qp_attr.retry_cnt = 7;
  qp_attr.rnr_retry = 7;
  qp_attr.sq_psn = 0;
  qp_attr.max_rd_atomic = 16;

  int ret = ibv_modify_qp(queue_pair_, &qp_attr,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);

  return ret == 0;
}

bool CompleteRdmaConnection::connect(const NetworkAddress& address) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (status_ != ConnectionStatus::DISCONNECTED) {
    return false;
  }

  if (!initializeVerbsContext()) {
    return false;
  }

  if (!createProtectionDomain()) {
    return false;
  }

  if (!createCompletionQueues(4096)) {
    return false;
  }

  if (!setupQueuePairs(1024, 1024, 4, 4)) {
    return false;
  }

  if (!modifyQueuePairToInit()) {
    return false;
  }

  status_ = ConnectionStatus::CONNECTED;
  remote_addr_ = address;

  return true;
}

void CompleteRdmaConnection::disconnect() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (queue_pair_) {
    ibv_destroy_qp(queue_pair_);
    queue_pair_ = nullptr;
  }

  if (send_cq_) {
    ibv_destroy_cq(send_cq_);
    send_cq_ = nullptr;
  }

  if (recv_cq_) {
    ibv_destroy_cq(recv_cq_);
    recv_cq_ = nullptr;
  }

  if (comp_channel_) {
    ibv_destroy_comp_channel(comp_channel_);
    comp_channel_ = nullptr;
  }

  if (protection_domain_) {
    ibv_dealloc_pd(protection_domain_);
    protection_domain_ = nullptr;
  }

  if (context_) {
    ibv_close_device(context_);
    context_ = nullptr;
  }

  memory_regions_.clear();
  status_ = ConnectionStatus::DISCONNECTED;
  verbs_initialized_ = false;
  qp_configured_ = false;
}

bool CompleteRdmaConnection::registerMemoryRegion(void* addr, size_t size, uint32_t& lkey, uint32_t& rkey) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!protection_domain_ || !addr) {
    return false;
  }

  auto it = memory_regions_.find(addr);
  if (it != memory_regions_.end()) {
    lkey = it->second->lkey;
    rkey = it->second->rkey;
    return true;
  }

  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  ibv_mr* mr = ibv_reg_mr(protection_domain_, addr, size, access);
  if (!mr) {
    return false;
  }

  auto region = std::make_unique<MemoryRegion>();
  region->addr = addr;
  region->size = size;
  region->mr = mr;
  region->lkey = mr->lkey;
  region->rkey = mr->rkey;

  lkey = mr->lkey;
  rkey = mr->rkey;

  memory_regions_[addr] = std::move(region);
  return true;
}

bool CompleteRdmaConnection::unregisterMemoryRegion(void* addr) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = memory_regions_.find(addr);
  if (it == memory_regions_.end()) {
    return false;
  }

  if (it->second->mr) {
    ibv_dereg_mr(it->second->mr);
  }

  memory_regions_.erase(it);
  return true;
}

bool CompleteRdmaConnection::postRDMAWrite(uint64_t remote_addr, uint32_t rkey, void* local_addr, size_t length) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!queue_pair_) {
    return false;
  }

  struct ibv_sge sge;
  sge.addr = reinterpret_cast<uint64_t>(local_addr);
  sge.length = length;
  sge.lkey = 0;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = next_message_id_++;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;

  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(queue_pair_, &wr, &bad_wr);

  return ret == 0;
}

bool CompleteRdmaConnection::postRDMARead(uint64_t remote_addr, uint32_t rkey, void* local_addr, size_t length) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!queue_pair_) {
    return false;
  }

  struct ibv_sge sge;
  sge.addr = reinterpret_cast<uint64_t>(local_addr);
  sge.length = length;
  sge.lkey = 0;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = next_message_id_++;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;

  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(queue_pair_, &wr, &bad_wr);

  return ret == 0;
}

int CompleteRdmaConnection::pollCompletionQueue(int max_entries, ibv_wc* wc) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!send_cq_) {
    return -1;
  }

  return ibv_poll_cq(send_cq_, max_entries, wc);
}

bool CompleteRdmaConnection::postReceive(void* addr, size_t length, uint32_t lkey, uint64_t wr_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!queue_pair_) {
    return false;
  }

  struct ibv_sge sge;
  sge.addr = reinterpret_cast<uint64_t>(addr);
  sge.length = length;
  sge.lkey = lkey;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_recv_wr* bad_wr;
  int ret = ibv_post_recv(queue_pair_, &wr, &bad_wr);

  return ret == 0;
}

ConnectionPtr createCompleteRdmaConnection() {
  return std::make_shared<CompleteRdmaConnection>();
}

}
}