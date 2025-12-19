#include <network/network.h>
#include <infiniband/verbs.h>
#include <cstring>
#include <memory>

namespace engine_c {
namespace network {

RdmaConnection::RdmaConnection() : status_(ConnectionStatus::DISCONNECTED) {
  memset(&rdma_ctx_, 0, sizeof(rdma_ctx_));
}

RdmaConnection::~RdmaConnection() {
  disconnect();
}

bool RdmaConnection::connect(const NetworkAddress& address) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (status_ != ConnectionStatus::DISCONNECTED) {
    return false;
  }

  rdma_ctx_.verbs_context = ibv_open_device(nullptr);
  if (!rdma_ctx_.verbs_context) {
    return false;
  }

  rdma_ctx_.protection_domain = ibv_alloc_pd(rdma_ctx_.verbs_context);
  if (!rdma_ctx_.protection_domain) {
    ibv_close_device(static_cast<ibv_context*>(rdma_ctx_.verbs_context));
    rdma_ctx_.verbs_context = nullptr;
    return false;
  }

  struct ibv_cq_init_attr cq_attr = {};
  cq_attr.cqe = 1024;
  rdma_ctx_.completion_queue = ibv_create_cq(
    static_cast<ibv_context*>(rdma_ctx_.verbs_context), &cq_attr, nullptr, nullptr, 0);
  if (!rdma_ctx_.completion_queue) {
    ibv_dealloc_pd(static_cast<ibv_pd*>(rdma_ctx_.protection_domain));
    ibv_close_device(static_cast<ibv_context*>(rdma_ctx_.verbs_context));
    rdma_ctx_.protection_domain = nullptr;
    rdma_ctx_.verbs_context = nullptr;
    return false;
  }

  struct ibv_qp_init_attr qp_init_attr = {};
  qp_init_attr.send_cq = static_cast<ibv_cq*>(rdma_ctx_.completion_queue);
  qp_init_attr.recv_cq = static_cast<ibv_cq*>(rdma_ctx_.completion_queue);
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.cap.max_send_wr = 1024;
  qp_init_attr.cap.max_recv_wr = 1024;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;

  rdma_ctx_.queue_pair = ibv_create_qp(
    static_cast<ibv_pd*>(rdma_ctx_.protection_domain), &qp_init_attr);
  if (!rdma_ctx_.queue_pair) {
    ibv_destroy_cq(static_cast<ibv_cq*>(rdma_ctx_.completion_queue));
    ibv_dealloc_pd(static_cast<ibv_pd*>(rdma_ctx_.protection_domain));
    ibv_close_device(static_cast<ibv_context*>(rdma_ctx_.verbs_context));
    rdma_ctx_.completion_queue = nullptr;
    rdma_ctx_.protection_domain = nullptr;
    rdma_ctx_.verbs_context = nullptr;
    return false;
  }

  rdma_ctx_.qp_num = static_cast<ibv_qp*>(rdma_ctx_.queue_pair)->qp_num;

  struct ibv_qp_attr qp_attr = {};
  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.port_num = 1;
  qp_attr.pkey_index = 0;

  int ret = ibv_modify_qp(static_cast<ibv_qp*>(rdma_ctx_.queue_pair),
                          &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT);
  if (ret) {
    ibv_destroy_qp(static_cast<ibv_qp*>(rdma_ctx_.queue_pair));
    ibv_destroy_cq(static_cast<ibv_cq*>(rdma_ctx_.completion_queue));
    ibv_dealloc_pd(static_cast<ibv_pd*>(rdma_ctx_.protection_domain));
    ibv_close_device(static_cast<ibv_context*>(rdma_ctx_.verbs_context));
    rdma_ctx_.queue_pair = nullptr;
    rdma_ctx_.completion_queue = nullptr;
    rdma_ctx_.protection_domain = nullptr;
    rdma_ctx_.verbs_context = nullptr;
    return false;
  }

  status_ = ConnectionStatus::CONNECTED;
  remote_addr_ = address;

  return true;
}

void RdmaConnection::disconnect() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (rdma_ctx_.queue_pair) {
    ibv_destroy_qp(static_cast<ibv_qp*>(rdma_ctx_.queue_pair));
    rdma_ctx_.queue_pair = nullptr;
  }

  if (rdma_ctx_.completion_queue) {
    ibv_destroy_cq(static_cast<ibv_cq*>(rdma_ctx_.completion_queue));
    rdma_ctx_.completion_queue = nullptr;
  }

  if (rdma_ctx_.protection_domain) {
    ibv_dealloc_pd(static_cast<ibv_pd*>(rdma_ctx_.protection_domain));
    rdma_ctx_.protection_domain = nullptr;
  }

  if (rdma_ctx_.verbs_context) {
    ibv_close_device(static_cast<ibv_context*>(rdma_ctx_.verbs_context));
    rdma_ctx_.verbs_context = nullptr;
  }

  status_ = ConnectionStatus::DISCONNECTED;
  pending_operations_.clear();
  memory_regions_.clear();
}

ConnectionStatus RdmaConnection::getStatus() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_;
}

bool RdmaConnection::sendMessage(const MessageHeader& header, const void* data) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (status_ != ConnectionStatus::CONNECTED) {
    return false;
  }

  auto header_it = memory_regions_.find(&header);
  if (header_it == memory_regions_.end()) {
    return false;
  }

  struct ibv_sge sge;
  sge.addr = reinterpret_cast<uint64_t>(&header);
  sge.length = sizeof(header);
  sge.lkey = header_it->second.first;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = next_message_id_++;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(static_cast<ibv_qp*>(rdma_ctx_.queue_pair), &wr, &bad_wr);
  if (ret) {
    status_ = ConnectionStatus::ERROR;
    return false;
  }

  if (header.data_size > 0 && data) {
    auto data_it = memory_regions_.find(const_cast<void*>(data));
    if (data_it == memory_regions_.end()) {
      return false;
    }

    sge.addr = reinterpret_cast<uint64_t>(data);
    sge.length = header.data_size;
    sge.lkey = data_it->second.first;

    memset(&wr, 0, sizeof(wr));
    wr.wr_id = next_message_id_++;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;

    ret = ibv_post_send(static_cast<ibv_qp*>(rdma_ctx_.queue_pair), &wr, &bad_wr);
    if (ret) {
      status_ = ConnectionStatus::ERROR;
      return false;
    }
  }

  return true;
}

bool RdmaConnection::receiveMessage(MessageHeader& header, void* data, size_t max_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (status_ != ConnectionStatus::CONNECTED) {
    return false;
  }

  struct ibv_wc wc;
  int ret = ibv_poll_cq(static_cast<ibv_cq*>(rdma_ctx_.completion_queue), 1, &wc);
  if (ret <= 0) {
    return false;
  }

  if (wc.status != IBV_WC_SUCCESS) {
    status_ = ConnectionStatus::ERROR;
    return false;
  }

  return true;
}

bool RdmaConnection::sendAsync(const MessageHeader& header, const void* data) {
  return sendMessage(header, data);
}

bool RdmaConnection::recvAsync(MessageHeader& header, void* data, size_t max_size) {
  return receiveMessage(header, data, max_size);
}

bool RdmaConnection::pollCompletion(std::vector<uint32_t>& completed_ids) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (status_ != ConnectionStatus::CONNECTED) {
    return false;
  }

  struct ibv_wc wc;
  int ret = ibv_poll_cq(static_cast<ibv_cq*>(rdma_ctx_.completion_queue), 1, &wc);
  if (ret <= 0) {
    return false;
  }

  if (wc.status == IBV_WC_SUCCESS) {
    completed_ids.push_back(wc.wr_id);
    return true;
  }

  return false;
}

std::string RdmaConnection::getLocalAddress() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return "rdma://" + std::to_string(rdma_ctx_.qp_num);
}

NetworkAddress RdmaConnection::getRemoteAddress() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return remote_addr_;
}

bool RdmaConnection::registerMemoryRegion(void* addr, size_t size, uint32_t& lkey, uint32_t& rkey) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (status_ != ConnectionStatus::CONNECTED || !rdma_ctx_.protection_domain) {
    return false;
  }

  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  ibv_mr* mr = ibv_reg_mr(static_cast<ibv_pd*>(rdma_ctx_.protection_domain), addr, size, access);
  if (!mr) {
    return false;
  }

  lkey = mr->lkey;
  rkey = mr->rkey;

  memory_regions_[addr] = {lkey, rkey};

  return true;
}

bool RdmaConnection::unregisterMemoryRegion(void* addr) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = memory_regions_.find(addr);
  if (it != memory_regions_.end()) {
    memory_regions_.erase(it);
    return true;
  }

  return false;
}

}
}