#include <infiniband/verbs.h>
#include <fmt/format.h>
#include <c10/util/Exception.h>
#include <plugins/rdma_executor/utils/rdma_utils.h>

namespace engine_c::rdma {

VerbsLib& VerbsLib::getInstance() {
  static VerbsLib instance;
  return instance;
}

void VerbsLib::load(const std::string& library_path) {
  std::lock_guard<std::mutex> lock(load_mutex_);
  if (handle_) {
    return;
  }
  handle_ = dlopen(library_path.c_str(), RTLD_LAZY);
  TORCH_CHECK(handle_ != nullptr, fmt::format("Failed to open {}", library_path));

  #ifdef LOAD_VERBS_FUNC
    static_assert(false);
  #endif

  #define LOAD_VERBS_FUNC(name) \
    name##_func = reinterpret_cast<name##_fn>(dlsym(handle_, #name)); \
    TORCH_CHECK(name##_func != nullptr, fmt::format("Failed to load function {}", #name));

  LOAD_VERBS_FUNC(ibv_get_device_list)
  LOAD_VERBS_FUNC(ibv_free_device_list)
  LOAD_VERBS_FUNC(ibv_get_device_name)
  LOAD_VERBS_FUNC(ibv_open_device)
  LOAD_VERBS_FUNC(ibv_close_device)
  LOAD_VERBS_FUNC(ibv_query_device)
  LOAD_VERBS_FUNC(ibv_query_port)
  LOAD_VERBS_FUNC(ibv_query_gid)
  LOAD_VERBS_FUNC(ibv_alloc_pd)
  LOAD_VERBS_FUNC(ibv_dealloc_pd)
  LOAD_VERBS_FUNC(ibv_reg_mr)
  LOAD_VERBS_FUNC(ibv_dereg_mr)
  LOAD_VERBS_FUNC(ibv_create_comp_channel)
  LOAD_VERBS_FUNC(ibv_destroy_comp_channel)
  LOAD_VERBS_FUNC(ibv_create_cq)
  LOAD_VERBS_FUNC(ibv_destroy_cq)
  LOAD_VERBS_FUNC(ibv_get_cq_event)
  LOAD_VERBS_FUNC(ibv_ack_cq_events)
  LOAD_VERBS_FUNC(ibv_create_qp)
  LOAD_VERBS_FUNC(ibv_destroy_qp)
  LOAD_VERBS_FUNC(ibv_modify_qp)
  LOAD_VERBS_FUNC(ibv_wc_status_str)
  LOAD_VERBS_FUNC(ibv_query_qp)
  #undef LOAD_VERBS_FUNC
}

VerbsLib::~VerbsLib() {
  std::lock_guard<std::mutex> lock(load_mutex_);
  if (handle_) {
    dlclose(handle_);
    handle_ = nullptr;
  }
}

ibv_device** VerbsLib::getDeviceList(int* num_devices) { 
  return ibv_get_device_list_func(num_devices); 
}

int VerbsLib::freeDeviceList(ibv_device** list) { 
  return ibv_free_device_list_func(list); 
}

const char* VerbsLib::getDeviceName(ibv_device* device) { 
  return ibv_get_device_name_func(device); 
}

ibv_context* VerbsLib::openDevice(ibv_device* device) { 
  return ibv_open_device_func(device); 
}

int VerbsLib::closeDevice(ibv_context* context) { 
  return ibv_close_device_func(context); 
}

int VerbsLib::queryDevice(ibv_context* context, ibv_device_attr* device_attr) { 
  return ibv_query_device_func(context, device_attr); 
}

int VerbsLib::queryPort(ibv_context* context, uint8_t port_num, ibv_port_attr* port_attr) { 
  return ibv_query_port_func(context, port_num, port_attr); 
}

int VerbsLib::queryGid(ibv_context* context, uint8_t port_num, int index, ibv_gid* gid) { 
  return ibv_query_gid_func(context, port_num, index, gid); 
}

ibv_pd* VerbsLib::allocPd(ibv_context* context) { 
  return ibv_alloc_pd_func(context); 
}

int VerbsLib::deallocPd(ibv_pd* pd) { 
  return ibv_dealloc_pd_func(pd); 
}

ibv_mr* VerbsLib::regMr(ibv_pd* pd, void* addr, size_t length, int access) { 
  return ibv_reg_mr_func(pd, addr, length, access); 
}

int VerbsLib::deregMr(ibv_mr* mr) { 
  return ibv_dereg_mr_func(mr); 
}

ibv_comp_channel* VerbsLib::createCompChannel(ibv_context* context) { 
  return ibv_create_comp_channel_func(context); 
}

int VerbsLib::destroyCompChannel(ibv_comp_channel* channel) { 
  return ibv_destroy_comp_channel_func(channel); 
}

int VerbsLib::getCqEvent(ibv_comp_channel* channel, ibv_cq** cq, void** cq_context) { 
  return ibv_get_cq_event_func(channel, cq, cq_context); 
}

void VerbsLib::ackCqEvents(ibv_cq* cq, unsigned int nevents) { 
  ibv_ack_cq_events_func(cq, nevents); 
}

ibv_cq* VerbsLib::createCq(ibv_context* context, int cqe, void* cq_context, ibv_comp_channel* channel, int comp_vector) { 
  return ibv_create_cq_func(context, cqe, cq_context, channel, comp_vector); 
}

int VerbsLib::destroyCq(ibv_cq* cq) { 
  return ibv_destroy_cq_func(cq); 
}

int VerbsLib::pollCq(ibv_cq* cq, int num_entries, ibv_wc* wc) { 
  return ibv_poll_cq(cq, num_entries, wc); 
}

ibv_qp* VerbsLib::createQp(ibv_pd* pd, ibv_qp_init_attr* attr) { 
  return ibv_create_qp_func(pd, attr); 
}

int VerbsLib::destroyQp(ibv_qp* qp) { 
  return ibv_destroy_qp_func(qp); 
}

int VerbsLib::modifyQp(ibv_qp* qp, ibv_qp_attr* attr, int attr_mask) { 
  return ibv_modify_qp_func(qp, attr, attr_mask); 
}

int VerbsLib::postSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad_wr) { 
  return ibv_post_send(qp, wr, bad_wr); 
}

int VerbsLib::postRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad_wr) { 
  return ibv_post_recv(qp, wr, bad_wr); 
}

int VerbsLib::reqNotifyCq(ibv_cq* cq, int solicited_only) { 
  return ibv_req_notify_cq(cq, solicited_only); 
}

const char* VerbsLib::wcStatusStr(enum ibv_wc_status status) { 
  return ibv_wc_status_str_func(status); 
}

int VerbsLib::queryQp(ibv_qp* qp, ibv_qp_attr* attr, int attr_mask, ibv_qp_init_attr *init_attr) {
  return ibv_query_qp_func(qp, attr, attr_mask, init_attr);
}

VerbsDevList::VerbsDevList() {
  auto& verbs_lib = VerbsLib::getInstance();
  devices_ = verbs_lib.getDeviceList(&num_devices_);
  TORCH_CHECK(devices_ != nullptr, "Failed to get IB devices");
}

VerbsDevList::~VerbsDevList() {
  if (devices_) {
    VerbsLib::getInstance().freeDeviceList(devices_);
  }
}

VerbsDevList::VerbsDevList(VerbsDevList&& other) noexcept 
  : devices_(other.devices_), num_devices_(other.num_devices_) {
  other.devices_ = nullptr;
  other.num_devices_ = 0;
}

VerbsDevList& VerbsDevList::operator=(VerbsDevList&& other) noexcept {
  if (this != &other) {
    if (devices_) VerbsLib::getInstance().freeDeviceList(devices_);
    devices_ = other.devices_;
    num_devices_ = other.num_devices_;
    other.devices_ = nullptr;
    other.num_devices_ = 0;
  }
  return *this;
}

ibv_device* VerbsDevList::getDevice(int index) const {
  TORCH_CHECK(index >= 0 and index < num_devices_, 
    fmt::format("Device index out of range: {}", index));
  return devices_[index];
}

std::string VerbsDevList::getDeviceName(int index) {
  ibv_device* dev = getDevice(index);
  return VerbsLib::getInstance().getDeviceName(dev);
}

int VerbsDevList::getNumDevices() const { 
  return num_devices_; 
}

VerbsContext::VerbsContext(ibv_device* device) {
  auto& verbs_lib = VerbsLib::getInstance();
  context_ = verbs_lib.openDevice(device);
  TORCH_CHECK(context_, fmt::format("Failed to open device: {}", verbs_lib.getDeviceName(device)));
}

VerbsContext::~VerbsContext() {
  if (context_) VerbsLib::getInstance().closeDevice(context_);
}

VerbsContext::VerbsContext(VerbsContext&& other) noexcept : context_(other.context_) {
  other.context_ = nullptr;
}

VerbsContext& VerbsContext::operator=(VerbsContext&& other) noexcept {
  if (this != &other) {
    if (context_) VerbsLib::getInstance().closeDevice(context_);
    context_ = other.context_;
    other.context_ = nullptr;
  }
  return *this;
}

ibv_device_attr VerbsContext::queryDevice() const {
  ibv_device_attr attr;
  TORCH_CHECK(VerbsLib::getInstance().queryDevice(context_, &attr) == 0, "Failed to query device attributes");
  return attr;
}

ibv_port_attr VerbsContext::queryPort(uint8_t port_num) const {
  ibv_port_attr attr;
  TORCH_CHECK(VerbsLib::getInstance().queryPort(context_, port_num, &attr) == 0, fmt::format("Failed to query port {} attributes", port_num));
  return attr;
}

ibv_gid VerbsContext::queryGid(uint8_t port_num, int index) const {
  ibv_gid gid;
  TORCH_CHECK(VerbsLib::getInstance().queryGid(context_, port_num, index, &gid) == 0, fmt::format("Failed to query GID (port {}, index {})", port_num, index));
  return gid;
}

ibv_context* VerbsContext::get() const { 
  return context_; 
}

VerbsProtectionDomain::VerbsProtectionDomain(const VerbsContext& ctx) {
  pd_ = VerbsLib::getInstance().allocPd(ctx.get());
  TORCH_CHECK(pd_, "Failed to allocate PD");
}

VerbsProtectionDomain::~VerbsProtectionDomain() {
  if (pd_) VerbsLib::getInstance().deallocPd(pd_);
}

VerbsProtectionDomain::VerbsProtectionDomain(VerbsProtectionDomain&& other) noexcept : pd_(other.pd_) {
  other.pd_ = nullptr;
}

VerbsProtectionDomain& VerbsProtectionDomain::operator=(VerbsProtectionDomain&& other) noexcept {
  if (this != &other) {
    if (pd_) VerbsLib::getInstance().deallocPd(pd_);
    pd_ = other.pd_;
    other.pd_ = nullptr;
  }
  return *this;
}

ibv_pd* VerbsProtectionDomain::get() const { 
  return pd_; 
}

VerbsCompChannel::VerbsCompChannel(const VerbsContext& ctx) {
  channel_ = VerbsLib::getInstance().createCompChannel(ctx.get());
  TORCH_CHECK(channel_, "Failed to create comp channel");
}

VerbsCompChannel::~VerbsCompChannel() {
  if (channel_) VerbsLib::getInstance().destroyCompChannel(channel_);
}

VerbsCompChannel::VerbsCompChannel(VerbsCompChannel&& other) noexcept : channel_(other.channel_) {
  other.channel_ = nullptr;
}

VerbsCompChannel& VerbsCompChannel::operator=(VerbsCompChannel&& other) noexcept {
  if (this != &other) {
    if (channel_) VerbsLib::getInstance().destroyCompChannel(channel_);
    channel_ = other.channel_;
    other.channel_ = nullptr;
  }
  return *this;
}

ibv_comp_channel* VerbsCompChannel::get() const { 
  return channel_; 
}

VerbsCompletionQueue::VerbsCompletionQueue(const VerbsContext& ctx, int cqe, void* cq_ctx,
                      const VerbsCompChannel* channel, int comp_vec) {
  cq_ = VerbsLib::getInstance().createCq(
    ctx.get(), cqe, cq_ctx, channel ? channel->get() : nullptr, comp_vec
  );
  TORCH_CHECK(cq_, "Failed to create CQ");
}

VerbsCompletionQueue::~VerbsCompletionQueue() {
  if (cq_) VerbsLib::getInstance().destroyCq(cq_);
}

VerbsCompletionQueue::VerbsCompletionQueue(VerbsCompletionQueue&& other) noexcept : cq_(other.cq_) {
  other.cq_ = nullptr;
}

VerbsCompletionQueue& VerbsCompletionQueue::operator=(VerbsCompletionQueue&& other) noexcept {
  if (this != &other) {
    if (cq_) VerbsLib::getInstance().destroyCq(cq_);
    cq_ = other.cq_;
    other.cq_ = nullptr;
  }
  return *this;
}

void VerbsCompletionQueue::getEvent(const VerbsCompChannel& channel, VerbsCompletionQueue*& out_cq, void*& out_ctx) {
  ibv_cq* cq = nullptr;
  TORCH_CHECK(VerbsLib::getInstance().getCqEvent(channel.get(), &cq, &out_ctx) == 0, "Failed to get CQ event");
  out_cq = static_cast<VerbsCompletionQueue*>(out_ctx);
}

int VerbsCompletionQueue::poll(int num_entries, ibv_wc* wc) const {
  return VerbsLib::getInstance().pollCq(cq_, num_entries, wc);
}

void VerbsCompletionQueue::ackEvents(unsigned int nevents) const {
  VerbsLib::getInstance().ackCqEvents(cq_, nevents);
}

ibv_cq* VerbsCompletionQueue::get() const { 
  return cq_; 
}

VerbsMemoryRegion::VerbsMemoryRegion(const VerbsProtectionDomain& pd, void* addr, size_t len, int access)
  : addr_(addr), len_(len) {
  mr_ = VerbsLib::getInstance().regMr(pd.get(), addr, len, access);
  TORCH_CHECK(mr_, "Failed to register MR");
  lkey_ = mr_->lkey;
  rkey_ = mr_->rkey;
}

VerbsMemoryRegion::~VerbsMemoryRegion() {
  if (mr_) VerbsLib::getInstance().deregMr(mr_);
}

VerbsMemoryRegion::VerbsMemoryRegion(VerbsMemoryRegion&& other) noexcept 
  : mr_(other.mr_), addr_(other.addr_), len_(other.len_), lkey_(other.lkey_), rkey_(other.rkey_) {
  other.mr_ = nullptr;
}

VerbsMemoryRegion& VerbsMemoryRegion::operator=(VerbsMemoryRegion&& other) noexcept {
  if (this != &other) {
    if (mr_) VerbsLib::getInstance().deregMr(mr_);
    mr_ = other.mr_;
    addr_ = other.addr_;
    len_ = other.len_;
    lkey_ = other.lkey_;
    rkey_ = other.rkey_;
    other.mr_ = nullptr;
  }
  return *this;
}

void* VerbsMemoryRegion::getAddr() const { 
  return addr_; 
}

size_t VerbsMemoryRegion::getLength() const { 
  return len_; 
}

uint32_t VerbsMemoryRegion::getLKey() const { 
  return lkey_; 
}

uint32_t VerbsMemoryRegion::getRKey() const { 
  return rkey_; 
}

ibv_mr* VerbsMemoryRegion::get() const { 
  return mr_; 
}

VerbsQueuePair::VerbsQueuePair(const VerbsProtectionDomain& pd, const ibv_qp_init_attr& init_attr) {
  qp_ = VerbsLib::getInstance().createQp(pd.get(), const_cast<ibv_qp_init_attr*>(&init_attr));
  TORCH_CHECK(qp_, "Failed to create QP");
  qp_num_ = qp_->qp_num;
}

VerbsQueuePair::~VerbsQueuePair() {
  if (qp_) VerbsLib::getInstance().destroyQp(qp_);
}

VerbsQueuePair::VerbsQueuePair(VerbsQueuePair&& other) noexcept 
  : qp_(other.qp_), qp_num_(other.qp_num_), connected_(other.connected_) {
  other.qp_ = nullptr;
  other.connected_ = false;
}

VerbsQueuePair& VerbsQueuePair::operator=(VerbsQueuePair&& other) noexcept {
  if (this != &other) {
    if (qp_) VerbsLib::getInstance().destroyQp(qp_);
    qp_ = other.qp_;
    qp_num_ = other.qp_num_;
    connected_ = other.connected_;
    other.qp_ = nullptr;
    other.connected_ = false;
  }
  return *this;
}

int VerbsQueuePair::modify(const ibv_qp_attr& attr, int attr_mask) const {
  return VerbsLib::getInstance().modifyQp(qp_, const_cast<ibv_qp_attr*>(&attr), attr_mask);
}

int VerbsQueuePair::postSend(ibv_send_wr* wr, ibv_send_wr** bad_wr) const {
  return VerbsLib::getInstance().postSend(qp_, wr, bad_wr);
}

int VerbsQueuePair::postRecv(ibv_recv_wr* wr, ibv_recv_wr** bad_wr) const {
  return VerbsLib::getInstance().postRecv(qp_, wr, bad_wr);
}

uint32_t VerbsQueuePair::getQpNum() const { 
  return qp_num_; 
}

ibv_qp* VerbsQueuePair::get() const { 
  return qp_; 
}

bool VerbsQueuePair::connected() {
  return connected_;
}

void VerbsQueuePair::set_connected(bool connected) {
  connected_ = connected;
}

VerbsRemotePeerInfo::VerbsRemotePeerInfo() : qp_num(0), lid(0) {
  std::memset(&gid, 0, sizeof(gid));
}

VerbsRemotePeerInfo::VerbsRemotePeerInfo(const VerbsRemotePeerInfo& other)
  : qp_num(other.qp_num), lid(other.lid) {
  std::memcpy(&gid, &other.gid, sizeof(gid));
}

VerbsRemotePeerInfo::VerbsRemotePeerInfo(VerbsRemotePeerInfo&& other) noexcept
  : qp_num(other.qp_num), lid(other.lid) {
  std::memcpy(&gid, &other.gid, sizeof(gid));
  other.qp_num = 0;
  other.lid = 0;
  std::memset(&other.gid, 0, sizeof(gid));
}

VerbsRemotePeerInfo& VerbsRemotePeerInfo::operator=(const VerbsRemotePeerInfo& other) {
  if (this != &other) {
    qp_num = other.qp_num;
    lid = other.lid;
    std::memcpy(&gid, &other.gid, sizeof(gid));
  }
  return *this;
}

VerbsRemotePeerInfo& VerbsRemotePeerInfo::operator=(VerbsRemotePeerInfo&& other) noexcept {
  if (this != &other) {
    qp_num = other.qp_num;
    lid = other.lid;
    std::memcpy(&gid, &other.gid, sizeof(gid));
    other.qp_num = 0;
    other.lid = 0;
    std::memset(&other.gid, 0, sizeof(gid));
  }
  return *this;
}

VerbsManager::QPConfig::QPConfig() 
  : qp_type(IBV_QPT_RC), 
    max_send_wr(1024), 
    max_recv_wr(1024), 
    max_send_sge(32), 
    max_recv_sge(32),
    max_inline_data(0) {}

VerbsManager::ConnectionInfo::ConnectionInfo() = default;

VerbsManager::ConnectionInfo::ConnectionInfo(const ConnectionInfo& other)
  : config(other.config), cq(other.cq) {
  for (const auto& pair : other.qps) qps[pair.first] = pair.second;
}

VerbsManager::ConnectionInfo::ConnectionInfo(ConnectionInfo&& other) noexcept
  : config(std::move(other.config)), cq(std::move(other.cq)), qps(std::move(other.qps)) {
}

VerbsManager::ConnectionInfo& VerbsManager::ConnectionInfo::operator=(const ConnectionInfo& other) {
  if (this != &other) {
    config = other.config;
    cq = other.cq;
    qps.clear();
    for (const auto& pair : other.qps) qps[pair.first] = pair.second;
  }
  return *this;
}

VerbsManager::ConnectionInfo& VerbsManager::ConnectionInfo::operator=(ConnectionInfo&& other) noexcept {
  if (this != &other) {
    config = std::move(other.config);
    cq = std::move(other.cq);
    qps = std::move(other.qps);
  }
  return *this;
}

VerbsManager::VerbsManager() {
  VerbsLib::getInstance().load("libibverbs.so");
}

VerbsManager::~VerbsManager() {
    std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
    connections_.clear();
}

VerbsManager::VerbsManager(VerbsManager&& other) noexcept 
  : device_list_(std::move(other.device_list_)),
    context_(std::move(other.context_)),
    pd_(std::move(other.pd_)),
    connections_(std::move(other.connections_)),
    next_connection_id_(other.next_connection_id_.load()),
    next_qp_id_(other.next_qp_id_.load()),
    initialized_(other.initialized_.load()) {
  other.next_connection_id_ = 1;
  other.next_qp_id_ = 1;
  other.initialized_ = false;
}

VerbsManager& VerbsManager::operator=(VerbsManager&& other) noexcept {
  if (this != &other) {
    std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
    device_list_ = std::move(other.device_list_);
    context_ = std::move(other.context_);
    pd_ = std::move(other.pd_);
    connections_ = std::move(other.connections_);
    next_connection_id_ = other.next_connection_id_.load();
    next_qp_id_ = other.next_qp_id_.load();
    initialized_ = other.initialized_.load();
    
    other.next_connection_id_ = 1;
    other.next_qp_id_ = 1;
    other.initialized_ = false;
  }
  return *this;
}

bool VerbsManager::initialize(const std::string& device_name, uint8_t port_num) {
  TORCH_CHECK(device_name.size() > 0);

  device_list_ = std::make_unique<VerbsDevList>();
  
  if (device_list_->getNumDevices() == 0) {
    return false;
  }

  ibv_device* device = nullptr;
  for (int i = 0; i < device_list_->getNumDevices(); ++i) {
    if (device_list_->getDeviceName(i) == device_name) {
      device = device_list_->getDevice(i);
      break;
    }
  }
    
  if (!device) {
    return false;
  }
      
  context_ = std::make_unique<VerbsContext>(device);
  pd_ = std::make_unique<VerbsProtectionDomain>(*context_);
      
  initialized_ = true;
  return true;
}

VerbsManager::ConnectionId VerbsManager::createConnection(const ConnectionConfig& config) {
  if (!initialized_.load(std::memory_order_relaxed)) {
    return 0;
  }
        
  ConnectionId conn_id = generateConnectionId();
        
  ConnectionInfo conn_info;
  conn_info.config = config;
  conn_info.cq = std::make_shared<VerbsCompletionQueue>(*context_, config.cq_size, nullptr);
    
  std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
  connections_[conn_id] = std::move(conn_info);
    
  return conn_id;
}

bool VerbsManager::destroyConnection(ConnectionId conn_id, QPId qp_id) {
  if (!initialized_) {
    return false;
  }
    
  std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
  auto it = connections_.find(conn_id);
  if (it == connections_.end()) {
    return false;
  }
  
  auto qp_it = it->second.qps.find(qp_id);
  if (qp_it == it->second.qps.end()) {
    return false;
  }
  
  it->second.qps.erase(qp_it);
  return true;
}

bool VerbsManager::connect(ConnectionId conn_id, QPId qp_id, const VerbsRemotePeerInfo& remote_peer_info) {
  if (!initialized_) {
    return false;
  }
  
  std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
  auto it = connections_.find(conn_id);
  if (it == connections_.end()) {
    return false;
  }
  
  auto qp_it = it->second.qps.find(qp_id);
  if (qp_it == it->second.qps.end()) {
    return false;
  }
  if (!modifyQPToRTR(conn_id, qp_id, 
            remote_peer_info.qp_num,
            remote_peer_info.lid,
            0,
            getPortNum(conn_id),
            0,
            &remote_peer_info.gid)) {
    return false;
  }
  
  if (!modifyQPToRTS(conn_id, qp_id)) {
    return false;
  }

  qp_it->second->set_connected(true);
  return true;
}

bool VerbsManager::disconnect(ConnectionId conn_id, QPId qp_id) {
  if (!initialized_) {
    return false;
  }
  
  std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
  auto it = connections_.find(conn_id);
  if (it == connections_.end()) {
    return false;
  }
  
  auto qp_it = it->second.qps.find(qp_id);
  if (qp_it == it->second.qps.end()) {
    return false;
  }
  
  qp_it->second->set_connected(false);
  return true;
}

VerbsManager::QPId VerbsManager::createQP(ConnectionId conn_id, const QPConfig& config) {
  if (!initialized_) {
    return 0;
  }
    
  std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
  auto it = connections_.find(conn_id);
  if (it == connections_.end()) {
    return 0;
  }
    
  QPId qp_id = generateQPId();
  
  ibv_qp_init_attr init_attr;
  std::memset(&init_attr, 0, sizeof(init_attr));
  init_attr.qp_type = config.qp_type;
  init_attr.send_cq = it->second.cq->get();
  init_attr.recv_cq = it->second.cq->get();
  init_attr.cap.max_send_wr = config.max_send_wr;
  init_attr.cap.max_recv_wr = config.max_recv_wr;
  init_attr.cap.max_send_sge = config.max_send_sge;
  init_attr.cap.max_recv_sge = config.max_recv_sge;
  init_attr.cap.max_inline_data = config.max_inline_data;
  init_attr.sq_sig_all = 0;

  auto qp = std::make_shared<VerbsQueuePair>(*pd_, init_attr);
  it->second.qps[qp_id] = qp;

  return qp_id;
}

bool VerbsManager::modifyQPToInit(ConnectionId conn_id, QPId qp_id) {
  if (!initialized_) {
    return false;
  }
  
  auto qp = getQP(conn_id, qp_id);
  if (!qp) {
    return false;
  }
  
  ibv_qp_attr attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  auto conn_it = connections_.find(conn_id);
  if (conn_it == connections_.end()) {
    return false;
  }
  attr.port_num = conn_it->second.config.port_num;
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | 
                         IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  

  return qp->modify(attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0;
}

bool VerbsManager::modifyQPToRTR(ConnectionId conn_id, QPId qp_id, 
                              uint32_t remote_qpn, uint16_t dlid, uint8_t sl, 
                              uint8_t port_num, uint16_t pkey_index, 
                              const ibv_gid* sgid) {
  if (!initialized_) {
    return false;
  }
    
  auto qp = getQP(conn_id, qp_id);
  if (!qp) {
    return false;
  }
    
  ibv_qp_attr attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_2048;
  attr.dest_qp_num = remote_qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.dlid = dlid;
  attr.ah_attr.sl = sl;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = port_num;
  attr.ah_attr.is_global = (sgid != nullptr && sgid->global.interface_id != 0) ? 1 : 0;
  if (attr.ah_attr.is_global) {
    attr.ah_attr.grh.hop_limit = 64;
    memcpy(&attr.ah_attr.grh.dgid, sgid, sizeof(ibv_gid));
    attr.ah_attr.grh.sgid_index = connections_[conn_id].config.gid_index;
  }
    
  int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
           IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
           IBV_QP_MIN_RNR_TIMER;

  int ret = qp->modify(attr, mask);

  return ret == 0;
}

bool VerbsManager::modifyQPToRTS(ConnectionId conn_id, QPId qp_id) {
  if (!initialized_) {
    return false;
  }
  
  auto qp = getQP(conn_id, qp_id);
  if (!qp) {
    return false;
  }
  
  ibv_qp_attr attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = 0;
  attr.max_rd_atomic = 1;
  
  int mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
             IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
  
  return qp->modify(attr, mask) == 0;
}

bool VerbsManager::postSend(ConnectionId conn_id, QPId qp_id, ibv_send_wr* wr, ibv_send_wr** bad_wr) {
  auto qp = getQP(conn_id, qp_id);
  int ret = qp->postSend(wr, bad_wr);
  return ret == 0;
}

bool VerbsManager::postRecv(ConnectionId conn_id, QPId qp_id, ibv_recv_wr* wr, ibv_recv_wr** bad_wr) {
  if (!initialized_) {
    return false;
  }
  
  auto qp = getQP(conn_id, qp_id);
  if (!qp) {
    return false;
  }
  
  return qp->postRecv(wr, bad_wr) == 0;
}

int VerbsManager::pollCQ(ConnectionId conn_id, int num_entries, ibv_wc* wc) {  
  auto cq = getCQ(conn_id);
  if (!cq) {
    return 0;
  }
  
  return cq->poll(num_entries, wc);
}

std::shared_ptr<VerbsQueuePair> VerbsManager::getQP(ConnectionId conn_id, QPId qp_id) {
  std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
  auto it = connections_.find(conn_id);
  if (it == connections_.end()) {
    return nullptr;
  }
  
  auto qp_it = it->second.qps.find(qp_id);
  if (qp_it == it->second.qps.end()) {
    return nullptr;
  }
  
  return qp_it->second;
}

std::shared_ptr<VerbsCompletionQueue> VerbsManager::getCQ(ConnectionId conn_id) {
  std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
  auto it = connections_.find(conn_id);
  if (it == connections_.end()) {
    return nullptr;
  }
  
  return it->second.cq;
}

std::shared_ptr<VerbsProtectionDomain> VerbsManager::getPD() {
  return pd_;
}

bool VerbsManager::isConnected(ConnectionId conn_id) const {
  std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
  auto it = connections_.find(conn_id);
  if (it == connections_.end()) {
    return false;
  }
  
  for (const auto& qp_pair : it->second.qps) {
    if (qp_pair.second->connected()) {
      return true;
    }
  }
  
  return false;
}

size_t VerbsManager::getConnectionCount() const {
  std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
  return connections_.size();
}

size_t VerbsManager::getQPCount(ConnectionId conn_id) const {
  std::lock_guard<std::recursive_mutex> lock(connections_mutex_);
  auto it = connections_.find(conn_id);
  if (it == connections_.end()) {
    return 0;
  }
  
  return it->second.qps.size();
}

uint8_t VerbsManager::getPortNum(ConnectionId conn_id) const {
  return connections_.at(conn_id).config.port_num;
}

VerbsRemotePeerInfo VerbsManager::getLocalMetadata(ConnectionId conn_id, QPId qp_id) {
  VerbsRemotePeerInfo info;
  
  auto qp = getQP(conn_id, qp_id);
  if (!qp) {
    return info;
  }
  
  info.qp_num = qp->getQpNum(); 
  return info;
}

VerbsManager::ConnectionId VerbsManager::generateConnectionId() {
  return next_connection_id_++;
}

VerbsManager::QPId VerbsManager::generateQPId() {
  return next_qp_id_++;
}

}

}