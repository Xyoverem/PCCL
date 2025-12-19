#include <plugins/rdma_executor/device.h>
#include <network/network.h>
#include <communication/communicator.h>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <memory>

namespace engine_c {

bool RdmaDevice::remoteCommAvailable() {
  auto& network_manager = engine_c::network::NetworkManager::getInstance();
  return network_manager.initialize(engine_c::network::NetworkType::RDMA_VERBS);
}

std::string RdmaDevice::activate() {
  auto comm = engine_c::communication::createCommunicator(engine_c::network::NetworkType::RDMA_VERBS);
  if (comm->initialize(0, 1)) {
    return "rdma://initialized";
  }
  return "";
}

std::string RdmaDevice::registerBuffer(void *addr, long size) {
  auto conn = engine_c::network::NetworkManager::getInstance().createConnection();
  if (!conn) {
    return "";
  }

  auto rdma_conn = std::dynamic_pointer_cast<engine_c::network::RdmaConnection>(conn);
  if (!rdma_conn) {
    return "";
  }

  uint32_t lkey, rkey;
  if (!rdma_conn->registerMemoryRegion(addr, size, lkey, rkey)) {
    return "";
  }

  nlohmann::json j;
  j["lkey"] = lkey;
  j["rkey"] = rkey;
  j["addr"] = reinterpret_cast<uint64_t>(addr);
  j["size"] = size;

  return j.dump();
}

void RdmaDevice::connect(std::string handle) {
  if (handle.empty()) {
    return;
  }

  try {
    auto j = nlohmann::json::parse(handle);
    std::string ip = j["ip"];
    int port = j["port"];

    engine_c::network::NetworkAddress addr(ip, port);
    auto& network_manager = engine_c::network::NetworkManager::getInstance();
    auto conn = network_manager.createConnection();

    if (conn) {
      conn->connect(addr);
    }
  } catch (const std::exception& e) {
    return;
  }
}

void RdmaDevice::disconnect(std::string handle) {
  auto& network_manager = engine_c::network::NetworkManager::getInstance();
  network_manager.shutdown();
}

}
