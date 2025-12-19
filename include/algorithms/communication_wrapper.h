#pragma once

#include <communication/communicator.h>
#include <memory>
#include <vector>
#include <functional>

namespace engine_c {
namespace algorithms {

class CommunicationWrapper {
public:
  CommunicationWrapper();
  ~CommunicationWrapper();

  bool initialize(int rank, int world_size,
                 network::NetworkType type = network::NetworkType::TCP_SOCKET);

  void finalize();

  bool allReduceRing(const void* sendbuf, void* recvbuf, size_t count,
                    communication::ReductionOp op = communication::ReductionOp::SUM,
                    uint32_t tag = 0);
  bool allReduceTree(const void* sendbuf, void* recvbuf, size_t count,
                    communication::ReductionOp op = communication::ReductionOp::SUM,
                    uint32_t tag = 0);
  bool allReduceRabenseifner(const void* sendbuf, void* recvbuf, size_t count,
                            communication::ReductionOp op = communication::ReductionOp::SUM,
                            uint32_t tag = 0);

  bool broadcast(void* buffer, size_t count, int root, uint32_t tag = 0);
  bool send(const void* buffer, size_t count, int dest, uint32_t tag = 0);
  bool recv(void* buffer, size_t count, int source, uint32_t tag = 0);

  bool isInitialized() const;
  int getRank() const;
  int getWorldSize() const;

  void setLatencyCallback(std::function<double(int, int)> callback);
  void setBandwidthCallback(std::function<double(int, int)> callback);

  double measureLatency(int peer_rank, size_t message_size = 1024);
  double measureBandwidth(int peer_rank, size_t message_size = 1024 * 1024);

private:
  std::unique_ptr<communication::Communicator> communicator_;
  bool initialized_;

  std::function<double(int, int)> latency_callback_;
  std::function<double(int, int)> bandwidth_callback_;

  std::vector<double> latency_cache_;
  std::vector<double> bandwidth_cache_;
  bool cache_valid_;

  void updateCaches();
};

}
}