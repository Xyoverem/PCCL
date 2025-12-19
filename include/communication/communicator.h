#pragma once

#include <network/network.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <atomic>
#include <future>

namespace engine_c {
namespace communication {

enum class CollectiveType {
  ALLREDUCE,
  ALLGATHER,
  REDUCE_SCATTER,
  BROADCAST,
  REDUCE,
  GATHER,
  SCATTER,
  ALLTOALL
};

enum class ReductionOp {
  SUM,
  MAX,
  MIN,
  PROD,
  LOR,
  LAND,
  LXOR,
  BOR,
  BAND,
  BXOR
};

struct CollectiveConfig {
  CollectiveType type;
  ReductionOp reduction_op;
  std::vector<int> ranks;
  int root_rank;
  uint32_t tag;
  size_t element_size;
  size_t count;
  std::string algorithm;

  CollectiveConfig()
    : type(CollectiveType::ALLREDUCE), reduction_op(ReductionOp::SUM),
      root_rank(0), tag(0), element_size(0), count(0), algorithm("auto") {}
};

class Communicator {
public:
  virtual ~Communicator() = default;

  virtual bool initialize(int rank, int world_size) = 0;
  virtual void finalize() = 0;

  virtual bool allReduce(const void* sendbuf, void* recvbuf, size_t count,
                        ReductionOp op, uint32_t tag = 0) = 0;
  virtual bool allGather(const void* sendbuf, void* recvbuf, size_t sendcount,
                        size_t element_size, uint32_t tag = 0) = 0;
  virtual bool reduceScatter(const void* sendbuf, void* recvbuf,
                            const size_t recvcounts[], ReductionOp op,
                            uint32_t tag = 0) = 0;
  virtual bool broadcast(void* buffer, size_t count, int root,
                        uint32_t tag = 0) = 0;
  virtual bool reduce(const void* sendbuf, void* recvbuf, size_t count,
                     ReductionOp op, int root, uint32_t tag = 0) = 0;
  virtual bool gather(const void* sendbuf, void* recvbuf, size_t sendcount,
                     size_t element_size, int root, uint32_t tag = 0) = 0;
  virtual bool scatter(const void* sendbuf, void* recvbuf, size_t recvcount,
                      size_t element_size, int root, uint32_t tag = 0) = 0;
  virtual bool alltoall(const void* sendbuf, void* recvbuf, size_t sendcount,
                       size_t element_size, uint32_t tag = 0) = 0;

  virtual bool allReduceAsync(const void* sendbuf, void* recvbuf, size_t count,
                             ReductionOp op, uint32_t tag, std::future<bool>& future) = 0;
  virtual bool allGatherAsync(const void* sendbuf, void* recvbuf, size_t sendcount,
                             size_t element_size, uint32_t tag, std::future<bool>& future) = 0;
  virtual bool broadcastAsync(void* buffer, size_t count, int root,
                             uint32_t tag, std::future<bool>& future) = 0;

  virtual bool send(const void* buffer, size_t count, int dest, int tag) = 0;
  virtual bool recv(void* buffer, size_t count, int source, int tag) = 0;
  virtual bool sendAsync(const void* buffer, size_t count, int dest, int tag,
                        std::future<bool>& future) = 0;
  virtual bool recvAsync(void* buffer, size_t count, int source, int tag,
                        std::future<bool>& future) = 0;

  virtual bool flush() = 0;
  virtual bool barrier() = 0;

  virtual int getRank() const = 0;
  virtual int getWorldSize() const = 0;

  virtual double getLastOperationTime() const = 0;
  virtual uint64_t getTotalBytesTransferred() const = 0;
  virtual double getAverageLatency() const = 0;
  virtual double getThroughput() const = 0;

  virtual void setCollectiveConfig(const CollectiveConfig& config) = 0;
  virtual CollectiveConfig getCollectiveConfig() const = 0;
};

using CommunicatorPtr = std::shared_ptr<Communicator>;

class NetworkCommunicator : public Communicator {
public:
  NetworkCommunicator();
  ~NetworkCommunicator() override;

  bool initialize(int rank, int world_size) override;
  void finalize() override;

  bool allReduce(const void* sendbuf, void* recvbuf, size_t count,
                ReductionOp op, uint32_t tag = 0) override;
  bool allGather(const void* sendbuf, void* recvbuf, size_t sendcount,
                size_t element_size, uint32_t tag = 0) override;
  bool reduceScatter(const void* sendbuf, void* recvbuf,
                    const size_t recvcounts[], ReductionOp op,
                    uint32_t tag = 0) override;
  bool broadcast(void* buffer, size_t count, int root,
                uint32_t tag = 0) override;
  bool reduce(const void* sendbuf, void* recvbuf, size_t count,
             ReductionOp op, int root, uint32_t tag = 0) override;
  bool gather(const void* sendbuf, void* recvbuf, size_t sendcount,
             size_t element_size, int root, uint32_t tag = 0) override;
  bool scatter(const void* sendbuf, void* recvbuf, size_t recvcount,
              size_t element_size, int root, uint32_t tag = 0) override;
  bool alltoall(const void* sendbuf, void* recvbuf, size_t sendcount,
               size_t element_size, uint32_t tag = 0) override;

  bool allReduceAsync(const void* sendbuf, void* recvbuf, size_t count,
                     ReductionOp op, uint32_t tag, std::future<bool>& future) override;
  bool allGatherAsync(const void* sendbuf, void* recvbuf, size_t sendcount,
                     size_t element_size, uint32_t tag, std::future<bool>& future) override;
  bool broadcastAsync(void* buffer, size_t count, int root,
                     uint32_t tag, std::future<bool>& future) override;

  bool send(const void* buffer, size_t count, int dest, int tag) override;
  bool recv(void* buffer, size_t count, int source, int tag) override;
  bool sendAsync(const void* buffer, size_t count, int dest, int tag,
                std::future<bool>& future) override;
  bool recvAsync(void* buffer, size_t count, int source, int tag,
                std::future<bool>& future) override;

  bool flush() override;
  bool barrier() override;

  int getRank() const override;
  int getWorldSize() const override;

  double getLastOperationTime() const override;
  uint64_t getTotalBytesTransferred() const override;
  double getAverageLatency() const override;
  double getThroughput() const override;

  void setCollectiveConfig(const CollectiveConfig& config) override;
  CollectiveConfig getCollectiveConfig() const override;

  bool connectToPeers(const std::vector<network::NetworkAddress>& peer_addresses);
  void setNetworkType(network::NetworkType type);

private:
  struct AsyncOperation {
    std::promise<bool> promise;
    std::future<bool> future;
    CollectiveType type;
    uint32_t tag;
    std::chrono::high_resolution_clock::time_point start_time;
  };

  bool ringAllReduce(const void* sendbuf, void* recvbuf, size_t count,
                    ReductionOp op, uint32_t tag);
  bool treeAllReduce(const void* sendbuf, void* recvbuf, size_t count,
                    ReductionOp op, uint32_t tag);
  bool rabenseifnerAllReduce(const void* sendbuf, void* recvbuf, size_t count,
                           ReductionOp op, uint32_t tag);

  void performReduction(void* dest, const void* src, size_t count, ReductionOp op);

  int rank_;
  int world_size_;
  bool initialized_;

  std::unique_ptr<network::NetworkManager> network_manager_;
  std::unordered_map<int, network::ConnectionPtr> connections_;

  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, std::unique_ptr<AsyncOperation>> async_operations_;
  std::atomic<uint32_t> next_tag_{1};

  CollectiveConfig collective_config_;

  std::atomic<double> last_operation_time_{0.0};
  std::atomic<uint64_t> total_bytes_transferred_{0};
  std::atomic<double> total_latency_{0.0};
  std::atomic<uint64_t> operation_count_{0};

  std::thread async_worker_thread_;
  std::atomic<bool> async_worker_running_{false};
  std::condition_variable async_cv_;
  std::mutex async_mutex_;
  std::queue<std::function<void()>> async_tasks_;
};

CommunicatorPtr createCommunicator(network::NetworkType type = network::NetworkType::TCP_SOCKET);

}
}