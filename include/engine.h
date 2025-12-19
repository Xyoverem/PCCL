#pragma once

#include "ATen/core/TensorBody.h"
#include <cluster/manager.h>
#include <base/chunk.h>
#include <base/operator.h>
#include <torch/extension.h>
#include <memory>
#include <string>
#include <cstring>

namespace engine_c {

class Engine {
public:
  Engine(int rank, int world_size);
  void initEngine();

  void regOp(const std::string &name, const std::string &filepath);
  void exeOp(const std::string &name, at::Tensor &input, at::Tensor output);

  std::string exportEndpoint();
  void joinCluster(const std::string &master_endpoint);
  void exitCluster();

private:
  int rank_, world_size_;
  std::unique_ptr<ClusterManager> cluster_;
  std::unique_ptr<BufferManager> buffers_;
  std::unique_ptr<OperatorManager> operators_;

  std::vector<std::string> available_remote_devices_;
  std::vector<std::string> available_memory_devices_;
};

}

