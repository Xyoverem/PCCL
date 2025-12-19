#pragma once

#include <base/ir.h>
#include <base/registry.h>
#include <memory>
#include <vector>
#include <unordered_map>

namespace engine_c {

enum class AllreduceAlgorithm {
  RING,
  TREE,
  RABENSEIFNER,
  DOUBLE_BINARY_TREE
};

struct AllreduceConfig {
  AllreduceAlgorithm algorithm;
  ReduceOp reduce_op;
  std::vector<int> participants;
  int buffer_size;
  bool enable_overlap;
  int pipeline_depth;

  AllreduceConfig(AllreduceAlgorithm alg = AllreduceAlgorithm::RING,
                  ReduceOp op = ReduceOp::SUM,
                  const std::vector<int>& parts = {},
                  int buf_size = 128 * 1024 * 1024,
                  bool overlap = false,
                  int depth = 2)
    : algorithm(alg), reduce_op(op), participants(parts),
      buffer_size(buf_size), enable_overlap(overlap), pipeline_depth(depth) {}
};

class AllreduceImpl {
public:
  AllreduceImpl(const AllreduceConfig& config);
  virtual ~AllreduceImpl() = default;

  virtual void execute(void* input, void* output, size_t data_size, DataType dtype) = 0;
  virtual void executeAsync(void* input, void* output, size_t data_size, DataType dtype) = 0;
  virtual void waitCompletion() = 0;

  const AllreduceConfig& getConfig() const { return config_; }

protected:
  AllreduceConfig config_;
  bool completed_;
};

class RingAllreduce : public AllreduceImpl {
public:
  RingAllreduce(const AllreduceConfig& config);
  ~RingAllreduce() = default;

  void execute(void* input, void* output, size_t data_size, DataType dtype) override;
  void executeAsync(void* input, void* output, size_t data_size, DataType dtype) override;
  void waitCompletion() override;

private:
  int rank_;
  int world_size_;
  int next_rank_;
  int prev_rank_;

  void scatterReduceRing(void* buffer, size_t data_size, DataType dtype);
  void allgatherRing(void* buffer, size_t data_size, DataType dtype);
};

class TreeAllreduce : public AllreduceImpl {
public:
  TreeAllreduce(const AllreduceConfig& config, int branching_factor = 2);
  ~TreeAllreduce() = default;

  void execute(void* input, void* output, size_t data_size, DataType dtype) override;
  void executeAsync(void* input, void* output, size_t data_size, DataType dtype) override;
  void waitCompletion() override;

private:
  int rank_;
  int world_size_;
  int branching_factor_;
  int parent_rank_;
  std::vector<int> child_ranks_;

  void buildTree();
  void reduceTree(void* buffer, size_t data_size, DataType dtype);
  void broadcastTree(void* buffer, size_t data_size, DataType dtype);
};

class RabenseifnerAllreduce : public AllreduceImpl {
public:
  RabenseifnerAllreduce(const AllreduceConfig& config);
  ~RabenseifnerAllreduce() = default;

  void execute(void* input, void* output, size_t data_size, DataType dtype) override;
  void executeAsync(void* input, void* output, size_t data_size, DataType dtype) override;
  void waitCompletion() override;

private:
  std::unique_ptr<RingAllreduce> ring_allreduce_;
  std::unique_ptr<TreeAllreduce> tree_allreduce_;
};

class AllreduceFactory {
public:
  static std::unique_ptr<AllreduceImpl> create(const AllreduceConfig& config);

  static std::unique_ptr<AllreduceImpl> createRing(const AllreduceConfig& config);
  static std::unique_ptr<AllreduceImpl> createTree(const AllreduceConfig& config, int branching_factor = 2);
  static std::unique_ptr<AllreduceImpl> createRabenseifner(const AllreduceConfig& config);
  static std::unique_ptr<AllreduceImpl> createDoubleBinaryTree(const AllreduceConfig& config);

  static AllreduceAlgorithm selectOptimalAlgorithm(size_t data_size, int world_size, bool bandwidth_limited = true);
};

}