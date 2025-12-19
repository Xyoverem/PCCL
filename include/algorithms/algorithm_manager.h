#pragma once

#include <base/registry.h>
#include <algorithms/allreduce.h>
#include <topology/topology.h>
#include <cluster/process_group.h>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <functional>

namespace engine_c {

struct AlgorithmSelector {
  AllreduceAlgorithm allreduce_algorithm;
  int branching_factor;
  bool enable_overlap;
  int pipeline_depth;
  int buffer_size;

  AlgorithmSelector()
    : allreduce_algorithm(AllreduceAlgorithm::RING),
      branching_factor(2),
      enable_overlap(false),
      pipeline_depth(2),
      buffer_size(128 * 1024 * 1024) {}
};

struct PerformanceMetrics {
  double bandwidth_utilization;
  double latency;
  double overlap_ratio;
  double efficiency;
  double total_time;

  PerformanceMetrics()
    : bandwidth_utilization(0.0),
      latency(0.0),
      overlap_ratio(0.0),
      efficiency(0.0),
      total_time(0.0) {}
};

struct AlgorithmProfile {
  std::string algorithm_name;
  AllreduceAlgorithm algorithm_type;
  PerformanceMetrics metrics;
  size_t data_size;
  int num_participants;
  std::unordered_map<std::string, std::string> parameters;

  AlgorithmProfile(const std::string& name, AllreduceAlgorithm type)
    : algorithm_name(name), algorithm_type(type), data_size(0), num_participants(0) {}
};

class AlgorithmBenchmark {
public:
  AlgorithmBenchmark();
  ~AlgorithmBenchmark() = default;

  PerformanceMetrics benchmarkAlgorithm(std::shared_ptr<AllreduceImpl> algorithm,
                                       void* input_data,
                                       size_t data_size,
                                       DataType dtype,
                                       int num_iterations = 10);

  PerformanceMetrics benchmarkProcessGroup(std::shared_ptr<ProcessGroup> process_group,
                                          void* input_data,
                                          size_t data_size,
                                          DataType dtype,
                                          int num_iterations = 10);

  void saveResults(const std::string& filepath) const;
  void loadResults(const std::string& filepath);

private:
  std::vector<AlgorithmProfile> benchmark_results_;

  double measureBandwidth(size_t data_size, double time_seconds) const;
  double measureLatency(const std::vector<double>& times) const;
  double calculateOverlapRatio(const std::vector<double>& compute_times,
                              const std::vector<double>& comm_times) const;
};

class AlgorithmOptimizer {
public:
  AlgorithmOptimizer();
  ~AlgorithmOptimizer() = default;

  AlgorithmSelector selectOptimalAlgorithm(size_t data_size,
                                          int num_participants,
                                          const TopologyMetrics& topology_metrics,
                                          bool enable_overlap = false);

  std::vector<AlgorithmSelector> generateCandidates(size_t data_size,
                                                   int num_participants,
                                                   const TopologyMetrics& topology_metrics);

  double predictPerformance(const AlgorithmSelector& selector,
                           size_t data_size,
                           int num_participants,
                           const TopologyMetrics& topology_metrics);

  void updateModel(const std::vector<AlgorithmProfile>& training_data);

private:
  struct ModelParameters {
    double bandwidth_weight;
    double latency_weight;
    double topology_weight;
    double size_weight;

    ModelParameters()
      : bandwidth_weight(0.4),
        latency_weight(0.3),
        topology_weight(0.2),
        size_weight(0.1) {}
  };

  ModelParameters model_params_;

  double calculateRingCost(size_t data_size, int num_participants, const TopologyMetrics& metrics);
  double calculateTreeCost(size_t data_size, int num_participants, const TopologyMetrics& metrics, int branching_factor);
  double calculateRabenseifnerCost(size_t data_size, int num_participants, const TopologyMetrics& metrics);
};

class AlgorithmManager {
public:
  AlgorithmManager();
  ~AlgorithmManager() = default;

  void initialize();

  std::shared_ptr<AllreduceImpl> createAllreduce(const AllreduceConfig& config);

  std::shared_ptr<AllreduceImpl> createOptimalAllreduce(size_t data_size,
                                                       ReduceOp reduce_op,
                                                       const std::vector<int>& participants,
                                                       std::shared_ptr<Topology> topology);

  AlgorithmSelector selectAlgorithm(size_t data_size,
                                   int num_participants,
                                   std::shared_ptr<Topology> topology,
                                   bool enable_overlap = false);

  PerformanceMetrics benchmarkAllreduce(const std::string& algorithm_name,
                                       size_t data_size,
                                       const std::vector<int>& participants,
                                       std::shared_ptr<Topology> topology);

  void autoTune(size_t data_size,
                const std::vector<int>& participants,
                std::shared_ptr<Topology> topology);

  void addCustomAlgorithm(const std::string& name,
                         std::function<std::shared_ptr<AllreduceImpl>(const AllreduceConfig&)> factory);

  void removeCustomAlgorithm(const std::string& name);

  std::vector<std::string> getAvailableAlgorithms() const;

  void saveProfiles(const std::string& filepath) const;
  void loadProfiles(const std::string& filepath);

  const std::vector<AlgorithmProfile>& getBenchmarkResults() const { return benchmark_results_; }

  void setTopologyManager(std::shared_ptr<TopologyManager> topology_manager);
  void setProcessGroupManager(std::shared_ptr<ProcessGroupManager> process_group_manager);

private:
  std::unique_ptr<AlgorithmBenchmark> benchmark_;
  std::unique_ptr<AlgorithmOptimizer> optimizer_;
  std::shared_ptr<TopologyManager> topology_manager_;
  std::shared_ptr<ProcessGroupManager> process_group_manager_;

  std::unordered_map<std::string, std::function<std::shared_ptr<AllreduceImpl>(const AllreduceConfig&)>> custom_algorithms_;
  std::vector<AlgorithmProfile> benchmark_results_;

  void registerBuiltInAlgorithms();
  std::shared_ptr<AllreduceImpl> createBuiltInAlgorithm(AllreduceAlgorithm algorithm, const AllreduceConfig& config);

  void updateCache(const std::string& key, const AlgorithmSelector& selector);
  AlgorithmSelector getFromCache(const std::string& key) const;

  std::unordered_map<std::string, AlgorithmSelector> algorithm_cache_;

  std::string generateCacheKey(size_t data_size, int num_participants, const TopologyMetrics& metrics) const;
};

}