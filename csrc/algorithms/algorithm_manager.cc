#include <algorithms/algorithm_manager.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace engine_c {

AlgorithmBenchmark::AlgorithmBenchmark() {}

PerformanceMetrics AlgorithmBenchmark::benchmarkAlgorithm(std::shared_ptr<AllreduceImpl> algorithm,
                                                          void* input_data,
                                                          size_t data_size,
                                                          DataType dtype,
                                                          int num_iterations) {
  PerformanceMetrics metrics;

  void* output_data = malloc(data_size);
  void* temp_input = malloc(data_size);

  std::vector<double> iteration_times(num_iterations);
  std::vector<double> compute_times(num_iterations);
  std::vector<double> comm_times(num_iterations);

  for (int i = 0; i < num_iterations; ++i) {
    memcpy(temp_input, input_data, data_size);

    auto start_time = std::chrono::high_resolution_clock::now();

    algorithm->execute(temp_input, output_data, data_size, dtype);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    iteration_times[i] = duration.count() / 1000.0;

    compute_times[i] = iteration_times[i] * 0.3;
    comm_times[i] = iteration_times[i] * 0.7;
  }

  double total_time = 0.0;
  for (double time : iteration_times) {
    total_time += time;
  }
  metrics.total_time = total_time / num_iterations;

  metrics.bandwidth_utilization = measureBandwidth(data_size, metrics.total_time);
  metrics.latency = measureLatency(iteration_times);
  metrics.overlap_ratio = calculateOverlapRatio(compute_times, comm_times);
  metrics.efficiency = metrics.bandwidth_utilization / 100.0;

  free(output_data);
  free(temp_input);

  return metrics;
}

PerformanceMetrics AlgorithmBenchmark::benchmarkProcessGroup(std::shared_ptr<ProcessGroup> process_group,
                                                           void* input_data,
                                                           size_t data_size,
                                                           DataType dtype,
                                                           int num_iterations) {
  PerformanceMetrics metrics;

  void* output_data = malloc(data_size);
  void* temp_input = malloc(data_size);

  std::vector<double> iteration_times(num_iterations);

  for (int i = 0; i < num_iterations; ++i) {
    memcpy(temp_input, input_data, data_size);

    auto start_time = std::chrono::high_resolution_clock::now();

    process_group->allreduce(temp_input, output_data, data_size, dtype, ReduceOp::SUM);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    iteration_times[i] = duration.count() / 1000.0;
  }

  double total_time = 0.0;
  for (double time : iteration_times) {
    total_time += time;
  }
  metrics.total_time = total_time / num_iterations;

  metrics.bandwidth_utilization = measureBandwidth(data_size, metrics.total_time);
  metrics.latency = measureLatency(iteration_times);
  metrics.overlap_ratio = 0.0;
  metrics.efficiency = metrics.bandwidth_utilization / 100.0;

  free(output_data);
  free(temp_input);

  return metrics;
}

void AlgorithmBenchmark::saveResults(const std::string& filepath) const {
  std::ofstream file(filepath);
  if (!file.is_open()) return;

  file << "# Algorithm Benchmark Results\n";
  file << "algorithm_name,data_size,num_participants,bandwidth_utilization,latency,overlap_ratio,efficiency,total_time\n";

  for (const auto& profile : benchmark_results_) {
    file << profile.algorithm_name << ","
         << profile.data_size << ","
         << profile.num_participants << ","
         << profile.metrics.bandwidth_utilization << ","
         << profile.metrics.latency << ","
         << profile.metrics.overlap_ratio << ","
         << profile.metrics.efficiency << ","
         << profile.metrics.total_time << "\n";
  }
}

void AlgorithmBenchmark::loadResults(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) return;

  benchmark_results_.clear();

  std::string line;
  std::getline(file, line);

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    std::string token;
    AlgorithmProfile profile("", AllreduceAlgorithm::RING);

    std::getline(iss, token, ',');
    profile.algorithm_name = token;

    std::getline(iss, token, ',');
    profile.data_size = std::stoull(token);

    std::getline(iss, token, ',');
    profile.num_participants = std::stoi(token);

    std::getline(iss, token, ',');
    profile.metrics.bandwidth_utilization = std::stod(token);

    std::getline(iss, token, ',');
    profile.metrics.latency = std::stod(token);

    std::getline(iss, token, ',');
    profile.metrics.overlap_ratio = std::stod(token);

    std::getline(iss, token, ',');
    profile.metrics.efficiency = std::stod(token);

    std::getline(iss, token, ',');
    profile.metrics.total_time = std::stod(token);

    benchmark_results_.push_back(profile);
  }
}

double AlgorithmBenchmark::measureBandwidth(size_t data_size, double time_seconds) const {
  if (time_seconds <= 0.0) return 0.0;
  double bandwidth_gbps = (static_cast<double>(data_size) / (1024.0 * 1024.0 * 1024.0)) / time_seconds;
  return bandwidth_gbps;
}

double AlgorithmBenchmark::measureLatency(const std::vector<double>& times) const {
  if (times.empty()) return 0.0;
  return times[times.size() / 2];
}

double AlgorithmBenchmark::calculateOverlapRatio(const std::vector<double>& compute_times,
                                                const std::vector<double>& comm_times) const {
  if (compute_times.size() != comm_times.size() || compute_times.empty()) return 0.0;

  double total_compute = 0.0, total_comm = 0.0;
  for (size_t i = 0; i < compute_times.size(); ++i) {
    total_compute += compute_times[i];
    total_comm += comm_times[i];
  }

  double avg_compute = total_compute / compute_times.size();
  double avg_comm = total_comm / comm_times.size();

  if (avg_compute + avg_comm == 0.0) return 0.0;
  return avg_compute / (avg_compute + avg_comm);
}

AlgorithmOptimizer::AlgorithmOptimizer() {}

AlgorithmSelector AlgorithmOptimizer::selectOptimalAlgorithm(size_t data_size,
                                                           int num_participants,
                                                           const TopologyMetrics& topology_metrics,
                                                           bool enable_overlap) {
  auto candidates = generateCandidates(data_size, num_participants, topology_metrics);

  AlgorithmSelector best_selector;
  double best_score = -1.0;

  for (const auto& selector : candidates) {
    double score = predictPerformance(selector, data_size, num_participants, topology_metrics);
    if (score > best_score) {
      best_score = score;
      best_selector = selector;
    }
  }

  if (enable_overlap) {
    best_selector.enable_overlap = true;
    best_selector.pipeline_depth = std::max(best_selector.pipeline_depth, 4);
  }

  return best_selector;
}

std::vector<AlgorithmSelector> AlgorithmOptimizer::generateCandidates(size_t data_size,
                                                                     int num_participants,
                                                                     const TopologyMetrics& topology_metrics) {
  std::vector<AlgorithmSelector> candidates;

  AlgorithmSelector ring_selector;
  ring_selector.allreduce_algorithm = AllreduceAlgorithm::RING;
  ring_selector.branching_factor = 2;
  candidates.push_back(ring_selector);

  if (num_participants > 2) {
    AlgorithmSelector tree_selector;
    tree_selector.allreduce_algorithm = AllreduceAlgorithm::TREE;
    tree_selector.branching_factor = std::min(num_participants, 4);
    candidates.push_back(tree_selector);
  }

  if (data_size > 1024 * 1024 && num_participants > 4) {
    AlgorithmSelector rabenseifner_selector;
    rabenseifner_selector.allreduce_algorithm = AllreduceAlgorithm::RABENSEIFNER;
    rabenseifner_selector.branching_factor = 2;
    candidates.push_back(rabenseifner_selector);
  }

  if (num_participants % 2 == 0 && num_participants >= 4) {
    AlgorithmSelector double_tree_selector;
    double_tree_selector.allreduce_algorithm = AllreduceAlgorithm::DOUBLE_BINARY_TREE;
    double_tree_selector.branching_factor = 2;
    candidates.push_back(double_tree_selector);
  }

  return candidates;
}

double AlgorithmOptimizer::predictPerformance(const AlgorithmSelector& selector,
                                             size_t data_size,
                                             int num_participants,
                                             const TopologyMetrics& topology_metrics) {
  double estimated_time = 0.0;

  switch (selector.allreduce_algorithm) {
    case AllreduceAlgorithm::RING:
      estimated_time = calculateRingCost(data_size, num_participants, topology_metrics);
      break;
    case AllreduceAlgorithm::TREE:
      estimated_time = calculateTreeCost(data_size, num_participants, topology_metrics, selector.branching_factor);
      break;
    case AllreduceAlgorithm::RABENSEIFNER:
      estimated_time = calculateRabenseifnerCost(data_size, num_participants, topology_metrics);
      break;
    case AllreduceAlgorithm::DOUBLE_BINARY_TREE:
      estimated_time = calculateTreeCost(data_size, num_participants, topology_metrics, 2) * 0.8;
      break;
  }

  if (selector.enable_overlap) {
    estimated_time *= (1.0 - 0.3 * (selector.pipeline_depth / 4.0));
  }

  double efficiency = topology_metrics.connectivity / 100.0;
  estimated_time /= efficiency;

  return 1.0 / estimated_time;
}

void AlgorithmOptimizer::updateModel(const std::vector<AlgorithmProfile>& training_data) {
  if (training_data.empty()) return;

  double total_bandwidth = 0.0, total_latency = 0.0, total_topology = 0.0;
  int count = 0;

  for (const auto& profile : training_data) {
    total_bandwidth += profile.metrics.bandwidth_utilization;
    total_latency += 1000.0 / profile.metrics.latency;
    total_topology += profile.metrics.efficiency;
    count++;
  }

  if (count > 0) {
    model_params_.bandwidth_weight = total_bandwidth / (count * 100.0);
    model_params_.latency_weight = total_latency / count;
    model_params_.topology_weight = total_topology / count;
    model_params_.size_weight = 1.0 - model_params_.bandwidth_weight -
                                model_params_.latency_weight - model_params_.topology_weight;
  }
}

double AlgorithmOptimizer::calculateRingCost(size_t data_size, int num_participants, const TopologyMetrics& metrics) {
  double alpha = metrics.average_latency;
  double beta = data_size / metrics.total_bandwidth;
  return 2 * (num_participants - 1) * (alpha + beta);
}

double AlgorithmOptimizer::calculateTreeCost(size_t data_size, int num_participants, const TopologyMetrics& metrics, int branching_factor) {
  double alpha = metrics.average_latency;
  double beta = data_size / metrics.total_bandwidth;
  int tree_depth = static_cast<int>(std::ceil(std::log(num_participants) / std::log(branching_factor)));
  return 2 * tree_depth * (alpha + beta);
}

double AlgorithmOptimizer::calculateRabenseifnerCost(size_t data_size, int num_participants, const TopologyMetrics& metrics) {
  double alpha = metrics.average_latency;
  double beta = data_size / metrics.total_bandwidth;
  return (num_participants - 1) * alpha + (2.0 * (num_participants - 1.0) / num_participants) * beta;
}

AlgorithmManager::AlgorithmManager() {
  benchmark_ = std::make_unique<AlgorithmBenchmark>();
  optimizer_ = std::make_unique<AlgorithmOptimizer>();
}

void AlgorithmManager::initialize() {
  registerBuiltInAlgorithms();
}

std::shared_ptr<AllreduceImpl> AlgorithmManager::createAllreduce(const AllreduceConfig& config) {
  switch (config.algorithm) {
    case AllreduceAlgorithm::RING:
      return AllreduceFactory::createRing(config);
    case AllreduceAlgorithm::TREE:
      return AllreduceFactory::createTree(config, 2);
    case AllreduceAlgorithm::RABENSEIFNER:
      return AllreduceFactory::createRabenseifner(config);
    case AllreduceAlgorithm::DOUBLE_BINARY_TREE:
      return AllreduceFactory::createDoubleBinaryTree(config);
    default:
      return AllreduceFactory::createRing(config);
  }
}

std::shared_ptr<AllreduceImpl> AlgorithmManager::createOptimalAllreduce(size_t data_size,
                                                                        ReduceOp reduce_op,
                                                                        const std::vector<int>& participants,
                                                                        std::shared_ptr<Topology> topology) {
  TopologyMetrics metrics = {};
  if (topology) {
    metrics = topology->calculateMetrics();
  }

  auto selector = selectAlgorithm(data_size, participants.size(), topology, false);

  AllreduceConfig config;
  config.algorithm = selector.allreduce_algorithm;
  config.reduce_op = reduce_op;
  config.participants = participants;
  config.buffer_size = selector.buffer_size;
  config.enable_overlap = selector.enable_overlap;
  config.pipeline_depth = selector.pipeline_depth;

  return createAllreduce(config);
}

AlgorithmSelector AlgorithmManager::selectAlgorithm(size_t data_size,
                                                   int num_participants,
                                                   std::shared_ptr<Topology> topology,
                                                   bool enable_overlap) {
  std::string cache_key = generateCacheKey(data_size, num_participants, topology ? topology->calculateMetrics() : TopologyMetrics{});

  auto cached_result = getFromCache(cache_key);
  if (cached_result.buffer_size > 0) {
    if (enable_overlap) {
      cached_result.enable_overlap = true;
    }
    return cached_result;
  }

  TopologyMetrics metrics = {};
  if (topology) {
    metrics = topology->calculateMetrics();
  }

  auto selector = optimizer_->selectOptimalAlgorithm(data_size, num_participants, metrics, enable_overlap);
  updateCache(cache_key, selector);

  return selector;
}

PerformanceMetrics AlgorithmManager::benchmarkAllreduce(const std::string& algorithm_name,
                                                       size_t data_size,
                                                       const std::vector<int>& participants,
                                                       std::shared_ptr<Topology> topology) {
  AllreduceConfig config;
  config.reduce_op = ReduceOp::SUM;
  config.participants = participants;

  if (algorithm_name == "ring") {
    config.algorithm = AllreduceAlgorithm::RING;
  } else if (algorithm_name == "tree") {
    config.algorithm = AllreduceAlgorithm::TREE;
  } else if (algorithm_name == "rabenseifner") {
    config.algorithm = AllreduceAlgorithm::RABENSEIFNER;
  } else {
    config.algorithm = AllreduceAlgorithm::RING;
  }

  auto algorithm = createAllreduce(config);

  void* input_data = malloc(data_size);
  memset(input_data, 1, data_size);

  auto metrics = benchmark_->benchmarkAlgorithm(algorithm, input_data, data_size, DataType::FLOAT32);

  AlgorithmProfile profile(algorithm_name, config.algorithm);
  profile.metrics = metrics;
  profile.data_size = data_size;
  profile.num_participants = participants.size();

  benchmark_results_.push_back(profile);

  free(input_data);
  return metrics;
}

void AlgorithmManager::autoTune(size_t data_size,
                               const std::vector<int>& participants,
                               std::shared_ptr<Topology> topology) {
  std::vector<std::string> algorithms = {"ring", "tree", "rabenseifner"};

  for (const auto& algorithm_name : algorithms) {
    benchmarkAllreduce(algorithm_name, data_size, participants, topology);
  }

  optimizer_->updateModel(benchmark_results_);
}

void AlgorithmManager::addCustomAlgorithm(const std::string& name,
                                         std::function<std::shared_ptr<AllreduceImpl>(const AllreduceConfig&)> factory) {
  custom_algorithms_[name] = factory;
}

void AlgorithmManager::removeCustomAlgorithm(const std::string& name) {
  custom_algorithms_.erase(name);
}

std::vector<std::string> AlgorithmManager::getAvailableAlgorithms() const {
  std::vector<std::string> algorithms = {"ring", "tree", "rabenseifner", "double_binary_tree"};

  for (const auto& pair : custom_algorithms_) {
    algorithms.push_back(pair.first);
  }

  return algorithms;
}

void AlgorithmManager::saveProfiles(const std::string& filepath) const {
  benchmark_->saveResults(filepath);
}

void AlgorithmManager::loadProfiles(const std::string& filepath) {
  benchmark_->loadResults(filepath);
  benchmark_results_ = benchmark_->getBenchmarkResults();
  optimizer_->updateModel(benchmark_results_);
}

void AlgorithmManager::setTopologyManager(std::shared_ptr<TopologyManager> topology_manager) {
  topology_manager_ = topology_manager;
}

void AlgorithmManager::setProcessGroupManager(std::shared_ptr<ProcessGroupManager> process_group_manager) {
  process_group_manager_ = process_group_manager;
}

void AlgorithmManager::registerBuiltInAlgorithms() {
}

std::shared_ptr<AllreduceImpl> AlgorithmManager::createBuiltInAlgorithm(AllreduceAlgorithm algorithm, const AllreduceConfig& config) {
  return AllreduceFactory::create(config);
}

void AlgorithmManager::updateCache(const std::string& key, const AlgorithmSelector& selector) {
  algorithm_cache_[key] = selector;
}

AlgorithmSelector AlgorithmManager::getFromCache(const std::string& key) const {
  auto it = algorithm_cache_.find(key);
  return (it != algorithm_cache_.end()) ? it->second : AlgorithmSelector();
}

std::string AlgorithmManager::generateCacheKey(size_t data_size, int num_participants, const TopologyMetrics& metrics) const {
  return std::to_string(data_size) + "_" + std::to_string(num_participants) + "_" +
         std::to_string(static_cast<int>(metrics.total_bandwidth)) + "_" +
         std::to_string(static_cast<int>(metrics.average_latency)) + "_" +
         std::to_string(metrics.connectivity);
}

}