#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <memory>
#include <algorithm>
#include <random>
#include <iomanip>
#include <fstream>
#include <network/network.h>
#include <communication/communicator.h>
#include <cluster/process_group.h>
#include <algorithms/communication_wrapper.h>

namespace engine_c {
namespace benchmark {

struct BenchmarkResult {
  std::string test_name;
  double latency_us;
  double bandwidth_mbps;
  double efficiency;
  int message_size;
  int iterations;
  double total_time_ms;
};

class PerformanceBenchmark {
public:
  PerformanceBenchmark();
  ~PerformanceBenchmark() = default;

  void runAllBenchmarks();
  void runLatencyBenchmark();
  void runBandwidthBenchmark();
  void runAllreduceBenchmark();
  void runProcessGroupBenchmark();

  void setNetworkType(network::NetworkType type);
  void setMessageSizes(const std::vector<int>& sizes);
  void setIterations(int iterations);
  void setWorldSize(int world_size);

  void saveResults(const std::string& filename);

private:
  network::NetworkType network_type_;
  std::vector<int> message_sizes_;
  int iterations_;
  int world_size_;

  std::vector<BenchmarkResult> results_;

  void printHeader();
  void printResult(const BenchmarkResult& result);
  void printSummary();

  double calculatePercentile(const std::vector<double>& values, double percentile);
  double calculateMean(const std::vector<double>& values);
  double calculateStdDev(const std::vector<double>& values, double mean);

  std::vector<double> measureLatencyMultipleRuns(int message_size, int runs);
  std::vector<double> measureBandwidthMultipleRuns(int message_size, int runs);

  void benchmarkTcpLatency(int message_size);
  void benchmarkTcpBandwidth(int message_size);
  void benchmarkRdmaLatency(int message_size);
  void benchmarkRdmaBandwidth(int message_size);

  void benchmarkAllreduceRing(int message_size);
  void benchmarkAllreduceTree(int message_size);
  void benchmarkAllreduceRabenseifner(int message_size);

  void benchmarkProcessGroupLatency(const std::string& group_type, int message_size);
  void benchmarkProcessGroupThroughput(const std::string& group_type, int message_size);
};

PerformanceBenchmark::PerformanceBenchmark()
  : network_type_(network::NetworkType::TCP_SOCKET),
    iterations_(1000),
    world_size_(4) {

  message_sizes_ = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608
  };
}

void PerformanceBenchmark::setNetworkType(network::NetworkType type) {
  network_type_ = type;
}

void PerformanceBenchmark::setMessageSizes(const std::vector<int>& sizes) {
  message_sizes_ = sizes;
}

void PerformanceBenchmark::setIterations(int iterations) {
  iterations_ = iterations;
}

void PerformanceBenchmark::setWorldSize(int world_size) {
  world_size_ = world_size;
}

void PerformanceBenchmark::runAllBenchmarks() {
  printHeader();

  std::cout << "Running comprehensive performance benchmarks..." << std::endl;
  std::cout << "Network type: " << (network_type_ == network::NetworkType::TCP_SOCKET ? "TCP" : "RDMA") << std::endl;
  std::cout << "World size: " << world_size_ << std::endl;
  std::cout << "Iterations per test: " << iterations_ << std::endl;
  std::cout << std::endl;

  runLatencyBenchmark();
  runBandwidthBenchmark();
  runAllreduceBenchmark();
  runProcessGroupBenchmark();

  printSummary();
}

void PerformanceBenchmark::runLatencyBenchmark() {
  std::cout << "=== Latency Benchmark ===" << std::endl;

  for (int size : message_sizes_) {
    if (network_type_ == network::NetworkType::TCP_SOCKET) {
      benchmarkTcpLatency(size);
    } else {
      benchmarkRdmaLatency(size);
    }
  }
}

void PerformanceBenchmark::runBandwidthBenchmark() {
  std::cout << "\n=== Bandwidth Benchmark ===" << std::endl;

  for (int size : message_sizes_) {
    if (network_type_ == network::NetworkType::TCP_SOCKET) {
      benchmarkTcpBandwidth(size);
    } else {
      benchmarkRdmaBandwidth(size);
    }
  }
}

void PerformanceBenchmark::runAllreduceBenchmark() {
  std::cout << "\n=== AllReduce Benchmark ===" << std::endl;

  std::vector<std::string> algorithms = {"ring", "tree", "rabenseifner"};

  for (int size : message_sizes_) {
    if (size > 1024 * 1024) break;

    for (const auto& algo : algorithms) {
      if (algo == "ring") {
        benchmarkAllreduceRing(size);
      } else if (algo == "tree") {
        benchmarkAllreduceTree(size);
      } else if (algo == "rabenseifner") {
        benchmarkAllreduceRabenseifner(size);
      }
    }
  }
}

void PerformanceBenchmark::runProcessGroupBenchmark() {
  std::cout << "\n=== Process Group Benchmark ===" << std::endl;

  std::vector<std::string> group_types = {"cpu", "cuda", "rdma"};

  for (const auto& type : group_types) {
    for (int size : {1024, 4096, 16384, 65536, 262144, 1048576}) {
      benchmarkProcessGroupLatency(type, size);
      benchmarkProcessGroupThroughput(type, size);
    }
  }
}

void PerformanceBenchmark::benchmarkTcpLatency(int message_size) {
  auto latencies = measureLatencyMultipleRuns(message_size, 10);

  double mean_latency = calculateMean(latencies);
  double std_dev = calculateStdDev(latencies, mean_latency);
  double p95 = calculatePercentile(latencies, 95.0);

  BenchmarkResult result;
  result.test_name = "TCP_Latency_" + std::to_string(message_size);
  result.latency_us = mean_latency;
  result.bandwidth_mbps = 0.0;
  result.efficiency = (std_dev / mean_latency) * 100.0;
  result.message_size = message_size;
  result.iterations = iterations_;
  result.total_time_ms = mean_latency * iterations_ / 1000.0;

  printResult(result);
  results_.push_back(result);
}

void PerformanceBenchmark::benchmarkTcpBandwidth(int message_size) {
  auto throughputs = measureBandwidthMultipleRuns(message_size, 10);

  double mean_throughput = calculateMean(throughputs);
  double std_dev = calculateStdDev(throughputs, mean_throughput);

  BenchmarkResult result;
  result.test_name = "TCP_Bandwidth_" + std::to_string(message_size);
  result.latency_us = 0.0;
  result.bandwidth_mbps = mean_throughput;
  result.efficiency = (std_dev / mean_throughput) * 100.0;
  result.message_size = message_size;
  result.iterations = 50;
  result.total_time_ms = (message_size * 50.0 / 1024.0 / 1024.0) / mean_throughput * 1000.0;

  printResult(result);
  results_.push_back(result);
}

void PerformanceBenchmark::benchmarkAllreduceRing(int message_size) {
  auto comm = algorithms::CommunicationWrapper();

  if (!comm.initialize(0, world_size_, network_type_)) {
    return;
  }

  std::vector<float> send_data(message_size / sizeof(float), 1.0f);
  std::vector<float> recv_data(message_size / sizeof(float));

  std::vector<double> times;
  times.reserve(10);

  for (int i = 0; i < 10; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    comm.allReduceRing(send_data.data(), recv_data.data(), message_size, communication::ReductionOp::SUM);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    times.push_back(duration.count());
  }

  comm.finalize();

  double mean_time = calculateMean(times);
  double bandwidth_mbps = (message_size / 1024.0 / 1024.0) / (mean_time / 1000000.0);

  BenchmarkResult result;
  result.test_name = "AllReduce_Ring_" + std::to_string(message_size);
  result.latency_us = mean_time;
  result.bandwidth_mbps = bandwidth_mbps;
  result.efficiency = (bandwidth_mbps / (10000.0 * (world_size_ - 1) / world_size_)) * 100.0;
  result.message_size = message_size;
  result.iterations = 10;
  result.total_time_ms = mean_time / 1000.0;

  printResult(result);
  results_.push_back(result);
}

void PerformanceBenchmark::benchmarkAllreduceTree(int message_size) {
  auto comm = algorithms::CommunicationWrapper();

  if (!comm.initialize(0, world_size_, network_type_)) {
    return;
  }

  std::vector<float> send_data(message_size / sizeof(float), 1.0f);
  std::vector<float> recv_data(message_size / sizeof(float));

  std::vector<double> times;
  times.reserve(10);

  for (int i = 0; i < 10; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    comm.allReduceTree(send_data.data(), recv_data.data(), message_size, communication::ReductionOp::SUM);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    times.push_back(duration.count());
  }

  comm.finalize();

  double mean_time = calculateMean(times);
  double bandwidth_mbps = (message_size / 1024.0 / 1024.0) / (mean_time / 1000000.0);

  BenchmarkResult result;
  result.test_name = "AllReduce_Tree_" + std::to_string(message_size);
  result.latency_us = mean_time;
  result.bandwidth_mbps = bandwidth_mbps;
  result.efficiency = (bandwidth_mbps / (10000.0 * (world_size_ - 1) / world_size_)) * 100.0;
  result.message_size = message_size;
  result.iterations = 10;
  result.total_time_ms = mean_time / 1000.0;

  printResult(result);
  results_.push_back(result);
}

void PerformanceBenchmark::benchmarkAllreduceRabenseifner(int message_size) {
  auto comm = algorithms::CommunicationWrapper();

  if (!comm.initialize(0, world_size_, network_type_)) {
    return;
  }

  std::vector<float> send_data(message_size / sizeof(float), 1.0f);
  std::vector<float> recv_data(message_size / sizeof(float));

  std::vector<double> times;
  times.reserve(10);

  for (int i = 0; i < 10; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    comm.allReduceRabenseifner(send_data.data(), recv_data.data(), message_size, communication::ReductionOp::SUM);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    times.push_back(duration.count());
  }

  comm.finalize();

  double mean_time = calculateMean(times);
  double bandwidth_mbps = (message_size / 1024.0 / 1024.0) / (mean_time / 1000000.0);

  BenchmarkResult result;
  result.test_name = "AllReduce_Rabenseifner_" + std::to_string(message_size);
  result.latency_us = mean_time;
  result.bandwidth_mbps = bandwidth_mbps;
  result.efficiency = (bandwidth_mbps / (10000.0 * (world_size_ - 1) / world_size_)) * 100.0;
  result.message_size = message_size;
  result.iterations = 10;
  result.total_time_ms = mean_time / 1000.0;

  printResult(result);
  results_.push_back(result);
}

void PerformanceBenchmark::benchmarkProcessGroupLatency(const std::string& group_type, int message_size) {
  auto manager = std::make_unique<ProcessGroupManager>();
  manager->initialize(0, world_size_);

  std::shared_ptr<ProcessGroup> group;

  std::vector<int> all_ranks(world_size_);
  std::iota(all_ranks.begin(), all_ranks.end(), 0);

  if (group_type == "cpu") {
    group = manager->createCPUGroup("test_group", all_ranks);
  } else if (group_type == "cuda") {
    group = manager->createCUDAGroup("test_group", all_ranks, {0});
  } else if (group_type == "rdma") {
    group = manager->createRDMAGroup("test_group", all_ranks, {});
  }

  if (!group) return;

  std::vector<char> send_data(message_size, 'x');
  std::vector<char> recv_data(message_size);

  std::vector<double> times;
  times.reserve(10);

  for (int i = 0; i < 10; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    group->allreduce(send_data.data(), recv_data.data(), message_size, DataType::UINT8, ReduceOp::SUM);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    times.push_back(duration.count());
  }

  double mean_time = calculateMean(times);

  BenchmarkResult result;
  result.test_name = "ProcessGroup_" + group_type + "_Latency_" + std::to_string(message_size);
  result.latency_us = mean_time;
  result.bandwidth_mbps = (message_size / 1024.0 / 1024.0) / (mean_time / 1000000.0);
  result.efficiency = 100.0;
  result.message_size = message_size;
  result.iterations = 10;
  result.total_time_ms = mean_time / 1000.0;

  printResult(result);
  results_.push_back(result);
}

void PerformanceBenchmark::benchmarkProcessGroupThroughput(const std::string& group_type, int message_size) {
  benchmarkProcessGroupLatency(group_type, message_size);
}

std::vector<double> PerformanceBenchmark::measureLatencyMultipleRuns(int message_size, int runs) {
  std::vector<double> latencies;
  latencies.reserve(runs);

  for (int i = 0; i < runs; ++i) {
    auto conn = network::NetworkManager::getInstance().createConnection();
    if (!conn) {
      latencies.push_back(-1.0);
      continue;
    }

    network::NetworkAddress addr("127.0.0.1", 12345);
    if (conn->connect(addr)) {
      network::MessageHeader header;
      header.message_id = i;
      header.data_size = message_size;
      header.source_rank = 0;
      header.dest_rank = 1;
      header.tag = 0;
      header.flags = 0;

      std::vector<char> data(message_size, 'x');

      auto start = std::chrono::high_resolution_clock::now();
      conn->sendMessage(header, data.data());
      auto end = std::chrono::high_resolution_clock::now();

      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      latencies.push_back(duration.count());
    } else {
      latencies.push_back(-1.0);
    }
  }

  return latencies;
}

std::vector<double> PerformanceBenchmark::measureBandwidthMultipleRuns(int message_size, int runs) {
  std::vector<double> throughputs;
  throughputs.reserve(runs);

  for (int i = 0; i < runs; ++i) {
    auto conn = network::NetworkManager::getInstance().createConnection();
    if (!conn) {
      throughputs.push_back(-1.0);
      continue;
    }

    network::NetworkAddress addr("127.0.0.1", 12345);
    if (conn->connect(addr)) {
      network::MessageHeader header;
      header.message_id = i;
      header.data_size = message_size;
      header.source_rank = 0;
      header.dest_rank = 1;
      header.tag = 0;
      header.flags = 0;

      std::vector<char> data(message_size, 'x');

      auto start = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < 50; ++j) {
        conn->sendMessage(header, data.data());
      }
      auto end = std::chrono::high_resolution_clock::now();

      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      double total_seconds = duration.count() / 1000000.0;
      double total_mb = (message_size * 50.0) / 1024.0 / 1024.0;
      double throughput = total_mb / total_seconds;

      throughputs.push_back(throughput);
    } else {
      throughputs.push_back(-1.0);
    }
  }

  return throughputs;
}

double PerformanceBenchmark::calculateMean(const std::vector<double>& values) {
  double sum = 0.0;
  int count = 0;

  for (double val : values) {
    if (val > 0) {
      sum += val;
      count++;
    }
  }

  return count > 0 ? sum / count : 0.0;
}

double PerformanceBenchmark::calculateStdDev(const std::vector<double>& values, double mean) {
  double sum_sq = 0.0;
  int count = 0;

  for (double val : values) {
    if (val > 0) {
      sum_sq += (val - mean) * (val - mean);
      count++;
    }
  }

  return count > 1 ? std::sqrt(sum_sq / (count - 1)) : 0.0;
}

double PerformanceBenchmark::calculatePercentile(const std::vector<double>& values, double percentile) {
  std::vector<double> sorted_values;
  for (double val : values) {
    if (val > 0) {
      sorted_values.push_back(val);
    }
  }

  if (sorted_values.empty()) return 0.0;

  std::sort(sorted_values.begin(), sorted_values.end());

  double index = (percentile / 100.0) * (sorted_values.size() - 1);
  int lower = static_cast<int>(std::floor(index));
  int upper = static_cast<int>(std::ceil(index));

  if (lower == upper) {
    return sorted_values[lower];
  }

  double weight = index - lower;
  return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight;
}

void PerformanceBenchmark::printHeader() {
  std::cout << std::string(120, '=') << std::endl;
  std::cout << "                    PCCL Performance Benchmark Suite" << std::endl;
  std::cout << std::string(120, '=') << std::endl;
}

void PerformanceBenchmark::printResult(const BenchmarkResult& result) {
  std::cout << std::left << std::setw(40) << result.test_name
            << std::setw(12) << std::fixed << std::setprecision(2) << result.latency_us << " μs"
            << std::setw(12) << std::fixed << std::setprecision(2) << result.bandwidth_mbps << " MB/s"
            << std::setw(10) << std::fixed << std::setprecision(1) << result.efficiency << "%"
            << std::setw(12) << result.message_size << " B"
            << std::setw(10) << result.iterations
            << std::setw(12) << std::fixed << std::setprecision(2) << result.total_time_ms << " ms"
            << std::endl;
}

void PerformanceBenchmark::printSummary() {
  std::cout << "\n" << std::string(120, '-') << std::endl;
  std::cout << "Benchmark Summary: " << results_.size() << " tests completed" << std::endl;

  double min_latency = std::numeric_limits<double>::max();
  double max_bandwidth = 0.0;
  double avg_efficiency = 0.0;

  for (const auto& result : results_) {
    if (result.latency_us > 0 && result.latency_us < min_latency) {
      min_latency = result.latency_us;
    }
    if (result.bandwidth_mbps > max_bandwidth) {
      max_bandwidth = result.bandwidth_mbps;
    }
    avg_efficiency += result.efficiency;
  }

  if (!results_.empty()) {
    avg_efficiency /= results_.size();
  }

  std::cout << "Best Latency:  " << std::fixed << std::setprecision(2) << min_latency << " μs" << std::endl;
  std::cout << "Max Bandwidth:  " << std::fixed << std::setprecision(2) << max_bandwidth << " MB/s" << std::endl;
  std::cout << "Avg Efficiency: " << std::fixed << std::setprecision(1) << avg_efficiency << "%" << std::endl;
  std::cout << std::string(120, '-') << std::endl;
}

void PerformanceBenchmark::saveResults(const std::string& filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  file << "Test Name,Latency (μs),Bandwidth (MB/s),Efficiency (%),Message Size (B),Iterations,Total Time (ms)" << std::endl;

  for (const auto& result : results_) {
    file << result.test_name << ","
         << result.latency_us << ","
         << result.bandwidth_mbps << ","
         << result.efficiency << ","
         << result.message_size << ","
         << result.iterations << ","
         << result.total_time_ms << std::endl;
  }

  file.close();
  std::cout << "Results saved to: " << filename << std::endl;
}

}
}

int main(int argc, char* argv[]) {
  using namespace engine_c::benchmark;

  auto benchmark = std::make_unique<PerformanceBenchmark>();

  if (argc > 1) {
    std::string network_type = argv[1];
    if (network_type == "rdma") {
      benchmark->setNetworkType(engine_c::network::NetworkType::RDMA_VERBS);
    }
  }

  if (argc > 2) {
    int world_size = std::atoi(argv[2]);
    if (world_size > 0) {
      benchmark->setWorldSize(world_size);
    }
  }

  benchmark->runAllBenchmarks();
  benchmark->saveResults("pccl_benchmark_results.csv");

  return 0;
}