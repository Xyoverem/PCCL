#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>
#include <cassert>
#include <random>

#include <network/network.h>
#include <communication/communicator.h>
#include <cluster/process_group.h>
#include <algorithms/communication_wrapper.h>
#include <algorithms/algorithm_manager.h>
#include <engine.h>

using namespace engine_c;

class IntegrationTest {
public:
  IntegrationTest() : tests_passed_(0), tests_total_(0) {}

  void runAllTests() {
    std::cout << "=== PCCL Integration Test Suite ===" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    testNetworkManager();
    testCommunicationWrapper();
    testProcessGroups();
    testAlgorithmManager();
    testEngineIntegration();
    testErrorHandling();

    printResults();
  }

private:
  int tests_passed_;
  int tests_total_;

  void testNetworkManager() {
    std::cout << "\n--- Network Manager Tests ---" << std::endl;

    runTest("NetworkManager Initialization", [this]() {
      auto& manager = network::NetworkManager::getInstance();
      bool result = manager.initialize(network::NetworkType::TCP_SOCKET);
      manager.shutdown();
      return result;
    });

    runTest("Connection Creation", [this]() {
      auto& manager = network::NetworkManager::getInstance();
      manager.initialize(network::NetworkType::TCP_SOCKET);
      auto conn = manager.createConnection();
      manager.shutdown();
      return conn != nullptr;
    });

    runTest("Multiple Connections", [this]() {
      auto& manager = network::NetworkManager::getInstance();
      manager.initialize(network::NetworkType::TCP_SOCKET);

      std::vector<network::ConnectionPtr> conns;
      for (int i = 0; i < 5; ++i) {
        auto conn = manager.createConnection();
        if (conn) conns.push_back(conn);
      }

      manager.shutdown();
      return conns.size() == 5;
    });
  }

  void testCommunicationWrapper() {
    std::cout << "\n--- Communication Wrapper Tests ---" << std::endl;

    runTest("CommunicationWrapper Initialization", [this]() {
      auto comm = algorithms::CommunicationWrapper();
      return comm.initialize(0, 2, network::NetworkType::TCP_SOCKET);
    });

    runTest("AllReduce Ring", [this]() {
      auto comm = algorithms::CommunicationWrapper();
      if (!comm.initialize(0, 2, network::NetworkType::TCP_SOCKET)) {
        return false;
      }

      std::vector<float> send_data(1000, 1.0f);
      std::vector<float> recv_data(1000);

      bool result = comm.allReduceRing(send_data.data(), recv_data.data(),
                                      send_data.size() * sizeof(float),
                                      communication::ReductionOp::SUM);
      comm.finalize();
      return result;
    });

    runTest("AllReduce Tree", [this]() {
      auto comm = algorithms::CommunicationWrapper();
      if (!comm.initialize(0, 2, network::NetworkType::TCP_SOCKET)) {
        return false;
      }

      std::vector<float> send_data(1000, 1.0f);
      std::vector<float> recv_data(1000);

      bool result = comm.allReduceTree(send_data.data(), recv_data.data(),
                                      send_data.size() * sizeof(float),
                                      communication::ReductionOp::SUM);
      comm.finalize();
      return result;
    });
  }

  void testProcessGroups() {
    std::cout << "\n--- Process Group Tests ---" << std::endl;

    runTest("ProcessGroupManager Creation", [this]() {
      auto manager = std::make_unique<ProcessGroupManager>();
      return manager != nullptr;
    });

    runTest("ProcessGroupManager Initialization", [this]() {
      auto manager = std::make_unique<ProcessGroupManager>();
      manager->initialize(0, 4);
      return manager->getGlobalRank() == 0 && manager->getGlobalSize() == 4;
    });

    runTest("CPU Process Group Creation", [this]() {
      auto manager = std::make_unique<ProcessGroupManager>();
      manager->initialize(0, 4);

      std::vector<int> ranks = {0, 1, 2, 3};
      auto group = manager->createCPUGroup("cpu_test", ranks);
      return group != nullptr && group->getSize() == 4;
    });

    runTest("CUDA Process Group Creation", [this]() {
      auto manager = std::make_unique<ProcessGroupManager>();
      manager->initialize(0, 4);

      std::vector<int> ranks = {0, 1, 2, 3};
      std::vector<int> device_ids = {0, 0, 0, 0};
      auto group = manager->createCUDAGroup("cuda_test", ranks, device_ids);
      return group != nullptr && group->getSize() == 4;
    });

    runTest("Process Group AllReduce", [this]() {
      auto manager = std::make_unique<ProcessGroupManager>();
      manager->initialize(0, 4);

      std::vector<int> ranks = {0, 1, 2, 3};
      auto group = manager->createCPUGroup("allreduce_test", ranks);

      if (!group) return false;

      std::vector<float> send_data(100, 1.0f);
      std::vector<float> recv_data(100);

      group->allreduce(send_data.data(), recv_data.data(),
                      send_data.size() * sizeof(float),
                      DataType::FLOAT32, ReduceOp::SUM);

      return recv_data[0] == 4.0f;
    });

    runTest("Process Group Barrier", [this]() {
      auto manager = std::make_unique<ProcessGroupManager>();
      manager->initialize(0, 4);

      std::vector<int> ranks = {0, 1, 2, 3};
      auto group = manager->createCPUGroup("barrier_test", ranks);

      if (!group) return false;

      auto start = std::chrono::high_resolution_clock::now();
      group->barrier();
      auto end = std::chrono::high_resolution_clock::now();

      return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() < 1000;
    });
  }

  void testAlgorithmManager() {
    std::cout << "\n--- Algorithm Manager Tests ---" << std::endl;

    runTest("AlgorithmManager Creation", [this]() {
      auto manager = std::make_unique<algorithms::AlgorithmManager>();
      return manager != nullptr;
    });

    runTest("Algorithm Selection", [this]() {
      auto manager = std::make_unique<algorithms::AlgorithmManager>();

      TopologyMetrics metrics;
      metrics.bandwidth = 10000.0;
      metrics.latency = 10.0;
      metrics.num_nodes = 4;
      metrics.topology_type = "ring";

      std::string algorithm = manager->selectOptimalAlgorithm("allreduce", 1024, metrics);
      return !algorithm.empty();
    });

    runTest("Performance Cache", [this]() {
      auto manager = std::make_unique<algorithms::AlgorithmManager>();

      TopologyMetrics metrics;
      metrics.bandwidth = 10000.0;
      metrics.latency = 10.0;
      metrics.num_nodes = 4;
      metrics.topology_type = "ring";

      std::string algorithm1 = manager->selectOptimalAlgorithm("allreduce", 1024, metrics);
      std::string algorithm2 = manager->selectOptimalAlgorithm("allreduce", 1024, metrics);

      return algorithm1 == algorithm2;
    });

    runTest("Algorithm Benchmarking", [this]() {
      auto manager = std::make_unique<algorithms::AlgorithmManager>();

      TopologyMetrics metrics;
      metrics.bandwidth = 10000.0;
      metrics.latency = 10.0;
      metrics.num_nodes = 4;
      metrics.topology_type = "ring";

      auto performance = manager->benchmarkAlgorithm("ring", 1024, metrics);
      return performance.bandwidth_mbps > 0 && performance.latency_us > 0;
    });
  }

  void testEngineIntegration() {
    std::cout << "\n--- Engine Integration Tests ---" << std::endl;

    runTest("Engine Creation", [this]() {
      auto engine = std::make_unique<Engine>();
      return engine != nullptr;
    });

    runTest("Engine Operator Registration", [this]() {
      auto engine = std::make_unique<Engine>();

      std::map<std::string, std::string> config = {
        {"host_id", "test"},
        {"rank", "0"},
        {"world_size", "2"}
      };

      engine->initialize(config);

      try {
        engine->regOp("test_allreduce", "/tmp/test_config.json");
        return true;
      } catch (...) {
        return false;
      }
    });

    runTest("Engine Execution", [this]() {
      auto engine = std::make_unique<Engine>();

      std::map<std::string, std::string> config = {
        {"host_id", "test"},
        {"rank", "0"},
        {"world_size", "2"}
      };

      engine->initialize(config);

      torch::Tensor input = torch::ones({1000});
      torch::Tensor output = torch::zeros({1000});

      try {
        engine->exeOp("test_allreduce", input, output);
        return true;
      } catch (...) {
        return false;
      }
    });
  }

  void testErrorHandling() {
    std::cout << "\n--- Error Handling Tests ---" << std::endl;

    runTest("Invalid Connection", [this]() {
      auto& manager = network::NetworkManager::getInstance();
      manager.initialize(network::NetworkType::TCP_SOCKET);

      network::NetworkAddress invalid_addr("invalid_host", -1);
      auto conn = manager.createConnection();
      bool result = conn->connect(invalid_addr);

      manager.shutdown();
      return !result;
    });

    runTest("Large Message Handling", [this]() {
      auto comm = algorithms::CommunicationWrapper();
      if (!comm.initialize(0, 2, network::NetworkType::TCP_SOCKET)) {
        return false;
      }

      std::vector<float> large_data(10 * 1024 * 1024, 1.0f);
      std::vector<float> recv_data(10 * 1024 * 1024);

      bool result = comm.allReduceRing(large_data.data(), recv_data.data(),
                                      large_data.size() * sizeof(float),
                                      communication::ReductionOp::SUM);
      comm.finalize();
      return result;
    });

    runTest("Invalid Rank Access", [this]() {
      auto manager = std::make_unique<ProcessGroupManager>();
      manager->initialize(0, 4);

      auto group = manager->getProcessGroup("nonexistent");
      return group == nullptr;
    });

    runTest("Memory Allocation Failure", [this]() {
      auto& manager = network::NetworkManager::getInstance();
      manager.initialize(network::NetworkType::TCP_SOCKET);

      auto conn = manager.createConnection();
      uint32_t lkey, rkey;

      bool result = conn->registerMemoryRegion(nullptr, 1024, lkey, rkey);
      manager.shutdown();
      return !result;
    });
  }

  void runTest(const std::string& test_name, std::function<bool()> test_func) {
    tests_total_++;
    std::cout << "  " << test_name << " ... ";

    try {
      bool result = test_func();
      if (result) {
        std::cout << "PASS" << std::endl;
        tests_passed_++;
      } else {
        std::cout << "FAIL" << std::endl;
      }
    } catch (const std::exception& e) {
      std::cout << "FAIL (Exception: " << e.what() << ")" << std::endl;
    } catch (...) {
      std::cout << "FAIL (Unknown exception)" << std::endl;
    }
  }

  void printResults() {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Test Results: " << tests_passed_ << "/" << tests_total_ << " passed" << std::endl;

    if (tests_passed_ == tests_total_) {
      std::cout << "✅ All tests passed!" << std::endl;
    } else {
      std::cout << "❌ " << (tests_total_ - tests_passed_) << " tests failed" << std::endl;
    }

    double success_rate = (tests_total_ > 0) ?
                         (static_cast<double>(tests_passed_) / tests_total_) * 100.0 : 0.0;
    std::cout << "Success Rate: " << std::fixed << std::setprecision(1) << success_rate << "%" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
  }
};

void runConcurrentTest() {
  std::cout << "\n--- Concurrent Communication Test ---" << std::endl;

  const int num_threads = 4;
  const int operations_per_thread = 100;
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      auto comm = algorithms::CommunicationWrapper();

      if (comm.initialize(i, num_threads, network::NetworkType::TCP_SOCKET)) {
        for (int j = 0; j < operations_per_thread; ++j) {
          std::vector<float> data(100, i + 1.0f);
          std::vector<float> result(100);

          if (comm.allReduceRing(data.data(), result.data(),
                               data.size() * sizeof(float),
                               communication::ReductionOp::SUM)) {
            success_count++;
          }
        }
        comm.finalize();
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  int expected_operations = num_threads * operations_per_thread;
  std::cout << "Concurrent operations: " << success_count << "/" << expected_operations;
  if (success_count == expected_operations) {
    std::cout << " ✅ PASS" << std::endl;
  } else {
    std::cout << " ❌ FAIL" << std::endl;
  }
}

void runStressTest() {
  std::cout << "\n--- Stress Test ---" << std::endl;

  const int num_iterations = 1000;
  const int message_size = 1024 * 1024;
  auto comm = algorithms::CommunicationWrapper();

  if (!comm.initialize(0, 2, network::NetworkType::TCP_SOCKET)) {
    std::cout << "Failed to initialize communicator for stress test" << std::endl;
    return;
  }

  std::vector<float> data(message_size / sizeof(float), 1.0f);
  std::vector<float> result(message_size / sizeof(float));

  auto start_time = std::chrono::high_resolution_clock::now();
  int success_count = 0;

  for (int i = 0; i < num_iterations; ++i) {
    if (comm.allReduceRing(data.data(), result.data(),
                         data.size() * sizeof(float),
                         communication::ReductionOp::SUM)) {
      success_count++;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  comm.finalize();

  std::cout << "Stress test completed:" << std::endl;
  std::cout << "  Operations: " << success_count << "/" << num_iterations << std::endl;
  std::cout << "  Total time: " << duration.count() << " ms" << std::endl;
  std::cout << "  Average time per operation: "
            << static_cast<double>(duration.count()) / num_iterations << " ms" << std::endl;
  std::cout << "  Throughput: "
            << (message_size * success_count / 1024.0 / 1024.0) / (duration.count() / 1000.0)
            << " MB/s" << std::endl;
}

int main(int argc, char* argv[]) {
  try {
    IntegrationTest test;
    test.runAllTests();

    runConcurrentTest();
    runStressTest();

    std::cout << "\n🎯 Integration testing completed!" << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Integration test failed with exception: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Integration test failed with unknown exception" << std::endl;
    return 1;
  }
}