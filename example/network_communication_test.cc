#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <memory>
#include <network/network.h>
#include <communication/communicator.h>
#include <algorithms/communication_wrapper.h>

using namespace engine_c;

void runNode(int rank, int world_size, const std::vector<network::NetworkAddress>& addresses) {
  auto comm = algorithms::CommunicationWrapper();

  std::cout << "Node " << rank << ": Initializing communication..." << std::endl;

  if (!comm.initialize(rank, world_size, network::NetworkType::TCP_SOCKET)) {
    std::cerr << "Node " << rank << ": Failed to initialize communication" << std::endl;
    return;
  }

  std::cout << "Node " << rank << ": Communication initialized successfully" << std::endl;

  std::vector<float> data(1000, rank + 1.0f);
  std::vector<float> result(1000);

  std::cout << "Node " << rank << ": Starting AllReduce test..." << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();
  bool success = comm.allReduceRing(data.data(), result.data(),
                                   data.size() * sizeof(float),
                                   communication::ReductionOp::SUM);
  auto end_time = std::chrono::high_resolution_clock::now();

  if (success) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Node " << rank << ": AllReduce completed in "
              << duration.count() << " ms" << std::endl;

    float expected_sum = 0.0f;
    for (int i = 0; i < world_size; ++i) {
      expected_sum += (i + 1.0f);
    }

    bool correct = true;
    for (size_t i = 0; i < result.size(); ++i) {
      if (std::abs(result[i] - expected_sum) > 1e-6) {
        correct = false;
        break;
      }
    }

    if (correct) {
      std::cout << "Node " << rank << ": AllReduce result is correct!" << std::endl;
    } else {
      std::cout << "Node " << rank << ": AllReduce result is INCORRECT!" << std::endl;
    }
  } else {
    std::cerr << "Node " << rank << ": AllReduce failed" << std::endl;
  }

  std::cout << "Node " << rank << ": Testing Tree AllReduce..." << std::endl;
  std::vector<float> tree_result(1000);

  start_time = std::chrono::high_resolution_clock::now();
  success = comm.allReduceTree(data.data(), tree_result.data(),
                              data.size() * sizeof(float),
                              communication::ReductionOp::SUM);
  end_time = std::chrono::high_resolution_clock::now();

  if (success) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Node " << rank << ": Tree AllReduce completed in "
              << duration.count() << " ms" << std::endl;
  }

  if (world_size >= 2) {
    std::cout << "Node " << rank << ": Testing latency measurement..." << std::endl;
    int peer = (rank + 1) % world_size;
    double latency = comm.measureLatency(peer, 1024);
    double bandwidth = comm.measureBandwidth(peer, 1024 * 1024);

    std::cout << "Node " << rank << "->" << peer << ": Latency = "
              << latency << " μs, Bandwidth = " << bandwidth << " MB/s" << std::endl;
  }

  std::cout << "Node " << rank << ": Finalizing communication..." << std::endl;
  comm.finalize();
  std::cout << "Node " << rank << ": Communication finalized" << std::endl;
}

int main(int argc, char* argv[]) {
  int num_nodes = 2;

  if (argc > 1) {
    num_nodes = std::atoi(argv[1]);
  }

  if (num_nodes < 2) {
    num_nodes = 2;
  }

  std::cout << "Starting network communication test with " << num_nodes << " nodes" << std::endl;

  std::vector<network::NetworkAddress> addresses;
  for (int i = 0; i < num_nodes; ++i) {
    addresses.emplace_back("127.0.0.1", 10000 + i);
  }

  std::vector<std::thread> threads;
  for (int i = 0; i < num_nodes; ++i) {
    threads.emplace_back(runNode, i, num_nodes, std::ref(addresses));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::cout << "Network communication test completed" << std::endl;
  return 0;
}