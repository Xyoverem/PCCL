#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <memory>
#include <cluster/process_group.h>
#include <communication/communicator.h>

using namespace engine_c;

void runProcessGroupTest(int rank, int world_size, const std::string& group_type) {
  std::cout << "Process " << rank << ": Starting " << group_type << " process group test" << std::endl;

  auto manager = std::make_unique<ProcessGroupManager>();
  manager->initialize(rank, world_size);

  std::shared_ptr<ProcessGroup> group;

  std::vector<int> all_ranks;
  for (int i = 0; i < world_size; ++i) {
    all_ranks.push_back(i);
  }

  if (group_type == "cpu") {
    group = manager->createCPUGroup("test_cpu_group", all_ranks);
  } else if (group_type == "cuda") {
    std::vector<int> device_ids(world_size, 0);
    group = manager->createCUDAGroup("test_cuda_group", all_ranks, device_ids);
  } else if (group_type == "rdma") {
    std::vector<std::string> endpoints;
    for (int i = 0; i < world_size; ++i) {
      endpoints.push_back("192.168.1." + std::to_string(100 + i) + ":5000");
    }
    group = manager->createRDMAGroup("test_rdma_group", all_ranks, endpoints);
  } else {
    group = manager->getGlobalProcessGroup();
  }

  if (!group) {
    std::cerr << "Process " << rank << ": Failed to create process group" << std::endl;
    return;
  }

  std::cout << "Process " << rank << ": Process group created with size " << group->getSize() << std::endl;

  std::vector<float> send_data(1000, rank + 1.0f);
  std::vector<float> recv_data(1000);

  std::cout << "Process " << rank << ": Starting AllReduce test..." << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();
  group->allreduce(send_data.data(), recv_data.data(),
                   send_data.size() * sizeof(float), DataType::FLOAT32, ReduceOp::SUM);
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Process " << rank << ": AllReduce completed in "
            << duration.count() << " ms" << std::endl;

  float expected_sum = 0.0f;
  for (int i = 0; i < world_size; ++i) {
    expected_sum += (i + 1.0f);
  }

  bool correct = true;
  for (size_t i = 0; i < recv_data.size(); ++i) {
    if (std::abs(recv_data[i] - expected_sum) > 1e-6) {
      correct = false;
      break;
    }
  }

  if (correct) {
    std::cout << "Process " << rank << ": AllReduce result is CORRECT!" << std::endl;
  } else {
    std::cout << "Process " << rank << ": AllReduce result is INCORRECT!" << std::endl;
  }

  std::cout << "Process " << rank << ": Testing AllGather..." << std::endl;
  std::vector<float> input_data(100, rank + 1.0f);
  std::vector<float> output_data(100 * world_size);

  start_time = std::chrono::high_resolution_clock::now();
  group->allgather(input_data.data(), output_data.data(),
                   input_data.size() * sizeof(float), output_data.size() * sizeof(float));
  end_time = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Process " << rank << ": AllGather completed in "
            << duration.count() << " ms" << std::endl;

  std::cout << "Process " << rank << ": Testing Broadcast..." << std::endl;
  std::vector<float> broadcast_data(500, rank + 1.0f);
  int root = 0;

  start_time = std::chrono::high_resolution_clock::now();
  group->broadcast(broadcast_data.data(), broadcast_data.size() * sizeof(float), root);
  end_time = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Process " << rank << ": Broadcast completed in "
            << duration.count() << " ms" << std::endl;

  if (world_size >= 2) {
    std::cout << "Process " << rank << ": Testing point-to-point communication..." << std::endl;

    int dest = (rank + 1) % world_size;
    int src = (rank - 1 + world_size) % world_size;

    if (rank < dest) {
      std::string message = "Hello from process " + std::to_string(rank);
      std::string recv_message(100, '\0');

      group->send(message.data(), message.size(), dest, 100);
      group->recv(const_cast<char*>(recv_message.data()), recv_message.size(), src, 100);

      std::cout << "Process " << rank << ": Sent and received message successfully" << std::endl;
    }
  }

  std::cout << "Process " << rank << ": Testing Barrier..." << std::endl;
  start_time = std::chrono::high_resolution_clock::now();
  group->barrier();
  end_time = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Process " << rank << ": Barrier completed in "
            << duration.count() << " ms" << std::endl;

  std::cout << "Process " << rank << ": Process group test completed" << std::endl;
}

int main(int argc, char* argv[]) {
  int num_processes = 4;
  std::string group_type = "cpu";

  if (argc > 1) {
    num_processes = std::atoi(argv[1]);
  }
  if (argc > 2) {
    group_type = argv[2];
  }

  std::cout << "Starting process group test with " << num_processes
            << " processes, type: " << group_type << std::endl;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_processes; ++i) {
    threads.emplace_back(runProcessGroupTest, i, num_processes, group_type);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::cout << "Process group test completed" << std::endl;
  return 0;
}