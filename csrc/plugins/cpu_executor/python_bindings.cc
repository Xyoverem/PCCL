#include <plugins/cpu_executor/device.h>
#include <plugins/cpu_executor/executor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <memory>

namespace py = pybind11;
namespace engine_c {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<CpuDevice>(m, "CpuDevice")
    .def(py::init<>())
    .def("allocatorAvailable", &CpuDevice::allocatorAvailable)
    .def("allocate", &CpuDevice::allocate,
         py::return_value_policy::reference)
    .def("deallocate", &CpuDevice::deallocate)
    .def("IPCAvailable", &CpuDevice::IPCAvailable)
    .def("allocateIpcBuffer", &CpuDevice::allocateIpcBuffer)
    .def("mapBuffer", &CpuDevice::mapBuffer);

  py::class_<CpuMemoryManager>(m, "CpuMemoryManager")
    .def(py::init<>())
    .def("allocate", &CpuMemoryManager::allocate,
         py::return_value_policy::reference)
    .def("free", &CpuMemoryManager::free)
    .def("getAllocatedBytes", &CpuMemoryManager::getAllocatedBytes)
    .def("getAllocationCount", &CpuMemoryManager::getAllocationCount)
    .def("clear", &CpuMemoryManager::clear);

  py::class_<CpuThreadPool>(m, "CpuThreadPool")
    .def(py::init<size_t>(), py::arg("num_threads") = std::thread::hardware_concurrency())
    .def("submit", [](CpuThreadPool& pool, py::function f) {
      pool.submit([f]() { f(); });
    })
    .def("getThreadCount", &CpuThreadPool::getThreadCount)
    .def("getQueueSize", &CpuThreadPool::getQueueSize);

  py::class_<CpuKernelRegistry>(m, "CpuKernelRegistry")
    .def(py::init<>())
    .def("registerKernel", &CpuKernelRegistry::registerKernel)
    .def("getKernel", &CpuKernelRegistry::getKernel,
         py::return_value_policy::reference)
    .def("hasKernel", &CpuKernelRegistry::hasKernel)
    .def("getKernelNames", &CpuKernelRegistry::getKernelNames)
    .def("clear", &CpuKernelRegistry::clear);

  py::class_<CpuKernelParams>(m, "CpuKernelParams")
    .def(py::init<>())
    .def_readonly("args", &CpuKernelParams::args)
    .def_readonly("grid_x", &CpuKernelParams::grid_x)
    .def_readonly("grid_y", &CpuKernelParams::grid_y)
    .def_readonly("grid_z", &CpuKernelParams::grid_z)
    .def_readonly("block_x", &CpuKernelParams::block_x)
    .def_readonly("block_y", &CpuKernelParams::block_y)
    .def_readonly("block_z", &CpuKernelParams::block_z)
    .def_readonly("grid_dim_x", &CpuKernelParams::grid_dim_x)
    .def_readonly("grid_dim_y", &CpuKernelParams::grid_dim_y)
    .def_readonly("grid_dim_z", &CpuKernelParams::grid_dim_z)
    .def_readonly("block_dim_x", &CpuKernelParams::block_dim_x)
    .def_readonly("block_dim_y", &CpuKernelParams::block_dim_y)
    .def_readonly("block_dim_z", &CpuKernelParams::block_dim_z);

  py::class_<DeviceExecutor, std::shared_ptr<DeviceExecutor>>(m, "DeviceExecutor");

  py::class_<CpuExecutor, DeviceExecutor, std::shared_ptr<CpuExecutor>>(m, "CpuExecutor")
    .def(py::init<>())
    .def(py::init<int>())
    .def("initialize", &CpuExecutor::initialize)
    .def("shutdown", &CpuExecutor::shutdown)
    .def("allocate", &CpuExecutor::allocate,
         py::return_value_policy::reference)
    .def("free", &CpuExecutor::free)
    .def("copy", &CpuExecutor::copy)
    .def("copyFromHost", &CpuExecutor::copyFromHost)
    .def("copyToHost", &CpuExecutor::copyToHost)
    .def("execute", &CpuExecutor::execute)
    .def("synchronize", &CpuExecutor::synchronize)
    .def("measureKernelTime", &CpuExecutor::measureKernelTime,
         py::arg("kernel"), py::arg("args"), py::arg("arg_size"),
         py::arg("grid"), py::arg("block"),
         py::arg("shared_mem") = 0,
         py::arg("iterations") = 100)
    .def("supportsP2P", &CpuExecutor::supportsP2P)
    .def("enableP2P", &CpuExecutor::enableP2P)
    .def("getDevicePointer", &CpuExecutor::getDevicePointer)
    .def("getDeviceId", &CpuExecutor::getDeviceId)
    .def("setDevice", &CpuExecutor::setDevice)
    .def("getTotalMemory", &CpuExecutor::getTotalMemory)
    .def("getAvailableMemory", &CpuExecutor::getAvailableMemory)
    .def("getMemoryManager", &CpuExecutor::getMemoryManager,
         py::return_value_policy::reference_internal)
    .def("getThreadPool", &CpuExecutor::getThreadPool,
         py::return_value_policy::reference_internal)
    .def("getKernelRegistry", &CpuExecutor::getKernelRegistry,
         py::return_value_policy::reference_internal);

  m.def("create_cpu_executor", [](int device_id) {
    return std::make_shared<CpuExecutor>(device_id);
  }, py::arg("device_id") = 0, py::return_value_policy::reference);

  m.def("create_cpu_device", []() {
    return std::make_unique<CpuDevice>();
  }, py::return_value_policy::reference);

  m.def("create_cpu_memory_manager", []() {
    return std::make_unique<CpuMemoryManager>();
  }, py::return_value_policy::reference);

  m.def("create_cpu_thread_pool", [](size_t num_threads) {
    return std::make_unique<CpuThreadPool>(num_threads);
  }, py::arg("num_threads") = std::thread::hardware_concurrency(),
    py::return_value_policy::reference);

  m.def("create_cpu_kernel_registry", []() {
    return std::make_unique<CpuKernelRegistry>();
  }, py::return_value_policy::reference);

  m.def("get_cpu_core_count", []() {
    return std::thread::hardware_concurrency();
  });

  m.def("get_cpu_memory_info", []() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    size_t total_memory = 0, available_memory = 0;

    while (std::getline(meminfo, line)) {
      if (line.find("MemTotal:") == 0) {
        std::istringstream iss(line);
        std::string label;
        iss >> label >> total_memory;
      } else if (line.find("MemAvailable:") == 0) {
        std::istringstream iss(line);
        std::string label;
        iss >> label >> available_memory;
      }
    }

    return std::make_tuple(total_memory * 1024, available_memory * 1024);
  });
}

}