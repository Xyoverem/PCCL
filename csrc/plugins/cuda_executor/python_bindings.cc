#include <plugins/cuda_executor/device.h>
#include <plugins/cuda_executor/executor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <memory>

namespace py = pybind11;
namespace engine_c {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<CudaDevice>(m, "CudaDevice")
    .def(py::init<>())
    .def("allocatorAvailable", &CudaDevice::allocatorAvailable)
    .def("allocate", &CudaDevice::allocate,
         py::return_value_policy::reference)
    .def("deallocate", &CudaDevice::deallocate)
    .def("IPCAvailable", &CudaDevice::IPCAvailable)
    .def("allocateIpcBuffer", &CudaDevice::allocateIpcBuffer)
    .def("mapBuffer", &CudaDevice::mapBuffer);

  py::class_<CudaStreamManager>(m, "CudaStreamManager")
    .def(py::init<>())
    .def("initialize", &CudaStreamManager::initialize)
    .def("getStream", &CudaStreamManager::getStream,
         py::arg("stream_id") = 0)
    .def("createStream", &CudaStreamManager::createStream)
    .def("destroyStream", &CudaStreamManager::destroyStream)
    .def("synchronizeStream", &CudaStreamManager::synchronizeStream,
         py::arg("stream_id") = 0)
    .def("createEvent", &CudaStreamManager::createEvent)
    .def("recordEvent", &CudaStreamManager::recordEvent,
         py::arg("event"), py::arg("stream") = 0)
    .def("waitForEvent", &CudaStreamManager::waitForEvent,
         py::arg("event"), py::arg("stream") = 0);

  py::class_<CudaMemoryManager>(m, "CudaMemoryManager")
    .def(py::init<>())
    .def("initialize", &CudaMemoryManager::initialize)
    .def("allocate", &CudaMemoryManager::allocate,
         py::return_value_policy::reference)
    .def("free", &CudaMemoryManager::free)
    .def("getAllocatedBytes", &CudaMemoryManager::getAllocatedBytes)
    .def("getAllocationCount", &CudaMemoryManager::getAllocationCount)
    .def("clear", &CudaMemoryManager::clear);

  py::class_<CudaKernelManager>(m, "CudaKernelManager")
    .def(py::init<>())
    .def("initialize", &CudaKernelManager::initialize)
    .def("getFunction", &CudaKernelManager::getFunction);

  py::class_<CudaBlasManager>(m, "CudaBlasManager")
    .def(py::init<>())
    .def("initialize", &CudaBlasManager::initialize)
    .def("getHandle", &CudaBlasManager::getHandle)
    .def("setStream", &CudaBlasManager::setStream)
    .def("synchronize", &CudaBlasManager::synchronize);

  py::class_<DeviceExecutor, std::shared_ptr<DeviceExecutor>>(m, "DeviceExecutor");

  py::class_<CudaExecutor, DeviceExecutor, std::shared_ptr<CudaExecutor>>(m, "CudaExecutor")
    .def(py::init<>())
    .def(py::init<int>())
    .def("initialize", &CudaExecutor::initialize)
    .def("shutdown", &CudaExecutor::shutdown)
    .def("allocate", &CudaExecutor::allocate,
         py::return_value_policy::reference)
    .def("free", &CudaExecutor::free)
    .def("copy", &CudaExecutor::copy)
    .def("copyFromHost", &CudaExecutor::copyFromHost)
    .def("copyToHost", &CudaExecutor::copyToHost)
    .def("execute", &CudaExecutor::execute)
    .def("synchronize", &CudaExecutor::synchronize)
    .def("measureKernelTime", &CudaExecutor::measureKernelTime,
         py::arg("kernel"), py::arg("args"), py::arg("arg_size"),
         py::arg("grid"), py::arg("block"),
         py::arg("shared_mem") = 0,
         py::arg("iterations") = 100)
    .def("supportsP2P", &CudaExecutor::supportsP2P)
    .def("enableP2P", &CudaExecutor::enableP2P)
    .def("getDevicePointer", &CudaExecutor::getDevicePointer)
    .def("getDeviceId", &CudaExecutor::getDeviceId)
    .def("getCurrentStream", &CudaExecutor::getCurrentStream)
    .def("setCurrentStream", &CudaExecutor::setCurrentStream)
    .def("getTotalMemory", &CudaExecutor::getTotalMemory)
    .def("getAvailableMemory", &CudaExecutor::getAvailableMemory)
    .def("getStreamManager", &CudaExecutor::getStreamManager,
         py::return_value_policy::reference_internal)
    .def("getMemoryManager", &CudaExecutor::getMemoryManager,
         py::return_value_policy::reference_internal)
    .def("getKernelManager", &CudaExecutor::getKernelManager,
         py::return_value_policy::reference_internal)
    .def("getBlasManager", &CudaExecutor::getBlasManager,
         py::return_value_policy::reference_internal);

  m.def("create_cuda_executor", [](int device_id) {
    return std::make_shared<CudaExecutor>(device_id);
  }, py::arg("device_id") = 0, py::return_value_policy::reference);

  m.def("create_cuda_device", []() {
    return std::make_unique<CudaDevice>();
  }, py::return_value_policy::reference);

  m.def("create_cuda_memory_manager", []() {
    return std::make_unique<CudaMemoryManager>();
  }, py::return_value_policy::reference);

  m.def("create_cuda_stream_manager", []() {
    return std::make_unique<CudaStreamManager>();
  }, py::return_value_policy::reference);

  m.def("create_cuda_kernel_manager", []() {
    return std::make_unique<CudaKernelManager>();
  }, py::return_value_policy::reference);

  m.def("create_cuda_blas_manager", []() {
    return std::make_unique<CudaBlasManager>();
  }, py::return_value_policy::reference);

  m.def("get_cuda_device_count", []() {
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    return (error == cudaSuccess) ? count : 0;
  });

  m.def("set_cuda_device", [](int device_id) {
    cudaError_t error = cudaSetDevice(device_id);
    return error == cudaSuccess;
  });

  m.def("get_cuda_device_name", [](int device_id) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
    return (error == cudaSuccess) ? std::string(prop.name) : std::string("Unknown");
  });

  m.def("get_cuda_device_properties", [](int device_id) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);

    if (error != cudaSuccess) {
      return py::dict();
    }

    return py::dict(
      "name"_a=prop.name,
      "major"_a=prop.major,
      "minor"_a=prop.minor,
      "totalGlobalMem"_a=prop.totalGlobalMem,
      "sharedMemPerBlock"_a=prop.sharedMemPerBlock,
      "maxThreadsPerBlock"_a=prop.maxThreadsPerBlock,
      "maxGridSize"_a=std::vector<int>{prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]},
      "maxThreadsDim"_a=std::vector<int>{prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]},
      "warpSize"_a=prop.warpSize,
      "memoryClockRate"_a=prop.memoryClockRate,
      "memoryBusWidth"_a=prop.memoryBusWidth,
      "l2CacheSize"_a=prop.l2CacheSize,
      "maxThreadsPerMultiProcessor"_a=prop.maxThreadsPerMultiProcessor,
      "multiProcessorCount"_a=prop.multiProcessorCount,
      "concurrentKernels"_a=prop.concurrentKernels,
      "integrated"_a=prop.integrated,
      "canMapHostMemory"_a=prop.canMapHostMemory,
      "computeMode"_a=static_cast<int>(prop.computeMode)
    );
  });

  m.def("cuda_can_access_peer", [](int device_id, int peer_device_id) {
    int can_access;
    cudaError_t error = cudaDeviceCanAccessPeer(&can_access, device_id, peer_device_id);
    return (error == cudaSuccess) ? (can_access != 0) : false;
  });

  m.def("cuda_enable_peer_access", [](int peer_device_id) {
    cudaError_t error = cudaDeviceEnablePeerAccess(peer_device_id, 0);
    return error == cudaSuccess || error == cudaErrorPeerAccessAlreadyEnabled;
  });

  m.def("cuda_get_memory_info", [](int device_id) {
    size_t free, total;
    cudaSetDevice(device_id);
    cudaError_t error = cudaMemGetInfo(&free, &total);

    if (error != cudaSuccess) {
      return std::make_tuple(0, 0);
    }

    return std::make_tuple(free, total);
  });
}

}