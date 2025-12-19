#include <plugins/rdma_executor/device.h>
#include <plugins/rdma_executor/executor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>

namespace py = pybind11;
namespace engine_c {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<RdmaDevice>(m, "RdmaDevice")
    .def(py::init<>())
    .def("remoteCommAvailable", &RdmaDevice::remoteCommAvailable)
    .def("activate", &RdmaDevice::activate)
    .def("registerBuffer", &RdmaDevice::registerBuffer)
    .def("connect", &RdmaDevice::connect)
    .def("disconnect", &RdmaDevice::disconnect);

  py::class_<RdmaMemoryManager>(m, "RdmaMemoryManager")
    .def(py::init<>())
    .def("initialize", &RdmaMemoryManager::initialize)
    .def("allocate", &RdmaMemoryManager::allocate,
         py::return_value_policy::reference)
    .def("free", &RdmaMemoryManager::free)
    .def("registerBuffer", &RdmaMemoryManager::registerBuffer)
    .def("unregisterBuffer", &RdmaMemoryManager::unregisterBuffer)
    .def("getAllocatedBytes", &RdmaMemoryManager::getAllocatedBytes)
    .def("getAllocationCount", &RdmaMemoryManager::getAllocationCount)
    .def("clear", &RdmaMemoryManager::clear);

  py::class_<RdmaConnectionManager>(m, "RdmaConnectionManager")
    .def(py::init<>())
    .def("initialize", &RdmaConnectionManager::initialize)
    .def("createConnection", &RdmaConnectionManager::createConnection)
    .def("connectToPeer", &RdmaConnectionManager::connectToPeer)
    .def("disconnectFromPeer", &RdmaConnectionManager::disconnectFromPeer)
    .def("isConnected", &RdmaConnectionManager::isConnected)
    .def("getConnectedPeers", &RdmaConnectionManager::getConnectedPeers)
    .def("enableP2P", &RdmaConnectionManager::enableP2P)
    .def("clear", &RdmaConnectionManager::clear)
    .def("getConnectionCount", &RdmaConnectionManager::getConnectionCount);

  py::class_<DeviceExecutor, std::shared_ptr<DeviceExecutor>>(m, "DeviceExecutor");

  py::class_<RdmaExecutor, DeviceExecutor, std::shared_ptr<RdmaExecutor>>(m, "RdmaExecutor")
    .def(py::init<>())
    .def(py::init<int>())
    .def("initialize", &RdmaExecutor::initialize)
    .def("shutdown", &RdmaExecutor::shutdown)
    .def("allocate", &RdmaExecutor::allocate,
         py::return_value_policy::reference)
    .def("free", &RdmaExecutor::free)
    .def("copy", &RdmaExecutor::copy)
    .def("copyFromHost", &RdmaExecutor::copyFromHost)
    .def("copyToHost", &RdmaExecutor::copyToHost)
    .def("execute", &RdmaExecutor::execute)
    .def("synchronize", &RdmaExecutor::synchronize)
    .def("measureKernelTime", &RdmaExecutor::measureKernelTime,
         py::arg("kernel"), py::arg("args"), py::arg("arg_size"),
         py::arg("grid"), py::arg("block"),
         py::arg("shared_mem") = 0,
         py::arg("iterations") = 100)
    .def("supportsP2P", &RdmaExecutor::supportsP2P)
    .def("enableP2P", &RdmaExecutor::enableP2P)
    .def("getDevicePointer", &RdmaExecutor::getDevicePointer)
    .def("getDeviceId", &RdmaExecutor::getDeviceId)
    .def("getRdmaDevice", &RdmaExecutor::getRdmaDevice,
         py::return_value_policy::reference_internal)
    .def("getMemoryManager", &RdmaExecutor::getMemoryManager,
         py::return_value_policy::reference_internal)
    .def("getConnectionManager", &RdmaExecutor::getConnectionManager,
         py::return_value_policy::reference_internal);

  m.def("create_rdma_executor", [](int device_id) {
    return std::make_shared<RdmaExecutor>(device_id);
  }, py::arg("device_id") = 0, py::return_value_policy::reference);

  m.def("create_rdma_device", []() {
    return std::make_unique<RdmaDevice>();
  }, py::return_value_policy::reference);

  m.def("create_rdma_memory_manager", []() {
    return std::make_unique<RdmaMemoryManager>();
  }, py::return_value_policy::reference);

  m.def("create_rdma_connection_manager", []() {
    return std::make_unique<RdmaConnectionManager>();
  }, py::return_value_policy::reference);

  m.def("rdma_is_available", []() {
    RdmaDevice device;
    return device.remoteCommAvailable();
  });

  m.def("rdma_get_device_list", []() {
    RdmaDevice device;
    std::vector<std::string> devices;

    if (device.remoteCommAvailable()) {
      std::string info = device.activate();
      devices.push_back(info);
    }

    return devices;
  });

  m.def("rdma_setup_environment", []() {
    RdmaDevice device;
    return device.remoteCommAvailable();
  });
}

}