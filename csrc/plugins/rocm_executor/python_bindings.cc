#include <plugins/rocm_executor/executor.h>
#include <plugins/rocm_executor/device.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <memory>

namespace py = pybind11;
namespace engine_c {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ROCmMemoryManager>(m, "ROCmMemoryManager")
    .def(py::init<>())
    .def("allocate", &ROCmMemoryManager::allocate,
         py::arg("size"), py::arg("hip_device") = 0,
         py::return_value_policy::reference)
    .def("free", &ROCmMemoryManager::free,
         py::arg("ptr"), py::arg("hip_device") = 0)
    .def("allocateUnified", &ROCmMemoryManager::allocateUnified,
         py::return_value_policy::reference)
    .def("freeUnified", &ROCmMemoryManager::freeUnified)
    .def("allocatePinned", &ROCmMemoryManager::allocatePinned,
         py::return_value_policy::reference)
    .def("freePinned", &ROCmMemoryManager::freePinned)
    .def("copyToDevice", &ROCmMemoryManager::copyToDevice)
    .def("copyFromDevice", &ROCmMemoryManager::copyFromDevice)
    .def("copyDeviceToDevice", &ROCmMemoryManager::copyDeviceToDevice)
    .def("canAccessPeer", &ROCmMemoryManager::canAccessPeer)
    .def("enablePeerAccess", &ROCmMemoryManager::enablePeerAccess);

  py::class_<ROCmKernelManager>(m, "ROCmKernelManager")
    .def(py::init<>())
    .def("loadModule", &ROCmKernelManager::loadModule,
         py::return_value_policy::reference)
    .def("getFunction", &ROCmKernelManager::getFunction,
         py::return_value_policy::reference)
    .def("launchKernel", &ROCmKernelManager::launchKernel)
    .def("createStream", &ROCmKernelManager::createStream,
         py::return_value_policy::reference)
    .def("destroyStream", &ROCmKernelManager::destroyStream)
    .def("synchronizeStream", &ROCmKernelManager::synchronizeStream)
    .def("recordEvent", &ROCmKernelManager::recordEvent)
    .def("waitForEvent", &ROCmKernelManager::waitForEvent);

  py::class_<ROCmDevice>(m, "ROCmDevice")
    .def(py::init<>())
    .def(py::init<int>())
    .def("initialize", &ROCmDevice::initialize)
    .def("shutdown", &ROCmDevice::shutdown)
    .def("getDeviceId", &ROCmDevice::getDeviceId)
    .def("getDeviceName", &ROCmDevice::getDeviceName,
         py::return_value_policy::reference_internal)
    .def("getGcnArch", &ROCmDevice::getGcnArch,
         py::return_value_policy::reference_internal)
    .def("getTotalMemory", &ROCmDevice::getTotalMemory)
    .def("getAvailableMemory", &ROCmDevice::getAvailableMemory)
    .def("getComputeUnits", &ROCmDevice::getComputeUnits)
    .def("getMaxThreadsPerBlock", &ROCmDevice::getMaxThreadsPerBlock)
    .def("getMaxThreadsPerSM", &ROCmDevice::getMaxThreadsPerSM)
    .def("getWarpSize", &ROCmDevice::getWarpSize)
    .def("getMemoryClockRate", &ROCmDevice::getMemoryClockRate)
    .def("getMemoryBandwidth", &ROCmDevice::getMemoryBandwidth)
    .def("getComputeClockRate", &ROCmDevice::getComputeClockRate)
    .def("setDevice", &ROCmDevice::setDevice)
    .def("synchronize", &ROCmDevice::synchronize)
    .def("getDefaultStream", &ROCmDevice::getDefaultStream)
    .def("getMemoryManager", &ROCmDevice::getMemoryManager,
         py::return_value_policy::reference_internal)
    .def("getKernelManager", &ROCmDevice::getKernelManager,
         py::return_value_policy::reference_internal)
    .def("supportsUnifiedAddressing", &ROCmDevice::supportsUnifiedAddressing)
    .def("supportsManagedMemory", &ROCmDevice::supportsManagedMemory)
    .def("supportsP2P", &ROCmDevice::supportsP2P)
    .def("enableP2P", &ROCmDevice::enableP2P);

  py::class_<ROCmDeviceManager>(m, "ROCmDeviceManager")
    .def(py::init<>())
    .def("initialize", &ROCmDeviceManager::initialize)
    .def("shutdown", &ROCmDeviceManager::shutdown)
    .def("getDeviceCount", &ROCmDeviceManager::getDeviceCount)
    .def("isInitialized", &ROCmDeviceManager::isInitialized)
    .def("getDevice", &ROCmDeviceManager::getDevice,
         py::return_value_policy::reference)
    .def("getDefaultDevice", &ROCmDeviceManager::getDefaultDevice)
    .def("setDefaultDevice", &ROCmDeviceManager::setDefaultDevice)
    .def("getDeviceList", &ROCmDeviceManager::getDeviceList)
    .def("getDeviceSummary", &ROCmDeviceManager::getDeviceSummary)
    .def("canAccessPeer", &ROCmDeviceManager::canAccessPeer)
    .def("enableP2PBetweenAllDevices", &ROCmDeviceManager::enableP2PBetweenAllDevices);

  py::enum_<ROCmKernelType>(m, "ROCmKernelType")
    .value("REDUCE_SUM", ROCmKernelType::REDUCE_SUM)
    .value("REDUCE_MAX", ROCmKernelType::REDUCE_MAX)
    .value("REDUCE_MIN", ROCmKernelType::REDUCE_MIN)
    .value("REDUCE_AVG", ROCmKernelType::REDUCE_AVG)
    .value("REDUCE_CUSTOM", ROCmKernelType::REDUCE_CUSTOM)
    .value("ALLREDUCE_RING", ROCmKernelType::ALLREDUCE_RING)
    .value("ALLREDUCE_TREE", ROCmKernelType::ALLREDUCE_TREE)
    .value("BROADCAST", ROCmKernelType::BROADCAST)
    .value("GATHER", ROCmKernelType::GATHER)
    .value("SCATTER", ROCmKernelType::SCATTER)
    .value("COPY", ROCmKernelType::COPY)
    .value("CUSTOM", ROCmKernelType::CUSTOM)
    .value("SEND", ROCmKernelType::SEND)
    .value("RECV", ROCmKernelType::RECV);

  py::class_<ROCmKernel>(m, "ROCmKernel")
    .def("getName", &ROCmKernel::getName,
         py::return_value_policy::reference_internal)
    .def("getType", &ROCmKernel::getType)
    .def("getDevice", &ROCmKernel::getDevice,
         py::return_value_policy::reference);

  py::class_<ROCmReduceKernel, ROCmKernel>(m, "ROCmReduceKernel")
    .def(py::init<ROCmDevice*, ROCmKernelType>(),
         py::arg("device"), py::arg("type") = ROCmKernelType::REDUCE_SUM)
    .def("launch", &ROCmReduceKernel::launch)
    .def("setBlockSize", &ROCmReduceKernel::setBlockSize)
    .def("setGridSize", &ROCmReduceKernel::setGridSize);

  py::class_<ROCmAllreduceKernel, ROCmKernel>(m, "ROCmAllreduceKernel")
    .def(py::init<ROCmDevice*>())
    .def("launch", &ROCmAllreduceKernel::launch)
    .def("setParticipants", &ROCmAllreduceKernel::setParticipants)
    .def("setAlgorithm", &ROCmAllreduceKernel::setAlgorithm)
    .def("setDataType", &ROCmAllreduceKernel::setDataType);

  py::class_<ROCmCopyKernel, ROCmKernel>(m, "ROCmCopyKernel")
    .def(py::init<ROCmDevice*>())
    .def("launch", &ROCmCopyKernel::launch);

  py::class_<ROCmCommKernel, ROCmKernel>(m, "ROCmCommKernel")
    .def(py::init<ROCmDevice*, ROCmKernelType>())
    .def("launch", &ROCmCommKernel::launch)
    .def("setSourceDevice", &ROCmCommKernel::setSourceDevice)
    .def("setDestinationDevice", &ROCmCommKernel::setDestinationDevice)
    .def("setTag", &ROCmCommKernel::setTag);

  py::class_<ROCmKernelRegistry>(m, "ROCmKernelRegistry")
    .def(py::init<>())
    .def("registerKernel", &ROCmKernelRegistry::registerKernel)
    .def("getKernel", &ROCmKernelRegistry::getKernel,
         py::return_value_policy::reference)
    .def("getKernelNames", &ROCmKernelRegistry::getKernelNames)
    .def("getReduceKernel", &ROCmKernelRegistry::getReduceKernel,
         py::return_value_policy::reference)
    .def("getAllreduceKernel", &ROCmKernelRegistry::getAllreduceKernel,
         py::return_value_policy::reference)
    .def("getCopyKernel", &ROCmKernelRegistry::getCopyKernel,
         py::return_value_policy::reference)
    .def("getSendKernel", &ROCmKernelRegistry::getSendKernel,
         py::return_value_policy::reference)
    .def("getRecvKernel", &ROCmKernelRegistry::getRecvKernel,
         py::return_value_policy::reference)
    .def("initializeBuiltinKernels", &ROCmKernelRegistry::initializeBuiltinKernels);

  py::class_<ROCmStreamManager>(m, "ROCmStreamManager")
    .def(py::init<ROCmDevice*>())
    .def("getStream", &ROCmStreamManager::getStream,
         py::arg("stream_id") = 0)
    .def("createStream", &ROCmStreamManager::createStream)
    .def("destroyStream", &ROCmStreamManager::destroyStream)
    .def("synchronizeStream", &ROCmStreamManager::synchronizeStream,
         py::arg("stream_id") = 0)
    .def("synchronizeAllStreams", &ROCmStreamManager::synchronizeAllStreams)
    .def("createEvent", &ROCmStreamManager::createEvent)
    .def("destroyEvent", &ROCmStreamManager::destroyEvent)
    .def("recordEvent", &ROCmStreamManager::recordEvent,
         py::arg("event"), py::arg("stream") = 0)
    .def("waitForEvent", &ROCmStreamManager::waitForEvent,
         py::arg("event"), py::arg("stream") = 0)
    .def("isStreamBusy", &ROCmStreamManager::isStreamBusy,
         py::arg("stream_id") = 0);

  py::class_<ROCmBLASManager>(m, "ROCmBLASManager")
    .def(py::init<ROCmDevice*>())
    .def("getHandle", &ROCmBLASManager::getHandle)
    .def("setStream", &ROCmBLASManager::setStream)
    .def("synchronize", &ROCmBLASManager::synchronize)
    .def("gemm", &ROCmBLASManager::gemm)
    .def("gemmStridedBatched", &ROCmBLASManager::gemmStridedBatched);

  py::class_<DeviceExecutor, std::shared_ptr<DeviceExecutor>>(m, "DeviceExecutor");

  py::class_<ROCmExecutor, DeviceExecutor, std::shared_ptr<ROCmExecutor>>(m, "ROCmExecutor")
    .def(py::init<>())
    .def(py::init<int>())
    .def("initialize", &ROCmExecutor::initialize)
    .def("shutdown", &ROCmExecutor::shutdown)
    .def("allocate", &ROCmExecutor::allocate,
         py::return_value_policy::reference)
    .def("free", &ROCmExecutor::free)
    .def("copy", &ROCmExecutor::copy)
    .def("copyFromHost", &ROCmExecutor::copyFromHost)
    .def("copyToHost", &ROCmExecutor::copyToHost)
    .def("execute", &ROCmExecutor::execute)
    .def("synchronize", &ROCmExecutor::synchronize)
    .def("measureKernelTime", &ROCmExecutor::measureKernelTime,
         py::arg("kernel"), py::arg("args"), py::arg("arg_size"),
         py::arg("grid"), py::arg("block"),
         py::arg("shared_mem") = 0,
         py::arg("iterations") = 100)
    .def("supportsP2P", &ROCmExecutor::supportsP2P)
    .def("enableP2P", &ROCmExecutor::enableP2P)
    .def("getDevicePointer", &ROCmExecutor::getDevicePointer)
    .def("getDeviceId", &ROCmExecutor::getDeviceId)
    .def("getROCmDevice", &ROCmExecutor::getROCmDevice,
         py::return_value_policy::reference)
    .def("getKernelRegistry", &ROCmExecutor::getKernelRegistry,
         py::return_value_policy::reference)
    .def("getStreamManager", &ROCmExecutor::getStreamManager,
         py::return_value_policy::reference)
    .def("getBLASManager", &ROCmExecutor::getBLASManager,
         py::return_value_policy::reference)
    .def("getCurrentStream", &ROCmExecutor::getCurrentStream)
    .def("setCurrentStream", &ROCmExecutor::setCurrentStream);

  py::class_<ROCmTopologyBuilder>(m, "ROCmTopologyBuilder")
    .def_static("buildP2PTopology", &ROCmTopologyBuilder::buildP2PTopology)
    .def_static("isP2PSupported", &ROCmTopologyBuilder::isP2PSupported)
    .def_static("estimateP2PBandwidth", &ROCmTopologyBuilder::estimateP2PBandwidth)
    .def_static("estimateP2PLatency", &ROCmTopologyBuilder::estimateP2PLatency);

  m.def("create_rocm_executor", [](int device_id) {
    return std::make_shared<ROCmExecutor>(device_id);
  }, py::arg("device_id") = 0, py::return_value_policy::reference);

  m.def("create_rocm_device_manager", []() {
    return std::make_unique<ROCmDeviceManager>();
  }, py::return_value_policy::reference);

  m.def("get_rocm_device_count", []() {
    int count = 0;
    hipError_t error = hipGetDeviceCount(&count);
    return (error == hipSuccess) ? count : 0;
  });

  m.def("set_rocm_device", [](int device_id) {
    hipError_t error = hipSetDevice(device_id);
    return error == hipSuccess;
  });

  m.def("get_rocm_device_name", [](int device_id) {
    hipDeviceProp_t props;
    hipError_t error = hipGetDeviceProperties(&props, device_id);
    return (error == hipSuccess) ? std::string(props.name) : std::string("Unknown");
  });

  m.def("rocm_memory_info", [](int device_id) {
    size_t free = 0, total = 0;
    hipSetDevice(device_id);
    hipError_t error = hipMemGetInfo(&free, &total);
    return std::make_tuple(
      (error == hipSuccess) ? static_cast<size_t>(free) : 0,
      (error == hipSuccess) ? static_cast<size_t>(total) : 0
    );
  });
}

}