#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <common.h>
#include <engine.h>
#include <base/ir.h>
#include <base/operator.h>
#include <algorithms/allreduce.h>
#include <topology/topology.h>
#include <cluster/process_group.h>
#include <algorithms/algorithm_manager.h>

namespace py = pybind11;

using namespace engine_c;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PCCL Python Bindings";

  py::enum_<OpType>(m, "OpType")
    .value("COPY", OpType::COPY)
    .value("REDUCE", OpType::REDUCE)
    .value("NOTIFY", OpType::NOTIFY)
    .value("GET_NOTIFIED", OpType::GET_NOTIFIED)
    .value("ALLREDUCE", OpType::ALLREDUCE)
    .value("ALLGATHER", OpType::ALLGATHER)
    .value("BROADCAST", OpType::BROADCAST)
    .value("REDUCE_SCATTER", OpType::REDUCE_SCATTER);

  py::enum_<ReduceOp>(m, "ReduceOp")
    .value("SUM", ReduceOp::SUM)
    .value("AVG", ReduceOp::AVG)
    .value("MAX", ReduceOp::MAX)
    .value("MIN", ReduceOp::MIN);

  py::enum_<AllreduceAlgorithm>(m, "AllreduceAlgorithm")
    .value("RING", AllreduceAlgorithm::RING)
    .value("TREE", AllreduceAlgorithm::TREE)
    .value("RABENSEIFNER", AllreduceAlgorithm::RABENSEIFNER)
    .value("DOUBLE_BINARY_TREE", AllreduceAlgorithm::DOUBLE_BINARY_TREE);

  py::enum_<InterconnectType>(m, "InterconnectType")
    .value("NVLINK", InterconnectType::NVLINK)
    .value("PCIE", InterconnectType::PCIE)
    .value("RDMA", InterconnectType::RDMA)
    .value("ETHERNET", InterconnectType::ETHERNET)
    .value("INFINIBAND", InterconnectType::INFINIBAND)
    .value("UNKNOWN", InterconnectType::UNKNOWN);

  py::class_<Value>(m, "Value")
    .def(py::init<DataType, const std::vector<int>&, int>())
    .def("getDataType", &Value::getDataType)
    .def("getShape", &Value::getShape)
    .def("getDeviceId", &Value::getDeviceId)
    .def("getSize", &Value::getSize)
    .def("setDataType", &Value::setDataType)
    .def("setShape", &Value::setShape)
    .def("setDeviceId", &Value::setDeviceId);

  py::class_<Op, std::shared_ptr<Op>>(m, "Op")
    .def("getOpType", &Op::getOpType)
    .def("getName", &Op::getName)
    .def("getOpId", &Op::getOpId)
    .def("addInput", &Op::addInput)
    .def("addOutput", &Op::addOutput)
    .def("addDependency", &Op::addDependency)
    .def("getInputs", &Op::getInputs)
    .def("getOutputs", &Op::getOutputs)
    .def("getDependencies", &Op::getDependencies)
    .def("execute", &Op::execute);

  py::class_<IRBuilder>(m, "IRBuilder")
    .def(py::init<>())
    .def("addOp", &IRBuilder::addOp)
    .def("addDependency", &IRBuilder::addDependency)
    .def("addValue", &IRBuilder::addValue)
    .def("getOp", &IRBuilder::getOp)
    .def("getValue", &IRBuilder::getValue)
    .def("createCopyOp", &IRBuilder::createCopyOp)
    .def("createReduceOp", &IRBuilder::createReduceOp)
    .def("createAllreduceOp", &IRBuilder::createAllreduceOp)
    .def("validateGraph", &IRBuilder::validateGraph);

  py::class_<GraphExecutor>(m, "GraphExecutor")
    .def(py::init<>())
    .def("setGraph", &GraphExecutor::setGraph)
    .def("execute", &GraphExecutor::execute)
    .def("executeAsync", &GraphExecutor::executeAsync)
    .def("waitCompletion", &GraphExecutor::waitCompletion)
    .def("isCompleted", &GraphExecutor::isCompleted)
    .def("getGraph", &GraphExecutor::getGraph);

  py::class_<Engine>(m, "Engine")
    .def(py::init<int, int>())
    .def("initEngine", &Engine::initEngine)
    .def("regOp", &Engine::regOp)
    .def("exeOp", [](Engine& self, const std::string& name, py::array_t<float> input, py::array_t<float> output) {
        auto input_buf = input.request();
        auto output_buf = output.request();
        at::Tensor input_tensor = at::from_blob(input_buf.ptr, input_buf.shape, at::kFloat);
        at::Tensor output_tensor = at::from_blob(output_buf.ptr, output_buf.shape, at::kFloat);
        self.exeOp(name, input_tensor, output_tensor);
    })
    .def("exportEndpoint", &Engine::exportEndpoint)
    .def("joinCluster", &Engine::joinCluster)
    .def("exitCluster", &Engine::exitCluster);

  py::class_<OperatorInfo>(m, "OperatorInfo")
    .def_readonly("name", &OperatorInfo::name)
    .def_readonly("op_type", &OperatorInfo::op_type)
    .def_readonly("executor_type", &OperatorInfo::executor_type)
    .def_readonly("default_params", &OperatorInfo::default_params);

  py::class_<OperatorManager>(m, "OperatorManager")
    .def(py::init<>())
    .def("registerOperator", py::overload_cast<const OperatorInfo&>(&OperatorManager::registerOperator))
    .def("registerOperator", py::overload_cast<const std::string&, OpType, ExecutorType,
         std::function<std::shared_ptr<Op>(const std::unordered_map<std::string, std::string>&)>,
         const std::unordered_map<std::string, std::string>&>(&OperatorManager::registerOperator))
    .def("isOperatorRegistered", &OperatorManager::isOperatorRegistered)
    .def("createOperator", &OperatorManager::createOperator)
    .def("getOperatorInfo", &OperatorManager::getOperatorInfo, py::return_value_policy::reference)
    .def("getRegisteredOperators", &OperatorManager::getRegisteredOperators)
    .def("executeOperator", &OperatorManager::executeOperator)
    .def("loadOperatorFromFile", &OperatorManager::loadOperatorFromFile)
    .def("clear", &OperatorManager::clear);

  py::class_<AllreduceConfig>(m, "AllreduceConfig")
    .def(py::init<>())
    .def_readwrite("algorithm", &AllreduceConfig::algorithm)
    .def_readwrite("reduce_op", &AllreduceConfig::reduce_op)
    .def_readwrite("participants", &AllreduceConfig::participants)
    .def_readwrite("buffer_size", &AllreduceConfig::buffer_size)
    .def_readwrite("enable_overlap", &AllreduceConfig::enable_overlap)
    .def_readwrite("pipeline_depth", &AllreduceConfig::pipeline_depth);

  py::class_<AllreduceImpl>(m, "AllreduceImpl")
    .def("execute", &AllreduceImpl::execute)
    .def("executeAsync", &AllreduceImpl::executeAsync)
    .def("waitCompletion", &AllreduceImpl::waitCompletion)
    .def("getConfig", &AllreduceImpl::getConfig);

  py::class_<AllreduceFactory>(m, "AllreduceFactory")
    .def_static("create", &AllreduceFactory::create)
    .def_static("createRing", &AllreduceFactory::createRing)
    .def_static("createTree", &AllreduceFactory::createTree)
    .def_static("createRabenseifner", &AllreduceFactory::createRabenseifner)
    .def_static("createDoubleBinaryTree", &AllreduceFactory::createDoubleBinaryTree)
    .def_static("selectOptimalAlgorithm", &AllreduceFactory::selectOptimalAlgorithm);

  py::class_<DeviceInfo>(m, "DeviceInfo")
    .def(py::init<>())
    .def_readwrite("device_id", &DeviceInfo::device_id)
    .def_readwrite("device_type", &DeviceInfo::device_type)
    .def_readwrite("device_name", &DeviceInfo::device_name)
    .def_readwrite("memory_bandwidth", &DeviceInfo::memory_bandwidth)
    .def_readwrite("compute_capability", &DeviceInfo::compute_capability)
    .def_readwrite("memory_size", &DeviceInfo::memory_size)
    .def_readwrite("numa_node", &DeviceInfo::numa_node);

  py::class_<LinkInfo>(m, "LinkInfo")
    .def(py::init<>())
    .def_readwrite("src_device", &LinkInfo::src_device)
    .def_readwrite("dst_device", &LinkInfo::dst_device)
    .def_readwrite("interconnect_type", &LinkInfo::interconnect_type)
    .def_readwrite("bandwidth", &LinkInfo::bandwidth)
    .def_readwrite("latency", &LinkInfo::latency)
    .def_readwrite("bidirectional", &LinkInfo::bidirectional);

  py::class_<TopologyMetrics>(m, "TopologyMetrics")
    .def(py::init<>())
    .def_readwrite("total_bandwidth", &TopologyMetrics::total_bandwidth)
    .def_readwrite("average_latency", &TopologyMetrics::average_latency)
    .def_readwrite("network_diameter", &TopologyMetrics::network_diameter)
    .def_readwrite("bisection_bandwidth", &TopologyMetrics::bisection_bandwidth)
    .def_readwrite("connectivity", &TopologyMetrics::connectivity);

  py::class_<Topology>(m, "Topology")
    .def(py::init<>())
    .def("addDevice", &Topology::addDevice)
    .def("addLink", &Topology::addLink)
    .def("getDevice", &Topology::getDevice, py::return_value_policy::reference)
    .def("getLink", &Topology::getLink, py::return_value_policy::reference)
    .def("calculateMetrics", &Topology::calculateMetrics)
    .def("getShortestPaths", &Topology::getShortestPaths)
    .def("isValid", &Topology::isValid)
    .def("isFullyConnected", &Topology::isFullyConnected)
    .def("printTopology", &Topology::printTopology);

  py::class_<TopologyBuilder>(m, "TopologyBuilder")
    .def_static("buildRingTopology", &TopologyBuilder::buildRingTopology)
    .def_static("buildTreeTopology", &TopologyBuilder::buildTreeTopology)
    .def_static("buildMeshTopology", &TopologyBuilder::buildMeshTopology)
    .def_static("buildHierarchicalTopology", &TopologyBuilder::buildHierarchicalTopology)
    .def_static("buildFullyConnectedTopology", &TopologyBuilder::buildFullyConnectedTopology);

  py::class_<TopologyManager>(m, "TopologyManager")
    .def(py::init<>())
    .def("discoverTopology", &TopologyManager::discoverTopology)
    .def("loadTopology", &TopologyManager::loadTopology)
    .def("saveTopology", &TopologyManager::saveTopology)
    .def("setActiveTopology", &TopologyManager::setActiveTopology)
    .def("getActiveTopology", &TopologyManager::getActiveTopology)
    .def("getTopology", &TopologyManager::getTopology)
    .def("addTopology", &TopologyManager::addTopology)
    .def("getAvailableTopologies", &TopologyManager::getAvailableTopologies)
    .def("buildOptimalTopology", &TopologyManager::buildOptimalTopology)
    .def("isDeviceAvailable", &TopologyManager::isDeviceAvailable)
    .def("getAvailableDevices", &TopologyManager::getAvailableDevices)
    .def("updateDeviceMetrics", &TopologyManager::updateDeviceMetrics);

  py::class_<ProcessGroup>(m, "ProcessGroup")
    .def("getName", &ProcessGroup::getName)
    .def("getType", &ProcessGroup::getType)
    .def("getRanks", &ProcessGroup::getRanks)
    .def("getSize", &ProcessGroup::getSize)
    .def("addProcess", &ProcessGroup::addProcess)
    .def("getProcessInfo", &ProcessGroup::getProcessInfo, py::return_value_policy::reference)
    .def("allreduce", &ProcessGroup::allreduce)
    .def("allgather", &ProcessGroup::allgather)
    .def("broadcast", &ProcessGroup::broadcast)
    .def("reduce", &ProcessGroup::reduce)
    .def("reduceScatter", &ProcessGroup::reduceScatter)
    .def("send", &ProcessGroup::send)
    .def("recv", &ProcessGroup::recv)
    .def("barrier", &ProcessGroup::barrier)
    .def("isRankInGroup", &ProcessGroup::isRankInGroup)
    .def("getRankIndex", &ProcessGroup::getRankIndex);

  py::class_<ProcessGroupManager>(m, "ProcessGroupManager")
    .def(py::init<>())
    .def("initialize", &ProcessGroupManager::initialize)
    .def("createProcessGroup", &ProcessGroupManager::createProcessGroup)
    .def("createCPUGroup", &ProcessGroupManager::createCPUGroup)
    .def("createCUDAGroup", &ProcessGroupManager::createCUDAGroup)
    .def("createRDMAGroup", &ProcessGroupManager::createRDMAGroup)
    .def("getProcessGroup", &ProcessGroupManager::getProcessGroup)
    .def("getGlobalProcessGroup", &ProcessGroupManager::getGlobalProcessGroup)
    .def("destroyProcessGroup", &ProcessGroupManager::destroyProcessGroup)
    .def("getProcessGroupNames", &ProcessGroupManager::getProcessGroupNames)
    .def("getGlobalRank", &ProcessGroupManager::getGlobalRank)
    .def("getGlobalSize", &ProcessGroupManager::getGlobalSize);

  py::class_<AlgorithmSelector>(m, "AlgorithmSelector")
    .def(py::init<>())
    .def_readwrite("allreduce_algorithm", &AlgorithmSelector::allreduce_algorithm)
    .def_readwrite("branching_factor", &AlgorithmSelector::branching_factor)
    .def_readwrite("enable_overlap", &AlgorithmSelector::enable_overlap)
    .def_readwrite("pipeline_depth", &AlgorithmSelector::pipeline_depth)
    .def_readwrite("buffer_size", &AlgorithmSelector::buffer_size);

  py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
    .def(py::init<>())
    .def_readwrite("bandwidth_utilization", &PerformanceMetrics::bandwidth_utilization)
    .def_readwrite("latency", &PerformanceMetrics::latency)
    .def_readwrite("overlap_ratio", &PerformanceMetrics::overlap_ratio)
    .def_readwrite("efficiency", &PerformanceMetrics::efficiency)
    .def_readwrite("total_time", &PerformanceMetrics::total_time);

  py::class_<AlgorithmManager>(m, "AlgorithmManager")
    .def(py::init<>())
    .def("initialize", &AlgorithmManager::initialize)
    .def("createAllreduce", &AlgorithmManager::createAllreduce)
    .def("createOptimalAllreduce", &AlgorithmManager::createOptimalAllreduce)
    .def("selectAlgorithm", &AlgorithmManager::selectAlgorithm)
    .def("benchmarkAllreduce", &AlgorithmManager::benchmarkAllreduce)
    .def("autoTune", &AlgorithmManager::autoTune)
    .def("addCustomAlgorithm", &AlgorithmManager::addCustomAlgorithm)
    .def("removeCustomAlgorithm", &AlgorithmManager::removeCustomAlgorithm)
    .def("getAvailableAlgorithms", &AlgorithmManager::getAvailableAlgorithms)
    .def("saveProfiles", &AlgorithmManager::saveProfiles)
    .def("loadProfiles", &AlgorithmManager::loadProfiles)
    .def("setTopologyManager", &AlgorithmManager::setTopologyManager)
    .def("setProcessGroupManager", &AlgorithmManager::setProcessGroupManager);

  m.def("createAllreduce", [](AllreduceAlgorithm algorithm, ReduceOp reduce_op,
                              const std::vector<int>& participants, size_t data_size) {
    AllreduceConfig config;
    config.algorithm = algorithm;
    config.reduce_op = reduce_op;
    config.participants = participants;
    config.buffer_size = 128 * 1024 * 1024;
    return AllreduceFactory::create(config);
  });

  m.def("executeAllreduce", [](std::shared_ptr<AllreduceImpl> allreduce_impl,
                               py::array_t<float> input, py::array_t<float> output) {
    auto input_buf = input.request();
    auto output_buf = output.request();
    void* input_ptr = input_buf.ptr;
    void* output_ptr = output_buf.ptr;
    size_t data_size = input_buf.size * sizeof(float);

    allreduce_impl->execute(input_ptr, output_ptr, data_size, DataType::FLOAT32);
  });

  m.def("createTopology", [](const std::vector<int>& device_ids) {
    return TopologyBuilder::buildRingTopology(device_ids);
  });

  m.def("getPerformanceStats", []() {
    return py::dict();
  });

  m.def("queryTopology", []() {
    return py::dict();
  });

  m.def("registerOperator", [](const std::string& name, py::object config) {
    return py::dict();
  });

  auto utils_submodule = m.def_submodule("utils");

  py::class_<utils::LaunchEnvironments>(utils_submodule, "LaunchEnvironments")
    .def_static("listOpt", &utils::LaunchEnvironments::listOpt)
    .def_static("getEnv", &utils::LaunchEnvironments::getEnv)
    .def_static("registerOpt", &utils::LaunchEnvironments::registerOpt);
}



