#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <common.h>
#include <engine/engine.h>

namespace py = pybind11;

using namespace engine_c;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "PCCL Python Bindings - High-performance Communication Library";

    py::class_<Engine, std::unique_ptr<Engine, py::nodelete>>(m, "Engine")
        .def_static("get_instance", &Engine::getInstance, py::return_value_policy::reference)
        .def("regOp", &Engine::regOp, py::arg("name"), py::arg("filename"))
        .def("exeOp", &Engine::exeOp, py::arg("name"), py::arg("input"), py::arg("output"))
        .def("exeOpAsync", &Engine::exeOpAsync, py::arg("name"), py::arg("input"), py::arg("output"))
        .def("syncOp", &Engine::syncOp, py::arg("name"))
        .def("resetSignals", &Engine::resetSignals, py::arg("name"))
        .def("exportEndpoint", &Engine::exportEndpoint, py::return_value_policy::reference)
        .def("updateEndpoint", &Engine::updateEndpoint, py::arg("rank"), py::arg("endpoint"));

    auto common_submodule = m.def_submodule("common");

    py::class_<common::Environs>(common_submodule, "Environs")
        .def_static("listOpt", &common::Environs::listOpt)
        .def_static("getEnv", &common::Environs::getEnv, py::arg("env"))
        .def_static("registerOpt", &common::Environs::registerOpt, py::arg("option"));

    m.attr("version") = "0.3.0";
}
