/* -------------------------------------------------------------------------- *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#define CLBK_IMPL_CPP_
#include "clbk/clbk.h"

namespace py = pybind11;
using namespace clbk;

PYBIND11_MODULE(clbk, m) {
  m.doc() = "A PyBind11 Demo of Callback-Callable";

  py::class_<Callable>(m, "Callable")
    .def(py::init<const Callable&>())
    .def("hasKWArg", &Callable::hasKWArg, py::arg("kwarg"))
    .def("returnGradient", &Callable::returnGradient)
    .def("returnForce", &Callable::returnForce)
    .def("getId", &Callable::getId)
    .def_readonly_static("RETURN_ENERGY", &Callable::RETURN_ENERGY)
    .def_readonly_static("RETURN_ENERGY_GRADIENT", &Callable::RETURN_ENERGY_GRADIENT)
    .def_readonly_static("RETURN_ENERGY_FORCE", &Callable::RETURN_ENERGY_FORCE)
    .def("add_kwarg", &Callable::add_kwarg, py::arg("kwarg"))
    .def(py::init<long, int>(), py::arg("cpython_id"), py::arg("return_flag"));
}
