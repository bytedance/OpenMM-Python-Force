/* -------------------------------------------------------------------------- *
 * MIT License                                                                *
 *                                                                            *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining      *
 * a copy of this software and associated documentation files (the            *
 * "Software"), to deal in the Software without restriction, including        *
 * without limitation the rights to use, copy, modify, merge, publish,        *
 * distribute, sublicense, and/or sell copies of the Software, and to         *
 * permit persons to whom the Software is furnished to do so, subject to      *
 * the following conditions:                                                  *
 *                                                                            *
 * The above copyright notice and this permission notice shall be             *
 * included in all copies or substantial portions of the Software.            *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,            *
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF         *
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.     *
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY       *
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,       *
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE          *
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                     *
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
