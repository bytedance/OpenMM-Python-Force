/* -------------------------------------------------------------------------- *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- *
 * The file doc/CREDITS.txt lists several external projects which served      *
 * as valuable guidance for this project. Although not all of these           *
 * projects are directly referenced in every source file, this source file    *
 * complies with all of their licenses.                                       *
 * -------------------------------------------------------------------------- */

#pragma once
#include <pybind11/numpy.h>

template <class T, class ANY>
static pybind11::array numpy_2d_array_from_blob(ANY* data, int natoms, int ndim = 3) {
  int array_nd = 2;
  return pybind11::array(pybind11::buffer_info(reinterpret_cast<void*>(data), sizeof(T),
                                               pybind11::format_descriptor<T>::format(), array_nd, {natoms, ndim},
                                               {ndim * sizeof(T), sizeof(T)}));
}
