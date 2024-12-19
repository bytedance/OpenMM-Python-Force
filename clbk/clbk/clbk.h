/* -------------------------------------------------------------------------- *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#pragma once

/**
 * How to Compile Callback-Callable
 * ================================
 *
 * Add a `.cpp` file to your project.
 *
 * ```cpp
 * // clbk.cpp
 * #define CLBK_IMPL_CPP_
 * #include "clbk/clbk.h"
 * ```
 *
 * The include path is stored in the CMake variable `CLBK_INCLUDE_DIR` from file `cmake/path.cmake`,
 * which can be added to `CMakeLists.txt` by command `INCLUDE()`.
 *
 * Make sure the include directories and libraries of Python3 and pybind11 are also passed to the compiler.
 */

#ifdef CLBK_IMPL_CPP_
#  include <pybind11/pybind11.h>

#  include <cstdint>
#  include <set>
#  include <stdexcept>
#endif

#include <memory>
#include <string>

namespace clbk {
class Callable {
private:
  class Impl;
  std::unique_ptr<Impl> impl_;

public:
  Callable() = delete;
  Callable(Callable&&) = default;

  template <class T>
  T nonownerCast() const;

  bool hasKWArg(std::string kwarg) const;
  bool returnGradient() const;
  bool returnForce() const;
  long getId() const;

  Callable(const Callable&);
  ~Callable();

  // export to python
public:
  static const int RETURN_ENERGY;
  static const int RETURN_ENERGY_GRADIENT;
  static const int RETURN_ENERGY_FORCE;
  void add_kwarg(std::string kwarg);
  Callable(long cpython_id, int return_flag);
};
}

#ifdef CLBK_IMPL_CPP_

//
// C++ Implementation
//

namespace py = pybind11;
using namespace clbk;

// Callable::Impl

// visibility("hidden") is required by pybind11
class [[gnu::visibility("hidden")]] Callable::Impl {
private:
  union ID {
    static_assert(sizeof(PyObject*) == sizeof(long), "sizeof(PyObject*) != sizeof(long)");
    PyObject* ptr_;
    long id_;

    ID(long cpython_id):
      id_(cpython_id) {}
  };

  static const std::set<std::string> allowedKWArgs_;
  std::set<std::string> kwargs_;
  ID u_;
  int return_flag_;

public:
  enum {
    RETURN_ENERGY = 0x01,
    RETURN_GRADIENT = 0x02,
    RETURN_FORCE = 0x04,
  };

  Impl(long cpython_id, int return_flag):
    kwargs_(),
    u_(cpython_id),
    return_flag_(return_flag) {}

  Impl(const Impl&) = default;

  py::object nonownerCast() const {
    return py::reinterpret_borrow<py::object>(u_.ptr_);
  }

  bool hasKWArg(std::string kwarg) const {
    return kwargs_.find(kwarg) != kwargs_.end();
  }

  bool returnGradient() const {
    return return_flag_ & RETURN_GRADIENT;
  }

  bool returnForce() const {
    return return_flag_ & RETURN_FORCE;
  }

  long getId() const {
    return u_.id_;
  }

  void add_kwarg(std::string kwarg) {
    if (allowedKWArgs_.find(kwarg) == allowedKWArgs_.end())
      throw std::invalid_argument(std::string("kwarg ") + kwarg + std::string(" is not allowed"));
    kwargs_.insert(kwarg);
  }
};

const std::set<std::string> Callable::Impl::allowedKWArgs_ = {
  "cell",
  "includeForces",
};

// Callable

// make it visiable to other libraries that link against API
template <>
[[gnu::visibility("default")]] py::object Callable::nonownerCast() const {
  return impl_->nonownerCast();
}

bool Callable::hasKWArg(std::string kwarg) const {
  return impl_->hasKWArg(kwarg);
}

bool Callable::returnGradient() const {
  return impl_->returnGradient();
}

bool Callable::returnForce() const {
  return impl_->returnForce();
}

long Callable::getId() const {
  return impl_->getId();
}

Callable::Callable(const Callable& c):
  impl_(new Callable::Impl(*c.impl_)) {}

Callable::~Callable() = default;

const int Callable::RETURN_ENERGY = Callable::Impl::RETURN_ENERGY;
const int Callable::RETURN_ENERGY_GRADIENT = Callable::Impl::RETURN_ENERGY | Callable::Impl::RETURN_GRADIENT;
const int Callable::RETURN_ENERGY_FORCE = Callable::Impl::RETURN_ENERGY | Callable::Impl::RETURN_FORCE;

void Callable::add_kwarg(std::string kwarg) {
  impl_->add_kwarg(kwarg);
}

Callable::Callable(long cpython_id, int return_flag):
  impl_(new Callable::Impl(cpython_id, return_flag)) {}

#endif
