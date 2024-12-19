/* -------------------------------------------------------------------------- *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- *
 * The file doc/CREDITS.txt lists serval external projects which served       *
 * as valuable guidance for this project. Although not all of these           *
 * projects are directly referenced in every source file, this source file    *
 * complies with all of their licenses.                                       *
 * -------------------------------------------------------------------------- */

#include <pybind11/pybind11.h>

#include <vector>

#include <openmm/OpenMMException.h>
#include <openmm/internal/ContextImpl.h>
#include <openmm/reference/ReferencePlatform.h>

#include "NumPy2DArrayFromBlob.h"
#include "ReferenceNumPyKernels.h"

using std::vector;
using namespace OpenMM;
namespace py = pybind11;

static ReferencePlatform::PlatformData& extractPlatformData(ContextImpl& context) {
  return *reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
}

static vector<Vec3>& extractPositions(ContextImpl& context) {
  return *extractPlatformData(context).positions;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
  return *extractPlatformData(context).forces;
}

static Vec3* extractBoxVectorsPtr(ContextImpl& context) {
  return extractPlatformData(context).periodicBoxVectors;
}

ReferenceCalcNumPyForceKernel::ReferenceCalcNumPyForceKernel(std::string name, const Platform& platform):
  CalcNumPyForceKernel(name, platform) {}

ReferenceCalcNumPyForceKernel::~ReferenceCalcNumPyForceKernel() {}

void ReferenceCalcNumPyForceKernel::initialize(const System& system, const NumPyForce& force) {
  callable_ptr_ = &force.getCallable();
}

double ReferenceCalcNumPyForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
  auto& pos = extractPositions(context);
  int numParticles = pos.size();
  int ndim = 2;

  py::gil_scoped_acquire gilAcquire;
  static_assert(sizeof(pos.data()[0]) == 3 * sizeof(double), "");
  auto posNp = numpy_2d_array_from_blob<double>(pos.data(), numParticles);
  py::dict kwargsPy;
  if (callable_ptr_->hasKWArg("cell")) {
    auto box = extractBoxVectorsPtr(context);
    kwargsPy["cell"] = numpy_2d_array_from_blob<double>(box, 3);
  }
  if (callable_ptr_->hasKWArg("includeForces")) {
    kwargsPy["includeForces"] = py::cast(static_cast<bool>(includeForces));
  }

  auto callPy = callable_ptr_->nonownerCast<py::object>();
  auto resultsPy = callPy(posNp, **kwargsPy);

  double energyNum;
  py::array gradsNpArray;
  if (callable_ptr_->returnGradient() or callable_ptr_->returnForce()) {
    py::tuple resultsTuple = resultsPy.cast<py::tuple>();
    energyNum = resultsTuple[0].cast<double>();
    gradsNpArray = resultsTuple[1].cast<py::array>();
  } else {
    energyNum = resultsPy.cast<double>();
  }

  if (includeForces) {
    int gradSign = 0;
    if (callable_ptr_->returnForce())
      gradSign = -1;
    else if (callable_ptr_->returnGradient())
      gradSign = 1;
    else
      throw OpenMMException(std::string("Callable must return force or gradient"));
    auto buffer = gradsNpArray.request();
    auto& forces = extractForces(context);
    if (gradsNpArray.dtype().is(py::dtype::of<double>())) {
      const double* gradsPtr = static_cast<const double*>(buffer.ptr);
      for (int i = 0; i < numParticles; ++i)
        for (int j = 0; j < 3; ++j)
          forces[i][j] -= gradSign * gradsPtr[3 * i + j];
    } else if (gradsNpArray.dtype().is(py::dtype::of<float>())) {
      const float* gradsPtr = static_cast<const float*>(buffer.ptr);
      for (int i = 0; i < numParticles; ++i)
        for (int j = 0; j < 3; ++j)
          forces[i][j] -= gradSign * gradsPtr[3 * i + j];
    }
  }
  py::gil_scoped_release gilRelease;

  double energyValue = 0.0;
  if (includeEnergy)
    energyValue = energyNum;
  return energyValue;
}
