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
 *                                                                            *
 * -------------------------------------------------------------------------- *
 *                                                                            *
 * The license for this source file is stated above. This paragraph and       *
 * the following one are not part of the license statement.                   *
 *                                                                            *
 * The file doc/CREDITS.txt lists serval external projects and their          *
 * licenses, which served as valuable guidance for this project. Although     *
 * not all of these projects are directly referenced in every source          *
 * file, this source file complies with the licenses of all listed            *
 * projects.                                                                  *
 * -------------------------------------------------------------------------- */

#include <pybind11/pybind11.h>

#include <vector>

#include <openmm/OpenMMException.h>
#include <openmm/internal/ContextImpl.h>
#include <openmm/reference/ReferencePlatform.h>
#include <torch/extension.h>

#include "ReferenceTorchKernels.h"

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

ReferenceCalcTorchForceKernel::ReferenceCalcTorchForceKernel(std::string name, const Platform& platform):
  CalcTorchForceKernel(name, platform) {}

ReferenceCalcTorchForceKernel::~ReferenceCalcTorchForceKernel() {}

void ReferenceCalcTorchForceKernel::initialize(const System& system, const TorchForce& force) {
  callable_ptr_ = &force.getCallable();
}

double ReferenceCalcTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
  auto dtype = torch::kFloat64;
  auto& pos = extractPositions(context);
  int numParticles = pos.size();
  auto posOptions = torch::TensorOptions().dtype(dtype).requires_grad(true);
  torch::Tensor posTensor = torch::from_blob(pos.data(), {numParticles, 3}, posOptions);

  py::gil_scoped_acquire gilAcquire;
  // torch::from_blob() doesn't take ownership; neither should py::cast().
  py::object posPy = py::cast(posTensor, py::return_value_policy::reference);
  py::dict kwargsPy;
  if (callable_ptr_->hasKWArg("cell")) {
    auto box = extractBoxVectorsPtr(context);
    auto boxOptions = torch::TensorOptions().dtype(dtype);
    torch::Tensor boxTensor = torch::from_blob(box, {3, 3}, boxOptions);
    // torch::from_blob() doesn't take ownership; neither should py::cast().
    kwargsPy["cell"] = py::cast(boxTensor, py::return_value_policy::reference);
  }
  if (callable_ptr_->hasKWArg("includeForces")) {
    kwargsPy["includeForces"] = py::cast(static_cast<bool>(includeForces));
  }

  auto callPy = callable_ptr_->nonownerCast<py::object>();
  auto resultsPy = callPy(posPy, **kwargsPy);

  // py to torch tensors
  torch::Tensor energyTensor, gradsTensor;
  if (callable_ptr_->returnGradient() or callable_ptr_->returnForce()) {
    py::tuple resultsTuple = resultsPy.cast<py::tuple>();
    energyTensor = resultsTuple[0].cast<torch::Tensor>();
    gradsTensor = resultsTuple[1].cast<torch::Tensor>();
    py::gil_scoped_release gilRelease;
  } else {
    energyTensor = resultsPy.cast<torch::Tensor>();
    py::gil_scoped_release gilRelease;
    if (includeForces) {
      energyTensor.backward();
      gradsTensor = posTensor.grad();
    }
  }

  // torch tensors to OpenMM
  if (includeForces) {
    int gradSign = 1;
    if (callable_ptr_->returnForce())
      gradSign = -1;
    auto& forces = extractForces(context);
    const void* gradsVoidPtr = gradsTensor.data_ptr();
    if (gradsTensor.dtype() == torch::kFloat64) {
      const double* gradsPtr = reinterpret_cast<const double*>(gradsVoidPtr);
      for (int i = 0; i < numParticles; ++i)
        for (int j = 0; j < 3; ++j)
          forces[i][j] -= gradSign * gradsPtr[3 * i + j];
    } else if (gradsTensor.dtype() == torch::kFloat32) {
      const float* gradsPtr = reinterpret_cast<const float*>(gradsVoidPtr);
      for (int i = 0; i < numParticles; ++i)
        for (int j = 0; j < 3; ++j)
          forces[i][j] -= gradSign * gradsPtr[3 * i + j];
    } else {
      throw OpenMMException(
        "ReferenceCalcTorchForceKernel::execute(): Gradient Tensor is not of dtype"
        " torch::kFloat32 or torch::kFloat64");
    }
  }

  double energyValue = 0.0;
  if (includeEnergy)
    energyValue = energyTensor.item<double>();
  return energyValue;
}
