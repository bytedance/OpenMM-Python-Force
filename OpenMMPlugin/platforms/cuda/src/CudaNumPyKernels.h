/* -------------------------------------------------------------------------- *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- *
 * The file doc/CREDITS.txt lists serval external projects which served       *
 * as valuable guidance for this project. Although not all of these           *
 * projects are directly referenced in every source file, this source file    *
 * complies with all of their licenses.                                       *
 * -------------------------------------------------------------------------- */

#pragma once
#include <vector>

#include <openmm/Platform.h>
#include <openmm/cuda/CudaContext.h>

#include "NumPyKernels.h"

namespace OpenMM {
class CudaCalcNumPyForceKernel: public CalcNumPyForceKernel {
private:
  CudaArray gradsBuffer_;
  std::vector<double> pos_, cell_;
  CudaContext& cu_;
  CUcontext primaryContext_;
  CUfunction addForcesKernel_;
  const clbk::Callable* callable_ptr_;
  int numParticles_;
  int paddedNumAtoms_;

public:
  CudaCalcNumPyForceKernel(std::string name, const Platform& platform, CudaContext& cu);
  ~CudaCalcNumPyForceKernel();

  void initialize(const System& system, const NumPyForce& force) override;
  double execute(ContextImpl& context, bool includeForces, bool includeEnergy) override;
};
}
