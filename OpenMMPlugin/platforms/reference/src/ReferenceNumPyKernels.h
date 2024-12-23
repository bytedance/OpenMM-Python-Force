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
#include <string>

#include <openmm/Platform.h>

#include "NumPyKernels.h"

namespace OpenMM {
class ReferenceCalcNumPyForceKernel: public CalcNumPyForceKernel {
private:
  const clbk::Callable* callable_ptr_;

public:
  ReferenceCalcNumPyForceKernel(std::string name, const Platform& Platform);
  ~ReferenceCalcNumPyForceKernel();

  void initialize(const System& system, const NumPyForce& force) override;
  double execute(ContextImpl& context, bool includeForces, bool includeEnergy) override;
};
}
