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

#include "TorchKernels.h"

namespace OpenMM {
class ReferenceCalcTorchForceKernel: public CalcTorchForceKernel {
private:
  const clbk::Callable* callable_ptr_;

public:
  ReferenceCalcTorchForceKernel(std::string name, const Platform& platform);
  ~ReferenceCalcTorchForceKernel();

  void initialize(const System& system, const TorchForce& force) override;
  double execute(ContextImpl& context, bool includeForces, bool includeEnergy) override;
};
}
