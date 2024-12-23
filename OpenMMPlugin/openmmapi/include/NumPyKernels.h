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

#include <openmm/KernelImpl.h>
#include <openmm/System.h>

#include "NumPyForce.h"

namespace OpenMM {
class CalcNumPyForceKernel: public KernelImpl {
public:
  static std::string Name() {
    return "CalcNumPyForce@CallbackPyForce";
  }

  CalcNumPyForceKernel(std::string name, const Platform& platform):
    KernelImpl(name, platform) {}

  virtual void initialize(const System& system, const NumPyForce& force) = 0;
  virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};
}
