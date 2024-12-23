/* -------------------------------------------------------------------------- *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- *
 * The file doc/CREDITS.txt lists several external projects which served      *
 * as valuable guidance for this project. Although not all of these           *
 * projects are directly referenced in every source file, this source file    *
 * complies with all of their licenses.                                       *
 * -------------------------------------------------------------------------- */

#include <openmm/internal/ContextImpl.h>

#include "NumPyKernels.h"
#include "internal/NumPyForceImpl.h"

using namespace OpenMM;

void NumPyForceImpl::initialize(ContextImpl& context) {
  calcKernel_ = context.getPlatform().createKernel(CalcNumPyForceKernel::Name(), context);
  calcKernel_.getAs<CalcNumPyForceKernel>().initialize(context.getSystem(), owner_);
}

double NumPyForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
  if (0 != (groups & (1 << owner_.getForceGroup())))
    return calcKernel_.getAs<CalcNumPyForceKernel>().execute(context, includeForces, includeEnergy);
  return 0.0;
}

std::map<std::string, double> NumPyForceImpl::getDefaultParameters() {
  std::map<std::string, double> parameters;
  return parameters;
}

std::vector<std::string> NumPyForceImpl::getKernelNames() {
  std::vector<std::string> names;
  names.emplace_back(CalcNumPyForceKernel::Name());
  return names;
}
