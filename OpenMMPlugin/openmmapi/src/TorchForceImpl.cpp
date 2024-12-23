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

#include "TorchKernels.h"
#include "internal/TorchForceImpl.h"

using namespace OpenMM;

void TorchForceImpl::initialize(ContextImpl& context) {
  calcKernel_ = context.getPlatform().createKernel(CalcTorchForceKernel::Name(), context);
  calcKernel_.getAs<CalcTorchForceKernel>().initialize(context.getSystem(), owner_);
}

double TorchForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
  if (0 != (groups & (1 << owner_.getForceGroup())))
    return calcKernel_.getAs<CalcTorchForceKernel>().execute(context, includeForces, includeEnergy);
  return 0.0;
}

std::map<std::string, double> TorchForceImpl::getDefaultParameters() {
  std::map<std::string, double> parameters;
  return parameters;
}

std::vector<std::string> TorchForceImpl::getKernelNames() {
  std::vector<std::string> names;
  names.emplace_back(CalcTorchForceKernel::Name());
  return names;
}
