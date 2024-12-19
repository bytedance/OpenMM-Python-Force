/* -------------------------------------------------------------------------- *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- *
 * The file doc/CREDITS.txt lists serval external projects which served       *
 * as valuable guidance for this project. Although not all of these           *
 * projects are directly referenced in every source file, this source file    *
 * complies with all of their licenses.                                       *
 * -------------------------------------------------------------------------- */

#include <openmm/OpenMMException.h>

#include "TorchForce.h"
#include "internal/TorchForceImpl.h"

using namespace OpenMM;

ForceImpl* TorchForce::createImpl() const {
  return new TorchForceImpl(*this);
}

TorchForce::TorchForce(const clbk::Callable& callable, const std::map<std::string, std::string>& properties):
  callable_(callable) {
  std::map<std::string, std::string> defaultProperties = {{"useCUDAGraphs", "false"}, {"CUDAGraphWarmupSteps", "10"}};
  properties_ = defaultProperties;
  for (const auto& kv : properties) {
    if (defaultProperties.find(kv.first) == defaultProperties.end())
      throw OpenMMException("TorchForce: Unknown property '" + kv.first + "'");
    properties_[kv.first] = kv.second;
  }
  this->setName("TorchForce@CallbackPyForce");
}

const clbk::Callable& TorchForce::getCallable() const {
  return callable_;
}

bool TorchForce::usesPeriodicBoundaryConditions() const {
  return callable_.hasKWArg("cell");
}

void TorchForce::setProperty(const std::string& name, const std::string& value) {
  if (properties_.find(name) == properties_.end())
    throw OpenMMException("TorchForce: Unknown property '" + name + "'");
  properties_[name] = value;
}

const std::map<std::string, std::string>& TorchForce::getProperties() const {
  return properties_;
}
