/* -------------------------------------------------------------------------- *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- *
 * The file doc/CREDITS.txt lists several external projects which served      *
 * as valuable guidance for this project. Although not all of these           *
 * projects are directly referenced in every source file, this source file    *
 * complies with all of their licenses.                                       *
 * -------------------------------------------------------------------------- */

#include "NumPyForce.h"
#include "internal/NumPyForceImpl.h"

using namespace OpenMM;

ForceImpl* NumPyForce::createImpl() const {
  return new NumPyForceImpl(*this);
}

NumPyForce::NumPyForce(const clbk::Callable& callable, const std::map<std::string, std::string>& properties):
  callable_(callable) {
  std::map<std::string, std::string> defaultProperties;
  properties_ = defaultProperties;
  for (const auto& kv : properties) {
    if (defaultProperties.find(kv.first) == defaultProperties.end())
      throw OpenMMException("NumPyForce: Unknown property '" + kv.first + "'");
    properties_[kv.first] = kv.second;
  }
  this->setName("NumPyForce@CallbackPyForce");
}

const clbk::Callable& NumPyForce::getCallable() const {
  return callable_;
}

bool NumPyForce::usesPeriodicBoundaryConditions() const {
  return callable_.hasKWArg("cell");
}

void NumPyForce::setProperty(const std::string& name, const std::string& value) {
  if (properties_.find(name) == properties_.end())
    throw OpenMMException("TorchForce: Unknown property '" + name + "'");
  properties_[name] = value;
}

const std::map<std::string, std::string>& NumPyForce::getProperties() const {
  return properties_;
}
