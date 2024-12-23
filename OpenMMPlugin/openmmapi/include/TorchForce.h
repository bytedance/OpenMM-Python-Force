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
#include <map>
#include <string>

#include <openmm/Force.h>

#include "Callable.h"

namespace OpenMM {
class OPENMM_EXPORT TorchForce: public Force {
private:
  std::map<std::string, std::string> properties_;
  clbk::Callable callable_;

protected:
  ForceImpl* createImpl() const override;

public:
  TorchForce(const clbk::Callable& callable, const std::map<std::string, std::string>& properties = {});

  const clbk::Callable& getCallable() const;
  bool usesPeriodicBoundaryConditions() const override;

  void setProperty(const std::string& name, const std::string& value);
  const std::map<std::string, std::string>& getProperties() const;
};
}
