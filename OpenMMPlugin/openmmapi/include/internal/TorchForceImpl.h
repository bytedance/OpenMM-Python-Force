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
#include <vector>

#include <openmm/Kernel.h>
#include <openmm/internal/ForceImpl.h>

#include "TorchForce.h"

namespace OpenMM {
class OPENMM_EXPORT TorchForceImpl: public ForceImpl {
private:
  Kernel calcKernel_;
  const TorchForce& owner_;

public:
  TorchForceImpl(const TorchForce& owner):
    owner_(owner) {}
  ~TorchForceImpl() = default;
  const TorchForce& getOwner() const override {
    return owner_;
  }
  void updateContextState(ContextImpl&, bool&) override {
    // This force does not update the state directly.
  }

  void initialize(ContextImpl& context) override;
  double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) override;

  std::map<std::string, double> getDefaultParameters() override;
  std::vector<std::string> getKernelNames() override;
};
}
