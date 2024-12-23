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

#include <openmm/Platform.h>
#include <openmm/cuda/CudaContext.h>

#include <ATen/cuda/CUDAGraph.h>
#include <torch/extension.h>

#include "TorchKernels.h"

namespace OpenMM {
class CudaCalcTorchForceKernel: public CalcTorchForceKernel {
private:
  std::map<bool, at::cuda::CUDAGraph> graphs_;
  std::vector<std::string> globalNames_;
  CudaContext& cu_;
  CUcontext primaryContext_;
  CUfunction copyInputsKernel_, addForcesKernel_;
  const clbk::Callable* callable_ptr_;
  torch::Tensor posTensor_, boxTensor_, energyTensor_, gradsTensor_;
  int warmupSteps_;
  bool useGraphs_;

public:
  CudaCalcTorchForceKernel(std::string name, const Platform& platform, CudaContext& cu);
  ~CudaCalcTorchForceKernel();

  void initialize(const System& system, const TorchForce& force) override;
  double execute(ContextImpl& context, bool includeForces, bool includeEnergy) override;
};
}
