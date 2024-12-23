/* -------------------------------------------------------------------------- *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- *
 * The file doc/CREDITS.txt lists several external projects which served      *
 * as valuable guidance for this project. Although not all of these           *
 * projects are directly referenced in every source file, this source file    *
 * complies with all of their licenses.                                       *
 * -------------------------------------------------------------------------- */

// <!--...-->
// <!--...-->

#include <openmm/OpenMMException.h>
#include <openmm/internal/ContextImpl.h>

#include "CudaCallbackPyKernelFactory.h"
#include "CudaNumPyKernels.h"
#if COMPILE_TORCH_FORCE
#  include "CudaTorchKernels.h"
#endif

using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
  try {
    Platform& platform = Platform::getPlatformByName("CUDA");
    CudaCallbackPyKernelFactory* factory = new CudaCallbackPyKernelFactory();
    platform.registerKernelFactory(CalcNumPyForceKernel::Name(), factory);
#if COMPILE_TORCH_FORCE
    platform.registerKernelFactory(CalcTorchForceKernel::Name(), factory);
#endif
  } catch (std::exception& e) {
    // ignore
  }
}

extern "C" OPENMM_EXPORT void registerCallbackPyCudaKernelFactories() {
  try {
    Platform::getPlatformByName("CUDA");
  } catch (...) {
    Platform::registerPlatform(new CudaPlatform());
  }
  registerKernelFactories();
}

KernelImpl* CudaCallbackPyKernelFactory::createKernelImpl(std::string name,
                                                          const Platform& platform,
                                                          ContextImpl& context) const {
  CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
  if (name == CalcNumPyForceKernel::Name())
    return new CudaCalcNumPyForceKernel(name, platform, cu);
#if COMPILE_TORCH_FORCE
  else if (name == CalcTorchForceKernel::Name())
    return new CudaCalcTorchForceKernel(name, platform, cu);
#endif
  throw OpenMMException(std::string("Tried to create kernel with illegal kernel name '") + name + "'");
}
