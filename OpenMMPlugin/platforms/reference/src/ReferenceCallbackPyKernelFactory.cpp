/* -------------------------------------------------------------------------- *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- *
 * The file doc/CREDITS.txt lists several external projects which served      *
 * as valuable guidance for this project. Although not all of these           *
 * projects are directly referenced in every source file, this source file    *
 * complies with all of their licenses.                                       *
 * -------------------------------------------------------------------------- */

#include <openmm/OpenMMException.h>
#include <openmm/internal/ContextImpl.h>
#include <openmm/reference/ReferencePlatform.h>

#include "ReferenceCallbackPyKernelFactory.h"
#include "ReferenceNumPyKernels.h"
#if COMPILE_TORCH_FORCE
#  include "ReferenceTorchKernels.h"
#endif

using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
  for (int i = 0; i < Platform::getNumPlatforms(); ++i) {
    auto& platform = Platform::getPlatform(i);
    if (dynamic_cast<ReferencePlatform*>(&platform) != nullptr) {
      ReferenceCallbackPyKernelFactory* factory = new ReferenceCallbackPyKernelFactory();
      platform.registerKernelFactory(CalcNumPyForceKernel::Name(), factory);
#if COMPILE_TORCH_FORCE
      platform.registerKernelFactory(CalcTorchForceKernel::Name(), factory);
#endif
    }
  }
}

extern "C" OPENMM_EXPORT void registerCallbackPyReferenceKernelFactories() {
  registerKernelFactories();
}

KernelImpl* ReferenceCallbackPyKernelFactory::createKernelImpl(std::string name,
                                                               const Platform& platform,
                                                               ContextImpl& context) const {
  // auto& data = *static_cast<ReferencePlatform*>(context.getPlatformData());
  if (name == CalcNumPyForceKernel::Name())
    return new ReferenceCalcNumPyForceKernel(name, platform);
#if COMPILE_TORCH_FORCE
  else if (name == CalcTorchForceKernel::Name())
    return new ReferenceCalcTorchForceKernel(name, platform);
#endif
  throw OpenMMException(std::string("Tried to create kernel with illegal kernel name '") + name + "'");
}
