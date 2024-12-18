/* -------------------------------------------------------------------------- *
 * MIT License                                                                *
 *                                                                            *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining      *
 * a copy of this software and associated documentation files (the            *
 * "Software"), to deal in the Software without restriction, including        *
 * without limitation the rights to use, copy, modify, merge, publish,        *
 * distribute, sublicense, and/or sell copies of the Software, and to         *
 * permit persons to whom the Software is furnished to do so, subject to      *
 * the following conditions:                                                  *
 *                                                                            *
 * The above copyright notice and this permission notice shall be             *
 * included in all copies or substantial portions of the Software.            *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,            *
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF         *
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.     *
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY       *
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,       *
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE          *
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                     *
 *                                                                            *
 * -------------------------------------------------------------------------- *
 *                                                                            *
 * The license for this source file is stated above. This paragraph and       *
 * the following one are not part of the license statement.                   *
 *                                                                            *
 * The file doc/CREDITS.txt lists serval external projects and their          *
 * licenses, which served as valuable guidance for this project. Although     *
 * not all of these projects are directly referenced in every source          *
 * file, this source file complies with the licenses of all listed            *
 * projects.                                                                  *
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