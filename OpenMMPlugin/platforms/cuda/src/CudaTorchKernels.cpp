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

#include <pybind11/pybind11.h>

#include <cuda_runtime_api.h>
#include <openmm/OpenMMException.h>
#include <openmm/common/ContextSelector.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "CommonCudaMacros.h"
#include "CudaCallbackPyKernelSources.h"
#include "CudaTorchKernels.h"

using namespace OpenMM;
namespace py = pybind11;

static void* getTensorPointer(CudaContext& cu, torch::Tensor& tensor) {
  void* data;
  if (cu.getUseDoublePrecision()) {
    assert(tensor.dtype() == torch::kFloat64);
    data = tensor.to(torch::kFloat64).data_ptr<double>();
  } else {
    assert(tensor.dtype() == torch::kFloat32);
    data = tensor.to(torch::kFloat32).data_ptr<float>();
  }
  return data;
}

CudaCalcTorchForceKernel::CudaCalcTorchForceKernel(std::string name, const Platform& platform, CudaContext& cu):
  CalcTorchForceKernel(name, platform),
  cu_(cu) {
  CHECK_RESULT(cuDevicePrimaryCtxRetain(&primaryContext_, cu_.getDevice()), "Failed to retain the primary context");
}

CudaCalcTorchForceKernel::~CudaCalcTorchForceKernel() {
  cuDevicePrimaryCtxRelease(cu_.getDevice());
}

void CudaCalcTorchForceKernel::initialize(const System& system, const TorchForce& force) {
  callable_ptr_ = &force.getCallable();

  // current CUcontext: ctx42
  // switch to the primary cuda context
  CURRENT_CUDA_CTX_PUSH(primaryContext_);
  // current CUcontext: ctxPrimary
  const torch::Device device(torch::kCUDA, cu_.getDeviceIndex());
  auto dtype = cu_.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32;
  auto options = torch::TensorOptions().device(device).dtype(dtype);
  int numParticles = system.getNumParticles();
  posTensor_ = torch::empty({numParticles, 3}, options.requires_grad(true));  // always requires_grad
  boxTensor_ = torch::empty({3, 3}, options);
  energyTensor_ = torch::empty({0}, options);
  gradsTensor_ = torch::empty({numParticles, 3}, options);
  CURRENT_CUDA_CTX_POP(primaryContext_);  // current CUcontext: ctx42

  // switch to an openmm cuda context
  ContextSelector selector(cu_);  // RAII
  // current CUcontext: ctxOpenMM
  std::map<std::string, std::string> defines;
  CUmodule program = cu_.createModule(CudaCallbackPyKernelSources::kernels, defines);
  copyInputsKernel_ = cu_.getKernel(program, "copyInputs");
  addForcesKernel_ = cu_.getKernel(program, "addForces");
  // these kernels must be executed in the ctxOpenMM

  warmupSteps_ = 10;
  useGraphs_ = false;
  const auto& properties = force.getProperties();
  const auto& useCUDAGraphsString = properties.at("useCUDAGraphs");
  if (useCUDAGraphsString == "true")
    useGraphs_ = true;
  else if (useCUDAGraphsString == "false" or useCUDAGraphsString == "")
    useGraphs_ = false;
  else
    throw OpenMMException("TorchForce: invalid value of \"useCUDAGraphs\" " + useCUDAGraphsString);
  if (useGraphs_) {
    const auto& warmupStepString = properties.at("CUDAGraphWarmupSteps");
    warmupSteps_ = std::stoi(warmupStepString);
    if (warmupSteps_ <= 0) {
      throw OpenMMException("TorchForce: \"CUDAGraphWarmupSteps\" must be a positive integer " + warmupStepString);
    }
  }

  // RAII ~dtor();
  // current CUcontext: ctx42
}

static void executeGraph(bool includeForces,
                         const clbk::Callable* callable_ptr_,
                         const py::object& posPy,
                         const py::object& kwargsPy,
                         torch::Tensor& posTensor_,
                         torch::Tensor& energyTensor_,
                         torch::Tensor& gradsTensor_) {
  py::gil_scoped_acquire gilAcquire;
  auto callPy = callable_ptr_->nonownerCast<py::object>();
  auto resultsPy = callPy(posPy, **kwargsPy);

  // py to torch tensors
  if (callable_ptr_->returnGradient() or callable_ptr_->returnForce()) {
    py::tuple resultsTuple = resultsPy.cast<py::tuple>();
    energyTensor_ = resultsTuple[0].cast<torch::Tensor>();
    gradsTensor_ = resultsTuple[1].cast<torch::Tensor>();
    py::gil_scoped_release gilRelease;
  } else {
    energyTensor_ = resultsPy.cast<torch::Tensor>();
    py::gil_scoped_release gilRelease;
    if (includeForces) {
      energyTensor_.backward();
      gradsTensor_ = posTensor_.grad().clone();
      posTensor_.grad().zero_();
    }
  }
}

double CudaCalcTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
  // current CUcontext: ctx42
  CURRENT_CUDA_CTX_PUSH(primaryContext_);
  // current CUcontext: ctxPrimary

  // copy data to torch tensors
  int numParticles = cu_.getNumAtoms();
  void* posData = getTensorPointer(cu_, posTensor_);
  void* boxData = getTensorPointer(cu_, boxTensor_);
  void* kernelArgsCopyInputs[] = {&posData,
                                  &boxData,
                                  &cu_.getPosq().getDevicePointer(),
                                  &cu_.getAtomIndexArray().getDevicePointer(),
                                  &numParticles,
                                  cu_.getPeriodicBoxVecXPointer(),
                                  cu_.getPeriodicBoxVecYPointer(),
                                  cu_.getPeriodicBoxVecZPointer()};

  // current CUcontext: ctxPrimary
  {
    ContextSelector sl(cu_);  // RAII
    // current CUcontext: ctxOpenMM
    cu_.executeKernel(copyInputsKernel_, kernelArgsCopyInputs, numParticles);
    SYNC_CUDA_CTX;
    // RAII ~dtor();
  }
  // current CUcontext: ctxPrimary

  py::gil_scoped_acquire gilAcquire;
  // convert torch tensors to python objects
  py::object posPy = py::cast(posTensor_, py::return_value_policy::reference);
  py::dict kwargsPy;
  if (callable_ptr_->hasKWArg("cell"))
    kwargsPy["cell"] = py::cast(boxTensor_, py::return_value_policy::reference);
  if (callable_ptr_->hasKWArg("includeForces"))
    kwargsPy["includeForces"] = py::cast(static_cast<bool>(includeForces));
  py::gil_scoped_release gilRelease;

  if (useGraphs_) {
    // record graph if not already done
    if (graphs_.find(includeForces) == graphs_.end()) {
      // cuda graph capture must occur in a non-default stream
      const auto stream = c10::cuda::getStreamFromPool(false, cu_.getDeviceIndex());
      c10::cuda::CUDAStreamGuard guard(stream);  // RAII;

      // warm up
      try {
        for (int i = 0; i < warmupSteps_; ++i)
          executeGraph(includeForces, callable_ptr_, posPy, kwargsPy, posTensor_, energyTensor_, gradsTensor_);
      } catch (std::exception& e) {
        throw OpenMMException(std::string("Failed to warmup the model: ") + e.what());
      }

      try {
        graphs_[includeForces].capture_begin();
        executeGraph(includeForces, callable_ptr_, posPy, kwargsPy, posTensor_, energyTensor_, gradsTensor_);
      } catch (std::exception& e) {
        graphs_[includeForces].capture_end();
        throw OpenMMException(std::string("Failed to capture the model into a CUDA graph: ") + e.what());
      }
      graphs_[includeForces].capture_end();
    }

    // use the same stream as the OpenMM context
    const auto openmmStream = cu_.getCurrentStream();
    const auto stream = c10::cuda::getStreamFromExternal(openmmStream, cu_.getDeviceIndex());
    c10::cuda::CUDAStream guard(stream);  // RAII
    graphs_[includeForces].replay();
  } else {
    executeGraph(includeForces, callable_ptr_, posPy, kwargsPy, posTensor_, energyTensor_, gradsTensor_);
  }

  // torch tenors to OpenMM
  if (includeForces) {
    int gradSign = 1;
    if (callable_ptr_->returnForce())
      gradSign = -1;
    void* gradsData = getTensorPointer(cu_, gradsTensor_);
    int paddedNumAtoms = cu_.getPaddedNumAtoms();
    void* kernelArgsAddForces[] = {&gradsData,
                                   &cu_.getForce().getDevicePointer(),
                                   &cu_.getAtomIndexArray().getDevicePointer(),
                                   &gradSign,
                                   &numParticles,
                                   &paddedNumAtoms};
    SYNC_CUDA_CTX;

    // current CUcontext: ctxPrimary
    ContextSelector sl(cu_);  // RAII
    // current CUcontext: ctxOpenMM
    cu_.executeKernel(addForcesKernel_, kernelArgsAddForces, numParticles);
    // If includeEnergy, Tensor.item() will synchronize the primary context.
    // If not, we can synchronize the primary context a few lines of code later.
    // RAII ~dtor();
    // current CUcontext: ctxPrimary
  }

  double energyValue = 0.0;
  if (includeEnergy) {
    energyValue = energyTensor_.item<double>();  // torch implicitly synchronizes the primary cuda context.
  } else {
    SYNC_CUDA_CTX;
  }

  // current CUcontext: ctxPrimary
  CURRENT_CUDA_CTX_POP(primaryContext_);
  // current CUcontext: ctx42
  return energyValue;
}
