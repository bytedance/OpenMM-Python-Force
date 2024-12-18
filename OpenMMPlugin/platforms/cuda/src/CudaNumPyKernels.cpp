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

#include <map>
#include <string>

#include <cuda_runtime_api.h>
#include <openmm/OpenMMException.h>
#include <openmm/common/ContextSelector.h>

#include "CommonCudaMacros.h"
#include "CudaCallbackPyKernelSources.h"
#include "CudaNumPyKernels.h"
#include "NumPy2DArrayFromBlob.h"

using std::vector;
using namespace OpenMM;
namespace py = pybind11;

CudaCalcNumPyForceKernel::CudaCalcNumPyForceKernel(std::string name, const Platform& platform, CudaContext& cu):
  CalcNumPyForceKernel(name, platform),
  cu_(cu) {
  CHECK_RESULT(cuDevicePrimaryCtxRetain(&primaryContext_, cu_.getDevice()), "Failed to retain the primary context");
}

CudaCalcNumPyForceKernel::~CudaCalcNumPyForceKernel() {
  cuDevicePrimaryCtxRelease(cu_.getDevice());
}

void CudaCalcNumPyForceKernel::initialize(const System& system, const NumPyForce& force) {
  callable_ptr_ = &force.getCallable();

  // current CUcontext: ctx42
  // switch to an openmm cuda context
  ContextSelector selector(cu_);  // RAII
  // current CUcontext: ctxOpenMM
  std::map<std::string, std::string> defines;
  CUmodule program = cu_.createModule(CudaCallbackPyKernelSources::kernels, defines);
  addForcesKernel_ = cu_.getKernel(program, "addForces");

  numParticles_ = cu_.getNumAtoms();
  paddedNumAtoms_ = cu_.getPaddedNumAtoms();
  pos_.resize(numParticles_ * 3);
  cell_.resize(3 * 3);
  if (cu_.getUseDoublePrecision()) {
    gradsBuffer_.initialize(cu_, numParticles_ * 3, sizeof(double), "NumPy Gradients Buffer (double)");
  } else {
    gradsBuffer_.initialize(cu_, numParticles_ * 3, sizeof(float), "NumPy Gradients Buffer (float)");
  }

  // RAII ~dtor();
  // current CUcontext: ctx42
}

double CudaCalcNumPyForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
  // current Cucontext: ctx42
  CURRENT_CUDA_CTX_PUSH(primaryContext_);
  // current CUcontext: ctxPrimary
  {
    ContextSelector sl(cu_);  // RAII
    // current CUcontext: ctxOpenMM
    const auto& atomIndexVec = cu_.getAtomIndex();
    if (cu_.getUseDoublePrecision()) {
      vector<double> posqBuffer(paddedNumAtoms_ * 4);
      cu_.getPosq().download(posqBuffer.data());
      for (int i = 0; i < numParticles_; ++i) {
        int index = atomIndexVec[i];
        pos_[3 * index + 0] = posqBuffer[4 * i + 0];
        pos_[3 * index + 1] = posqBuffer[4 * i + 1];
        pos_[3 * index + 2] = posqBuffer[4 * i + 2];
      }
    } else {
      vector<float> posqBuffer(paddedNumAtoms_ * 4);
      cu_.getPosq().download(posqBuffer.data());
      for (int i = 0; i < numParticles_; ++i) {
        int index = atomIndexVec[i];
        pos_[3 * index + 0] = posqBuffer[4 * i + 0];
        pos_[3 * index + 1] = posqBuffer[4 * i + 1];
        pos_[3 * index + 2] = posqBuffer[4 * i + 2];
      }
    }

    Vec3 periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ;
    cu_.getPeriodicBoxVectors(periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
    cell_[0] = periodicBoxVecX[0];
    cell_[1] = periodicBoxVecX[1];
    cell_[2] = periodicBoxVecX[2];
    cell_[3] = periodicBoxVecY[0];
    cell_[4] = periodicBoxVecY[1];
    cell_[5] = periodicBoxVecY[2];
    cell_[6] = periodicBoxVecZ[0];
    cell_[7] = periodicBoxVecZ[1];
    cell_[8] = periodicBoxVecZ[2];
    // RAII ~dtor();
  }
  // current CUcontext: ctxPrimary

  py::gil_scoped_acquire gilAcquire;
  auto posNp = numpy_2d_array_from_blob<double>(pos_.data(), numParticles_);
  py::dict kwargsPy;
  if (callable_ptr_->hasKWArg("cell")) {
    kwargsPy["cell"] = numpy_2d_array_from_blob<double>(cell_.data(), 3);
  }
  if (callable_ptr_->hasKWArg("includeForces")) {
    kwargsPy["includeForces"] = py::cast(static_cast<bool>(includeForces));
  }

  auto callPy = callable_ptr_->nonownerCast<py::object>();
  auto resultsPy = callPy(posNp, **kwargsPy);

  double energyNum;
  py::array gradsNpArray;
  if (callable_ptr_->returnGradient() or callable_ptr_->returnForce()) {
    py::tuple resultsTuple = resultsPy.cast<py::tuple>();
    energyNum = resultsTuple[0].cast<double>();
    gradsNpArray = resultsTuple[1].cast<py::array>();
  } else {
    energyNum = resultsPy.cast<double>();
  }

  if (includeForces) {
    int gradSign = 0;
    if (callable_ptr_->returnForce())
      gradSign = -1;
    else if (callable_ptr_->returnGradient())
      gradSign = 1;
    else
      throw OpenMMException(std::string("Callable must return force or gradient"));
    auto buffer = gradsNpArray.request();
    const double* gradsPtr = static_cast<const double*>(buffer.ptr);

    // current CUcontext: ctxPrimary
    ContextSelector sl(cu_);  // RAII
    // current CUcontext: ctxOpenMM
    if (cu_.getUseDoublePrecision()) {
      vector<double> gradsVec(numParticles_ * 3);
      for (int i = 0; i < numParticles_; ++i)
        for (int j = 0; j < 3; ++j)
          gradsVec[3 * i + j] = gradsPtr[3 * i + j];
      gradsBuffer_.upload(gradsVec);
    } else {
      vector<float> gradsVec(numParticles_ * 3);
      for (int i = 0; i < numParticles_; ++i)
        for (int j = 0; j < 3; ++j)
          gradsVec[3 * i + j] = gradsPtr[3 * i + j];
      gradsBuffer_.upload(gradsVec);
    }
    void* kernelArgsAddForces[] = {&gradsBuffer_.getDevicePointer(),
                                   &cu_.getForce().getDevicePointer(),
                                   &cu_.getAtomIndexArray().getDevicePointer(),
                                   &gradSign,
                                   &numParticles_,
                                   &paddedNumAtoms_};
    cu_.executeKernel(addForcesKernel_, kernelArgsAddForces, numParticles_);
    SYNC_CUDA_CTX;
    // RAII ~dtor();
    // current CUcontext: ctxPrimary
  }
  py::gil_scoped_release gilRelease;

  double energyValue = 0.0;
  if (includeEnergy)
    energyValue = energyNum;

  // current CUcontext: ctxPrimary
  CURRENT_CUDA_CTX_POP(primaryContext_);
  // current CUcontext: ctx42
  return energyValue;
}
