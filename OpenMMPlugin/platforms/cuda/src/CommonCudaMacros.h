/* -------------------------------------------------------------------------- *
 * Copyright (c) 2024--present ByteDance Ltd. and/or its affiliates           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- *
 * The file doc/CREDITS.txt lists serval external projects which served       *
 * as valuable guidance for this project. Although not all of these           *
 * projects are directly referenced in every source file, this source file    *
 * complies with all of their licenses.                                       *
 * -------------------------------------------------------------------------- */

#pragma once

// macro for checking the result of synchronization operation on CUDA
// copied from `openmm/platforms/cuda/src/CudaParallelKernels.cpp`

#define CHECK_RESULT(result, prefix)                                           \
  if (result != CUDA_SUCCESS) {                                                \
    std::stringstream m;                                                       \
    m << prefix << ": " << cu_.getErrorString(result) << " (" << result << ")" \
      << " at " << __FILE__ << ":" << __LINE__;                                \
    throw OpenMMException(m.str());                                            \
  }
#define CURRENT_CUDA_CTX_PUSH(PRI_CTX_) CHECK_RESULT(cuCtxPushCurrent(PRI_CTX_), "Failed to push the CUDA context")
#define CURRENT_CUDA_CTX_POP(PRI_CTX_)                                             \
  do {                                                                             \
    CUcontext ctx_scoped_;                                                         \
    CHECK_RESULT(cuCtxPopCurrent(&ctx_scoped_), "Failed to pop the CUDA context"); \
    auto check_scoped_ = CUDA_ERROR_UNKNOWN;                                       \
    if (ctx_scoped_ == PRI_CTX_)                                                   \
      check_scoped_ = CUDA_SUCCESS;                                                \
    CHECK_RESULT(check_scoped_, "PyTorch messed up the context stack");            \
  } while (0)
#define SYNC_CUDA_CTX CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context")
