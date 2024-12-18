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
