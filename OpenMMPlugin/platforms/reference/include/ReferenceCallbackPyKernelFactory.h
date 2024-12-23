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
#include <openmm/KernelFactory.h>

namespace OpenMM {
class ReferenceCallbackPyKernelFactory: public KernelFactory {
public:
  KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const override;
};
}
