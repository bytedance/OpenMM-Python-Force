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
#include <map>
#include <string>

#include <openmm/Force.h>

#include "Callable.h"

namespace OpenMM {
class OPENMM_EXPORT TorchForce: public Force {
private:
  std::map<std::string, std::string> properties_;
  clbk::Callable callable_;

protected:
  ForceImpl* createImpl() const override;

public:
  TorchForce(const clbk::Callable& callable, const std::map<std::string, std::string>& properties = {});

  const clbk::Callable& getCallable() const;
  bool usesPeriodicBoundaryConditions() const override;

  void setProperty(const std::string& name, const std::string& value);
  const std::map<std::string, std::string>& getProperties() const;
};
}
