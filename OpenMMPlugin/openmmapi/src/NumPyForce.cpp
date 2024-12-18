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

#include "NumPyForce.h"
#include "internal/NumPyForceImpl.h"

using namespace OpenMM;

ForceImpl* NumPyForce::createImpl() const {
  return new NumPyForceImpl(*this);
}

NumPyForce::NumPyForce(const clbk::Callable& callable, const std::map<std::string, std::string>& properties):
  callable_(callable) {
  std::map<std::string, std::string> defaultProperties;
  properties_ = defaultProperties;
  for (const auto& kv : properties) {
    if (defaultProperties.find(kv.first) == defaultProperties.end())
      throw OpenMMException("NumPyForce: Unknown property '" + kv.first + "'");
    properties_[kv.first] = kv.second;
  }
  this->setName("NumPyForce@CallbackPyForce");
}

const clbk::Callable& NumPyForce::getCallable() const {
  return callable_;
}

bool NumPyForce::usesPeriodicBoundaryConditions() const {
  return callable_.hasKWArg("cell");
}

void NumPyForce::setProperty(const std::string& name, const std::string& value) {
  if (properties_.find(name) == properties_.end())
    throw OpenMMException("TorchForce: Unknown property '" + name + "'");
  properties_[name] = value;
}

const std::map<std::string, std::string>& NumPyForce::getProperties() const {
  return properties_;
}
