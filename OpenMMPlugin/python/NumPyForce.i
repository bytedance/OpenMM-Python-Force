%{
#include "NumPyForce.h"
%}

namespace OpenMM {
class NumPyForce: public Force {
public:
  NumPyForce(const clbk::Callable& callable, const std::map<std::string, std::string>& properties = {});

  const clbk::Callable& getCallable() const;
  bool usesPeriodicBoundaryConditions() const override;

  void setProperty(const std::string& name, const std::string& value);
  const std::map<std::string, std::string>& getProperties() const;

  %extend {
    static OpenMM::NumPyForce& cast(OpenMM::Force& force) {
      return dynamic_cast<OpenMM::NumPyForce&>(force);
    }

    static bool isinstance(OpenMM::Force& force) {
      return (dynamic_cast<OpenMM::NumPyForce*>(&force) != nullptr);
    }
  }
};
}
