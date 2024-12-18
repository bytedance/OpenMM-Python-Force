%{
#include "TorchForce.h"
%}

namespace OpenMM {
class TorchForce: public Force {
public:
  TorchForce(const clbk::Callable& callable, const std::map<std::string, std::string>& properties = {});

  const clbk::Callable& getCallable() const;
  bool usesPeriodicBoundaryConditions() const override;

  void setProperty(const std::string& name, const std::string& value);
  const std::map<std::string, std::string>& getProperties() const;

  %extend {
    static OpenMM::TorchForce& cast(OpenMM::Force& force) {
      return dynamic_cast<OpenMM::TorchForce&>(force);
    }

    static bool isinstance(OpenMM::Force& force) {
      return (dynamic_cast<OpenMM::TorchForce*>(&force) != nullptr);
    }
  }
};
}
