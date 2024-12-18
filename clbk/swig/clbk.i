%{
#include "clbk/clbk.h"
%}

%include <std_string.i>

namespace clbk {
class Callable {
public:
  static const int RETURN_ENERGY;
  static const int RETURN_ENERGY_GRADIENT;
  static const int RETURN_ENERGY_FORCE;
  void add_kwarg(std::string kwarg);
  Callable(long cpython_id, int return_flag);
};
}
