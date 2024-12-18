%module CallbackPyForce

%include "factory.i"
%import(module = "openmm") "OpenMMForce.i"

/*
 * Convert C++ exceptions to Python exceptions.
 */
%exception {
  try {
    $action
  } catch (std::exception& e) {
    PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
    return nullptr;
  }
}

%include <std_string.i>
%include <std_map.i>
namespace std {
%template(property_map) map<string, string>;
}
%include "swig/clbk.i"
%include "NumPyForce.i"
#if COMPILE_TORCH_FORCE
%include "TorchForce.i"
#endif

%constant char* __version__ = "0.1.0";
