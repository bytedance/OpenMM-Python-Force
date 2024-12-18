CMAKE_MINIMUM_REQUIRED(VERSION 3.27)

SET(WRAP_FILE SwigWrapper.cpp)
IF(COMPILE_TORCH_FORCE)
  ADD_CUSTOM_COMMAND(
  OUTPUT
    "${WRAP_FILE}"
  COMMAND
    "${SWIG_EXECUTABLE}"
      -python -c++
      -o "${WRAP_FILE}"
      -DCOMPILE_TORCH_FORCE=1
      "-I${CLBK_INCLUDE_DIR}"
      "-I${OPENMM_DIR}/include"
      "-I${TORCH_INSTALL_PREFIX}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/Module.i"
  DEPENDS
    OpenMMForce.i
    Module.i
    NumPyForce.i
    TorchForce.i
  )
ELSE()
  ADD_CUSTOM_COMMAND(
  OUTPUT
    "${WRAP_FILE}"
  COMMAND
    "${SWIG_EXECUTABLE}"
      -python -c++
      -o "${WRAP_FILE}"
      -DCOMPILE_TORCH_FORCE=0
      "-I${CLBK_INCLUDE_DIR}"
      "-I${OPENMM_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/Module.i"
  DEPENDS
    OpenMMForce.i
    Module.i
    NumPyForce.i
  )
ENDIF()

SET(libLocal libSwig)
ADD_LIBRARY("${libLocal}"
SHARED EXCLUDE_FROM_ALL
  "${CMAKE_CURRENT_BINARY_DIR}/${WRAP_FILE}"
)
SET_TARGET_PROPERTIES("${libLocal}"
PROPERTIES
  OUTPUT_NAME "_${MODULE_NAME}"
  PREFIX ""
)
IF(APPLE OR LINUX)
  SET_TARGET_PROPERTIES("${libLocal}"
  PROPERTIES
    SUFFIX ".so"
  )
ENDIF()
TARGET_LINK_LIBRARIES("${libLocal}"
PUBLIC
  libAPI
)

INCLUDE("${CMAKE_SOURCE_DIR}/cmake/CommonDirsAndLibs.cmake")
