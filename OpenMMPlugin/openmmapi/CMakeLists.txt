CMAKE_MINIMUM_REQUIRED(VERSION 3.27)

SET(srcLocal
  src/Callable.cpp
  src/NumPyForce.cpp
  src/NumPyForceImpl.cpp
)
IF(COMPILE_TORCH_FORCE)
  LIST(APPEND srcLocal
    src/TorchForce.cpp
    src/TorchForceImpl.cpp
  )
ENDIF()
SET(libLocal libAPI)
ADD_LIBRARY("${libLocal}"
SHARED
  ${srcLocal}
)
SET_TARGET_PROPERTIES("${libLocal}"
PROPERTIES
  OUTPUT_NAME "${LIB_NAME}"
)
TARGET_INCLUDE_DIRECTORIES("${libLocal}"
PUBLIC
  "${CLBK_INCLUDE_DIR}"
)
INSTALL(TARGETS "${libLocal}"
LIBRARY
  DESTINATION lib
)

INCLUDE("${CMAKE_SOURCE_DIR}/cmake/CommonDirsAndLibs.cmake")
