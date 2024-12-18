CMAKE_MINIMUM_REQUIRED(VERSION 3.27)

SET(srcLocal
  src/ReferenceCallbackPyKernelFactory.cpp
  src/ReferenceNumPyKernels.cpp
)
IF(COMPILE_TORCH_FORCE)
  LIST(APPEND srcLocal
    src/ReferenceTorchKernels.cpp
  )
ENDIF()
SET(libLocal libReference)
ADD_LIBRARY("${libLocal}"
SHARED
  ${srcLocal}
)
SET_TARGET_PROPERTIES("${libLocal}"
PROPERTIES
  OUTPUT_NAME "${LIB_NAME}_Reference"
)
INSTALL(TARGETS "${libLocal}"
LIBRARY
  DESTINATION lib/plugins
)
TARGET_LINK_LIBRARIES("${libLocal}"
PUBLIC
  libAPI
)
TARGET_INCLUDE_DIRECTORIES("${libLocal}"
PRIVATE
  "${CMAKE_SOURCE_DIR}/platforms/common/include"
)

INCLUDE("${CMAKE_SOURCE_DIR}/cmake/CommonDirsAndLibs.cmake")
