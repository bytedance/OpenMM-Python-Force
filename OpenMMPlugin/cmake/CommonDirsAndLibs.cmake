## link dirs
SET(linkdirsLocal
  "${OPENMM_DIR}/lib"
)
IF(COMPILE_TORCH_FORCE)
  LIST(APPEND linkdirsLocal
    "${TORCH_INSTALL_PREFIX}/lib"
  )
ENDIF()

## link libs
SET(linklibsLocal
  OpenMM
  Python3::Module
  Python3::NumPy
  pybind11::module
)
IF(COMPILE_TORCH_FORCE)
  LIST(APPEND linklibsLocal
    "${TORCH_LIBRARIES}"
    torch_python
  )
ENDIF()

## link
TARGET_LINK_DIRECTORIES("${libLocal}"
PUBLIC
  ${linkdirsLocal}
)
TARGET_LINK_LIBRARIES("${libLocal}"
PUBLIC
  ${linklibsLocal}
)

## include dirs
SET(isysdirsLocal
  "${OPENMM_DIR}/include"
)
IF(COMPILE_TORCH_FORCE)
  LIST(APPEND isysdirsLocal
    "${TORCH_INSTALL_PREFIX}/include"
  )
ENDIF()
TARGET_INCLUDE_DIRECTORIES("${libLocal}" SYSTEM
PRIVATE
  ${isysdirsLocal}
)
TARGET_INCLUDE_DIRECTORIES("${libLocal}"
PRIVATE
  "${CMAKE_SOURCE_DIR}/openmmapi/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
