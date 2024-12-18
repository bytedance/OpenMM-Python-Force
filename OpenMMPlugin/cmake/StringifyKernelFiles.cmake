FUNCTION(StringifyKernelFiles
# INPUT
  kernelsDirectory PROJECT_CUDA_SOURCE_CLASS
# OUPUT
  PROJECT_CUDA_FILE_DECLARATIONS PROJECT_CUDA_FILE_DEFINITIONS)

  FILE(GLOB _cuFiles "${kernelsDirectory}/*.cu")

  SET(_fDeclarations "")
  SET(_fDefinitions "")
  FOREACH(f ${_cuFiles})
    # h
    GET_FILENAME_COMPONENT(fStem ${f} NAME_WE) # abc/def/gh.ext => gh
    SET(_fDeclarations "${_fDeclarations}\nstatic const std::string ${fStem};")

    # cpp
    # Read the file contents.
    FILE(READ ${f} fContent NEWLINE_CONSUME)
    # Write as C++11 raw string literals.
    SET(_fDefinitions "${_fDefinitions}\nconst std::string ${PROJECT_CUDA_SOURCE_CLASS}::${fStem} = R\"(\n${fContent})\";")
  ENDFOREACH()

  # output
  SET(${PROJECT_CUDA_FILE_DECLARATIONS} "${_fDeclarations}" PARENT_SCOPE)
  SET(${PROJECT_CUDA_FILE_DEFINITIONS} "${_fDefinitions}" PARENT_SCOPE)
ENDFUNCTION()
