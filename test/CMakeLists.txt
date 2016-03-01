set(Python_ADDITIONAL_VERSIONS 2.7)
include(FindPythonInterp)
if( NOT PYTHONINTERP_FOUND )
  message(FATAL_ERROR
          "Unable to find Python interpreter, required for builds and testing\n\n"
          "Please install Python or specify the PYTHON_EXECUTABLE CMake variable.")
endif()

set(BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..  )

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  @ONLY)

add_lit_testsuite(check-hipify "Running HIPify regression tests"
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  DEPENDS hipify FileCheck lit
  )

add_custom_target(check)
add_dependencies(check check-hipify)
add_custom_target(test)
add_dependencies(test check-hipify)
set_target_properties(check PROPERTIES FOLDER "Tests")

