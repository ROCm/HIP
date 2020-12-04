###############################################################################
# Tests.cmake
###############################################################################

# Add tests
include_directories(${HIP_SRC_PATH}/tests/src)
hit_add_directory_recursive(${HIP_CTEST_CONFIG_DEFAULT} ${HIP_SRC_PATH}/tests/src "directed_tests")

# Add unit tests
include_directories(${HIP_SRC_PATH}/tests/unit)
hit_add_directory_recursive(${HIP_CTEST_CONFIG_DEFAULT} ${HIP_SRC_PATH}/tests/unit "unit_tests")

# Add performance tests
include_directories(${HIP_SRC_PATH}/tests/performance)
hit_add_directory_recursive(${HIP_CTEST_CONFIG_PERFORMANCE} ${HIP_SRC_PATH}/tests/performance "performance_tests")

# Add top-level tests to build_tests
add_custom_target(build_tests DEPENDS directed_tests unit_tests)

# Add top-level tests to build performance_tests.
# To build performance tests, just run "make build_perf"
add_custom_target(build_perf DEPENDS performance_tests)

# Add custom target: perf.
# To run performance tests, just run "make perf"
add_custom_target(perf COMMAND "${CMAKE_CTEST_COMMAND}" -C "${HIP_CTEST_CONFIG_PERFORMANCE}" -R "performance_tests/" --verbose)

# Add custom target: check
add_custom_target(check COMMAND "${CMAKE_COMMAND}" --build . --target test DEPENDS build_tests)

# vim: ts=4:sw=4:expandtab:smartindent