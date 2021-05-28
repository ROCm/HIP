# Common Tests - Test independent of all platforms
set(TEST_SRC
    Stress_printf_ComplexKernels.cc
    Stress_printf_SimpleKernels.cc
)

# Create shared lib of all tests
add_library(printf SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_stress_test printf)
