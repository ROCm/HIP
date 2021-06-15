# Common Tests - Test independent of all platforms
set(TEST_SRC
    add.cc
)

# Create shared lib of all tests
add_library(ABMAddKernels SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests ABMAddKernels)
