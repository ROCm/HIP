# Common Tests - Test independent of all platforms
set(TEST_SRC
    memcpy.cc
    hipMemcpyMThreadMSize.cc
    hipMallocManagedStress.cc
)

# Create shared lib of all tests
add_library(memory SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_stress_test memory)
