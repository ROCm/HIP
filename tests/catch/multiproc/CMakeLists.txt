# AMD Tests
set(LINUX_TEST_SRC
    hipMallocConcurrency.cc
    childMalloc.cc
)

if(UNIX)
    # Create shared lib of all tests
    add_library(MultiProc SHARED EXCLUDE_FROM_ALL ${LINUX_TEST_SRC})

    # Add dependency on build_tests to build it on this custom target
    add_dependencies(build_tests MultiProc)
endif()
