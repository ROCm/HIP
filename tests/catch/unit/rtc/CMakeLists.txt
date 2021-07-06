# AMD Tests
set(AMD_TEST_SRC
    saxpy.cc
)

if(HIP_PLATFORM MATCHES "amd")
    # Create shared lib of all tests
    add_library(RTC SHARED EXCLUDE_FROM_ALL ${AMD_TEST_SRC})

    # Add dependency on build_tests to build it on this custom target
    add_dependencies(build_tests RTC)
endif()
