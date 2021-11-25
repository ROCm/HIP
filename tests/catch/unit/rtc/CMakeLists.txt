# AMD Tests
set(TEST_SRC
    saxpy.cc
)

# AMD only tests
set(AMD_TEST_SRC
    customOptions.cc
)

if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
endif()

# Create shared lib of all tests
add_library(RTC SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests RTC)
