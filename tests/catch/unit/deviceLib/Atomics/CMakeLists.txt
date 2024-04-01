# Common Tests - Test independent of all platforms
file(GLOB TEST_SRC
  *.cc
)

if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
    hip_add_exe_to_target(NAME DeviceAtomics
        TEST_SRC ${TEST_SRC}
        TEST_TARGET_NAME build_tests)
elseif(HIP_PLATFORM MATCHES "nvidia")
    hip_add_exe_to_target(NAME DeviceAtomics
        TEST_SRC ${TEST_SRC}
        TEST_TARGET_NAME build_tests
        COMPILE_OPTIONS --Wno-deprecated-declarations)
elseif(HIP_PLATFORM MATCHES "spirv")
    hip_add_exe_to_target(NAME DeviceAtomics
        TEST_SRC ${TEST_SRC}
        TEST_TARGET_NAME build_tests
        COMPILE_OPTIONS --Wno-deprecated-declarations)
endif()