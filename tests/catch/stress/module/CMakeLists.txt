# Common Tests - Test independent of all platforms
if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
    hipExtModuleLaunchKernel_CornerTest.cc
    hipModuleLaunchKernel_CornerTests.cc
)
else()
set(TEST_SRC
    hipModuleLaunchKernel_CornerTests.cc
)
endif()

add_custom_target(kernels.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${HIP_COMMON_DIR}/tests/catch/stress/module/kernels.cc -o  ${CMAKE_CURRENT_BINARY_DIR}/../../stress/module/kernels.code -I${HIP_PATH}/include/ -I${HIP_COMMON_DIR}/tests/catch/include)

# Create shared lib of all tests

# Add dependency on build_tests to build it on this custom target
hip_add_exe_to_target(NAME module
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME stress_test)
add_dependencies(stress_test kernels.code)
