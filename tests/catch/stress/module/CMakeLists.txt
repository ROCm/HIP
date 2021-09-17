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

add_custom_target(kernels.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${HIP_COMMON_DIR}/tests/catch/stress/module/kernels.cc -o ${HIP_PATH}/catch/hipTestMain/kernels.code -I${HIP_PATH}/include/ -I${HIP_COMMON_DIR}/tests/catch/include)

# Create shared lib of all tests
add_library(module SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_stress_test module kernels.code)
