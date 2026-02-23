# Common Tests - Test independent of all platforms
file(GLOB TEST_SRC
  *.cc
)

# chipStar/SPIRV: Bessel functions j0, j1, jn use __muldc3/__divdc3 which emit i128;
# hipspv-link does not support i128 (InvalidBitWidth).
if(HIP_PLATFORM MATCHES "spirv")
  list(REMOVE_ITEM TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/j0.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/j1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/jn.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/y0.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/y1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/yn.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/norm3d.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rnorm3d.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/norm4d.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rnorm4d.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cyl_bessel_i0.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cyl_bessel_i1.cc)
endif()

if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
    hip_add_exe_to_target(NAME DoublePrecisionMathFunctionsTests
        TEST_SRC ${TEST_SRC}
        TEST_TARGET_NAME build_tests)
elseif(HIP_PLATFORM MATCHES "nvidia")
    hip_add_exe_to_target(NAME DoublePrecisionMathFunctionsTests
        TEST_SRC ${TEST_SRC}
        TEST_TARGET_NAME build_tests
        COMPILE_OPTIONS -Wno-deprecated-declarations)
elseif(HIP_PLATFORM MATCHES "spirv")
    hip_add_exe_to_target(NAME DoublePrecisionMathFunctionsTests
        TEST_SRC ${TEST_SRC}
        TEST_TARGET_NAME build_tests
        COMPILE_OPTIONS -Wno-deprecated-declarations)
else()
    message(FATAL_ERROR "HIP_PLATFORM must be either amd, nvidia, or spirv")
endif()