# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipGetErrorName.cc
    hipGetErrorString.cc
    hipGetLastError.cc
    hipPeekAtLastError.cc
)

if (UNIX)
  set(__LINKER_LIBS__ pthread)
endif()

hip_add_exe_to_target(NAME ErrorHandlingTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
		      LINKER_LIBS ${__LINKER_LIBS__}
		      COMPILE_OPTIONS -std=c++14)
