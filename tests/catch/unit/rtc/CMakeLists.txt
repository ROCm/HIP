# Common Tests - Test independent of all platforms
set(TEST_SRC
    saxpy.cc
    warpsize.cc
)

# AMD only tests
set(AMD_TEST_SRC
    customOptions.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME RTC
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
  hip_add_exe_to_target(NAME RTC
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc)
endif()
