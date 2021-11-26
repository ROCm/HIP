# AMD Tests
set(TEST_SRC
    saxpy.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME RTC
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  hip_add_exe_to_target(NAME RTC
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
endif()
