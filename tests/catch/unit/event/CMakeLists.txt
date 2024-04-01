# Common Tests - Test independent of all platforms
set(TEST_SRC
    Unit_hipEvent_Negative.cc
    Unit_hipEvent.cc
    Unit_hipEventElapsedTime.cc
    Unit_hipEventRecord.cc
    Unit_hipEventIpc.cc
    hipEventDestroy.cc
    hipEventCreate.cc
    hipEventCreateWithFlags.cc
    hipEventSynchronize.cc
    Unit_hipEventMGpuMThreads.cc
)

# The test used wait mechanism and doesnt play well with all arch of nvidia
if(HIP_PLATFORM MATCHES "amd")
  set(AMD_SRC
      Unit_hipEventQuery.cc
  )
  set(TEST_SRC ${TEST_SRC} ${AMD_SRC})
endif()

hip_add_exe_to_target(NAME EventTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
