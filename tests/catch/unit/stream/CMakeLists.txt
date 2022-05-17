if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
    hipStreamCreate.cc
    hipStreamGetFlags.cc
    hipStreamGetPriority.cc
    hipMultiStream.cc
    hipStreamAddCallback.cc
    hipStreamCreateWithFlags.cc
    hipStreamCreateWithPriority.cc
    hipStreamGetCUMask.cc
    hipAPIStreamDisable.cc
    streamCommon.cc
    hipStreamWithCUMask.cc
    hipStreamACb_MultiThread.cc
    )
else()
set(TEST_SRC
    hipStreamCreate.cc
    hipStreamGetFlags.cc
    hipStreamGetPriority.cc
    hipMultiStream.cc
    hipStreamACb_MultiThread.cc
    hipStreamAddCallback.cc
    hipStreamCreateWithFlags.cc
    hipStreamCreateWithPriority.cc
    hipAPIStreamDisable.cc
    streamCommon.cc
    )
endif()

hip_add_exe_to_target(NAME StreamTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
