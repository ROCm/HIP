if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
    hipStreamCreate.cc
    hipStreamGetFlags.cc
    hipStreamGetPriority.cc
    hipMultiStream.cc
    hipStreamAddCallback.cc
    hipStreamCreateWithFlags.cc
    hipStreamCreateWithPriority.cc
    hipStreamSynchronize.cc
    hipStreamGetCUMask.cc
    hipAPIStreamDisable.cc
    hipStreamQuery.cc
)

#skipped in windows - duplicate HipTest::vector_square sym (compiler issue)
if(UNIX)
    set(TEST_SRC ${TEST_SRC}
                 hipStreamWithCUMask.cc
                 hipStreamACb_MultiThread.cc)
endif()

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
    hipStreamSynchronize.cc
    hipStreamQuery.cc
)
endif()

hip_add_exe_to_target(NAME StreamTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
