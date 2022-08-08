if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
    hipStreamCreate.cc
    hipStreamGetFlags.cc
    hipStreamGetPriority.cc
    hipMultiStream.cc
    hipStreamAddCallback.cc
    hipStreamCreateWithFlags.cc
    hipStreamCreateWithPriority.cc
    hipStreamDestroy.cc
    hipStreamGetCUMask.cc
    hipAPIStreamDisable.cc
    streamCommon.cc
    hipStreamValue.cc
    hipStreamWithCUMask.cc
    hipStreamACb_MultiThread.cc
    hipStreamSynchronize.cc
    hipStreamQuery.cc
    hipStreamWaitEvent.cc
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
    hipStreamDestroy.cc
    hipAPIStreamDisable.cc
    # hipStreamAttachMemAsync.cc   # Disabling it on nvidia due to issue in function definition of hipStreamAttachMemAsync
                                   # Fixing would break ABI, to be re-enabled when the fix is made.
    streamCommon.cc
    hipStreamValue.cc
    hipStreamSynchronize.cc
    hipStreamQuery.cc
    )

    # set_source_files_properties(hipStreamAttachMemAsync.cc PROPERTIES COMPILE_FLAGS -std=c++17)
endif()

hip_add_exe_to_target(NAME StreamTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
