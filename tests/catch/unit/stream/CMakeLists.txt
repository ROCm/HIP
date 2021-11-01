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
)
endif()

# Create shared lib of all tests
add_library(StreamTest SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests StreamTest)