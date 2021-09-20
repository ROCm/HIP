if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
    hipStreamCreate.cc
    hipStreamGetFlags.cc
    hipStreamGetPriority.cc
    hipMultiStream.cc
    hipStreamACb_MultiThread.cc
    hipStreamAddCallback.cc
    hipStreamCreateWithFlags.cc
    hipStreamCreateWithPriority.cc
    hipStreamWithCUMask.cc
    hipStreamGetCUMask.cc
    hipAPIStreamDisable.cc
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
)
endif()

# Create shared lib of all tests
add_library(StreamTest SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests StreamTest)