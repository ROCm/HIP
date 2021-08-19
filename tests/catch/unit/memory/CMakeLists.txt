# Common Tests - Test independent of all platforms
if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
    memset.cc
    malloc.cc
    hipMemcpy2DToArray.cc
    hipMemcpy2DToArrayAsync.cc
    hipMemcpyPeer.cc
    hipMemcpyPeerAsync.cc
    hipMemcpy3D.cc
    hipMemcpy3DAsync.cc
    hipMemcpyParam2D.cc
    hipMemcpyParam2DAsync.cc
    hipMemcpy2D.cc
    hipMemcpy2DAsync.cc
    hipMemcpy2DFromArray.cc
    hipMemcpy2DFromArrayAsync.cc
    hipMemcpyAtoH.cc
    hipMemcpyHtoA.cc
    hipMemcpyDtoD.cc
    hipMemcpyDtoDAsync.cc
    hipMemcpyAsync.cc
    hipMemcpy.cc
    hipMemcpyWithStream.cc
    hipMemcpyAllApiNegative.cc
    hipMemcpyWithStreamMultiThread.cc
    hipMemcpy_MultiThread.cc
    hipHostMalloc.cc
    hipHostRegister.cc
    hipMemPtrGetInfo.cc
    hipPointerGetAttributes.cc
    hipHostGetFlags.cc
    hipMemoryAllocateCoherent.cc
    hipMallocManaged_MultiScenario.cc
    hipManagedKeyword.cc
)
else()
set(TEST_SRC
    memset.cc
    malloc.cc
    hipMemcpy2DToArray.cc
    hipMemcpy2DToArrayAsync.cc
    hipMemcpyPeer.cc
    hipMemcpyPeerAsync.cc
    hipMemcpy3D.cc
    hipMemcpy3DAsync.cc
    hipMemcpyParam2D.cc
    hipMemcpyParam2DAsync.cc
    hipMemcpy2D.cc
    hipMemcpy2DAsync.cc
    hipMemcpy2DFromArray.cc
    hipMemcpy2DFromArrayAsync.cc
    hipMemcpyAtoH.cc
    hipMemcpyHtoA.cc
    hipMemcpyDtoD.cc
    hipMemcpyDtoDAsync.cc
    hipMemcpyAsync.cc
    hipMemcpy.cc
    hipMemcpyWithStream.cc
    hipMemcpyAllApiNegative.cc
    hipMemcpyWithStreamMultiThread.cc
    hipMemcpy_MultiThread.cc
    hipHostMalloc.cc
    hipHostRegister.cc
    hipHostGetFlags.cc
    hipMemoryAllocateCoherent.cc
    hipMallocManaged_MultiScenario.cc
    hipManagedKeyword.cc
)
endif()
# Create shared lib of all tests
add_library(MemoryTest SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests MemoryTest)
