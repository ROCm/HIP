# Common Tests - Test independent of all platforms
if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
    memset.cc
    malloc.cc
    hipMemcpy2DToArray.cc
    hipMemcpy2DToArrayAsync.cc
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
    hipMemcpyAllApiNegative.cc
    hipMemcpy_MultiThread.cc
    hipHostRegister.cc
    hipMemPtrGetInfo.cc
    hipPointerGetAttributes.cc
    hipHostGetFlags.cc
    hipMallocManaged_MultiScenario.cc
    hipMemsetInvalidPtr.cc
    hipMemset.cc
    hipMemsetAsyncMultiThread.cc
    hipMemset3D.cc
    hipMemset2D.cc
    hipHostMallocTests.cc
    hipMemset3DFunctional.cc
    hipMemset3DNegative.cc
    hipMemset3DRegressMultiThread.cc
    hipMallocManagedFlagsTst.cc
    hipMemPrefetchAsyncExtTsts.cc
    hipMemAdviseMmap.cc
)
else()
set(TEST_SRC
    memset.cc
    malloc.cc
    hipMemcpy2DToArray.cc
    hipMemcpy2DToArrayAsync.cc
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
    hipMemcpyAllApiNegative.cc
    hipMemcpy_MultiThread.cc
    hipHostRegister.cc
    hipHostGetFlags.cc
    hipMallocManaged_MultiScenario.cc
    hipMemsetInvalidPtr.cc
    hipMemset.cc
    hipMemsetAsyncMultiThread.cc
    hipMemset3D.cc
    hipMemset2D.cc
    hipHostMallocTests.cc
    hipMemset3DFunctional.cc
    hipMemset3DNegative.cc
    hipMemset3DRegressMultiThread.cc
    hipMallocManagedFlagsTst.cc
    hipMemPrefetchAsyncExtTsts.cc
    hipMemAdviseMmap.cc
)
endif()

#skipped in windows -duplicate HipTest::vectorADD sym (compiler issue)
if(UNIX)
    set(TEST_SRC ${TEST_SRC}
                 hipMemcpyPeer.cc
                 hipMemcpyPeerAsync.cc
                 hipMemcpyWithStream.cc
                 hipMemcpyWithStreamMultiThread.cc
                 hipMemsetAsyncAndKernel.cc
                 hipMemset2DAsyncMultiThreadAndKernel.cc
                 hipMallocManaged.cc
                 hipMallocConcurrency.cc
                 hipMemcpyDtoD.cc
                 hipMemcpyDtoDAsync.cc
                 hipMemoryAllocateCoherent.cc
                 hipHostMalloc.cc
                 hipMemcpy.cc
                 hipMemcpyAsync.cc)
endif()

# Create shared lib of all tests
add_library(MemoryTest SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests MemoryTest)
