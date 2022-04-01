# Copyright (c) 2021 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
    hipMallocPitch.cc
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
    hipMemCoherencyTst.cc
    hipMallocManaged.cc
    hipMemRangeGetAttribute.cc
    hipHmmOvrSubscriptionTst.cc
    hipMemcpyFromSymbol.cc
    hipMemcpyFromSymbolAsync.cc
    hipMemcpyToSymbol.cc
    hipMemcpyToSymbolAsync.cc
    hipPtrGetAttribute.cc
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
    hipMallocPitch.cc
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
    hipMallocManaged.cc
    hipMemRangeGetAttribute.cc
    hipHmmOvrSubscriptionTst.cc
    hipMemcpyFromSymbol.cc
    hipMemcpyFromSymbolAsync.cc
    hipMemcpyToSymbol.cc
    hipMemcpyToSymbolAsync.cc
    hipPtrGetAttribute.cc
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

hip_add_exe_to_target(NAME MemoryTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
