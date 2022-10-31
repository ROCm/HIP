/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip_module_launch_kernel_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

static hipError_t hipModuleLaunchKernelWrapper(hipFunction_t f, uint32_t gridX, uint32_t gridY,
                                               uint32_t gridZ, uint32_t blockX, uint32_t blockY,
                                               uint32_t blockZ, size_t sharedMemBytes,
                                               hipStream_t hStream, void** kernelParams,
                                               void** extra, hipEvent_t, hipEvent_t, uint32_t) {
  return hipModuleLaunchKernel(f, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
                               hStream, kernelParams, extra);
}

TEST_CASE("Unit_hipModuleLaunchKernel_Positive_Basic") {
  HIP_CHECK(hipFree(nullptr));
  ModuleLaunchKernelPositiveBasic<hipModuleLaunchKernelWrapper>();
}

TEST_CASE("Unit_hipModuleLaunchKernel_Positive_Parameters") {
  HIP_CHECK(hipFree(nullptr));
  ModuleLaunchKernelPositiveParameters<hipModuleLaunchKernelWrapper>();
}

TEST_CASE("Unit_hipModuleLaunchKernel_Negative_Parameters") {
  HIP_CHECK(hipFree(nullptr));
  ModuleLaunchKernelNegativeParameters<hipModuleLaunchKernelWrapper>();
}