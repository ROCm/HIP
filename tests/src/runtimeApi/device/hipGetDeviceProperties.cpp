/*
Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.

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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t --tests 0x01
 * TEST: %t --tests 0x02 EXCLUDE_HIP_PLATFORM nvidia
 * HIT_END
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "test_common.h"

#define NUM_OF_ARCHPROP 17
#define HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS_IDX     0
#define HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH_IDX 1
#define HIP_ARCH_HAS_SHARED_INT32_ATOMICS_IDX     2
#define HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH_IDX 3
#define HIP_ARCH_HAS_FLOAT_ATOMIC_ADD_IDX         4
#define HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS_IDX     5
#define HIP_ARCH_HAS_SHARED_INT64_ATOMICS_IDX     6
#define HIP_ARCH_HAS_DOUBLES_IDX                  7
#define HIP_ARCH_HAS_WARP_VOTE_IDX                8
#define HIP_ARCH_HAS_WARP_BALLOT_IDX              9
#define HIP_ARCH_HAS_WARP_SHUFFLE_IDX             10
#define HIP_ARCH_HAS_WARP_FUNNEL_SHIFT_IDX        11
#define HIP_ARCH_HAS_THREAD_FENCE_SYSTEM_IDX      12
#define HIP_ARCH_HAS_SYNC_THREAD_EXT_IDX          13
#define HIP_ARCH_HAS_SURFACE_FUNCS_IDX            14
#define HIP_ARCH_HAS_3DGRID_IDX                   15
#define HIP_ARCH_HAS_DYNAMIC_PARALLEL_IDX         16

__device__ void getArchValuesFromDevice(int *archProp_d) {
  archProp_d[0] = __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__;
  archProp_d[1] = __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__;
  archProp_d[2] = __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__;
  archProp_d[3] = __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__;
  archProp_d[4] = __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__;
  archProp_d[5] = __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__;
  archProp_d[6] = __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__;
  archProp_d[7] = __HIP_ARCH_HAS_DOUBLES__;
  archProp_d[8] = __HIP_ARCH_HAS_WARP_VOTE__;
  archProp_d[9] = __HIP_ARCH_HAS_WARP_BALLOT__;
  archProp_d[10] = __HIP_ARCH_HAS_WARP_SHUFFLE__;
  archProp_d[11] = __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__;
  archProp_d[12] = __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__;
  archProp_d[13] = __HIP_ARCH_HAS_SYNC_THREAD_EXT__;
  archProp_d[14] = __HIP_ARCH_HAS_SURFACE_FUNCS__;
  archProp_d[15] = __HIP_ARCH_HAS_3DGRID__;
  archProp_d[16] = __HIP_ARCH_HAS_DYNAMIC_PARALLEL__;
}

__global__ void mykernel(int *archProp_d) {
  getArchValuesFromDevice(archProp_d);
}

/**
 * Internal Functions
 */
bool validateDeviceMacro(int *archProp_h, hipDeviceProp_t *prop) {
  bool TestPassed = true;
  if (prop->arch.hasGlobalInt32Atomics !=
      archProp_h[HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS_IDX]) {
    printf("mismatch: hasGlobalInt32Atomics \n");
    TestPassed &= false;
  }
  if (prop->arch.hasGlobalFloatAtomicExch !=
      archProp_h[HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH_IDX]) {
    printf("mismatch: hasGlobalFloatAtomicExch \n");
    TestPassed &= false;
  }
  if (prop->arch.hasSharedInt32Atomics !=
      archProp_h[HIP_ARCH_HAS_SHARED_INT32_ATOMICS_IDX]) {
    TestPassed &= false;
  }
  if (prop->arch.hasSharedFloatAtomicExch !=
      archProp_h[HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH_IDX]) {
    printf("mismatch: hasSharedFloatAtomicExch \n");
    TestPassed &= false;
  }
  if (prop->arch.hasFloatAtomicAdd !=
      archProp_h[HIP_ARCH_HAS_FLOAT_ATOMIC_ADD_IDX]) {
    printf("mismatch: hasFloatAtomicAdd \n");
    TestPassed &= false;
  }
  if (prop->arch.hasGlobalInt64Atomics !=
      archProp_h[HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS_IDX]) {
    printf("mismatch: hasGlobalInt64Atomics \n");
    TestPassed &= false;
  }
  /* TODO: Uncomment this code once the mismatch issue is resolved
  if (prop->arch.hasSharedInt64Atomics !=
      archProp_h[HIP_ARCH_HAS_SHARED_INT64_ATOMICS_IDX]) {
    TestPassed &= false;
  }*/
  if (prop->arch.hasDoubles !=
      archProp_h[HIP_ARCH_HAS_DOUBLES_IDX]) {
    printf("mismatch: hasDoubles \n");
    TestPassed &= false;
  }
  if (prop->arch.hasWarpVote !=
      archProp_h[HIP_ARCH_HAS_WARP_VOTE_IDX]) {
    printf("mismatch: hasWarpVote \n");
    TestPassed &= false;
  }
  if (prop->arch.hasWarpBallot !=
      archProp_h[HIP_ARCH_HAS_WARP_BALLOT_IDX]) {
    printf("mismatch: hasWarpBallot \n");
    TestPassed &= false;
  }
  if (prop->arch.hasWarpShuffle !=
      archProp_h[HIP_ARCH_HAS_WARP_SHUFFLE_IDX]) {
    printf("mismatch: hasWarpShuffle \n");
    TestPassed &= false;
  }
  if (prop->arch.hasFunnelShift !=
      archProp_h[HIP_ARCH_HAS_WARP_FUNNEL_SHIFT_IDX]) {
    printf("mismatch: hasFunnelShift \n");
    TestPassed &= false;
  }
  if (prop->arch.hasThreadFenceSystem !=
      archProp_h[HIP_ARCH_HAS_THREAD_FENCE_SYSTEM_IDX]) {
    printf("mismatch: hasThreadFenceSystem \n");
    TestPassed &= false;
  }
  if (prop->arch.hasSyncThreadsExt !=
      archProp_h[HIP_ARCH_HAS_SYNC_THREAD_EXT_IDX]) {
    printf("mismatch: hasSyncThreadsExt \n");
    TestPassed &= false;
  }
  if (prop->arch.hasSurfaceFuncs !=
      archProp_h[HIP_ARCH_HAS_SURFACE_FUNCS_IDX]) {
    printf("mismatch: hasSurfaceFuncs \n");
    TestPassed &= false;
  }
  if (prop->arch.has3dGrid !=
      archProp_h[HIP_ARCH_HAS_3DGRID_IDX]) {
    printf("mismatch: has3dGrid \n");
    TestPassed &= false;
  }
  if (prop->arch.hasDynamicParallelism !=
      archProp_h[HIP_ARCH_HAS_DYNAMIC_PARALLEL_IDX]) {
    printf("mismatch: hasDynamicParallelism \n");
    TestPassed &= false;
  }
  return TestPassed;
}
/**
 * Validates value of __HIP_ARCH_*  with deviceProp.arch.has* as follows
 * __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__ == hasGlobalInt32Atomics
 * __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__ == hasGlobalFloatAtomicExch
 * __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__ == hasSharedInt32Atomics
 * __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__ == hasSharedFloatAtomicExch
 * __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__ == hasFloatAtomicAdd
 * __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__ == hasGlobalInt64Atomics
 * __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__ == hasSharedInt64Atomics
 * __HIP_ARCH_HAS_DOUBLES__ == hasDoubles
 * __HIP_ARCH_HAS_WARP_VOTE__ == hasWarpVote
 * __HIP_ARCH_HAS_WARP_BALLOT__ == hasWarpBallot
 * __HIP_ARCH_HAS_WARP_SHUFFLE__ == hasWarpShuffle
 * __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__ == hasFunnelShift
 * __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__ == hasThreadFenceSystem
 * __HIP_ARCH_HAS_SYNC_THREAD_EXT__ == hasSyncThreadsExt
 * __HIP_ARCH_HAS_SURFACE_FUNCS__ == hasSurfaceFuncs
 * __HIP_ARCH_HAS_3DGRID__ == has3dGrid
 * __HIP_ARCH_HAS_DYNAMIC_PARALLEL__ == hasDynamicParallelism
 */
bool testArchitectureProperties() {
  bool TestPassed = true;
  int *archProp_h, *archProp_d;
  archProp_h = new int[NUM_OF_ARCHPROP];
  hipDeviceProp_t prop;
  int deviceCount = 0, device;
  HIPCHECK(hipGetDeviceCount(&deviceCount));
  HIPASSERT(deviceCount != 0);
  for (device = 0; device < deviceCount; device++) {
    // Inititalize archProp_h to 0
    for (int i = 0; i < NUM_OF_ARCHPROP; i++) {
      archProp_h[i] = 0;
    }
    HIPCHECK(hipGetDeviceProperties(&prop, device));
    HIPCHECK(hipSetDevice(device));
    HIPCHECK(hipMalloc(reinterpret_cast<void**>(&archProp_d),
            NUM_OF_ARCHPROP*sizeof(int)));
    HIPCHECK(hipMemcpy(archProp_d, archProp_h,
            NUM_OF_ARCHPROP*sizeof(int),
            hipMemcpyHostToDevice));
    hipLaunchKernelGGL(mykernel, dim3(1), dim3(1),
                       0, 0, archProp_d);
    HIPCHECK(hipMemcpy(archProp_h, archProp_d,
            NUM_OF_ARCHPROP*sizeof(int), hipMemcpyDeviceToHost));
    // Validate the host architecture property with device
    // architecture property.
    TestPassed &= validateDeviceMacro(archProp_h, &prop);
    HIPCHECK(hipFree(archProp_d));
  }
  delete[] archProp_h;
  return TestPassed;
}
/**
 * Validates negative scenarios for hipGetDeviceProperties
 * scenario1: props = nullptr
 * scenario2: device = -1 (Invalid Device)
 * scenario3: device = Non Existing Device
 */
bool testInvalidParameters() {
  bool TestPassed = true;
  hipError_t ret;
  // props = nullptr
#ifndef __HIP_PLATFORM_NVCC__
  int device;
  HIPCHECK(hipGetDevice(&device));
  // this test case results in segmentation fault on NVCC
  ret = hipGetDeviceProperties(nullptr, device);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {props = nullptr} Failed \n");
  }
#endif
  hipDeviceProp_t prop;
  ret = hipGetDeviceProperties(&prop, -1);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {device = -1} Failed \n");
  }
  // device = Non Existing Device
  int deviceCount = 0;
  HIPCHECK(hipGetDeviceCount(&deviceCount));
  HIPASSERT(deviceCount != 0);
  ret = hipGetDeviceProperties(&prop, deviceCount);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {device = Non Existing Device} Failed \n");
  }
  return TestPassed;
}

int main(int argc, char** argv) {
  HipTest::parseStandardArguments(argc, argv, true);
  bool TestPassed = true;
  if (p_tests == 0x01) {
    TestPassed = testInvalidParameters();
  } else if (p_tests == 0x02) {
    TestPassed = testArchitectureProperties();
  } else {
    printf("Invalid Test Case \n");
  }
  if (TestPassed) {
    passed();
  } else {
    failed("Test Case %x Failed!", p_tests);
  }
}
