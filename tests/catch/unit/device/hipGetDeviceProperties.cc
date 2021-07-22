/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>


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
static void validateDeviceMacro(int *archProp_h, hipDeviceProp_t *prop) {
  CHECK_FALSE(prop->arch.hasGlobalInt32Atomics !=
      archProp_h[HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS_IDX]);

  CHECK_FALSE(prop->arch.hasGlobalFloatAtomicExch !=
      archProp_h[HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH_IDX]);

  CHECK_FALSE(prop->arch.hasSharedInt32Atomics !=
      archProp_h[HIP_ARCH_HAS_SHARED_INT32_ATOMICS_IDX]);

  CHECK_FALSE(prop->arch.hasSharedFloatAtomicExch !=
      archProp_h[HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH_IDX]);

  CHECK_FALSE(prop->arch.hasFloatAtomicAdd !=
      archProp_h[HIP_ARCH_HAS_FLOAT_ATOMIC_ADD_IDX]);

  CHECK_FALSE(prop->arch.hasGlobalInt64Atomics !=
      archProp_h[HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS_IDX]);

  CHECK_FALSE(prop->arch.hasSharedInt64Atomics !=
      archProp_h[HIP_ARCH_HAS_SHARED_INT64_ATOMICS_IDX]);

  CHECK_FALSE(prop->arch.hasDoubles !=
      archProp_h[HIP_ARCH_HAS_DOUBLES_IDX]);

  CHECK_FALSE(prop->arch.hasWarpVote !=
      archProp_h[HIP_ARCH_HAS_WARP_VOTE_IDX]);

  CHECK_FALSE(prop->arch.hasWarpBallot !=
      archProp_h[HIP_ARCH_HAS_WARP_BALLOT_IDX]);

  CHECK_FALSE(prop->arch.hasWarpShuffle !=
      archProp_h[HIP_ARCH_HAS_WARP_SHUFFLE_IDX]);

  CHECK_FALSE(prop->arch.hasFunnelShift !=
      archProp_h[HIP_ARCH_HAS_WARP_FUNNEL_SHIFT_IDX]);

  CHECK_FALSE(prop->arch.hasThreadFenceSystem !=
      archProp_h[HIP_ARCH_HAS_THREAD_FENCE_SYSTEM_IDX]);

  CHECK_FALSE(prop->arch.hasSyncThreadsExt !=
      archProp_h[HIP_ARCH_HAS_SYNC_THREAD_EXT_IDX]);

  CHECK_FALSE(prop->arch.hasSurfaceFuncs !=
      archProp_h[HIP_ARCH_HAS_SURFACE_FUNCS_IDX]);

  CHECK_FALSE(prop->arch.has3dGrid !=
      archProp_h[HIP_ARCH_HAS_3DGRID_IDX]);

  CHECK_FALSE(prop->arch.hasDynamicParallelism !=
      archProp_h[HIP_ARCH_HAS_DYNAMIC_PARALLEL_IDX]);
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
#if HT_AMD
TEST_CASE("Unit_hipGetDeviceProperties_ArchPropertiesTst") {
  int *archProp_h, *archProp_d;
  archProp_h = new int[NUM_OF_ARCHPROP];
  hipDeviceProp_t prop;
  int deviceCount = 0, device;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount != 0);
  for (device = 0; device < deviceCount; device++) {
    // Inititalize archProp_h to 0
    for (int i = 0; i < NUM_OF_ARCHPROP; i++) {
      archProp_h[i] = 0;
    }
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&archProp_d),
            NUM_OF_ARCHPROP*sizeof(int)));
    HIP_CHECK(hipMemcpy(archProp_d, archProp_h,
            NUM_OF_ARCHPROP*sizeof(int),
            hipMemcpyHostToDevice));
    hipLaunchKernelGGL(mykernel, dim3(1), dim3(1),
                       0, 0, archProp_d);
    HIP_CHECK(hipMemcpy(archProp_h, archProp_d,
            NUM_OF_ARCHPROP*sizeof(int), hipMemcpyDeviceToHost));
    // Validate the host architecture property with device
    // architecture property.
    validateDeviceMacro(archProp_h, &prop);
    HIP_CHECK(hipFree(archProp_d));
  }
  delete[] archProp_h;
}
#endif
/**
 * Validates negative scenarios for hipGetDeviceProperties
 * scenario1: props = nullptr
 * scenario2: device = -1 (Invalid Device)
 * scenario3: device = Non Existing Device
 */
TEST_CASE("Unit_hipGetDeviceProperties_NegTst") {
  hipDeviceProp_t prop;

#if HT_AMD
  SECTION("props is nullptr") {
    int device;
    HIP_CHECK(hipGetDevice(&device));
    // this test case results in segmentation fault on NVCC
    REQUIRE_FALSE(hipSuccess == hipGetDeviceProperties(nullptr, device));
  }
#endif

  SECTION("device is -1") {
    REQUIRE_FALSE(hipSuccess == hipGetDeviceProperties(&prop, -1));
  }

  SECTION("device is -1") {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    REQUIRE(deviceCount != 0);
    REQUIRE_FALSE(hipSuccess == hipGetDeviceProperties(&prop, deviceCount));
  }
}
