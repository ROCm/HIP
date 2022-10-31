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

#include "execution_control_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

TEST_CASE("Unit_hipLaunchCooperativeKernel_Positive_Basic") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  SECTION("Cooperative kernel with no arguments") {
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(coop_kernel), dim3{2, 2, 1},
                                         dim3{1, 1, 1}, nullptr, 0, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Kernel with arguments using kernelParams") {
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, sizeof(*result_dev.ptr())));

    int* result_ptr = result_dev.ptr();
    void* kernel_args[1] = {&result_ptr};
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_42), dim3{1, 1, 1},
                                         dim3{1, 1, 1}, kernel_args, 0, nullptr));

    int result = 0;
    HIP_CHECK(hipMemcpy(&result, result_dev.ptr(), sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 42);
  }
}

TEST_CASE("Unit_hipLaunchCooperativeKernel_Positive_Parameters") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  SECTION("blockDim.x == maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0);
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                         dim3{x, 1, 1}, nullptr, 0, nullptr));
  }

  SECTION("blockDim.y == maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0);
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                         dim3{y, 1, 1}, nullptr, 0, nullptr));
  }

  SECTION("blockDim.z == maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0);
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                         dim3{z, 1, 1}, nullptr, 0, nullptr));
  }
}

TEST_CASE("Unit_hipLaunchCooperativeKernel_Negative_Parameters") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  SECTION("f == nullptr") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(static_cast<void*>(nullptr), dim3{1, 1, 1},
                                               dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidDeviceFunction);
  }

  SECTION("gridDim.x == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{0, 1, 1},
                                               dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("gridDim.y == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 0, 1},
                                               dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("gridDim.z == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 0},
                                               dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.x == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                               dim3{0, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.y == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                               dim3{1, 0, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.z == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                               dim3{1, 1, 0}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.x > maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                               dim3{x, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.y > maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                               dim3{1, y, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.z > maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                               dim3{1, 1, z}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.x * blockDim.y * blockDim.z > maxThreadsPerBlock") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0);
    const unsigned int dim = std::ceil(std::cbrt(max));
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                               dim3{dim, dim, dim}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION(
      "gridDim.x * gridDim.y * gridDim.z > maxActiveBlocksPerMultiprocessor * "
      "multiProcessorCount") {
    int max_blocks;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                                                           reinterpret_cast<void*>(kernel), 1, 0));
    const unsigned int multiproc_count =
        GetDeviceAttribute(hipDeviceAttributeMultiprocessorCount, 0);
    const unsigned int dim = std::ceil(std::cbrt(max_blocks * multiproc_count));
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{dim, dim, dim},
                                               dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorCooperativeLaunchTooLarge);
  }

  SECTION("sharedMemBytes > maxSharedMemoryPerBlock") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                               dim3{1, 1, 1}, nullptr, max, nullptr),
                    hipErrorCooperativeLaunchTooLarge);
  }

  SECTION("Invalid stream") {
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                               dim3{1, 1, 1}, nullptr, 0, stream),
                    hipErrorContextIsDestroyed);
  }
}