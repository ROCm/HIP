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

#pragma once

#include "hip_module_common.hh"

#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

inline ModuleGuard InitModule() {
  HIP_CHECK(hipFree(nullptr));
  return ModuleGuard::LoadModule("launch_kernel_module.code");
}

inline ModuleGuard mg{InitModule()};

using ExtModuleLaunchKernelSig = hipError_t(hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                            uint32_t, uint32_t, size_t, hipStream_t, void**, void**,
                                            hipEvent_t, hipEvent_t, uint32_t);

template <ExtModuleLaunchKernelSig* func> void ModuleLaunchKernelPositiveBasic() {
  SECTION("Kernel with no arguments") {
    hipFunction_t f = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(func(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Kernel with arguments using kernelParams") {
    hipFunction_t f = GetKernel(mg.module(), "Kernel42");
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, sizeof(*result_dev.ptr())));
    int* result_ptr = result_dev.ptr();
    void* kernel_args[1] = {&result_ptr};
    HIP_CHECK(func(f, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args, nullptr, nullptr, nullptr, 0u));
    int result = 0;
    HIP_CHECK(hipMemcpy(&result, result_dev.ptr(), sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 42);
  }

  SECTION("Kernel with arguments using extra") {
    hipFunction_t f = GetKernel(mg.module(), "Kernel42");
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, sizeof(*result_dev.ptr())));
    int* result_ptr = result_dev.ptr();
    size_t size = sizeof(result_ptr);
    // clang-format off
    void *extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &result_ptr, 
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, 
        HIP_LAUNCH_PARAM_END
    };
    // clang-format on
    HIP_CHECK(func(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, extra, nullptr, nullptr, 0u));
    int result = 0;
    HIP_CHECK(hipMemcpy(&result, result_dev.ptr(), sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 42);
  }
}

template <ExtModuleLaunchKernelSig* func> void ModuleLaunchKernelPositiveParameters() {
  const auto LaunchNOPKernel = [=](unsigned int gridDimX, unsigned int gridDimY,
                                   unsigned int gridDimZ, unsigned int blockDimX,
                                   unsigned int blockDimY, unsigned int blockDimZ) {
    hipFunction_t f = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, nullptr,
                   nullptr, nullptr, nullptr, nullptr, 0u));
    HIP_CHECK(hipDeviceSynchronize());
  };

  SECTION("gridDimX == maxGridDimX") {
    const unsigned int x = GetDeviceAttribute(0, hipDeviceAttributeMaxGridDimX);
    LaunchNOPKernel(x, 1, 1, 1, 1, 1);
  }

  SECTION("gridDimY == maxGridDimY") {
    const unsigned int y = GetDeviceAttribute(0, hipDeviceAttributeMaxGridDimY);
    LaunchNOPKernel(1, y, 1, 1, 1, 1);
  }

  SECTION("gridDimZ == maxGridDimZ") {
    const unsigned int z = GetDeviceAttribute(0, hipDeviceAttributeMaxGridDimZ);
    LaunchNOPKernel(1, 1, z, 1, 1, 1);
  }

  SECTION("blockDimX == maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(0, hipDeviceAttributeMaxBlockDimX);
    LaunchNOPKernel(1, 1, 1, x, 1, 1);
  }

  SECTION("blockDimY == maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(0, hipDeviceAttributeMaxBlockDimY);
    LaunchNOPKernel(1, 1, 1, 1, y, 1);
  }

  SECTION("blockDimZ == maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(0, hipDeviceAttributeMaxBlockDimZ);
    LaunchNOPKernel(1, 1, 1, 1, 1, z);
  }
}

template <ExtModuleLaunchKernelSig* func> void ModuleLaunchKernelNegativeParameters() {
  hipFunction_t f = GetKernel(mg.module(), "NOPKernel");

// Disabled on AMD due to defect
#if HT_NVIDIA
  SECTION("f == nullptr") {
    HIP_CHECK_ERROR(
        func(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
        hipErrorInvalidResourceHandle);
  }
#endif

  SECTION("gridDimX == 0") {
    HIP_CHECK_ERROR(func(f, 0, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("gridDimY == 0") {
    HIP_CHECK_ERROR(func(f, 1, 0, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("gridDimZ == 0") {
    HIP_CHECK_ERROR(func(f, 1, 1, 0, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("blockDimX == 0") {
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 0, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("blockDimY == 0") {
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 0, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("blockDimZ == 0") {
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

// Disabled on AMD due to defect
#if HT_NVIDIA
  SECTION("gridDimX > maxGridDimX") {
    const unsigned int x = GetDeviceAttribute(0, hipDeviceAttributeMaxGridDimX) + 1u;
    HIP_CHECK_ERROR(func(f, x, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("gridDimY > maxGridDimY") {
    const unsigned int y = GetDeviceAttribute(0, hipDeviceAttributeMaxGridDimY) + 1u;
    HIP_CHECK_ERROR(func(f, 1, y, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("gridDimZ > maxGridDimZ") {
    const unsigned int z = GetDeviceAttribute(0, hipDeviceAttributeMaxGridDimZ) + 1u;
    HIP_CHECK_ERROR(func(f, 1, 1, z, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }
#endif

// Disabled on AMD due to defect
#if HT_NVIDIA
  SECTION("blockDimX > maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(0, hipDeviceAttributeMaxBlockDimX) + 1u;
    HIP_CHECK_ERROR(func(f, 1, 1, 1, x, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("blockDimY > maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(0, hipDeviceAttributeMaxBlockDimY) + 1u;
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, y, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("blockDimZ > maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(0, hipDeviceAttributeMaxBlockDimZ) + 1u;
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, z, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }
#endif

// Disabled on AMD due to defect
#if HT_NVIDIA
  SECTION("blockDimX * blockDimY * blockDimZ > MaxThreadsPerBlock") {
    const unsigned int max = GetDeviceAttribute(0, hipDeviceAttributeMaxThreadsPerBlock);
    const unsigned int dim = std::ceil(std::cbrt(max)) + 1;
    HIP_CHECK_ERROR(
        func(f, 1, 1, 1, dim, dim, dim, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
        hipErrorInvalidValue);
  }
#endif

// Disabled on AMD due to defect
#if HT_NVIDIA
  SECTION("sharedMemBytes > max shared memory per block") {
    const unsigned int max = GetDeviceAttribute(0, hipDeviceAttributeMaxSharedMemoryPerBlock) + 1u;
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, 1, max, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }
#endif

// Disabled on AMD due to defect
#if HT_NVIDIA
  SECTION("Invalid stream") {
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, 0, 0, stream, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorContextIsDestroyed);
  }
#endif

  SECTION("Passing kernel_args and extra simultaneously") {
    hipFunction_t f = GetKernel(mg.module(), "Kernel42");
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    int* result_ptr = result_dev.ptr();
    size_t size = sizeof(result_ptr);
    void* kernel_args[1] = {&result_ptr};
    // clang-format off
    void *extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &result_ptr, 
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, 
        HIP_LAUNCH_PARAM_END
    };
    // clang-format on
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args, extra, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

// Disabled on AMD due to defect
#if HT_NVIDIA
  SECTION("Invalid extra") {
    hipFunction_t f = GetKernel(mg.module(), "Kernel42");
    void* extra[0] = {};
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, extra, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }
#endif
}