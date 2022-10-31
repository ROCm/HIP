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
#include <hip/hip_ext.h>

TEST_CASE("Unit_hipExtModuleLaunchKernel_Positive_Basic") {
  ModuleLaunchKernelPositiveBasic<hipExtModuleLaunchKernel>();

  SECTION("Timed kernel launch with events") {
    hipEvent_t start_event = nullptr, stop_event = nullptr;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));
    const auto kernel = GetKernel(mg.module(), "Delay");
    int clock_rate = 0;
    HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeClockRate, 0));
    uint32_t interval = 100;
    uint32_t ticks_per_second = clock_rate;
    void* kernel_params[2] = {&interval, &ticks_per_second};
    HIP_CHECK(hipExtModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_params, nullptr,
                                       start_event, stop_event));
    HIP_CHECK(hipDeviceSynchronize());
    auto elapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed, start_event, stop_event));
    REQUIRE(static_cast<uint32_t>(elapsed) >= interval);
  }
}

TEST_CASE("Unit_hipExtModuleLaunchKernel_Positive_Parameters") {
  ModuleLaunchKernelPositiveParameters<hipExtModuleLaunchKernel>();

  SECTION("Pass only start event") {
    hipEvent_t start_event = nullptr;
    HIP_CHECK(hipEventCreate(&start_event));
    const auto kernel = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(hipExtModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr,
                                       start_event, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventQuery(start_event));
  }

  SECTION("Pass only stop event") {
    hipEvent_t stop_event = nullptr;
    HIP_CHECK(hipEventCreate(&stop_event));
    const auto kernel = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(hipExtModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr,
                                       nullptr, stop_event));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventQuery(stop_event));
  }
}

TEST_CASE("Unit_hipExtModuleLaunchKernel_Negative_Parameters") {
  ModuleLaunchKernelNegativeParameters<hipExtModuleLaunchKernel>();
}