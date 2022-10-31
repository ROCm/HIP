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

TEST_CASE("Unit_hipLaunchCooperativeKernelMultiDevice_Positive_Basic") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  const auto device_count = HipTest::getDeviceCount();

  std::vector<hipLaunchParams> params_list(device_count);

  int device = 0;
  for (auto& params : params_list) {
    params.func = reinterpret_cast<void*>(coop_kernel);
    params.gridDim = dim3{1, 1, 1};
    params.blockDim = dim3{1, 1, 1};
    params.args = nullptr;
    params.sharedMem = 0;
    HIP_CHECK(hipSetDevice(device++));
    HIP_CHECK(hipStreamCreate(&params.stream));
  }

  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u));

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamSynchronize(params.stream));
  }

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamDestroy(params.stream));
  }
}

TEST_CASE("Unit_hipLaunchCooperativeKernelMultiDevice_Negative_Parameters") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  const auto device_count = HipTest::getDeviceCount();

  std::vector<hipLaunchParams> params_list(device_count);

  int device = 0;
  for (auto& params : params_list) {
    params.func = reinterpret_cast<void*>(coop_kernel);
    params.gridDim = dim3{1, 1, 1};
    params.blockDim = dim3{1, 1, 1};
    params.args = nullptr;
    params.sharedMem = 0;
    HIP_CHECK(hipSetDevice(device++));
    HIP_CHECK(hipStreamCreate(&params.stream));
  }

  SECTION("launchParamsList == nullptr") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernelMultiDevice(nullptr, device_count, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("numDevices == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernelMultiDevice(params_list.data(), 0, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("numDevices > device count") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernelMultiDevice(params_list.data(), device_count + 1, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("invalid flags") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 999),
                    hipErrorInvalidValue);
  }

  if (device_count > 1) {
    SECTION("launchParamsList.func doesn't match across all devices") {
      params_list[1].func = reinterpret_cast<void*>(kernel);
      HIP_CHECK_ERROR(hipLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
                      hipErrorInvalidValue);
    }

    SECTION("launchParamsList.gridDim doesn't match across all kernels") {
      params_list[1].gridDim = dim3{2, 2, 2};
      HIP_CHECK_ERROR(hipLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
                      hipErrorInvalidValue);
    }

    SECTION("launchParamsList.blockDim doesn't match across all kernels") {
      params_list[1].blockDim = dim3{2, 2, 2};
      HIP_CHECK_ERROR(hipLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
                      hipErrorInvalidValue);
    }

    SECTION("launchParamsList.sharedMem doesn't match across all kernels") {
      params_list[1].sharedMem = 1024;
      HIP_CHECK_ERROR(hipLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
                      hipErrorInvalidValue);
    }
  }

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamDestroy(params.stream));
  }
}

TEST_CASE("Unit_hipLaunchCooperativeKernelMultiDevice_Negative_MultiKernelSameDevice") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  HIP_CHECK(hipSetDevice(0));

  std::vector<hipLaunchParams> params_list(2);

  for (auto& params : params_list) {
    params.func = reinterpret_cast<void*>(coop_kernel);
    params.gridDim = dim3{1, 1, 1};
    params.blockDim = dim3{1, 1, 1};
    params.args = nullptr;
    params.sharedMem = 0;
    HIP_CHECK(hipStreamCreate(&params.stream));
  }

  HIP_CHECK_ERROR(hipLaunchCooperativeKernelMultiDevice(params_list.data(), 2, 0u),
                  hipErrorInvalidValue);

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamDestroy(params.stream));
  }
}