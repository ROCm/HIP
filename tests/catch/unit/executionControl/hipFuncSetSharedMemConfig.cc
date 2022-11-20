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

namespace {
constexpr std::array<hipSharedMemConfig, 3> kSharedMemConfigs{
    hipSharedMemBankSizeDefault, hipSharedMemBankSizeFourByte, hipSharedMemBankSizeEightByte};
}  // anonymous namespace

TEST_CASE("Unit_hipFuncSetSharedMemConfig_Positive_Basic") {
  const auto shared_mem_config =
      GENERATE(from_range(begin(kSharedMemConfigs), end(kSharedMemConfigs)));

  HIP_CHECK(hipFuncSetSharedMemConfig(reinterpret_cast<void*>(kernel), shared_mem_config));

  kernel<<<1, 1>>>();
  HIP_CHECK(hipDeviceSynchronize());
}

TEST_CASE("Unit_hipFuncSetSharedMemConfig_Negative_Parameters") {
  SECTION("func == nullptr") {
    HIP_CHECK_ERROR(hipFuncSetSharedMemConfig(nullptr, hipSharedMemBankSizeDefault),
                    hipErrorInvalidDeviceFunction);
  }
  SECTION("invalid shared mem config") {
    HIP_CHECK_ERROR(hipFuncSetSharedMemConfig(reinterpret_cast<void*>(kernel),
                                              static_cast<hipSharedMemConfig>(-1)),
                    hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipFuncSetSharedMemConfig_Negative_Not_Supported") {
  HIP_CHECK_ERROR(
      hipFuncSetSharedMemConfig(reinterpret_cast<void*>(kernel), hipSharedMemBankSizeDefault),
      hipErrorNotSupported);
}