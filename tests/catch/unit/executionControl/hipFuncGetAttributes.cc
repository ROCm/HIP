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

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>

constexpr size_t kConstSizeBytes = 128;
__constant__ char const_data[kConstSizeBytes];

__global__ void attribute_test_kernel() {}

TEST_CASE("Unit_hipFuncGetAttributes_Positive_Basic") {
  hipFuncAttributes attr;
  HIP_CHECK(hipFuncGetAttributes(&attr, reinterpret_cast<void*>(attribute_test_kernel)));

  SECTION("binaryVersion") {
#if HT_NVIDIA
    const auto major = GetDeviceAttribute(hipDeviceAttributeComputeCapabilityMajor, 0);
    const auto minor = GetDeviceAttribute(hipDeviceAttributeComputeCapabilityMinor, 0);
    REQUIRE(attr.binaryVersion == major * 10 + minor);
#elif HT_AMD
    REQUIRE(attr.binaryVersion > 0);
#endif
  }

  SECTION("cacheModeCA") { REQUIRE((attr.cacheModeCA == 0 || attr.cacheModeCA == 1)); }

  SECTION("constSizeBytes") { REQUIRE(attr.constSizeBytes == kConstSizeBytes); }

  SECTION("maxThreadsPerBlock") {
    REQUIRE(attr.maxThreadsPerBlock == GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0));
  }

  SECTION("numRegs") { REQUIRE(attr.numRegs >= 0); }

  SECTION("ptxVersion") { REQUIRE(attr.ptxVersion > 0); }

  SECTION("sharedSizeBytes") {
    REQUIRE(attr.sharedSizeBytes <=
            GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
  }
}

TEST_CASE("Unit_hipFuncGetAttributes_Negative_Parameters") {
  SECTION("attr == nullptr") {
    HIP_CHECK_ERROR(hipFuncGetAttributes(nullptr, reinterpret_cast<void*>(attribute_test_kernel)),
                    hipErrorInvalidValue);
  }
  SECTION("func == nullptr") {
    hipFuncAttributes attr;
    HIP_CHECK_ERROR(hipFuncGetAttributes(&attr, nullptr), hipErrorInvalidDeviceFunction);
  }
}