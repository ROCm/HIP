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

#include "hip_module_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>

static hipModule_t GetModule() {
  HIP_CHECK(hipFree(nullptr));
  static const auto mg = ModuleGuard::LoadModule("get_function_module.code");
  return mg.module();
}

TEST_CASE("Unit_hipFuncGetAttribute_Positive_Basic") {
  hipFunction_t kernel = GetKernel(GetModule(), "GlobalKernel");

  int value;

  SECTION("binaryVersion") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, kernel));
#if HT_NVIDIA
    const auto major = GetDeviceAttribute(0, hipDeviceAttributeComputeCapabilityMajor);
    const auto minor = GetDeviceAttribute(0, hipDeviceAttributeComputeCapabilityMinor);
    REQUIRE(value == major * 10 + minor);
#elif HT_AMD
    REQUIRE(value > 0);
#endif
  }

  SECTION("cacheModeCA") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA, kernel));
    REQUIRE((value == 0 || value == 1));
  }

  SECTION("maxThreadsPerBlock") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel));
    REQUIRE(value == GetDeviceAttribute(0, hipDeviceAttributeMaxThreadsPerBlock));
  }

  SECTION("numRegs") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_NUM_REGS, kernel));
    REQUIRE(value >= 0);
  }

  SECTION("ptxVersion") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_PTX_VERSION, kernel));
    REQUIRE(value > 0);
  }

  SECTION("sharedSizeBytes") {
    HIP_CHECK(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel));
    REQUIRE(value <= GetDeviceAttribute(0, hipDeviceAttributeMaxSharedMemoryPerBlock));
  }
}

TEST_CASE("Unit_hipFuncGetAttribute_Negative_Parameters") {
  hipFunction_t kernel = GetKernel(GetModule(), "GlobalKernel");

  int value;

  SECTION("value == nullptr") {
    HIP_CHECK_ERROR(hipFuncGetAttribute(nullptr, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, kernel),
                    hipErrorInvalidValue);
  }

  SECTION("invalid attribute") {
    HIP_CHECK_ERROR(hipFuncGetAttribute(&value, static_cast<hipFunction_attribute>(-1), kernel),
                    hipErrorInvalidValue);
  }

  SECTION("hfunc == nullptr") {
    HIP_CHECK_ERROR(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, nullptr),
                    hipErrorInvalidResourceHandle);
  }
}