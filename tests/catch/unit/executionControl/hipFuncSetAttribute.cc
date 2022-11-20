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

TEST_CASE("Unit_hipFuncSetAttribute_Positive_MaxDynamicSharedMemorySize") {
  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                hipFuncAttributeMaxDynamicSharedMemorySize, 1024));

  hipFuncAttributes attributes;
  HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<void*>(kernel)));

  REQUIRE(attributes.maxDynamicSharedSizeBytes == 1024);
}

TEST_CASE("Unit_hipFuncSetAttribute_Positive_PreferredSharedMemoryCarveout") {
  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                hipFuncAttributePreferredSharedMemoryCarveout, 50));

  hipFuncAttributes attributes;
  HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<void*>(kernel)));

  REQUIRE(attributes.preferredShmemCarveout == 50);
}

TEST_CASE("Unit_hipFuncSetAttribute_Positive_Parameters") {
  SECTION("hipFuncAttributeMaxDynamicSharedMemorySize == 0") {
    HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                  hipFuncAttributeMaxDynamicSharedMemorySize, 0));
  }

  SECTION(
      "hipFuncAttributeMaxDynamicSharedMemorySize == maxSharedMemoryPerBlock - sharedSizeBytes") {
    // The sum of this value and the function attribute sharedSizeBytes cannot exceed the device
    // attribute cudaDevAttrMaxSharedMemoryPerBlockOptin
    int max_shared;
    HIP_CHECK(hipDeviceGetAttribute(&max_shared, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));

    hipFuncAttributes attributes;
    HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<void*>(kernel)));

    HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  max_shared - attributes.sharedSizeBytes));
  }

  SECTION("hipFuncAttributePreferredSharedMemoryCarveout == 0") {
    HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                  hipFuncAttributePreferredSharedMemoryCarveout, 0));
  }

  SECTION("hipFuncAttributePreferredSharedMemoryCarveout == 100") {
    HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                  hipFuncAttributePreferredSharedMemoryCarveout, 100));
  }

  SECTION("hipFuncAttributePreferredSharedMemoryCarveout == -1 (default)") {
    HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                  hipFuncAttributePreferredSharedMemoryCarveout, -1));
  }
}

TEST_CASE("Unit_hipFuncSetAttribute_Negative_Parameters") {
  SECTION("func == nullptr") {
    HIP_CHECK_ERROR(hipFuncSetAttribute(nullptr, hipFuncAttributePreferredSharedMemoryCarveout, 50),
                    hipErrorInvalidDeviceFunction);
  }

  SECTION("invalid attribute") {
    HIP_CHECK_ERROR(
        hipFuncSetAttribute(reinterpret_cast<void*>(kernel), static_cast<hipFuncAttribute>(-1), 50),
        hipErrorInvalidValue);
  }

  SECTION("hipFuncAttributeMaxDynamicSharedMemorySize < 0") {
    HIP_CHECK_ERROR(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                        hipFuncAttributeMaxDynamicSharedMemorySize, -1),
                    hipErrorInvalidValue);
  }

  SECTION(
      "hipFuncAttributeMaxDynamicSharedMemorySize > maxSharedMemoryPerBlock - sharedSizeBytes") {
    // The sum of this value and the function attribute sharedSizeBytes cannot exceed the device
    // attribute cudaDevAttrMaxSharedMemoryPerBlockOptin
    int max_shared;
    HIP_CHECK(hipDeviceGetAttribute(&max_shared, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));

    hipFuncAttributes attributes;
    HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<void*>(kernel)));

    HIP_CHECK_ERROR(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                        hipFuncAttributeMaxDynamicSharedMemorySize,
                                        max_shared - attributes.sharedSizeBytes + 1),
                    hipErrorInvalidValue);
  }

  SECTION("hipFuncAttributePreferredSharedMemoryCarveout < -1") {
    HIP_CHECK_ERROR(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                        hipFuncAttributePreferredSharedMemoryCarveout, -2),
                    hipErrorInvalidValue);
  }

  SECTION("hipFuncAttributePreferredSharedMemoryCarveout > 100") {
    HIP_CHECK_ERROR(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                        hipFuncAttributePreferredSharedMemoryCarveout, 101),
                    hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipFuncSetAttribute_Positive_MaxDynamicSharedMemorySize_Not_Supported") {
  hipFuncAttributes old_attributes;
  HIP_CHECK(hipFuncGetAttributes(&old_attributes, reinterpret_cast<void*>(kernel)));

  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                hipFuncAttributeMaxDynamicSharedMemorySize, 1024));

  hipFuncAttributes new_attributes;
  HIP_CHECK(hipFuncGetAttributes(&new_attributes, reinterpret_cast<void*>(kernel)));

  REQUIRE(old_attributes.maxDynamicSharedSizeBytes == new_attributes.maxDynamicSharedSizeBytes);
}

TEST_CASE("Unit_hipFuncSetAttribute_Positive_PreferredSharedMemoryCarveout_Not_Supported") {
  hipFuncAttributes old_attributes;
  HIP_CHECK(hipFuncGetAttributes(&old_attributes, reinterpret_cast<void*>(kernel)));

  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<void*>(kernel),
                                hipFuncAttributePreferredSharedMemoryCarveout, 50));

  hipFuncAttributes new_attributes;
  HIP_CHECK(hipFuncGetAttributes(&new_attributes, reinterpret_cast<void*>(kernel)));

  REQUIRE(old_attributes.preferredShmemCarveout == new_attributes.preferredShmemCarveout);
}