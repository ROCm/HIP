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
#include <utils.hh>

TEST_CASE("Unit_hipHostGetDevicePointer_Negative") {
  int* hPtr{nullptr};
  int* dPtr{nullptr};
  HIP_CHECK(hipHostMalloc(&hPtr, sizeof(int)));

  if (!DeviceAttributesSupport(0, hipDeviceAttributeCanMapHostMemory)) {
    HIP_CHECK_ERROR(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), hPtr, 0),
                    hipErrorNotSupported);
    return;
  }

  SECTION("Nullptr as device") {
    HIP_CHECK_ERROR(hipHostGetDevicePointer(nullptr, hPtr, 0), hipErrorInvalidValue);
  }

  SECTION("Nullptr as host") {
    int* dPtr{nullptr};
    HIP_CHECK_ERROR(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), nullptr, 0),
                    hipErrorInvalidValue);
  }

  SECTION("Non pinned memory as host") {
    int* hPtr = reinterpret_cast<int*>(malloc(sizeof(*hPtr)));
    HIP_CHECK_ERROR(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), hPtr, 0),
                    hipErrorInvalidValue);
    free(hPtr);
  }

  SECTION("Flags non-zero") {
    HIP_CHECK_ERROR(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), hPtr, 1),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipHostFree(hPtr));
}

template <typename T> __global__ void set(T* ptr, T val) { *ptr = val; }

TEST_CASE("Unit_hipHostGetDevicePointer_UseCase") {
  if(!DeviceAttributesSupport(0, hipDeviceAttributeCanMapHostMemory)) {
    HipTest::HIP_SKIP_TEST("Device does not support mapping host memory"); 
    return;
  }

  int* hPtr{nullptr};
  HIP_CHECK(hipHostMalloc(&hPtr, sizeof(int)));

  auto kernel = set<int>;
  constexpr int value = 10;

  SECTION("Set the value on device - Get device ptr") {
    int* dPtr{nullptr};
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), hPtr, 0));
    REQUIRE(dPtr != nullptr);

    kernel<<<1, 1>>>(dPtr, value);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(*hPtr == value);
  }

  SECTION("Set the value on device - by hipHostRegister") {
    int res{0};                                        // Stuff on stack
    HIP_CHECK(hipHostRegister(&res, sizeof(int), 0));  // Lets map stack memory :)

    int* dPtr{nullptr};
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), &res, 0))

    kernel<<<1, 1>>>(dPtr, value);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipHostUnregister(&res));

    REQUIRE(res == value);
  }

  HIP_CHECK(hipHostFree(hPtr));
}