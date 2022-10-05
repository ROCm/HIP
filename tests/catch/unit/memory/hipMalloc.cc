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

TEST_CASE("Unit_hipMalloc_Positive_Basic") {
  constexpr size_t page_size = 4096;
  void* ptr = nullptr;
  const auto alloc_size =
      GENERATE_COPY(10, page_size / 2, page_size, page_size * 3 / 2, page_size * 2);
  HIP_CHECK(hipMalloc(&ptr, alloc_size));
  CHECK(ptr != nullptr);
  CHECK(reinterpret_cast<intptr_t>(ptr) % 256 == 0);
  HIP_CHECK(hipFree(ptr));
}

TEST_CASE("Unit_hipMalloc_Positive_Zero_Size") {
  void* ptr = reinterpret_cast<void*>(0x1);
  HIP_CHECK(hipMalloc(&ptr, 0));
  REQUIRE(ptr == nullptr);
}

TEST_CASE("Unit_hipMalloc_Positive_Alignment") {
  void *ptr1 = nullptr, *ptr2 = nullptr;
  HIP_CHECK(hipMalloc(&ptr1, 1));
  HIP_CHECK(hipMalloc(&ptr2, 10));
  CHECK(reinterpret_cast<intptr_t>(ptr1) % 256 == 0);
  CHECK(reinterpret_cast<intptr_t>(ptr2) % 256 == 0);
  HIP_CHECK(hipFree(ptr1));
  HIP_CHECK(hipFree(ptr2));
}

TEST_CASE("Unit_hipMalloc_Negative_Parameters") {
  SECTION("ptr == nullptr") { HIP_CHECK_ERROR(hipMalloc(nullptr, 4096), hipErrorInvalidValue); }
  SECTION("size == max size_t") {
    void* ptr;
    HIP_CHECK_ERROR(hipMalloc(&ptr, std::numeric_limits<size_t>::max()), hipErrorOutOfMemory);
  }
}