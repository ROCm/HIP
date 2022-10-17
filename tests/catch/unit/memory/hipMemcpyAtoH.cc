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

#include "array_memcpy_tests_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include <resource_guards.hh>

TEST_CASE("Unit_hipMemcpyAtoH_Default") {
  using namespace std::placeholders;

  const auto width = GENERATE(512, 1024, 2048);
  const auto allocation_size = width * sizeof(int);

  MemcpyAtoHShell<false, int>(std::bind(hipMemcpyAtoH, _1, _2, 0, allocation_size), width);
}

TEST_CASE("Unit_hipMemcpyAtoH_Synchronization_Behavior") {
  using namespace std::placeholders;

  const auto width = GENERATE(512, 1024, 2048);
  const auto height = 0;
  const auto allocation_size = width * sizeof(int);

  MemcpyAtoHPageableSyncBehavior(std::bind(hipMemcpyAtoH, _1, _2, 0, allocation_size), width, height, true);
  MemcpyAtoHPinnedSyncBehavior(std::bind(hipMemcpyAtoH, _1, _2, 0, allocation_size), width, height, true);
}

/*
This testcase verifies size 0 check of hipMemcpyAtoH API
Excluded the testcase for amd,as there is already a bug raised
SWDEV-274683
*/
#if HT_NVIDIA
TEST_CASE("Unit_hipMemcpyAtoH_ZeroCount") {
  const auto width = 1024;
  const auto height = 0;
  const auto allocation_size = width * sizeof(int);

  const unsigned int flag = hipArrayDefault;

  ArrayAllocGuard2D<int> array_alloc(width, height, flag);
  LinearAllocGuard<uint8_t> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);

  int fill_value = 42;
  std::fill_n(host_alloc.host_ptr(), width, fill_value);
  HIP_CHECK(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.host_ptr(), sizeof(int)*width, sizeof(int)*width, 1, hipMemcpyHostToDevice));
  fill_value = 41;
  std::fill_n(host_alloc.host_ptr(), width, fill_value);
  HIP_CHECK(hipMemcpyAtoH(host_alloc.ptr(), array_alloc.ptr(), 0, 0));

  ArrayFindIfNot(host_alloc.host_ptr(), static_cast<uint8_t>(fill_value), width);
}
#endif

TEST_CASE("Unit_hipMemcpyAtoH_Negative_Parameters") {
  using namespace std::placeholders;

  const auto width = 1024;
  const auto height = 0;
  const auto allocation_size = width * sizeof(int);

  const unsigned int flag = hipArrayDefault;

  ArrayAllocGuard2D<int> array_alloc(width, height, flag);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);

  SECTION("dst == nullptr") {
    HIP_CHECK_ERROR(hipMemcpyAtoH(nullptr, array_alloc.ptr(), 0, allocation_size), hipErrorInvalidValue);
  }
  SECTION("src == nullptr") {
    HIP_CHECK_ERROR(hipMemcpyAtoH(host_alloc.ptr(), nullptr, 0, allocation_size), hipErrorInvalidValue);
  }
  SECTION("Offset is greater than allocated size") {
    HIP_CHECK_ERROR(hipMemcpyAtoH(host_alloc.ptr(), array_alloc.ptr(), allocation_size + 10, allocation_size), hipErrorInvalidValue);
  }
  SECTION("Count is greater than allocated size") {
    HIP_CHECK_ERROR(hipMemcpyAtoH(host_alloc.ptr(), array_alloc.ptr(), 0, allocation_size + 10), hipErrorInvalidValue);
  }
  SECTION("2D array is allocated") {
    const auto width_2d = 32;
    const auto height_2d = width_2d;
    const auto allocation_size_2d = width_2d * height_2d * sizeof(int);

    ArrayAllocGuard2D<int> array_alloc_2d(width_2d, height_2d, flag);
    LinearAllocGuard<int> host_alloc_2d(LinearAllocs::hipHostMalloc, allocation_size_2d);
    HIP_CHECK_ERROR(hipMemcpyAtoH(host_alloc_2d.ptr(), array_alloc_2d.ptr(), 0, allocation_size_2d), hipErrorInvalidValue);
  }
}
