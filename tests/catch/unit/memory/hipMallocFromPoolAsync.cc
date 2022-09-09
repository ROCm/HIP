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
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <hip_test_common.hh>

#include <limits>

constexpr hipMemPoolProps kPoolPropsForMalloc = {
  hipMemAllocationTypePinned,
  hipMemHandleTypeNone,
  {
    hipMemLocationTypeDevice,
    0
  },
  nullptr,
  {0}
};

TEST_CASE("Unit_hipMallocFromPoolAsync_negative") {
    HIP_CHECK(hipSetDevice(0));
    int mem_pool_support = 0;
    HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
    if (!mem_pool_support) {
        SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
        return;
    }

    hipMemPool_t mem_pool = nullptr;
    void *p = nullptr;
    size_t max_size = std::numeric_limits<size_t>::max();
    hipStream_t stream{nullptr};
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolPropsForMalloc));
    
    SECTION("dev_ptr is nullptr") {
        REQUIRE(hipMallocFromPoolAsync(nullptr, 100, mem_pool, stream) != hipSuccess);
    }

    SECTION("invalid mempool handle") {
        REQUIRE(hipMallocFromPoolAsync(static_cast<void**>(&p), 100, reinterpret_cast<hipMemPool_t>(-1), stream) != hipSuccess);
    }

    SECTION("invalid stream handle") {
        REQUIRE(hipMallocFromPoolAsync(static_cast<void**>(&p), 100, mem_pool, reinterpret_cast<hipStream_t>(-1)) != hipSuccess);
    }

    SECTION("out of memory") {
        REQUIRE(hipMallocFromPoolAsync(static_cast<void**>(&p), max_size, mem_pool, stream) != hipSuccess);
    }

    HIP_CHECK(hipStreamSynchronize(stream));
	HIP_CHECK(hipMemPoolDestroy(mem_pool));
    HIP_CHECK(hipStreamDestroy(stream));

}
