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

/* Test Case Description:
   1) This testcase verifies the  basic scenario - supported on
     all devices
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <chrono>

constexpr hipMemPoolProps kPoolProps = {
  hipMemAllocationTypePinned,
  hipMemHandleTypeNone,
  {
    hipMemLocationTypeDevice,
    0
  },
  nullptr,
  {0}
};

/*
   This testcase verifies HIP Mem Pool API basic scenario - supported on all devices
 */

TEST_CASE("Unit_hipMemPoolApi_Basic") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  int numElements = 64 * 1024 * 1024;
  float *A = nullptr, *B = nullptr;

  hipMemPool_t mem_pool = nullptr;
  int device = 0;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, device));
  HIP_CHECK(hipDeviceSetMemPool(device, mem_pool));
  HIP_CHECK(hipDeviceGetMemPool(&mem_pool, device));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMallocAsync(&A, numElements * sizeof(float), stream));
  INFO("hipMallocAsync result: " << A);

  HIP_CHECK(hipFreeAsync(A, stream));
  // Reset the default memory pool usage for the following tests
  hipMemPoolAttr attr = hipMemPoolAttrUsedMemHigh;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value64));

  size_t min_bytes_to_hold = 1024 * 1024;
  HIP_CHECK(hipMemPoolTrimTo(mem_pool, min_bytes_to_hold));

  attr = hipMemPoolReuseFollowEventDependencies;
  int value = 0;

  HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));

  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value));

  hipMemAccessDesc desc_list = {};
  int count = 1;
  HIP_CHECK(hipMemPoolSetAccess(mem_pool, &desc_list, count));

  hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
  hipMemLocation location = {};
  HIP_CHECK(hipMemPoolGetAccess(&flags, mem_pool, &location));

  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));
  HIP_CHECK(hipMallocFromPoolAsync(&B, numElements * sizeof(float), mem_pool, stream));
  HIP_CHECK(hipMemPoolDestroy(mem_pool));

  HIP_CHECK(hipStreamDestroy(stream));
}

constexpr auto wait_ms = 500;

__global__ void kernel500ms(float* hostRes, int clkRate) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  hostRes[tid] = tid + 1;
  __threadfence_system();
  // expecting that the data is getting flushed to host here!
  uint64_t start = clock64()/clkRate, cur;
  if (clkRate > 1) {
    do { cur = clock64()/clkRate-start;}while (cur < wait_ms);
  } else {
    do { cur = clock64()/start;}while (cur < wait_ms);
  }
}

TEST_CASE("Unit_hipMemPoolApi_BasicAlloc") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  hipMemPool_t mem_pool;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));

  float* B, *C;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  size_t numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float), mem_pool, stream));

  numElements = 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C), numElements * sizeof(float), mem_pool, stream));

  int blocks = 1024;
  int clkRate;
  hipMemPoolAttr attr;
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));

  kernel500ms<<<32, blocks, 0, stream>>>(B, clkRate);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream));

  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_before_sync));

  HIP_CHECK(hipStreamSynchronize(stream));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_after_sync));
  // Sync must releaae memory to OS
  REQUIRE(res_after_sync < res_before_sync);

  int value = 0;
  attr = hipMemPoolReuseFollowEventDependencies;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value));
  // Default enabled
  REQUIRE(1 == value);

  attr = hipMemPoolReuseAllowOpportunistic;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value));
  // Default enabled
  REQUIRE(1 == value);

  attr = hipMemPoolReuseAllowInternalDependencies;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value));
  // Default enabled
  REQUIRE(1 == value);

  attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Default is 0
  REQUIRE(0 == value64);

  attr = hipMemPoolAttrReservedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Must be bigger than current
  REQUIRE(value64 > res_after_sync);

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current usage query works - just small buffer left
  REQUIRE(sizeof(float) * 1024 == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE(sizeof(float) * (8 * 1024 * 1024 + 1024) == value64);

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C), stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipMemPoolApi_BasicTrim") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  hipMemPool_t mem_pool;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));

  float* B, *C;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  size_t numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float), mem_pool, stream));

  numElements = 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C), numElements * sizeof(float), mem_pool, stream));

  int blocks = 2;
  int clkRate;
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));

  kernel500ms<<<32, blocks, 0, stream>>>(B, clkRate);

  hipMemPoolAttr attr;
  attr = hipMemPoolAttrReleaseThreshold;
  // The pool must hold 128MB
  std::uint64_t threshold = 128 * 1024 * 1024;
  HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &threshold));

  // Not a real free, since kernel isn't done
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream));

  // Get reserved memory before trim
  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_trim = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_before_trim));

  size_t min_bytes_to_hold = sizeof(float) * 1024;
  HIP_CHECK(hipMemPoolTrimTo(mem_pool, min_bytes_to_hold));

  std::uint64_t res_after_trim = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_after_trim));
  // Trim must be a nop because execution isn't done
  REQUIRE(res_before_trim == res_after_trim);

  HIP_CHECK(hipStreamSynchronize(stream));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_after_sync));
  // Since hipMemPoolAttrReleaseThreshold is 128 MB sync does nothing to the freed memory
  REQUIRE(res_after_trim == res_after_sync);

  HIP_CHECK(hipMemPoolTrimTo(mem_pool, min_bytes_to_hold));

  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_after_trim));
  // Validate memory after real trim. The pool must hold less memory than before
  REQUIRE(res_after_trim < res_after_sync);

  attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the threshold query works
  REQUIRE(threshold == value64);

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current usage query works - just small buffer left
  REQUIRE(sizeof(float) * 1024 == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE(sizeof(float) * (8 * 1024 * 1024 + 1024) == value64);

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C), stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipMemPoolApi_BasicReuse") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  hipMemPool_t mem_pool;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));

  float *A, *B, *C;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  size_t numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float), mem_pool, stream));

  numElements = 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C), numElements * sizeof(float), mem_pool, stream));

  int blocks = 2;
  int clkRate;
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));

  kernel500ms<<<32, blocks, 0, stream>>>(A, clkRate);

  hipMemPoolAttr attr;
  // Not a real free, since kernel isn't done
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A), stream));

  numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float), mem_pool, stream));
  // Runtime must reuse the pointer
  REQUIRE(A == B);

  // Make a sync before the second kernel launch to make sure memory B isn't gone
  HIP_CHECK(hipStreamSynchronize(stream));

  // Second kernel launch with new memory
  kernel500ms<<<32, blocks, 0, stream>>>(B, clkRate);

  HIP_CHECK(hipStreamSynchronize(stream));

  attr = hipMemPoolAttrUsedMemCurrent;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current usage reports the both buffers
  REQUIRE(sizeof(float) * (8 * 1024 * 1024 + 1024) == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE(sizeof(float) * (8 * 1024 * 1024 + 1024) == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream));
  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current usage reports just one buffer, because the above free doesn't hold memory
  REQUIRE(sizeof(float) * 1024 == value64);

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C), stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipMemPoolApi_Opportunistic") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  hipMemPool_t mem_pool;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));

  hipMemPoolAttr attr;
  int blocks = 2;
  int clkRate;
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));

  float *A, *B, *C;
  hipStream_t stream, stream2;

  // Create 2 async non-blocking streams
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking));

  size_t numElements = 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C), numElements * sizeof(float), mem_pool, stream));
  int value = 0;

  SECTION("Disallow Opportunistic - No Reuse") {
    numElements = 8 * 1024 * 1024;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float), mem_pool, stream));

    // Disable all default pool states
    attr = hipMemPoolReuseFollowEventDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));
    attr = hipMemPoolReuseAllowOpportunistic;
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));
    attr = hipMemPoolReuseAllowInternalDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));

    // Run kernel for 500 ms in the first stream
    kernel500ms<<<32, blocks, 0, stream>>>(A, clkRate);

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A), stream));

    // Sleep for 1 second GPU should be idle by now
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    numElements = 8 * 1024 * 1024;
    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float), mem_pool, stream2));
    // Without Opportunistic state runtime must allocate another buffer
    REQUIRE(A != B);

    // Run kernel with the new memory in the second stream
    kernel500ms<<<32, blocks, 0, stream2>>>(B, clkRate);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamSynchronize(stream2));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream2));
  }

  SECTION("Allow Opportunistic - Reuse") {
    numElements = 8 * 1024 * 1024;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float), mem_pool, stream));

    value = 1;
    attr = hipMemPoolReuseAllowOpportunistic;
    // Enable Opportunistic
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));

    // Run kernel for 500 ms in the first stream
    kernel500ms<<<32, blocks, 0, stream>>>(A, clkRate);

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A), stream));

    // Sleep for 1 second GPU should be idle by now
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    numElements = 8 * 1024 * 1024;
    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float), mem_pool, stream2));
    // With Opportunistic state runtime will reuse freed buffer A
    REQUIRE(A == B);

    // Run kernel with the new memory in the second stream
    kernel500ms<<<32, blocks, 0, stream2>>>(B, clkRate);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamSynchronize(stream2));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream2));
  }

  SECTION("Allow Opportunistic - No Reuse") {
    numElements = 8 * 1024 * 1024;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float), mem_pool, stream));

    value = 1;
    attr = hipMemPoolReuseAllowOpportunistic;
    // Enable Opportunistic
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));

    // Run kernel for 500 ms in the first stream
    kernel500ms<<<32, blocks, 0, stream>>>(A, clkRate);

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A), stream));

    numElements = 8 * 1024 * 1024;
    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float), mem_pool, stream2));
    // With Opportunistic state runtime can't reuse freed buffer A, because it's still busy with the kernel
    REQUIRE(A != B);

    // Run kernel with the new memory in the second stream
    kernel500ms<<<32, blocks, 0, stream2>>>(B, clkRate);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamSynchronize(stream2));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream2));
  }

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C), stream));
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipStreamDestroy(stream2));
}

TEST_CASE("Unit_hipMemPoolApi_Default") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  hipMemPool_t mem_pool;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, 0));

  float *A, *B, *C;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  size_t numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float), stream));

  numElements = 1024;
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&C), numElements * sizeof(float), stream));

  int blocks = 2;
  int clkRate;
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));

  kernel500ms<<<32, blocks, 0, stream>>>(A, clkRate);

  hipMemPoolAttr attr;
  // Not a real free, since kernel isn't done
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A), stream));

  numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float), stream));
  // Runtime must reuse the pointer
  REQUIRE(A == B);

  // Make a sync before the second kernel launch to make sure memory B isn't gone
  HIP_CHECK(hipStreamSynchronize(stream));

  // Second kernel launch with new memory
  kernel500ms<<<32, blocks, 0, stream>>>(B, clkRate);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream));

  HIP_CHECK(hipStreamSynchronize(stream));

  std::uint64_t value64 = 0;
  attr = hipMemPoolAttrReservedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current reserved just 4KB alloc
  REQUIRE(sizeof(float) * 1024 == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE(sizeof(float) * (8 * 1024 * 1024 + 1024) == value64);

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current usage reports just one buffer, because the above free doesn't hold memory
  REQUIRE(sizeof(float) * 1024 == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C), stream));
  HIP_CHECK(hipStreamDestroy(stream));
}
