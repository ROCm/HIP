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
#include <resource_guards.hh>

TEST_CASE("Unit_hipMemGetAddressRange_Positive") {
  hipDeviceptr_t base_ptr;
  size_t mem_size = 0;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const int offset = GENERATE(0, 20, 40, 60, 80);

  SECTION("Host address range") {
    using LA = LinearAllocs;
    LinearAllocGuard<int> host_alloc(LA::hipHostMalloc, allocation_size);

    HIP_CHECK(hipMemGetAddressRange(&base_ptr, &mem_size, reinterpret_cast<hipDeviceptr_t>(host_alloc.ptr() + offset)));

    REQUIRE(reinterpret_cast<hipDeviceptr_t>(host_alloc.ptr()) == base_ptr);
    REQUIRE(mem_size == allocation_size);
  }
  SECTION("Device address range") {
    using LA = LinearAllocs;
    const auto device_allocation_type = GENERATE(LA::hipMalloc, LA::hipMallocManaged);
    LinearAllocGuard<int> device_alloc(device_allocation_type, allocation_size);

    HIP_CHECK(hipMemGetAddressRange(&base_ptr, &mem_size, reinterpret_cast<hipDeviceptr_t>(device_alloc.ptr() + offset)));

    REQUIRE(reinterpret_cast<hipDeviceptr_t>(device_alloc.ptr()) == base_ptr);
    REQUIRE(mem_size == allocation_size);    
  }
  SECTION("Pitch address range") {
    size_t width = 32;
    size_t height = 32;
    LinearAllocGuard2D<int> device_alloc(width, height);

    HIP_CHECK(hipMemGetAddressRange(&base_ptr, &mem_size, reinterpret_cast<hipDeviceptr_t>(device_alloc.ptr() + offset)));

    REQUIRE(reinterpret_cast<hipDeviceptr_t>(device_alloc.ptr()) == base_ptr);
    REQUIRE(mem_size == (device_alloc.pitch() * height));    
  }
}

TEST_CASE("Unit_hipMemGetAddressRange_Negative") {
  hipDeviceptr_t base_ptr;
  size_t mem_size = 0;
  const auto allocation_size = kPageSize / 2;
  const int offset = kPageSize;
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);

  hipDeviceptr_t dummy_ptr;

  SECTION("Device pointer is invalid") {
    HIP_CHECK_ERROR(hipMemGetAddressRange(&base_ptr, &mem_size, dummy_ptr), hipErrorInvalidDevicePointer);
  }
  SECTION("Offset is greater than allocated size") {
    HIP_CHECK_ERROR(hipMemGetAddressRange(&base_ptr, &mem_size, reinterpret_cast<hipDeviceptr_t>(host_alloc.ptr() + offset)), hipErrorInvalidDevicePointer);
  }
}
