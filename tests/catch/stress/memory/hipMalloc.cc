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

#include <memory>

// Stress allocation tests
// Try to allocate as much memory as possible
// But since max allocation can fail, we need to be happy with atleast 1/4th of memory
TEST_CASE("Stress_hipMalloc_HighSizeAlloc") {
  size_t devMemTotal{0}, devMemFree{0};
  HIP_CHECK(hipMemGetInfo(&devMemFree, &devMemTotal));
  REQUIRE(devMemFree > 0);
  REQUIRE(devMemTotal > 0);

  char* d_ptr{nullptr};
  size_t counter{0};

  INFO("Free Mem Available: " << devMemFree << " bytes out of " << devMemTotal << " bytes!");
  while (hipMalloc(&d_ptr, devMemFree) != hipSuccess && devMemFree > 1) {
    counter++;
    devMemFree >>= 1;  // reduce the memory to be allocated by half
    INFO("Attempt to allocate " << devMemFree << " bytes out of " << devMemTotal
                                << " bytes failed!");
    REQUIRE(counter <= 2);  // Make sure that we are atleast able to allocate 1/4th of max memory
  }

  HIP_CHECK(hipMemset(d_ptr, 1, devMemFree));
  auto ptr = std::unique_ptr<unsigned char[]>{new unsigned char[devMemFree]};
  HIP_CHECK(hipMemcpy(ptr.get(), d_ptr, devMemFree, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_ptr));
  REQUIRE(std::all_of(ptr.get(), ptr.get() + devMemFree, [](unsigned char n) { return n == 1; }));
}
