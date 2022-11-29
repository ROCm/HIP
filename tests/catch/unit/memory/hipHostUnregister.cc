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

namespace hipHostUnregisterTests {
constexpr unsigned int allFlags = hipHostRegisterDefault &  // 0
    hipHostRegisterPortable &                               // 1
    hipHostRegisterMapped &                                 // 2
    hipHostRegisterIoMemory                                 // 4
#if HT_NVIDIA
    & cudaHostRegisterReadOnly;  // 8
#else
    ;
#endif

inline bool hipHostRegisterSupported() {
#if HT_NVIDIA
  // unable to query for cudaDevAttrHostRegisterSupported equivalent
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-40");
  HipTest::HIP_SKIP_TEST("hipHostRegister is not supported on this device");
  return false;
#else
  return true;
#endif
}


TEST_CASE("Unit_hipHostUnregister_MemoryNotAccessableAfterUnregister") {
  if (!hipHostRegisterSupported()) {
    return;
  }
  // try all combinations of flags
  for (unsigned int flag = 0; flag <= allFlags; ++flag) {
    DYNAMIC_SECTION("Using flag: " << flag) {
      auto x = std::unique_ptr<int>(new int);
      HIP_CHECK(hipHostRegister(x.get(), sizeof(int), flag));

      void* device_memory;
      HIP_CHECK(hipHostGetDevicePointer(&device_memory, x.get(), 0));

      HIP_CHECK(hipHostUnregister(x.get()));
      HIP_CHECK_ERROR(hipHostGetDevicePointer(&device_memory, x.get(), 0), hipErrorInvalidValue);
    }
  }
}

TEST_CASE("Unit_hipHostUnregister_NullPtr") {
  HIP_CHECK_ERROR(hipHostUnregister(nullptr), hipErrorInvalidValue);
}

TEST_CASE("Unit_hipHostUnregister_Ptr_Different_Than_Specified_To_Register") {
  std::vector<int> alloc(2);
  HIP_CHECK(hipHostRegister(alloc.data(), alloc.size(), 0));
  HIP_CHECK_ERROR(hipHostUnregister(&alloc.data()[1]), hipErrorHostMemoryNotRegistered);
}

TEST_CASE("Unit_hipHostUnregister_NotRegisteredPointer") {
  auto x = std::unique_ptr<int>(new int);
  HIP_CHECK_ERROR(hipHostUnregister(x.get()), hipErrorHostMemoryNotRegistered);
}

TEST_CASE("Unit_hipHostUnregister_AlreadyUnregisteredPointer") {
  if (!hipHostRegisterSupported()) {
    return;
  }
  // try all combinations of flags
  for (unsigned int flag = 0; flag <= allFlags; ++flag) {
    DYNAMIC_SECTION("Using flag: " << flag) {
      auto x = std::unique_ptr<int>(new int);
      HIP_CHECK(hipHostRegister(x.get(), sizeof(int), flag));
      HIP_CHECK(hipHostUnregister(x.get()));
      HIP_CHECK_ERROR(hipHostUnregister(x.get()), hipErrorHostMemoryNotRegistered);
    }
  }
}

}  // namespace hipHostUnregisterTests
