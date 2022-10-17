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

TEST_CASE("Unit_hipMemcpy2DToArray_Default") {
  using namespace std::placeholders;

  const auto width = GENERATE(16, 32, 48);
  const auto height = GENERATE(1, 16, 32, 48);

  SECTION("Host to Array") {
    Memcpy2DHosttoAShell<false, int>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height, hipMemcpyHostToDevice), width, height);
  }

  SECTION("Host to Array with default kind") {
    Memcpy2DHosttoAShell<false, int>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height, hipMemcpyDefault), width, height);
  }

  SECTION("Device to Array") {
    SECTION("Peer access disabled") {
      Memcpy2DDevicetoAShell<false, false, int>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height, hipMemcpyDeviceToDevice), width, height);
    }
    SECTION("Peer access enabled") {
      Memcpy2DDevicetoAShell<false, true, int>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height, hipMemcpyDeviceToDevice), width, height);
    }
  }

  SECTION("Device to Array with default kind") {
    SECTION("Peer access disabled") {
      Memcpy2DDevicetoAShell
      <false, false, int>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height, hipMemcpyDefault), width, height);
    }
    SECTION("Peer access enabled") {
      Memcpy2DDevicetoAShell
      <false, true, int>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height, hipMemcpyDefault), width, height);
    }
  }
}


TEST_CASE("Unit_hipMemcpy2DToArray_Synchronization_Behavior") {
  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Array") {
    const auto width = GENERATE(16, 32, 48);
    const auto height = GENERATE(16, 32, 48);

    MemcpyHtoASyncBehavior(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, width * sizeof(int), width * sizeof(int), height, hipMemcpyHostToDevice), width, height, true);
  }

  SECTION("Device to Array") {
    const auto width = GENERATE(16, 32, 48);
    const auto height = GENERATE(16, 32, 48);

    MemcpyDtoASyncBehavior(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height, hipMemcpyDeviceToDevice), width, height, false);
  }
}

TEST_CASE("Unit_hipMemcpy2DToArray_ZeroWidthHeight") {
  using namespace std::placeholders;
  const auto width = 16;
  const auto height = 16;

  SECTION("Array to host") {
    SECTION("Height is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), 0, hipMemcpyHostToDevice), width, height);
    }
    SECTION("Width is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, 0, height, hipMemcpyHostToDevice), width, height);
    }
  }
  SECTION("Array to device") {
    SECTION("Height is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), 0, hipMemcpyDeviceToDevice), width, height);
    }
    SECTION("Width is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, 0, height, hipMemcpyDeviceToDevice), width, height);
    }
  }
}

TEST_CASE("Unit_hipMemcpy2DToArray_Negative_Parameters") {
  using namespace std::placeholders;

  const auto width = 32;
  const auto height = 32;
  const auto allocation_size = 2 * width * height * sizeof(int);

  const unsigned int flag = hipArrayDefault;

  ArrayAllocGuard2D<int> array_alloc(width, height, flag);
  LinearAllocGuard2D<int> device_alloc(width, height);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);

  SECTION("Host to Array") {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(nullptr, 0, 0, host_alloc.ptr(), 2 *width * sizeof(int), width * sizeof(int), height, hipMemcpyHostToDevice), hipErrorInvalidHandle);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, nullptr, 2 * width * sizeof(int), width * sizeof(int), height, hipMemcpyHostToDevice), hipErrorInvalidValue);
    }
    SECTION("spitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), width * sizeof(int) - 10, width * sizeof(int), height, hipMemcpyHostToDevice), hipErrorInvalidPitchValue);
    }
    SECTION("Offset + width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 1, 0, host_alloc.ptr(), 2 * width * sizeof(int), width * sizeof(int), height, hipMemcpyHostToDevice), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 1, host_alloc.ptr(), 2 * width * sizeof(int), width * sizeof(int), height, hipMemcpyHostToDevice), hipErrorInvalidValue);
    }
    SECTION("Width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), 2 * width * sizeof(int), width * sizeof(int) + 1, height, hipMemcpyHostToDevice), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), 2 * width * sizeof(int), width * sizeof(int), height + 1, hipMemcpyHostToDevice), hipErrorInvalidValue);
    }
    SECTION("Memcpy kind is invalid") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), 2 * width * sizeof(int), width * sizeof(int), height, static_cast<hipMemcpyKind>(-1)), hipErrorInvalidMemcpyDirection);
    }
  }
  SECTION("Device to Array") {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(nullptr, 0, 0, device_alloc.ptr(), device_alloc.pitch(), width * sizeof(int), height, hipMemcpyDeviceToDevice), hipErrorInvalidHandle);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, nullptr, device_alloc.pitch(), width * sizeof(int), height, hipMemcpyDeviceToDevice), hipErrorInvalidValue);
    }
    SECTION("spitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), width * sizeof(int) - 10, width * sizeof(int), height, hipMemcpyDeviceToDevice), hipErrorInvalidPitchValue);
    }
    SECTION("Offset + width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 1, 0, device_alloc.ptr(), device_alloc.pitch(), width * sizeof(int), height, hipMemcpyDeviceToDevice), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 1, device_alloc.ptr(), device_alloc.pitch(), width * sizeof(int), height, hipMemcpyDeviceToDevice), hipErrorInvalidValue);
    }
    SECTION("Width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), device_alloc.pitch(), width * sizeof(int) + 1, height, hipMemcpyDeviceToDevice), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), device_alloc.pitch(), width * sizeof(int), height + 1, hipMemcpyDeviceToDevice), hipErrorInvalidValue);
    }
    SECTION("Memcpy kind is invalid") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), device_alloc.pitch(), width * sizeof(int), height, static_cast<hipMemcpyKind>(-1)), hipErrorInvalidMemcpyDirection);
    }
  }
}