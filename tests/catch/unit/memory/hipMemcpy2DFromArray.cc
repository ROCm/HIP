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
/*
Testcase Scenarios :
Unit_hipMemcpy2DFromArray_Positive_Default - Test basic memcpy between 2D array and host/device with hipMemcpy2DFromArray api
Unit_hipMemcpy2DFromArray_Positive_Synchronization_Behavior - Test synchronization behavior for hipMemcpy2DFromArray api
Unit_hipMemcpy2DFromArray_Positive_ZeroWidthHeight - Test that no data is copied when width/height is set to 0
Unit_hipMemcpy2DFromArray_Negative_Parameters - Test unsuccessful execution of hipMemcpy2DFromArray api when parameters are invalid
*/
#include "array_memcpy_tests_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include <resource_guards.hh>

TEST_CASE("Unit_hipMemcpy2DFromArray_Positive_Default") {
  using namespace std::placeholders;

  const auto width = GENERATE(16, 32, 48);
  const auto height = GENERATE(1, 16, 32, 48);

  SECTION("Array to host") {
    Memcpy2DHostFromAShell<false, int>(std::bind(hipMemcpy2DFromArray, _1, _2, _3, 0, 0, width * sizeof(int), height, hipMemcpyDeviceToHost), width, height);
  }

  SECTION("Array to host with default kind") {
    Memcpy2DHostFromAShell<false, int>(std::bind(hipMemcpy2DFromArray, _1, _2, _3, 0, 0, width * sizeof(int), height, hipMemcpyDefault), width, height);
  }

  SECTION("Array to device") {
    SECTION("Peer access disabled") {
      Memcpy2DDeviceFromAShell<false, false, int>(std::bind(hipMemcpy2DFromArray, _1, _2, _3, 0, 0, width * sizeof(int), height, hipMemcpyDeviceToDevice), width, height);
    }
    SECTION("Peer access enabled") {
      Memcpy2DDeviceFromAShell<false, true, int>(std::bind(hipMemcpy2DFromArray, _1, _2, _3, 0, 0, width * sizeof(int), height, hipMemcpyDeviceToDevice), width, height);
    }
  }

  SECTION("Array to device with default kind") {
    SECTION("Peer access disabled") {
      Memcpy2DDeviceFromAShell<false, false, int>(std::bind(hipMemcpy2DFromArray, _1, _2, _3, 0, 0, width * sizeof(int), height, hipMemcpyDefault), width, height);
    }
    SECTION("Peer access enabled") {
      Memcpy2DDeviceFromAShell<false, true, int>(std::bind(hipMemcpy2DFromArray, _1, _2, _3, 0, 0, width * sizeof(int), height, hipMemcpyDefault), width, height);
    }
  }
}

TEST_CASE("Unit_hipMemcpy2DFromArray_Positive_Synchronization_Behavior") {
  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Array to host") {
    const auto width = GENERATE(16, 32, 48);
    const auto height = GENERATE(16, 32, 48);

    MemcpyAtoHPageableSyncBehavior(std::bind(hipMemcpy2DFromArray, _1, width * sizeof(int), _2, 0, 0, width * sizeof(int), height, hipMemcpyDeviceToHost), width, height, true);
    MemcpyAtoHPinnedSyncBehavior(std::bind(hipMemcpy2DFromArray, _1, width * sizeof(int), _2, 0, 0, width * sizeof(int), height, hipMemcpyDeviceToHost), width, height, true);
  }

  SECTION("Array to device") {
    const auto width = GENERATE(16, 32, 48);
    const auto height = GENERATE(16, 32, 48);

    MemcpyAtoDSyncBehavior(std::bind(hipMemcpy2DFromArray, _1, _2, _3, 0, 0, width * sizeof(int), height, hipMemcpyDeviceToDevice), width, height, false);
  }
}

TEST_CASE("Unit_hipMemcpy2DFromArray_Positive_ZeroWidthHeight") {
  using namespace std::placeholders;
  const auto width = 16;
  const auto height = 16;

  SECTION("Array to host") {
    SECTION("Height is 0") {
      Memcpy2DFromArrayZeroWidthHeight<false>(std::bind(hipMemcpy2DFromArray, _1, _2, _3, 0, 0, width * sizeof(int), 0, hipMemcpyDeviceToHost), width, height);
    }
    SECTION("Width is 0") {
      Memcpy2DFromArrayZeroWidthHeight<false>(std::bind(hipMemcpy2DFromArray, _1, _2, _3, 0, 0, 0, height, hipMemcpyDeviceToHost), width, height);
    }
  }
  SECTION("Array to device") {
    SECTION("Height is 0") {
      Memcpy2DFromArrayZeroWidthHeight<false>(std::bind(hipMemcpy2DFromArray, _1, _2, _3, 0, 0, width * sizeof(int), 0, hipMemcpyDeviceToDevice), width, height);
    }
    SECTION("Width is 0") {
      Memcpy2DFromArrayZeroWidthHeight<false>(std::bind(hipMemcpy2DFromArray, _1, _2, _3, 0, 0, 0, height, hipMemcpyDeviceToDevice), width, height);
    }
  }
}

TEST_CASE("Unit_hipMemcpy2DFromArray_Negative_Parameters") {
  using namespace std::placeholders;

  const auto width = 32;
  const auto height = 32;
  const auto allocation_size = 2 * width * height * sizeof(int);

  const unsigned int flag = hipArrayDefault;

  ArrayAllocGuard<int> array_alloc(make_hipExtent(width, height, 0), flag);
  LinearAllocGuard2D<int> device_alloc(width, height);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);

  SECTION("Array to host") {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(nullptr, 2 * width * sizeof(int), array_alloc.ptr(), 0, 0, width * sizeof(int), height, hipMemcpyDeviceToHost), hipErrorInvalidValue);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(host_alloc.ptr(), 2 * width * sizeof(int), nullptr, 0, 0, width * sizeof(int), height, hipMemcpyDeviceToHost), hipErrorInvalidHandle);
    }
    SECTION("dpitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(host_alloc.ptr(), width * sizeof(int) - 10, array_alloc.ptr(), 0, 0, width * sizeof(int), height, hipMemcpyDeviceToHost), hipErrorInvalidPitchValue);
    }
    SECTION("Offset + width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(host_alloc.ptr(), 2 * width * sizeof(int), array_alloc.ptr(), 1, 0, width * sizeof(int), height, hipMemcpyDeviceToHost), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(host_alloc.ptr(), 2 * width * sizeof(int), array_alloc.ptr(), 0, 1, width * sizeof(int), height, hipMemcpyDeviceToHost), hipErrorInvalidValue);
    }
    SECTION("Width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(host_alloc.ptr(), 2 * width * sizeof(int), array_alloc.ptr(), 0, 0, width * sizeof(int) + 1, height, hipMemcpyDeviceToHost), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(host_alloc.ptr(), 2 * width * sizeof(int), array_alloc.ptr(), 0, 0, width * sizeof(int), height + 1, hipMemcpyDeviceToHost), hipErrorInvalidValue);
    }
    SECTION("Memcpy kind is invalid") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(host_alloc.ptr(), 2 * width * sizeof(int), array_alloc.ptr(), 0, 0, width * sizeof(int), height, static_cast<hipMemcpyKind>(-1)), hipErrorInvalidMemcpyDirection);
    }
  }
  SECTION("Array to device") {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(nullptr, device_alloc.pitch(), array_alloc.ptr(), 0, 0, width * sizeof(int), height, hipMemcpyDeviceToDevice), hipErrorInvalidValue);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(device_alloc.ptr(), device_alloc.pitch(), nullptr, 0, 0, width * sizeof(int), height, hipMemcpyDeviceToDevice), hipErrorInvalidHandle);
    }
    SECTION("dpitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(device_alloc.ptr(), width * sizeof(int) - 10, array_alloc.ptr(), 0, 0, width * sizeof(int), height, hipMemcpyDeviceToDevice), hipErrorInvalidPitchValue);
    }
    SECTION("Offset + width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(device_alloc.ptr(), device_alloc.pitch(), array_alloc.ptr(), 1, 0, width * sizeof(int), height, hipMemcpyDeviceToDevice), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(device_alloc.ptr(), device_alloc.pitch(), array_alloc.ptr(), 0, 1, width * sizeof(int), height, hipMemcpyDeviceToDevice), hipErrorInvalidValue);
    }
    SECTION("Width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(device_alloc.ptr(), device_alloc.pitch(), array_alloc.ptr(), 0, 0, width * sizeof(int) + 1, height, hipMemcpyDeviceToDevice), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(device_alloc.ptr(), device_alloc.pitch(), array_alloc.ptr(), 0, 0, width * sizeof(int), height + 1, hipMemcpyDeviceToDevice), hipErrorInvalidValue);
    }
    SECTION("Memcpy kind is invalid") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArray(device_alloc.ptr(), device_alloc.pitch(), array_alloc.ptr(), 0, 0, width * sizeof(int), height, static_cast<hipMemcpyKind>(-1)), hipErrorInvalidMemcpyDirection);
    }
  }
}
