/*
Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * hipGetDeviceCount tests
 * Scenario: Validates the value of numDevices when devices are hidden.
 */

#include <hip_test_common.hh>
#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>

#define MAX_SIZE 30
#define VISIBLE_DEVICE 0

/**
 * Validate behavior of hipGetDeviceCount for masked devices.
 */
TEST_CASE("Unit_hipGetDeviceCount_MaskedDevices") {
  int numDevices = 0;
  char visibleDeviceString[MAX_SIZE] = {};
  snprintf(visibleDeviceString, MAX_SIZE, "%d", VISIBLE_DEVICE);

#ifdef __HIP_PLATFORM_NVCC__
  unsetenv("CUDA_VISIBLE_DEVICES");
  setenv("CUDA_VISIBLE_DEVICES", visibleDeviceString, 1);
#else
  unsetenv("ROCR_VISIBLE_DEVICES");
  unsetenv("HIP_VISIBLE_DEVICES");
  setenv("ROCR_VISIBLE_DEVICES", visibleDeviceString, 1);
  setenv("HIP_VISIBLE_DEVICES", visibleDeviceString, 1);
#endif

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  REQUIRE(numDevices == 1);
}
#endif
