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
#include <vector>

/*
    This testcase verifies HIP Mem VMM API basic scenario - supported on all devices
 */

TEST_CASE("Unit_hipMemVmm_Basic") {
  int vmm = 0;
  HIP_CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  INFO("hipDeviceAttributeVirtualMemoryManagementSupported: " << vmm);

  if (vmm == 0) {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }

  size_t granularity = 0;

  hipMemAllocationProp memAllocationProp;
  memAllocationProp.type = hipMemAllocationTypePinned;
  memAllocationProp.location.id = 0;
  memAllocationProp.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &memAllocationProp, hipMemAllocationGranularityRecommended));

  size_t size = 4 * 1024;
  void* reservedAddress{nullptr};
  HIP_CHECK(hipMemAddressReserve(&reservedAddress, size, granularity, nullptr, 0));

  hipMemGenericAllocationHandle_t gaHandle{nullptr};
  HIP_CHECK(hipMemCreate(&gaHandle, size, &memAllocationProp, 0));

  HIP_CHECK(hipMemMap(reservedAddress, size, 0, gaHandle, 0));

  hipMemAccessDesc desc;
  std::vector<char> values(size);
  const char value = 1;

  HIP_CHECK(hipMemSetAccess(reservedAddress, size, &desc, 1));
  HIP_CHECK(hipMemset(reservedAddress, value, size));
  HIP_CHECK(hipMemcpy(&values[0], reservedAddress, size, hipMemcpyDeviceToHost));

  for (size_t i=0; i < size; ++i) {
    REQUIRE(values[i] == value);
  }

  HIP_CHECK(hipMemUnmap(reservedAddress, size));

  HIP_CHECK(hipMemRelease(gaHandle));
  HIP_CHECK(hipMemAddressFree(reservedAddress, size));
}

