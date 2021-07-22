/*
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
 * Test to compare
 * 1.pciBusID from hipDeviceGetPCIBusId and hipDeviceGetAttribute **
 * 2.{pciDomainID, pciBusID, pciDeviceID} values hipDeviceGetPCIBusId vs lspci **
 */

#include <hip_test_common.hh>

#define MAX_DEVICE_LENGTH 20

namespace hipDeviceGetPCIBusIdTests {

void getPciBusId(int deviceCount,
                 char **hipDeviceList) {
  for (int i = 0; i < deviceCount; i++) {
    HIP_CHECK(hipDeviceGetPCIBusId(hipDeviceList[i], MAX_DEVICE_LENGTH, i));
  }
}
}  // namespace hipDeviceGetPCIBusIdTests

TEST_CASE("Unit_hipDeviceGetPCIBusId_Check_PciBusID_WithAttr") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE_FALSE(deviceCount == 0);
  printf("No.of gpus in the system: %d\n", deviceCount);
  // Allocate an array of pointer to characters
  char **hipDeviceList = new char*[deviceCount];
  REQUIRE_FALSE(hipDeviceList == nullptr);
  for (int i = 0; i < deviceCount; i++) {
    hipDeviceList[i] = new char[MAX_DEVICE_LENGTH];
    REQUIRE_FALSE(hipDeviceList[i] == nullptr);
  }
  hipDeviceGetPCIBusIdTests::getPciBusId(deviceCount, hipDeviceList);

  for (int i = 0; i < deviceCount; i++) {
    int pciBusID = -1;
    int pciDeviceID = -1;
    int pciDomainID = -1;
    int tempPciBusId = -1;
    sscanf(hipDeviceList[i], "%04x:%02x:%02x", &pciDomainID, &pciBusID,
           &pciDeviceID);
    HIP_CHECK(hipDeviceGetAttribute(&tempPciBusId,
                                   hipDeviceAttributePciBusId, i));
    REQUIRE_FALSE(pciBusID != tempPciBusId);
  }
  // Deallocate
  for (int i = 0; i < deviceCount; i++) {
    delete hipDeviceList[i];
  }
  delete[] hipDeviceList;
  printf("pciBusID output of both hipDeviceGetPCIBusId and"
         " hipDeviceGetAttribute matched for all gpus\n");
}


/**
 * Validates negative scenarios for hipDeviceGetPCIBusId
 * scenario1: pciBusId = nullptr
 * scenario2: device = -1 (Invalid Device)
 * scenario3: device = Non Existing Device
 * scenario4: len = 0
 * scenario5: len < 0
 */
TEST_CASE("Unit_hipDeviceGetPCIBusId_NegTst") {
  char pciBusId[MAX_DEVICE_LENGTH];
  int device;
  HIP_CHECK(hipGetDevice(&device));

  // pciBusId is nullptr
  SECTION("pciBusId is nullptr") {
    REQUIRE_FALSE(hipDeviceGetPCIBusId(nullptr, MAX_DEVICE_LENGTH, device)
                  == hipSuccess);
  }

  // len = 0
  SECTION("len is 0") {
    REQUIRE_FALSE(hipDeviceGetPCIBusId(pciBusId, 0, device) == hipSuccess);
  }

  // len < 0
  SECTION("len is less than 0") {
    REQUIRE_FALSE(hipDeviceGetPCIBusId(pciBusId, -1, device) == hipSuccess);
  }

  // device = -1
  SECTION("device is -1") {
    REQUIRE_FALSE(hipDeviceGetPCIBusId(pciBusId, MAX_DEVICE_LENGTH, -1)
                  == hipSuccess);
  }
  // device = Non Existing Device
  SECTION("device is -1") {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    REQUIRE_FALSE(deviceCount == 0);
    REQUIRE_FALSE(hipDeviceGetPCIBusId(pciBusId, MAX_DEVICE_LENGTH,
                  deviceCount) == hipSuccess);
  }
}
