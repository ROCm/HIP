/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t --tests 0x1
 * TEST: %t --tests 0x2
 * TEST: %t --tests 0x3
 * TEST: %t --tests 0x4
 * HIT_END
 */

#include <stdio.h>
#include "test_common.h"
/**
 * Validates negative scenarios for hipDeviceGetByPCIBusId
 * scenario: Validates device number from pciBusIdstr string
 */
bool testPciBusId(void) {
  bool testResult = true;
  char pciBusId[13];
  int deviceCount = 0;
  HIPCHECK(hipGetDeviceCount(&deviceCount));
  HIPASSERT(deviceCount != 0);
  for (int i = 0; i < deviceCount; i++) {
    int pciBusID = -1;
    int pciDeviceID = -1;
    int pciDomainID = -1;
    int tempPciBusId = -1;
    int tempDeviceId = -1;
    HIPCHECK(hipDeviceGetPCIBusId(&pciBusId[0], 13, i));
    sscanf(pciBusId, "%04x:%02x:%02x", &pciDomainID,
           &pciBusID, &pciDeviceID);
    HIPCHECK(hipDeviceGetAttribute(&tempPciBusId,
           hipDeviceAttributePciBusId, i));
    if (pciBusID != tempPciBusId) {
      testResult = false;
      break;
    }
    HIPCHECK(hipDeviceGetByPCIBusId(&tempDeviceId, pciBusId));
    if (tempDeviceId != i) {
      testResult = false;
      break;
    }
  }
  return testResult;
}

/**
 * Validates negative scenarios for hipDeviceGetByPCIBusId
 * scenario: device = nullptr and pciBusIdstr = nullptr
 */
bool testNullPtr() {
  bool TestPassed = true;
  int device = -1;
  hipError_t ret;
  char pciBusIdstr[13];
  ret = hipDeviceGetByPCIBusId(nullptr, pciBusIdstr);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {device = nullptr} Failed \n");
  }
  ret = hipDeviceGetByPCIBusId(&device, nullptr);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {pciBusIdstr = nullptr} Failed \n");
  }
  return TestPassed;
}

/**
 * Validates negative scenarios for hipDeviceGetByPCIBusId
 * scenario1: Pass an empty like ""
 * scenario1: Pass an shorter string "0000:"
 */
bool testInputString() {
  bool TestPassed = true;
  int device = -1;
  hipError_t ret;
  ret = hipDeviceGetByPCIBusId(&device, "");
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {empty input string:\"\"} Failed \n");
  }
  ret = hipDeviceGetByPCIBusId(&device, "0000:");
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {shorter input string: \"0000:\"} Failed \n");
  }
  return TestPassed;
}

/**
 * Validates negative scenarios for hipDeviceGetByPCIBusId
 * scenario: Pass wrong bus id in pciBusIdstr
 */
bool testWrongBusID() {
  bool TestPassed = true;
  int deviceCount = 0;
  HIPCHECK(hipGetDeviceCount(&deviceCount));
  HIPASSERT(deviceCount != 0);
  int pciBusId[deviceCount], pciDeviceID[deviceCount],
      pciDomainID[deviceCount];
  // get bus id of all the devices
  for (int i = 0; i < deviceCount; i++) {
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, i));
    pciBusId[i] = prop.pciBusID;
    pciDeviceID[i] = prop.pciDeviceID;
    pciDomainID[i] = prop.pciDomainID;
    printf("device %d: pciDomainID=%x, pciBusID=%x, pciDeviceID=%x \n",
           i, prop.pciDomainID, prop.pciBusID, prop.pciDomainID);
  }
  // get a non existing bus id
  int id = 0;
  for (; id < 256; id++) {
    bool bFound = false;
    // check if id is the pci busid of any existing device
    for (int j = 0; j < deviceCount; j++) {
      if (id == pciBusId[j]) {
        bFound = true;
        break;
      }
    }
    if (!bFound)
       break;
  }
  // now pass the non existing bus id as string
  char pciBusIdstr[12];
  int device = -1;
  hipError_t ret;
  snprintf(pciBusIdstr, sizeof(pciBusIdstr), "%04x:%02x:%02x", pciDomainID[0],
           id, pciDeviceID[0]);
  ret = hipDeviceGetByPCIBusId(&device, pciBusIdstr);
  if (ret == hipSuccess) {
    TestPassed = false;
    printf("Test: hipDeviceGetByPCIBusId(&device,%s) Failed \n", pciBusIdstr);
  }
  return TestPassed;
}

int main(int argc, char* argv[]) {
  bool TestPassed = true;
  HipTest::parseStandardArguments(argc, argv, true);

  if (p_tests == 0x1) {
    TestPassed = testPciBusId();
  } else if (p_tests == 0x2) {
    TestPassed = testNullPtr();
  } else if (p_tests == 0x3) {
    TestPassed = testInputString();
  } else if (p_tests == 0x4) {
    TestPassed = testWrongBusID();
  } else {
    printf("Invalid Test Case \n");
    exit(1);
  }
  if (TestPassed) {
    passed();
  } else {
    failed("Test Case %x Failed!", p_tests);
  }
}
