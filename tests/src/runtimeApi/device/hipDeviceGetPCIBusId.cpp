/*
 * Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.
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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST_NAMED: %t  hipDeviceGetPCIBusId-vs-hipDeviceGetAttribute --tests 0x1
 * TEST_NAMED: %t  hipDeviceGetPCIBusId-vs-lspci --tests 0x2
 * HIT_END
 */

#include "test_common.h"
#define MAX_DEVICE_LENGTH 20

static bool getPciBusId(int deviceCount, char hipDeviceList[][MAX_DEVICE_LENGTH]) {
  for (int i = 0; i < deviceCount; i++) {
    HIPCHECK(hipDeviceGetPCIBusId(hipDeviceList[i], MAX_DEVICE_LENGTH, i));
  }
  return true;
}

bool comparePciBusIDWithHipDeviceGetAttribute() {
  bool testResult = true;
  int deviceCount = 0;
  HIPCHECK(hipGetDeviceCount(&deviceCount));
  HIPASSERT(deviceCount != 0);
  printf("No.of gpus in the system: %d\n", deviceCount);
  char hipDeviceList[deviceCount][MAX_DEVICE_LENGTH];
  char pciDeviceList[deviceCount][MAX_DEVICE_LENGTH];

  getPciBusId(deviceCount, hipDeviceList);

  for (int i = 0; i < deviceCount; i++) {
    int pciBusID = -1;
    int pciDeviceID = -1;
    int pciDomainID = -1;
    int tempPciBusId = -1;
    sscanf(hipDeviceList[i], "%04x:%02x:%02x", &pciDomainID, &pciBusID,
           &pciDeviceID);
    HIPCHECK(hipDeviceGetAttribute(&tempPciBusId, hipDeviceAttributePciBusId, i));
    if (pciBusID != tempPciBusId) {
      testResult = false;
      printf("pciBusID from hipDeviceGetPCIBusId mismatched to that from "
             "hipDeviceGetAttribute for gpu %d\n", i);
    }
  }

  printf("pciBusID output of both hipDeviceGetPCIBusId and"
         " hipDeviceGetAttribute matched for all gpus\n");
  return testResult;
}

bool compareHipDeviceGetPCIBusIdWithLspci() {
  FILE *fpipe;
  bool testResult = false;

  {
    // Check if lspci is installed, if not, don't proceed
    char const *cmd = "lspci --version";
    char *lspciCheck;
    char temp[20];
    fpipe = popen(cmd, "r");

    if (fpipe == nullptr) {
      printf("Unable to create command file\n");
      return testResult;
    }

    lspciCheck = fgets(temp, 20, fpipe);
    pclose(fpipe);

    if (!lspciCheck) {
      printf("lspci not found. Skipping the test\n");
      return true;
    }
  }

  int deviceCount = 0;
  HIPCHECK(hipGetDeviceCount(&deviceCount));
  HIPASSERT(deviceCount != 0);
  printf("No.of gpus in the system: %d\n", deviceCount);
  char hipDeviceList[deviceCount][MAX_DEVICE_LENGTH];
  char pciDeviceList[deviceCount][MAX_DEVICE_LENGTH];

  getPciBusId(deviceCount, hipDeviceList);

  // Get lspci device list and compare with hip device list
  char const *command = "lspci -D | grep controller | grep AMD/ATI | "
                        "cut -d ' ' -f 1";
  fpipe = popen(command, "r");

  if (fpipe == nullptr) {
    printf("Unable to create command file\n");
    return testResult;
  }

  int index = 0;
  int deviceMatchCount = 0;

  while (fgets(pciDeviceList[index], sizeof(pciDeviceList[index]), fpipe)) {
    bool bMatchFound = false;
    for (int deviceNo = 0; deviceNo < deviceCount; deviceNo++) {
      if (!strncmp(pciDeviceList[index], hipDeviceList[deviceNo], 10)) {
        deviceMatchCount++;
        bMatchFound = true;
      }
    }
    if (bMatchFound == false) {
      printf("PCI device: %s is not reported by HIP\n", pciDeviceList[index]);
    }
    index++;
  }

  pclose(fpipe);

  if (deviceMatchCount == deviceCount) {
    printf("hip and lspci output for {pciDomainID, pciBusID, pciDeviceID} "
           "matched for all gpus\n");
    testResult = true;
  } else {
    printf("Mismatch in number GPUs reported by HIP with lscpi\n");
  }
  return testResult;
}

int main(int argc, char* argv[]) {
  bool testResult = true;
  HipTest::parseStandardArguments(argc, argv, true);

  if (p_tests & 0x1) {
    testResult &= comparePciBusIDWithHipDeviceGetAttribute();
  }

  if (p_tests & 0x2) {
#ifdef __unix__
    testResult &= compareHipDeviceGetPCIBusIdWithLspci();
#else
    printf("Detected non-linux OS. Skipping the test\n");
#endif
  }

  if (testResult) {
    passed();
  } else {
    failed("one or more tests failed\n");
  }
}
