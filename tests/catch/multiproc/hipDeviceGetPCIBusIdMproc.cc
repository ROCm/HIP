/*
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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
 * Tests to
 * 1. Compare {pciDomainID, pciBusID, pciDeviceID} values
 *    hipDeviceGetPCIBusId vs lspci
 * 2. Validate behavior of hipDeviceGetPCIBusId for masked devices
 */

#include <hip_test_common.hh>
#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>

#define MAX_DEVICE_LENGTH 20
#define MAX_SIZE 30
#define VISIBLE_DEVICE 0

namespace hipDeviceGetPCIBusIdTests {

/**
 * Fetches Gpu device count
 */
void getDeviceCount(int *pdevCnt) {
  int fd[2], val = 0;
  pid_t childpid;

  // create pipe descriptors
  pipe(fd);

  // disable visible_devices env from shell
#ifdef __HIP_PLATFORM_NVCC__
  unsetenv("CUDA_VISIBLE_DEVICES");
#else
  unsetenv("ROCR_VISIBLE_DEVICES");
  unsetenv("HIP_VISIBLE_DEVICES");
#endif

  childpid = fork();

  if (childpid > 0) {  // Parent
    close(fd[1]);
    // parent will wait to read the device cnt
    read(fd[0], &val, sizeof(val));

    // close the read-descriptor
    close(fd[0]);

    // wait for child exit
    wait(NULL);

    *pdevCnt = val;
  } else if (!childpid) {  // Child
    int devCnt = 1;
    // writing only, no need for read-descriptor
    close(fd[0]);

    HIP_CHECK(hipGetDeviceCount(&devCnt));
    // send the value on the write-descriptor:
    write(fd[1], &devCnt, sizeof(devCnt));

    // close the write descriptor:
    close(fd[1]);
    exit(0);
  } else {  // failure
    *pdevCnt = 0;
    return;
  }
}

/**
 * Runs test on masked devices
 */
bool testWithMaskedDevices(int actualNumGPUs) {
  bool testResult = true;
  int fd[2];
  pipe(fd);
  pid_t cPid;
  cPid = fork();
  if (cPid == 0) {  // child
    hipError_t err;
    char pciBusId[MAX_DEVICE_LENGTH];
    char visibleDeviceString[MAX_SIZE] = {};
    snprintf(visibleDeviceString, MAX_SIZE, "%d", VISIBLE_DEVICE);

    // disable visible_devices env from shell
#ifdef __HIP_PLATFORM_NVCC__
    unsetenv("CUDA_VISIBLE_DEVICES");
    setenv("CUDA_VISIBLE_DEVICES", visibleDeviceString, 1);
    HIP_CHECK(hipInit(0));
#else
    unsetenv("ROCR_VISIBLE_DEVICES");
    unsetenv("HIP_VISIBLE_DEVICES");
    setenv("ROCR_VISIBLE_DEVICES", visibleDeviceString, 1);
    setenv("HIP_VISIBLE_DEVICES", visibleDeviceString, 1);
#endif

    for (int count = 1;
        count < actualNumGPUs; count++) {
      err = hipDeviceGetPCIBusId(pciBusId, MAX_DEVICE_LENGTH, count);
      if (err == hipSuccess) {
        testResult &= false;
      } else {
        printf("hipGetDeviceProperties: Error Code Returned: '%s'(%d)\n",
              hipGetErrorString(err), err);
      }
    }
    close(fd[0]);
    printf("testResult = %d \n", testResult);
    write(fd[1], &testResult, sizeof(testResult));
    close(fd[1]);
    exit(0);

  } else if (cPid > 0) {  // parent
    close(fd[1]);
    read(fd[0], &testResult, sizeof(testResult));
    close(fd[0]);
    wait(NULL);

  } else {
    printf("fork() failed\n");
    HIP_ASSERT(false);
  }

  return testResult;
}


bool getPciBusId(int deviceCount,
                 char **hipDeviceList) {
  for (int i = 0; i < deviceCount; i++) {
    HIP_CHECK(hipDeviceGetPCIBusId(hipDeviceList[i], MAX_DEVICE_LENGTH, i));
  }
  return true;
}
}  // namespace hipDeviceGetPCIBusIdTests


/**
 * Scenario: Validate behavior of hipDeviceGetPCIBusId for masked devices.
 */
TEST_CASE("Unit_hipDeviceGetPCIBusId_MaskedDevices") {
  int count = -1;
  constexpr int ReqGPUs = 2;
  bool ret;

  hipDeviceGetPCIBusIdTests::getDeviceCount(&count);

  if (count >= ReqGPUs) {
    ret = hipDeviceGetPCIBusIdTests::testWithMaskedDevices(count);
    REQUIRE(ret == true);
  } else {
    SUCCEED("Not enough GPUs to run the masked GPU tests");
  }
}

/* Compare {pciDomainID, pciBusID, pciDeviceID} values
 * hipDeviceGetPCIBusId vs lspci
 */
TEST_CASE("Unit_hipDeviceGetPCIBusId_CheckPciBusIDWithLspci") {
  FILE *fpipe;
  {
    // Check if lspci is installed, if not, don't proceed
    char const *cmd = "lspci --version";
    char *lspciCheck{nullptr};
    constexpr auto MaxLen = 50;
    char temp[MaxLen]{};

    fpipe = popen(cmd, "r");
    REQUIRE_FALSE(fpipe == nullptr);

    lspciCheck = fgets(temp, MaxLen, fpipe);
    pclose(fpipe);

    if (lspciCheck == nullptr) {
      WARN("Skipping test as lspci is not found in system");
      return;
    }
  }

  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE_FALSE(deviceCount == 0);
  // Allocate an array of pointer to characters
  char **hipDeviceList = new char*[deviceCount];
  REQUIRE_FALSE(hipDeviceList == nullptr);
  char **pciDeviceList = new char*[deviceCount];
  REQUIRE_FALSE(pciDeviceList == nullptr);
  for (int i = 0; i < deviceCount; i++) {
    hipDeviceList[i] = new char[MAX_DEVICE_LENGTH];
    REQUIRE_FALSE(hipDeviceList[i] == nullptr);
    pciDeviceList[i] = new char[MAX_DEVICE_LENGTH];
    REQUIRE_FALSE(pciDeviceList[i] == nullptr);
  }

  hipDeviceGetPCIBusIdTests::getPciBusId(deviceCount, hipDeviceList);
  char const *command = nullptr;
  // Get lspci device list and compare with hip device list
  if ((TestContext::get()).isNvidia()) {
    command = "lspci -D | grep controller | grep NVIDIA | "
              "cut -d ' ' -f 1";
  } else {
    command = "lspci -D | grep controller | grep AMD/ATI | "
              "cut -d ' ' -f 1";
  }
  fpipe = popen(command, "r");
  REQUIRE_FALSE(fpipe == nullptr);

  int index = 0;
  int deviceMatchCount = 0;
  constexpr auto cmpLen = 10;
  while (fgets(pciDeviceList[index], MAX_DEVICE_LENGTH, fpipe)) {
    bool bMatchFound = false;
    for (int deviceNo = 0; deviceNo < deviceCount; deviceNo++) {
      if (!strncasecmp(pciDeviceList[index], hipDeviceList[deviceNo],
                                                             cmpLen)) {
        deviceMatchCount++;
        bMatchFound = true;
      }
    }
    if (bMatchFound == false) {
      printf("PCI device: %s is not reported by HIP\n",
                                   pciDeviceList[index]);
    }
    index++;
    if (index >= deviceCount) break;
  }
  // Deallocate
  for (int i = 0; i < deviceCount; i++) {
    delete hipDeviceList[i];
  }
  delete[] hipDeviceList;
  for (int i = 0; i < deviceCount; i++) {
    delete pciDeviceList[i];
  }
  delete[] pciDeviceList;
  pclose(fpipe);

  REQUIRE(deviceMatchCount == deviceCount);
}
#endif
