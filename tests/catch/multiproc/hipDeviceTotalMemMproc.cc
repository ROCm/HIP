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
 * hipDeviceTotalMem tests
 * Scenario: Validate behavior of hipDeviceTotalMem for masked devices.
 */

#include <hip_test_common.hh>
#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>

#define MAX_SIZE 30
#define VISIBLE_DEVICE 0

/**
 * Fetches Gpu device count
 */
static void getDeviceCount(int *pdevCnt) {
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
 * Func tries to fetch total memory of masked devices and returns pass/fail.
 */
static bool getTotalMemoryOfMaskedDevices(int actualNumGPUs) {
  bool testResult = true;
  int fd[2];

  pipe(fd);
  pid_t cPid;
  cPid = fork();
  if (cPid == 0) {  // child
    hipError_t err;
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
      size_t totMem;
      err = hipDeviceTotalMem(&totMem, count);
      if (err == hipSuccess) {
        testResult &= false;
      } else {
        printf("hipDeviceTotalMem: Error Code Returned: '%s'(%d)\n",
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


/**
 * Scenario: Validate behavior of hipDeviceTotalMem for masked devices.
 */
TEST_CASE("Unit_hipDeviceTotalMem_MaskedDevices") {
  int count = -1;
  constexpr int ReqGPUs = 2;
  bool ret;

  getDeviceCount(&count);

  if (count >= ReqGPUs) {
    ret = getTotalMemoryOfMaskedDevices(count);
    REQUIRE(ret == true);
  } else {
    SUCCEED("Not enough GPUs to run the masked GPU tests");
  }
}

#endif
