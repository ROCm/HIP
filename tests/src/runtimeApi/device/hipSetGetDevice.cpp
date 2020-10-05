/*
 * Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
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
 * Test designed to run on Linux based platforms
 * Verifies functionality of
 * -- hipSetDevice and hipGetDevice with different ROCR_VISIBLE_DEVICES and
 *    HIP_VISIBLE_DEVICES values set
 */

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11 EXCLUDE_HIP_PLATFORM nvcc
 * TEST_NAMED: %t hipSetGetDevice-invalidDevice
 * TEST_NAMED: %t hipSetGetDevice-allValidDevice
 * TEST_NAMED: %t hipSetGetDevice-validDev1 --computeDevCnt 1
 * TEST_NAMED: %t hipSetGetDevice-validDev2 --computeDevCnt 2
 * TEST_NAMED: %t hipSetGetDevice-validDev3 --computeDevCnt 3
 * TEST_NAMED: %t hipSetGetDevice-validDev4 --computeDevCnt 4
 * TEST_NAMED: %t hipSetGetDevice-validDev5 --computeDevCnt 5
 * TEST_NAMED: %t hipSetGetDevice-validDev6 --computeDevCnt 6
 * TEST_NAMED: %t hipSetGetDevice-validDev7 --computeDevCnt 7
 * TEST_NAMED: %t hipSetGetDevice-validDev8 --computeDevCnt 8
 * HIT_END
 */

#include "test_common.h"
#include <sys/wait.h>
#include <unistd.h>

void getDeviceCount(int *numDevices) {
  int fd[2], val = 0;
  pipe(fd);

  pid_t childPid;
  childPid = fork();

  if (childPid > 0) {  // parent
    close(fd[1]);
    read(fd[0], &val, sizeof(val));
    close(fd[0]);
    *numDevices = val;

  } else if (childPid == 0) {   // child
    int devCnt = 0;
    close(fd[0]);

    unsetenv("ROCR_VISIBLE_DEVICES");
    unsetenv("HIP_VISIBLE_DEVICES");

    hipGetDeviceCount(&devCnt);

    write(fd[1], &devCnt, sizeof(devCnt));
    close(fd[1]);
    exit(0);

  } else {
    failed("fork() failed. Exiting the test\n");
  }
}

#define MAX_SIZE 30

// Pass either -1 in deviceNumber or invalid device number
bool testInvalidDevice(int numDevices, bool useRocrEnv, int deviceNumber) {
  bool testResult = true;
  int device;
  int tempCount = 0;
  int setDeviceErrorCheck = 0;
  int getDeviceErrorCheck = 0;
  int getDeviceCountErrorCheck = 0;
  int fd[2];
  pipe(fd);

  pid_t cPid;
  cPid = fork();

  char visibleDeviceString[MAX_SIZE] = {};
  snprintf(visibleDeviceString, MAX_SIZE, "%d", deviceNumber);

  if (cPid == 0) {  // child
    hipError_t err;

    if (true == useRocrEnv) {
      setenv("ROCR_VISIBLE_DEVICES", visibleDeviceString, 1);
    } else {
      setenv("HIP_VISIBLE_DEVICES", visibleDeviceString, 1);
    }

    err = hipGetDeviceCount(&tempCount);
    if (err == hipErrorNoDevice) {
      getDeviceCountErrorCheck = 1;
    } else {
      printf("hipGetDeviceCount returns wrong value: %u\n", hipError_t(err));
    }

    for (int i = 0; i < numDevices; i++) {
      err = hipSetDevice(i);
      if (err == hipErrorInvalidDevice) {
        setDeviceErrorCheck+= 1;
      } else {
        printf("hipSetDevice returns wrong value: %u\n", hipError_t(err));
      }

      err = hipGetDevice(&device);
      if (err == hipErrorNoDevice) {
        getDeviceErrorCheck+= 1;
      } else {
        printf("hipGetDevice returns wrong value: %u\n", hipError_t(err));
      }
    }

    if ((getDeviceCountErrorCheck == 1) && (setDeviceErrorCheck == numDevices)
        && (getDeviceErrorCheck == numDevices)) {
       testResult = true;

    } else {
      printf("Test failed for invalid device\n");
      testResult = false;
    }

    close(fd[0]);
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
    testResult = false;
  }
  return testResult;
}

int deviceListLength = 1;
int parseExtraArguments(int argc, char* argv[]) {
  int i = 0;
  for (i = 1; i < argc; i++) {
    const char* arg = argv[i];
    if (!strcmp(arg, " ")) {
      // skip NULL args.
    } else if (!strcmp(arg, "--computeDevCnt")) {
        if (++i >= argc || !HipTest::parseInt(argv[i], &deviceListLength)) {
          failed("Bad deviceListLength argument");
        }
    } else {
        failed("Bad argument");
    }
  }
  return i;
}

bool testValidDevices(int numDevices, bool useRocrEnv, int *deviceList,
                      int deviceListLength) {
  bool testResult = true;
  int tempCount = 0;
  int device;
  int setDeviceErrorCheck = 0;
  int getDeviceErrorCheck = 0;
  int getDeviceCountErrorCheck = 0;
  int *deviceListPtr = deviceList;
  char visibleDeviceString[MAX_SIZE] = {};

  if ((NULL == deviceList) || ((deviceListLength < 1) ||
      deviceListLength > numDevices)) {
    printf("Invalid argument for number of devices. Skipping current test\n");
    return testResult;
  }

  for (int i = 0; i < deviceListLength; i++) {
    if (NULL == deviceListPtr) {
      printf("Invalid gpu index. Skipping current test\n");
      return testResult;
    }
    snprintf(visibleDeviceString + strlen(visibleDeviceString), MAX_SIZE, "%d,",
            *deviceListPtr++);
  }

  visibleDeviceString[strlen(visibleDeviceString)-1] = 0;

  int fd[2];
  pipe(fd);

  pid_t cPid;
  cPid = fork();

  if (cPid == 0) {
    unsetenv("ROCR_VISIBLE_DEVICES");
    unsetenv("HIP_VISIBLE_DEVICES");

    if (true == useRocrEnv) {
      setenv("ROCR_VISIBLE_DEVICES", visibleDeviceString, 1);
    } else {
      setenv("HIP_VISIBLE_DEVICES", visibleDeviceString, 1);
    }

    hipError_t err;
    err = hipGetDeviceCount(&tempCount);

    if (tempCount == deviceListLength) {
      getDeviceCountErrorCheck = 1;
    } else {
      printf("hipGetDeviceCount failed. return value: %u\n", hipError_t(err));
    }

    for (int i = 0; i < numDevices; i++) {
      err = hipSetDevice(i);
      if (err == hipErrorInvalidDevice) {
        setDeviceErrorCheck+= 1;
      }

      err = hipGetDevice(&device);
      if (err == hipErrorNoDevice) {
        getDeviceErrorCheck+= 1;
      }
    }

    if ((getDeviceCountErrorCheck == 1) && (setDeviceErrorCheck ==
        (numDevices-deviceListLength)) && (getDeviceErrorCheck == 0)) {
      testResult = true;

    } else {
      printf("Test failed for device count %d\n", deviceListLength);
      testResult = false;
    }

    close(fd[0]);
    write(fd[1], &testResult, sizeof(testResult));
    close(fd[1]);
    exit(0);

  } else if (cPid > 0) {
    close(fd[1]);
    read(fd[0], &testResult, sizeof(testResult));
    close(fd[0]);
    wait(NULL);

  } else {
    printf("fork() failed\n");
    testResult = false;
  }
  return testResult;
}

bool testValidDevicesBasic() {
  bool testResult = true;
  int numDevices = 0;
  int device;
  int validateCount = 0;
  HIPCHECK(hipGetDeviceCount(&numDevices));
  printf("Available compute devices in the system: %d\n", numDevices);

  for (int i = 0; i < numDevices; i++) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipGetDevice(&device));
    if (device == i) {
      validateCount+= 1;
    }
  }
  if (numDevices != validateCount) {
    testResult = false;
  }

  return testResult;
}

int main(int argc, char* argv[]) {
  bool testResult = true;
  int numDevices = 0;
  int device;
  int deviceList[MAX_SIZE];
  int extraArgs = 0;

#ifdef __unix__
  getDeviceCount(&numDevices);

  if (numDevices == 0) {
    failed("No gpus found. exiting\n");
  }

  printf("Available compute devices in the system: %d\n", numDevices);

  extraArgs = HipTest::parseStandardArguments(argc, argv, false);
  parseExtraArguments(extraArgs, argv);

  if (extraArgs == 1) {
    printf("\nRunning test for invalid compute device\n");
    // Test setting -1 to ROCR_VISIBLE_DEVICES
    testResult &= testInvalidDevice(numDevices, true, -1);
    // Test setting -1 to HIP_VISIBLE_DEVICES
    testResult &= testInvalidDevice(numDevices, false, -1);
    // Test setting invalid device to ROCR_VISIBLE_DEVICES
    testResult &= testInvalidDevice(numDevices, true, numDevices);
    // Test setting invalide device to HIP_VISIBLE_DEVICES
    testResult &= testInvalidDevice(numDevices, false, numDevices);

    // Test for all available devices
    printf("\nRunning test for all available compute devices\n");

    for (int i = 0; i < numDevices; i++) {
      deviceList[i] = i;
    }

    testResult &= testValidDevices(numDevices, true, deviceList, numDevices);
    testResult &= testValidDevices(numDevices, false, deviceList, numDevices);
  }

  // Assigning values to deviceList in reverse order

  for (int i=0; i < deviceListLength; i++) {
    deviceList[i] = deviceListLength-1-i;
  }

  // Test for subset of available gpus
  if (extraArgs == 3) {
    printf("\nRunning test for %d compute devices\n", deviceListLength);
    testResult &= testValidDevices(numDevices, true, deviceList,
                                   deviceListLength);
    testResult &= testValidDevices(numDevices, false, deviceList,
                                   deviceListLength);
  }

#else

  printf("Running basic test on Windows\n");
  testResult &= testValidDevicesBasic();

#endif

  if (testResult == true) {
    passed();

  } else {
    failed("One or more tests failed\n");
  }
}
