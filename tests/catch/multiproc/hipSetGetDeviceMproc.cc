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

#include <hip_test_common.hh>

#ifdef __linux__
#include <sys/wait.h>
#include <unistd.h>


#define MAX_SIZE 30

/**
 * Fetches Gpu device count
 */
static void getDeviceCount(int *pdevCnt) {
  int fd[2], val = 0;
  pid_t childpid;

  // create pipe descriptors
  pipe(fd);

  // disable visible_devices env from shell
  unsetenv("ROCR_VISIBLE_DEVICES");
  unsetenv("HIP_VISIBLE_DEVICES");


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
  }
}


// Pass either -1 in deviceNumber or invalid device number
static void testInvalidDevice(int numDevices, bool useRocrEnv,
                                                   int deviceNumber) {
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
#ifdef __HIP_PLATFORM_NVCC__
    setenv("CUDA_VISIBLE_DEVICES", visibleDeviceString, 1);
#else
    if (true == useRocrEnv) {
      setenv("ROCR_VISIBLE_DEVICES", visibleDeviceString, 1);
    } else {
      setenv("HIP_VISIBLE_DEVICES", visibleDeviceString, 1);
    }
#endif
    err = hipGetDeviceCount(&tempCount);
    if (err != hipSuccess) {
      getDeviceCountErrorCheck = 1;
    }
    for (int i = 0; i < numDevices; i++) {
      err = hipSetDevice(i);
      if (err != hipSuccess) {
        setDeviceErrorCheck+= 1;
      }

      err = hipGetDevice(&device);
      if (err != hipSuccess) {
        getDeviceErrorCheck+= 1;
      }
    }

    if ((getDeviceCountErrorCheck == 1) && (setDeviceErrorCheck == numDevices)
        && (getDeviceErrorCheck == numDevices)) {
      testResult = true;

    } else {
      printf("Test failed for invalid device, getDeviceCountErrorCheck %d,"
             "setDeviceErrorCheck %d, getDeviceErrorCheck %d\n",
             getDeviceCountErrorCheck, setDeviceErrorCheck,
             getDeviceErrorCheck);

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
    HIP_ASSERT(false);
  }

  REQUIRE(testResult == true);
}


static void testValidDevices(int numDevices, bool useRocrEnv, int *deviceList,
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
    INFO("Invalid argument for number of devices. Skipping current test");
    REQUIRE(false);
  }

  for (int i = 0; i < deviceListLength; i++) {
    snprintf(visibleDeviceString + strlen(visibleDeviceString), MAX_SIZE, "%d,",
        *deviceListPtr++);
  }

  visibleDeviceString[strlen(visibleDeviceString)-1] = 0;

  int fd[2];
  pipe(fd);

  pid_t cPid;
  cPid = fork();

  if (cPid == 0) {
#ifdef __HIP_PLATFORM_NVCC__
    unsetenv("CUDA_VISIBLE_DEVICES");
    setenv("CUDA_VISIBLE_DEVICES", visibleDeviceString, 1);
#else
    unsetenv("ROCR_VISIBLE_DEVICES");
    unsetenv("HIP_VISIBLE_DEVICES");
    if (true == useRocrEnv) {
      setenv("ROCR_VISIBLE_DEVICES", visibleDeviceString, 1);
    } else {
      setenv("HIP_VISIBLE_DEVICES", visibleDeviceString, 1);
    }
#endif


    hipError_t err;
    err = hipGetDeviceCount(&tempCount);

    if (tempCount == deviceListLength) {
      getDeviceCountErrorCheck = 1;
    } else {
      printf("hipGetDeviceCount failed. return value: %u\n", hipError_t(err));
    }

    for (int i = 0; i < numDevices; i++) {
      err = hipSetDevice(i);
      if (err != hipSuccess) {
        setDeviceErrorCheck+= 1;
      }

      err = hipGetDevice(&device);
      if (err != hipSuccess) {
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
    HIP_ASSERT(false);
  }

  REQUIRE(testResult == true);
}


static void Initialize(int *deviceList, int numDevices, int count,
    char min_visibleDeviceString[], char max_visibleDeviceString[]) {
  int *deviceListPtr = deviceList;
  for (int i =0; i < count; i++) {
    if (i == count-1) {
      snprintf(min_visibleDeviceString + strlen(min_visibleDeviceString),
              MAX_SIZE, "%d", *deviceListPtr++);
    } else {
      snprintf(min_visibleDeviceString + strlen(min_visibleDeviceString),
              MAX_SIZE, "%d,", *deviceListPtr++);
    }
  }
  for (int i =0; i < numDevices; i++) {
    if (i == numDevices-1) {
      snprintf(max_visibleDeviceString + strlen(max_visibleDeviceString),
               MAX_SIZE, "%d", i);
    } else {
      snprintf(max_visibleDeviceString + strlen(max_visibleDeviceString),
               MAX_SIZE, "%d,", i);
    }
  }
}

static void testMaxRvdMinHvd(int numDevices, int *deviceList, int count) {
  bool testResult = true;
  int device;
  int validateCount = 0;
  char min_visibleDeviceString[MAX_SIZE] = {0};
  char max_visibleDeviceString[MAX_SIZE] = {0};
  int fd[2];
  pipe(fd);
  pid_t cPid;
  cPid = fork();
  if (cPid == 0) {  // child
    Initialize(deviceList, numDevices,
        count, min_visibleDeviceString, max_visibleDeviceString);
    unsetenv("ROCR_VISIBLE_DEVICES");
    unsetenv("HIP_VISIBLE_DEVICES");
    setenv("ROCR_VISIBLE_DEVICES", max_visibleDeviceString, 1);
    setenv("HIP_VISIBLE_DEVICES", min_visibleDeviceString, 1);
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipGetDevice(&device));
      if (device == i) {
         validateCount+= 1;
      }
    }
    if (count != validateCount) {
      testResult = false;
    }
  } else if (cPid > 0) {
    close(fd[1]);
    read(fd[0], &testResult, sizeof(testResult));
    close(fd[0]);
    wait(NULL);
  } else {
    printf("fork() failed\n");
    HIP_ASSERT(false);
  }

  REQUIRE(testResult == true);
}

static void testRvdCvd(int numDevices, int *deviceList, int count) {
  bool testResult = true;
  int device;
  int validateCount = 0;
  char min_visibleDeviceString[MAX_SIZE] = {0};
  char max_visibleDeviceString[MAX_SIZE] = {0};
  int fd[2];
  pipe(fd);
  pid_t cPid;
  cPid = fork();
  if (cPid == 0) {  // child
    Initialize(deviceList, numDevices, count,
             min_visibleDeviceString, max_visibleDeviceString);
    unsetenv("ROCR_VISIBLE_DEVICES");
    unsetenv("HIP_VISIBLE_DEVICES");
    setenv("ROCR_VISIBLE_DEVICES", max_visibleDeviceString, 1);
    setenv("CUDA_VISIBLE_DEVICES", min_visibleDeviceString, 1);
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipGetDevice(&device));
      if (device == i) {
        validateCount+= 1;
      }
    }
    if (count != validateCount) {
      testResult = false;
    }
  } else if (cPid > 0) {
    close(fd[1]);
    read(fd[0], &testResult, sizeof(testResult));
    close(fd[0]);
    wait(NULL);
  } else {
    printf("fork() failed\n");
    HIP_ASSERT(false);
  }

  REQUIRE(testResult == true);
}

static void testMinRvdMaxHvd(int numDevices, int *deviceList, int count) {
  bool testResult = true;
  int device;
  int validateCount = 0;
  char min_visibleDeviceString[MAX_SIZE] = {0};
  char max_visibleDeviceString[MAX_SIZE] = {0};
  int fd[2];
  pipe(fd);
  pid_t cPid;
  cPid = fork();
  if (cPid == 0) {  // child
    Initialize(deviceList, numDevices, count,
              min_visibleDeviceString, max_visibleDeviceString);
    unsetenv("ROCR_VISIBLE_DEVICES");
    unsetenv("HIP_VISIBLE_DEVICES");
    setenv("ROCR_VISIBLE_DEVICES", min_visibleDeviceString, 1);
    setenv("HIP_VISIBLE_DEVICES", max_visibleDeviceString, 1);
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipGetDevice(&device));
      if (device == i) {
         validateCount+= 1;
      }
    }
    if (count != validateCount) {
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
    HIP_ASSERT(false);
  }

  REQUIRE(testResult == true);
}

/**
 * Scenario sets Invalid visible device list and checks behavior.
 */
TEST_CASE("Unit_hipSetDevice_InvalidVisibleDeviceList") {
  int numDevices = 0;

  getDeviceCount(&numDevices);
  REQUIRE(numDevices != 0);

  SECTION("Test setting -1 to HIP_VISIBLE_DEVICES") {
    testInvalidDevice(numDevices, false, -1);
  }

  SECTION("Test setting invalid device to HIP_VISIBLE_DEVICES") {
    testInvalidDevice(numDevices, false, numDevices);
  }
#ifndef __HIP_PLATFORM_NVCC__
  SECTION("Test setting -1 to ROCR_VISIBLE_DEVICES") {
    testInvalidDevice(numDevices, true, -1);
  }

  SECTION("Test setting invalid device to ROCR_VISIBLE_DEVICES") {
    testInvalidDevice(numDevices, true, numDevices);
  }
#endif
}

/**
 * Scenario sets valid visible device list and checks behavior.
 */
TEST_CASE("Unit_hipSetDevice_ValidVisibleDeviceList") {
  int numDevices = 0;
  int deviceList[MAX_SIZE];

  getDeviceCount(&numDevices);
  REQUIRE(numDevices != 0);

  // Test for all available devices
  for (int i = 0; i < numDevices; i++) {
    deviceList[i] = i;
  }

  SECTION("Test setting valid hip visible device list") {
    testValidDevices(numDevices, false, deviceList, numDevices);
  }
#ifndef __HIP_PLATFORM_NVCC__
  SECTION("Test setting valid rocr visible device list") {
    testValidDevices(numDevices, true, deviceList, numDevices);
  }
#endif
}

/**
 * Scenario sets subset of available devices and checks behavior.
 */
TEST_CASE("Unit_hipSetDevice_SubsetOfAvailableDevices") {
  int numDevices = 0;
  int deviceList[MAX_SIZE];
  int deviceListLength = 1;

  getDeviceCount(&numDevices);
  REQUIRE(numDevices != 0);

  // Test for subset of available gpus
  for (int i=0; i < deviceListLength; i++) {
    deviceList[i] = deviceListLength-1-i;
  }

#ifndef __HIP_PLATFORM_NVCC__
  testValidDevices(numDevices, true, deviceList,
        deviceListLength);
#endif
  testValidDevices(numDevices, false, deviceList,
        deviceListLength);
}

#ifndef __HIP_PLATFORM_NVCC__
/* Following tests apply only for AMD Platforms */

/**
 * Scenario tests getDevice behavior with Minimal Len of RVD
 * and Maximal Len of HVD
 */
TEST_CASE("Unit_hipSetDevice_MinRvdMaxHvdDevicesList") {
  int numDevices = 0;
  int deviceList[MAX_SIZE];
  int count = 0;

  getDeviceCount(&numDevices);

  REQUIRE(numDevices != 0);

  if (numDevices == 1) {
    deviceList[0] = 0;
    count = 1;
  } else {
    for (int i=0; i < numDevices; i++) {
      if (i%2 == 0) {
        deviceList[count] = i;
        count++;
      }
    }
  }

  testMinRvdMaxHvd(numDevices, deviceList, count);
}

/**
 * Scenario tests getDevice behavior with Maximal Len of RVD
 * and Minimal Len of HVD
 */
TEST_CASE("Unit_hipSetDevice_MaxRvdMinHvdDevicesList") {
  int numDevices = 0;
  int deviceList[MAX_SIZE];
  int count = 0;

  getDeviceCount(&numDevices);

  REQUIRE(numDevices != 0);

  if (numDevices == 1) {
    deviceList[0] = 0;
    count = 1;
  } else {
    for (int i=0; i < numDevices; i++) {
      if (i%2 == 0) {
        deviceList[count] = i;
        count++;
      }
    }
  }

  testMaxRvdMinHvd(numDevices, deviceList, count);
}

/**
 * Scenario tests getDevice behavior with combination of RVD and CVD
 */
TEST_CASE("Unit_hipSetDevice_RvdCvdDevicesList") {
  int numDevices = 0;
  int deviceList[MAX_SIZE];
  int count = 0;

  getDeviceCount(&numDevices);

  REQUIRE(numDevices != 0);

  if (numDevices == 1) {
    deviceList[0] = 0;
    count = 1;
  } else {
    for (int i=0; i < numDevices; i++) {
      if (i%2 == 0) {
        deviceList[count] = i;
        count++;
      }
    }
  }

  testRvdCvd(numDevices, deviceList, count);
}
#endif  // __HIP_PLATFORM_NVCC__

#endif  // __linux__
