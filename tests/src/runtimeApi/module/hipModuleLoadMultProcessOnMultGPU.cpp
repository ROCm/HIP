/*
Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD_CMD: kernel_composite_test.code %hc --genco %S/kernel_composite_test.cpp -o kernel_composite_test.code
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t --tests 0x1
 * TEST: %t --tests 0x2
 * TEST: %t --tests 0x3
 * HIT_END
 */
#include <stdio.h>
#include <stdlib.h>

#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#endif
#include <iostream>
#include <fstream>
#include <cstddef>
#include <vector>
#include "test_common.h"

#define TEST_ITERATIONS 1000
#define CODEOBJ_FILE "kernel_composite_test.code"
#define CODEOBJ_GLOB_KERNEL1 "testWeightedCopy"
#define CODEOBJ_GLOB_KERNEL2 "getAvg"
#define BLOCKSPERCULDULD 6
#define THREADSPERBLOCKLDULD 256

unsigned int globTestID = 0;

/**
 * Fetches Gpu device count
 */
void getDeviceCount(int *pdevCnt) {
#ifdef __linux__
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

    HIPCHECK(hipGetDeviceCount(&devCnt));
    // send the value on the write-descriptor:
    write(fd[1], &devCnt, sizeof(devCnt));

    // close the write descriptor:
    close(fd[1]);
    exit(0);
  } else {  // failure
    *pdevCnt = 1;
    return;
  }

#else
  HIPCHECK(hipGetDeviceCount(pdevCnt));
#endif
}

/**
 * Validates hipModuleLoadUnload if globTestID = 1
 * Validates hipModuleLoadDataUnload if globTestID = 2
 * Validates hipModuleLoadDataExUnload if globTestID = 3
 */
bool testhipModuleLoadUnloadFunc(const std::vector<char>& buffer) {
  size_t N = 16*16;
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d;
  int *A_h, *B_h;
  unsigned blocks = HipTest::setNumBlocks(BLOCKSPERCULDULD,
                             THREADSPERBLOCKLDULD, N);
  int deviceid;
  hipGetDevice(&deviceid);
  printf("pid = %u deviceid = %d\n", getpid(), deviceid);
  // allocate host and device buffer
  HIPCHECK(hipMalloc(&A_d, Nbytes));
  HIPCHECK(hipMalloc(&B_d, Nbytes));

  A_h = reinterpret_cast<int *>(malloc(Nbytes));
  if (NULL == A_h) {
    failed("Failed to allocate using malloc");
  }
  B_h = reinterpret_cast<int *>(malloc(Nbytes));
  if (NULL == B_h) {
    failed("Failed to allocate using malloc");
  }
  // set host buffers
  for (int idx = 0; idx < N; idx++) {
    A_h[idx] = deviceid;
  }
  // Copy buffer from host to device

  HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

  hipModule_t Module;
  hipFunction_t Function;
  if (1 == globTestID) {
    HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  } else if (2 == globTestID) {
    HIPCHECK(hipModuleLoadData(&Module, &buffer[0]));
  } else if (3 == globTestID) {
    HIPCHECK(hipModuleLoadDataEx(&Module,
            &buffer[0], 0, nullptr, nullptr));
  }
  HIPCHECK(hipModuleGetFunction(&Function, Module,
                               CODEOBJ_GLOB_KERNEL1));
  float deviceGlobalFloatH = 3.14;
  int   deviceGlobalInt1H = 100*deviceid;
  int   deviceGlobalInt2H = 50*deviceid;
  short deviceGlobalShortH = 25*deviceid;
  char  deviceGlobalCharH = 13*deviceid;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
  HIPCHECK(hipModuleGetGlobal(&deviceGlobal,
           &deviceGlobalSize,
           Module, "deviceGlobalFloat"));
  HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal),
           &deviceGlobalFloatH,
           deviceGlobalSize));
  HIPCHECK(hipModuleGetGlobal(&deviceGlobal,
           &deviceGlobalSize,
           Module, "deviceGlobalInt1"));
  HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal),
           &deviceGlobalInt1H,
           deviceGlobalSize));
  HIPCHECK(hipModuleGetGlobal(&deviceGlobal,
           &deviceGlobalSize,
           Module,
           "deviceGlobalInt2"));
  HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal),
           &deviceGlobalInt2H, deviceGlobalSize));
  HIPCHECK(hipModuleGetGlobal(&deviceGlobal,
           &deviceGlobalSize,
           Module, "deviceGlobalShort"));
  HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal),
           &deviceGlobalShortH, deviceGlobalSize));
  HIPCHECK(hipModuleGetGlobal(&deviceGlobal,
           &deviceGlobalSize, Module, "deviceGlobalChar"));
  HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal),
           &deviceGlobalCharH, deviceGlobalSize));
  // Launch Function kernel function

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  struct {
    void* _Ad;
    void* _Bd;
  } args;
  args._Ad = reinterpret_cast<void*>(A_d);
  args._Bd = reinterpret_cast<void*>(B_d);
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  HIPCHECK(hipModuleLaunchKernel(Function, 1, 1, 1,
            N, 1, 1, 0, stream, NULL,
            reinterpret_cast<void**>(&config)));
  // Copy buffer from decice to host
  HIPCHECK(hipMemcpyAsync(B_h, B_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIPCHECK(hipDeviceSynchronize());
  HIPCHECK(hipStreamDestroy(stream));

  // Check the results
  for (int idx = 0; idx < N; idx++) {
    if (B_h[idx] != (deviceGlobalInt1H*A_h[idx]
            + deviceGlobalInt2H
            + static_cast<int>(deviceGlobalShortH) +
            + static_cast<int>(deviceGlobalCharH)
            + static_cast<int>(deviceGlobalFloatH*deviceGlobalFloatH))) {
        printf("Matrix Addition Failed\n");
        // exit the current process with failure
        return false;
    }
  }
  HIPCHECK(hipModuleUnload(Module));
  // free memory
  HIPCHECK(hipFree(B_d));
  HIPCHECK(hipFree(A_d));
  free(B_h);
  free(A_h);
  printf("pid:%u PASSED\n", getpid());
  return true;
}

/**
 * Spawn 1 Process for each device
 *
 */
void spawnProc(int deviceCount, const std::vector<char>& buffer) {
  int numDevices = deviceCount;
  bool TestPassed = true;
#ifdef __linux__
  pid_t pid = 0;
  // spawn a process for each device
  for (int deviceNo = 0; deviceNo < numDevices; deviceNo++) {
    if ((pid = fork()) < 0) {
      printf("Child_Concurrency_MultiGpu : fork() returned error %d\n",
             pid);
      failed("Test Failed!");
    } else if (!pid) {  // Child process
      bool TestPassedChild = true;
      // set the device id for the current process
      HIPCHECK(hipSetDevice(deviceNo));
      TestPassedChild = testhipModuleLoadUnloadFunc(buffer);

      if (TestPassedChild) {
        exit(0);  // child exit with success status
      } else {
        printf("Child_Concurrency_MultiGpu : childpid %d failed\n",
               getpid());
        exit(1);  // child exit with failure status
      }
    }
  }
  int cumStatus = 0;
  // Parent shall wait for child to complete
  for (int i = 0; i < numDevices; i++) {
    int pidwait = 0, exitStatus;
    pidwait = wait(&exitStatus);
    cumStatus |= WEXITSTATUS(exitStatus);
  }
  if (cumStatus) {
    TestPassed &= false;
  }
#else
  for (int deviceNo = 0; deviceNo < numDevices; deviceNo++) {
    // set the device id for the current process
    HIPCHECK(hipSetDevice(deviceNo));
    TestPassed &= testhipModuleLoadUnloadFunc(buffer);
  }
#endif
  if (TestPassed) {
    passed();
  } else {
    failed("hipMallocChild_Concurrency_MultiGpu Failed!");
  }
}

int main(int argc, char* argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  int numDevices = 0;
  getDeviceCount(&numDevices);
  if (1 == numDevices) {
    printf("Testing on Single GPU machine.\n");
  }
  std::ifstream file(CODEOBJ_FILE,
                    std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    failed("could not open code object '%s'\n", CODEOBJ_FILE);
  }
  file.close();
  if (p_tests == 0x1) {
    globTestID = 1;
    spawnProc(numDevices, buffer);
  } else if (p_tests == 0x2) {
    globTestID = 2;
    spawnProc(numDevices, buffer);
  } else if (p_tests == 0x3) {
    globTestID = 3;
    spawnProc(numDevices, buffer);
  }
}
