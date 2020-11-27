/*
Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * Test for transferring data beween devices using host pinned memory
 */

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST_NAMED: %t d2dMemCpyWithPinnedHostMemory_pinned --pinned
 * TEST_NAMED: %t d2dMemCpyWithPinnedHostMemory_registered --registered
 * HIT_END
 */

#include <stdlib.h>
#include "test_common.h"
#define N 1000000

enum MallopinType {mallocPinned, mallocRegistered, mallocNone};

MallopinType p_malloc_mode = mallocNone;

int *Ad0{nullptr}, *Bd0{nullptr}, *Cd0{nullptr}, *Ad1{nullptr},
    *Bd1{nullptr}, *Cd1{nullptr};
int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};

void memAllocate(MallopinType pinType) {
  size_t Nbytes = N * sizeof(int);
  if (pinType == mallocPinned) {
    std::cout << "Allocating pinned host memory\n";
    HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), Nbytes));
    HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&B_h), Nbytes));
    HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&C_h), Nbytes));

  } else {
    std::cout << "Allocating registered host memory\n";
    A_h = reinterpret_cast<int*>(malloc(Nbytes));
    HIPCHECK(hipHostRegister(A_h, Nbytes, hipHostRegisterDefault));
    B_h = reinterpret_cast<int*>(malloc(Nbytes));
    HIPCHECK(hipHostRegister(B_h, Nbytes, hipHostRegisterDefault));
    C_h = reinterpret_cast<int*>(malloc(Nbytes));
    HIPCHECK(hipHostRegister(C_h, Nbytes, hipHostRegisterDefault));
  }

  HIPCHECK(hipSetDevice(0));
  HIPCHECK(hipMalloc(&Ad0, Nbytes));
  HIPCHECK(hipMalloc(&Bd0, Nbytes));
  HIPCHECK(hipMalloc(&Cd0, Nbytes));
}

void memClear(MallopinType pinType) {
  if (pinType == mallocPinned) {
    HIPCHECK(hipHostFree(A_h));
    HIPCHECK(hipHostFree(B_h));
    HIPCHECK(hipHostFree(C_h));
  } else {
    HIPCHECK(hipHostUnregister(A_h));
    free(A_h);
    HIPCHECK(hipHostUnregister(B_h));
    free(B_h);
    HIPCHECK(hipHostUnregister(C_h));
    free(C_h);
  }

  HIPCHECK(hipFree(Ad0));
  HIPCHECK(hipFree(Bd0));
  HIPCHECK(hipFree(Cd0));
  HIPCHECK(hipFree(Ad1));
  HIPCHECK(hipFree(Bd1));
  HIPCHECK(hipFree(Cd1));
}

bool testMemCopy(int gpuCnt, MallopinType pinType) {
  size_t Nbytes = N * sizeof(int);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  memAllocate(pinType);

  for (int i = 0; i < N; i++) {
      A_h[i] = i;
      B_h[i] = i;
  }

  HIPCHECK(hipMemcpy(Ad0, A_h, Nbytes, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(Bd0, B_h, Nbytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, 0, static_cast<const int*>(Ad0),
                     static_cast<const int*>(Bd0), Cd0, N);

  HIPCHECK(hipMemcpy(C_h, Cd0, Nbytes, hipMemcpyDeviceToHost));
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  unsigned int seed = time(0);
  HIPCHECK(hipSetDevice(rand_r(&seed) % (gpuCnt-1)+1));

  int device;
  hipGetDevice(&device);
  std::cout <<"hipMemcpy is set to happen between device 0 and device "
            <<device << std::endl;

  HIPCHECK(hipMalloc(&Ad1, Nbytes));
  HIPCHECK(hipMalloc(&Bd1, Nbytes));
  HIPCHECK(hipMalloc(&Cd1, Nbytes));

  for (int j = 0; j < N; j++) {
      A_h[j] = 0;
      B_h[j] = 0;
      C_h[j] = 0;
  }

  hipMemcpy(A_h, Ad0, Nbytes, hipMemcpyDeviceToHost);
  hipMemcpy(Ad1, A_h, Nbytes, hipMemcpyHostToDevice);
  hipMemcpy(B_h, Bd0, Nbytes, hipMemcpyDeviceToHost);
  hipMemcpy(Bd1, B_h, Nbytes, hipMemcpyHostToDevice);

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, 0, static_cast<const int*>(Ad1),
                     static_cast<const int*>(Bd1), Cd1, N);

  HIPCHECK(hipMemcpy(C_h, Cd1, Nbytes, hipMemcpyDeviceToHost));

  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  memClear(pinType);
  return true;
}

bool testMemCopyAsync(int gpuCnt, MallopinType pinType) {
  size_t Nbytes = N * sizeof(int);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  memAllocate(pinType);

  for (int i = 0; i < N; i++) {
      A_h[i] = i;
      B_h[i] = i;
  }

  HIPCHECK(hipMemcpy(Ad0, A_h, Nbytes, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(Bd0, B_h, Nbytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, 0, static_cast<const int*>(Ad0),
                     static_cast<const int*>(Bd0), Cd0, N);

  HIPCHECK(hipMemcpy(C_h, Cd0, Nbytes, hipMemcpyDeviceToHost));

  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  // Setting random gpu from all available gpus except gpu0
  unsigned int seed = time(0);
  HIPCHECK(hipSetDevice(rand_r(&seed) % (gpuCnt-1)+1));

  int device;
  hipGetDevice(&device);
  std::cout <<"hipMemcpyAsync is set to happen between device 0 and device "
            <<device << std::endl;

  HIPCHECK(hipMalloc(&Ad1, Nbytes));
  HIPCHECK(hipMalloc(&Bd1, Nbytes));
  HIPCHECK(hipMalloc(&Cd1, Nbytes));

  hipStream_t gpu1Stream;
  HIPCHECK(hipStreamCreate(&gpu1Stream));

  for (int j = 0; j < N; j++) {
      A_h[j] = 0;
      B_h[j] = 0;
      C_h[j] = 0;
  }

  // Perform d2d transfer using host pinned memory as staging buffer
  hipMemcpy(A_h, Ad0, Nbytes, hipMemcpyDeviceToHost);
  hipMemcpyAsync(Ad1, A_h, Nbytes, hipMemcpyHostToDevice, gpu1Stream);
  hipMemcpy(B_h, Bd0, Nbytes, hipMemcpyDeviceToHost);
  hipMemcpyAsync(Bd1, B_h, Nbytes, hipMemcpyHostToDevice, gpu1Stream);

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, gpu1Stream, static_cast<const int*>(Ad1),
                     static_cast<const int*>(Bd1), Cd1, N);

  HIPCHECK(hipMemcpyAsync(C_h, Cd1, Nbytes, hipMemcpyDeviceToHost, gpu1Stream));
  HIPCHECK(hipStreamSynchronize(gpu1Stream));

  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  memClear(pinType);
  HIPCHECK(hipStreamDestroy(gpu1Stream));
  return true;
}

int parseStandardArguments(int argc, char* argv[]) {
  for (int i = 1; i < argc; i++) {
    const char* arg = argv[i];

    if (!strcmp(arg, " ")) {
      // skip NULL args.
    } else if (!strcmp(arg, "--pinned")) {
      p_malloc_mode = mallocPinned;
    } else if (!strcmp(arg, "--registered")) {
      p_malloc_mode = mallocRegistered;
    } else {
      failed("Bad argument '%s'", arg);
    }
  }
  return 0;
}

int main(int argc, char* argv[]) {
  bool testResult1 = true;
  bool testResult2 = true;
  parseStandardArguments(argc, argv);

  if (p_malloc_mode == mallocNone) {
    std::cout << "info: invalid malloc type. Empty pass\n";
    passed();
  }
  int numDevices = 0;
  HIPCHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    testResult1 &= testMemCopy(numDevices, p_malloc_mode);
    if (!(testResult1)) {
      std::cout << "d2d failed with hipMemcpy using pinned host buffers\n";
    }

    testResult2 &= testMemCopyAsync(numDevices, p_malloc_mode);
    if (!(testResult2)) {
      std::cout << "d2d failed with hipMemcpyAsync using pinned host buffers\n";
    }

    if (testResult1 && testResult2) {
      passed();
    } else {
      failed("One or more tests failed\n");
    }
  } else {
    std::cout << "Machine does not have more than one gpu, Empty Pass"
              << std::endl;
    passed();
  }
}
