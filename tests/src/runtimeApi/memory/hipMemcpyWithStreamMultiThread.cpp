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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
 * Different test for checking functionality of
 * hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,
 * hipMemcpyKind kind, hipStream_t stream);
 */

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include <vector>
#include <thread>
#include <chrono>
#include "test_common.h"

#define LEN 64
#define SIZE LEN << 2
#define THREADS 2
#define MAX_THREADS 16


#define test_passed(test_name)  printf("%s %s  PASSED!%s\n", \
                    KGRN, #test_name, KNRM);
#define test_failed(test_name)  printf("%s %s  FAILED!%s\n", \
                    KRED, #test_name, KNRM);

enum class ops
{   TestwithOnestream,
    TestwithTwoStream,
    TestOnMultiGPUwithOneStream,
    TestkindDtoH,
    TestkindDtoD,
    TestkindHtoH,
    TestkindDefault,
    TestkindDefaultForDtoD,
    TestDtoDonSameDevice,
    END_OF_LIST
};


class HipMemcpyWithStreamMultiThreadtests {
  // Test hipMemcpyWithStream with one streams and launch kernel in
  // that stream, verify the data.
  void TestwithOnestream(void);
  // Test hipMemcpyWithStream with two streams and launch kernels in
  // two streams, verify the data.
  void TestwithTwoStream(void);
  // Test hipMemcpyWithStream with one stream for each gpu and launch
  // kernels in each, verify the data
  void TestOnMultiGPUwithOneStream(void);
  // Test hipMemcpyWithStream to copy data from
  // device to host (hipMemcpyDeviceToHost).
  void TestkindDtoH(void);
  // Test hipMemcpyWithStream with hipMemcpyDeviceToDevice on MultiGPU.
  void TestkindDtoD(void);
  // Test hipMemcpyWithStream with hipMemcpyHostToHost.
  void TestkindHtoH(void);
  // Test hipMemcpyWithStream with hipMemcpyDefault.
  void TestkindDefault(void);
  // Test hipMemcpyWithStream with hipMemcpyDefault for
  // device to device transfer case.
  void TestkindDefaultForDtoD(void);
  // Test hipMemcpyWithStream with hipMemcpyDeviceToDevice on same device.
  void TestDtoDonSameDevice(void);

 public:
  // run all the tests on multithreaded.
  void TestwithMultiThreaded(ops op);
};

struct joinable_thread : std::thread {
    template <class... Xs>
    explicit joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...)
    {} // NOLINT

    joinable_thread& operator=(joinable_thread&& other) = default;
    joinable_thread(joinable_thread&& other)            = default;

    ~joinable_thread() {
        if (this->joinable())
            this->join();
    }
};

void HipMemcpyWithStreamMultiThreadtests::TestwithOnestream(void) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  HIPCHECK(hipMemcpyWithStream(A_d, A_h, Nbytes,
                               hipMemcpyHostToDevice, stream));
  HIPCHECK(hipMemcpyWithStream(B_d, B_h, Nbytes,
                               hipMemcpyHostToDevice, stream));
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, stream, static_cast<const int*>(A_d),
                     static_cast<const int*>(B_d), C_d, N);
  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIPCHECK(hipStreamDestroy(stream));
}

void HipMemcpyWithStreamMultiThreadtests::TestwithTwoStream(void) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;
  int noOfstreams = 2;
  int *A_d[noOfstreams], *B_d[noOfstreams], *C_d[noOfstreams];
  int *A_h[noOfstreams], *B_h[noOfstreams], *C_h[noOfstreams];

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  for (int i=0; i < noOfstreams; ++i) {
    HipTest::initArrays(&A_d[i], &B_d[i], &C_d[i],
                        &A_h[i], &B_h[i], &C_h[i], N, false);
  }

  hipStream_t stream[noOfstreams];
  for (int i=0; i < noOfstreams; ++i) {
    HIPCHECK(hipStreamCreate(&stream[i]));
  }

  for (int i=0; i < noOfstreams; ++i) {
    HIPCHECK(hipMemcpyWithStream(A_d[i], A_h[i], Nbytes,
             hipMemcpyHostToDevice, stream[i]));
    HIPCHECK(hipMemcpyWithStream(B_d[i], B_h[i], Nbytes,
             hipMemcpyHostToDevice, stream[i]));
  }

  for (int i=0; i < noOfstreams; ++i) {
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, stream[i], static_cast<const int*>(A_d[i]),
                       static_cast<const int*>(B_d[i]), C_d[i], N);
  }

  for (int i=0; i < noOfstreams; ++i) {
    HIPCHECK(hipStreamSynchronize(stream[i]));
    HIPCHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
    HipTest::checkVectorADD(A_h[i], B_h[i], C_h[i], N);
  }

  for (int i=0; i < noOfstreams; ++i) {
    HipTest::freeArrays(A_d[i], B_d[i], C_d[i], A_h[i], B_h[i], C_h[i], false);
    HIPCHECK(hipStreamDestroy(stream[i]));
  }
}

void HipMemcpyWithStreamMultiThreadtests::TestDtoDonSameDevice(void) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;
  int noOfstreams = 2;
  int *A_d[noOfstreams], *B_d[noOfstreams], *C_d[noOfstreams];
  int *A_h[noOfstreams], *B_h[noOfstreams], *C_h[noOfstreams];

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HipTest::initArrays(&A_d[0], &B_d[0], &C_d[0],
                      &A_h[0], &B_h[0], &C_h[0], N, false);


  hipStream_t stream[noOfstreams];
  for (int i=0; i < noOfstreams; ++i) {
    HIPCHECK(hipSetDevice(0));
    HIPCHECK(hipStreamCreate(&stream[i]));
  }

  HIPCHECK(hipSetDevice(0));
  HIPCHECK(hipMalloc(&A_d[1], Nbytes));
  HIPCHECK(hipMalloc(&B_d[1], Nbytes));
  HIPCHECK(hipMalloc(&C_d[1], Nbytes));
  C_h[1] = reinterpret_cast<int*>(malloc(Nbytes));
  HIPASSERT(C_h[1] != NULL);

  HIPCHECK(hipMemcpyWithStream(A_d[0], A_h[0], Nbytes,
                               hipMemcpyHostToDevice, stream[0]));
  HIPCHECK(hipMemcpyWithStream(B_d[0], B_h[0], Nbytes,
                               hipMemcpyHostToDevice, stream[0]));

  HIPCHECK(hipMemcpyWithStream(A_d[1], A_d[0], Nbytes,
                               hipMemcpyDeviceToDevice, stream[1]));
  HIPCHECK(hipMemcpyWithStream(B_d[1], B_d[0], Nbytes,
                               hipMemcpyDeviceToDevice, stream[1]));


  for (int i=0; i < noOfstreams; ++i) {
    HIPCHECK(hipSetDevice(0));
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, stream[i], static_cast<const int*>(A_d[i]),
                       static_cast<const int*>(B_d[i]), C_d[i], N);
  }

  for (int i=0; i < noOfstreams; ++i) {
    HIPCHECK(hipSetDevice(0));
    HIPCHECK(hipStreamSynchronize(stream[i]));
    HIPCHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
    HipTest::checkVectorADD(A_h[0], B_h[0], C_h[i], N);
  }


  HipTest::freeArrays(A_d[0], B_d[0], C_d[0], A_h[0], B_h[0], C_h[0], false);

  if (A_d[1]) {
    HIPCHECK(hipFree(A_d[1]));
  }
  if (B_d[1]) {
    HIPCHECK(hipFree(B_d[1]));
  }
  if (C_d[1]) {
    HIPCHECK(hipFree(C_d[1]));
  }
  if (C_h[1]) {
    free(C_h[1]);
  }


  for (int i=0; i < noOfstreams; ++i) {
    HIPCHECK(hipStreamDestroy(stream[i]));
  }
}

void HipMemcpyWithStreamMultiThreadtests::TestOnMultiGPUwithOneStream(void) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIPCHECK(hipGetDeviceCount(&numDevices));
  // If you have single GPU machine the return
  if (numDevices <= 1) {
    return;
  }
  int *A_d[numDevices], *B_d[numDevices], *C_d[numDevices];
  int *A_h[numDevices], *B_h[numDevices], *C_h[numDevices];

  hipStream_t stream[numDevices];
  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipStreamCreate(&stream[i]));
  }

  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HipTest::initArrays(&A_d[i], &B_d[i], &C_d[i],
                        &A_h[i], &B_h[i], &C_h[i], N, false);
  }


  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMemcpyWithStream(A_d[i], A_h[i], Nbytes,
             hipMemcpyHostToDevice, stream[i]));
    HIPCHECK(hipMemcpyWithStream(B_d[i], B_h[i], Nbytes,
             hipMemcpyHostToDevice, stream[i]));
  }

  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, stream[i], static_cast<const int*>(A_d[i]),
                       static_cast<const int*>(B_d[i]), C_d[i], N);
  }

  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipStreamSynchronize(stream[i]));
    HIPCHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
    HipTest::checkVectorADD(A_h[i], B_h[i], C_h[i], N);
  }

  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HipTest::freeArrays(A_d[i], B_d[i], C_d[i], A_h[i], B_h[i], C_h[i], false);
    HIPCHECK(hipStreamDestroy(stream[i]));
  }
}

void HipMemcpyWithStreamMultiThreadtests::TestkindDtoH(void) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  HIPCHECK(hipMemcpyWithStream(A_d, A_h, Nbytes,
                               hipMemcpyHostToDevice, stream));
  HIPCHECK(hipMemcpyWithStream(B_d, B_h, Nbytes,
                               hipMemcpyHostToDevice, stream));
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, stream, static_cast<const int*>(A_d),
                     static_cast<const int*>(B_d), C_d, N);
  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipMemcpyWithStream(C_h, C_d, Nbytes,
                               hipMemcpyDeviceToHost, stream));
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIPCHECK(hipStreamDestroy(stream));
}


void HipMemcpyWithStreamMultiThreadtests::TestkindDtoD(void) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;


  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIPCHECK(hipGetDeviceCount(&numDevices));
  // If you have single GPU machine the return
  if (numDevices <= 1) {
    return;
  }

  int *A_d[numDevices], *B_d[numDevices], *C_d[numDevices];
  int *A_h[numDevices], *B_h[numDevices], *C_h[numDevices];

  hipStream_t stream[numDevices];
  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipStreamCreate(&stream[i]));
  }

  // Initialize and create the host and device elements for first device
  HIPCHECK(hipSetDevice(0));
  HipTest::initArrays(&A_d[0], &B_d[0], &C_d[0],
                      &A_h[0], &B_h[0], &C_h[0], N, false);

  for (int i=1; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i))
    HIPCHECK(hipMalloc(&A_d[i], Nbytes));
    HIPCHECK(hipMalloc(&B_d[i], Nbytes));
    HIPCHECK(hipMalloc(&C_d[i], Nbytes));
    C_h[i] = reinterpret_cast<int*>(malloc(Nbytes));
    HIPASSERT(C_h[i] != NULL);
  }



  HIPCHECK(hipSetDevice(0));
  HIPCHECK(hipMemcpyWithStream(A_d[0], A_h[0], Nbytes,
           hipMemcpyHostToDevice, stream[0]));
  HIPCHECK(hipMemcpyWithStream(B_d[0], B_h[0], Nbytes,
           hipMemcpyHostToDevice, stream[0]));

  // Copying device data from 1st GPU to the rest of the the GPUs that is
  // numDevices in the setup. 1st GPU start numbering from 0,1,2..n etc.
  for (int i=1; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMemcpyWithStream(A_d[i], A_d[0], Nbytes,
             hipMemcpyDeviceToDevice, stream[i]));
    HIPCHECK(hipMemcpyWithStream(B_d[i], B_d[0], Nbytes,
             hipMemcpyDeviceToDevice, stream[i]));
  }


  // Launching the kernel including the 1st GPU to the no of GPUs present
  // in the setup. 1st GPU start numbering from 0,1,2..n etc.
  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, stream[i], static_cast<const int*>(A_d[i]),
                       static_cast<const int*>(B_d[i]), C_d[i], N);
  }

  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipStreamSynchronize(stream[i]));
    HIPCHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
    HipTest::checkVectorADD(A_h[0], B_h[0], C_h[i], N);
  }

  HipTest::freeArrays(A_d[0], B_d[0], C_d[0], A_h[0], B_h[0], C_h[0], false);
  HIPCHECK(hipStreamDestroy(stream[0]));

  for (int i=1; i < numDevices; ++i) {
    if (A_d[i]) {
      HIPCHECK(hipFree(A_d[i]));
    }
    if (B_d[i]) {
      HIPCHECK(hipFree(B_d[i]));
    }
    if (C_d[i]) {
      HIPCHECK(hipFree(C_d[i]));
    }
    if (C_h[i]) {
      free(C_h[i]);
    }
    HIPCHECK(hipStreamDestroy(stream[i]));
  }
}

void HipMemcpyWithStreamMultiThreadtests::TestkindDefault(void) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;


  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  HIPCHECK(hipMemcpyWithStream(A_d, A_h, Nbytes, hipMemcpyDefault, stream));
  HIPCHECK(hipMemcpyWithStream(B_d, B_h, Nbytes, hipMemcpyDefault, stream));
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, stream, static_cast<const int*>(A_d),
                     static_cast<const int*>(B_d), C_d, N);
  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipMemcpyWithStream(C_h, C_d, Nbytes, hipMemcpyDefault, stream));
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIPCHECK(hipStreamDestroy(stream));
}

void HipMemcpyWithStreamMultiThreadtests::TestkindDefaultForDtoD(void) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;


  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIPCHECK(hipGetDeviceCount(&numDevices));
  // Test case will not run on single GPU setup.
  if (numDevices <= 1) {
    return;
  }

  int *A_d[numDevices], *B_d[numDevices], *C_d[numDevices];
  int *A_h[numDevices], *B_h[numDevices], *C_h[numDevices];

  // Initialize and create the host and device elements for first device
  HipTest::initArrays(&A_d[0], &B_d[0], &C_d[0],
                      &A_h[0], &B_h[0], &C_h[0], N, false);

  for (int i=1; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMalloc(&A_d[i], Nbytes));
    HIPCHECK(hipMalloc(&B_d[i], Nbytes));
    HIPCHECK(hipMalloc(&C_d[i], Nbytes));
    C_h[i] = reinterpret_cast<int*>(malloc(Nbytes));
    HIPASSERT(C_h[i] != NULL);
  }

  hipStream_t stream[numDevices];
  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipStreamCreate(&stream[i]));
  }

  HIPCHECK(hipSetDevice(0));
  HIPCHECK(hipMemcpyWithStream(A_d[0], A_h[0], Nbytes,
           hipMemcpyHostToDevice, stream[0]));
  HIPCHECK(hipMemcpyWithStream(B_d[0], B_h[0], Nbytes,
           hipMemcpyHostToDevice, stream[0]));

  // Copying device data from 1st GPU to the rest of the the GPUs
  // using hipMemcpyDefault kind  that is numDevices in the setup.
  // 1st GPU start numbering from 0,1,2..n etc.
  for (int i=1; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMemcpyWithStream(A_d[i], A_d[0], Nbytes,
             hipMemcpyDefault, stream[i]));
    HIPCHECK(hipMemcpyWithStream(B_d[i], B_d[0], Nbytes,
             hipMemcpyDefault, stream[i]));
  }

  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, stream[i], static_cast<const int*>(A_d[i]),
                       static_cast<const int*>(B_d[i]), C_d[i], N);
  }

  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipStreamSynchronize(stream[i]));
    HIPCHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
    // Output of each GPU is getting validated with input of 1st GPU.
    HipTest::checkVectorADD(A_h[0], B_h[0], C_h[i], N);
  }

  HipTest::freeArrays(A_d[0], B_d[0], C_d[0], A_h[0], B_h[0], C_h[0], false);
  HIPCHECK(hipStreamDestroy(stream[0]));

  for (int i=1; i < numDevices; ++i) {
    if (A_d[i]) {
      HIPCHECK(hipFree(A_d[i]));
    }
    if (B_d[i]) {
      HIPCHECK(hipFree(B_d[i]));
    }
    if (C_d[i]) {
      HIPCHECK(hipFree(C_d[i]));
    }
    if (C_h[i]) {
      free(C_h[i]);
    }
    HIPCHECK(hipStreamDestroy(stream[i]));
  }
}

void HipMemcpyWithStreamMultiThreadtests::TestkindHtoH(void) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;
  int *A_h, *B_h;

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  // Allocate memory to A_h and B_h
  A_h = static_cast<int*>(malloc(Nbytes));
  HIPASSERT(A_h != NULL);
  B_h = static_cast<int*>(malloc(Nbytes));
  HIPASSERT(B_h != NULL);

  for (size_t i = 0; i < N; ++i) {
    if (A_h) {
      (A_h)[i] = 3.146f + i;  // Pi
    }
  }

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  HIPCHECK(hipMemcpyWithStream(B_h, A_h, Nbytes, hipMemcpyHostToHost, stream));
  HIPCHECK(hipStreamSynchronize(stream));

  for (size_t i = 0; i < N; i++) {
    HIPASSERT(A_h[i] == B_h[i]);
  }

  if (A_h) {
    free(A_h);
  }
  if (B_h) {
    free(B_h);
  }
  HIPCHECK(hipStreamDestroy(stream));
}


void HipMemcpyWithStreamMultiThreadtests::TestwithMultiThreaded(ops op) {
  int n = min(THREADS * std::thread::hardware_concurrency(), MAX_THREADS);
  std::vector<joinable_thread> threads;

  for (uint32_t i = 0; i < n; i++) {
    threads.emplace_back(std::thread{[&] {
      switch ( op ) {
        case ops::TestwithOnestream:
          TestwithOnestream();
          break;
        case ops::TestwithTwoStream:
          TestwithTwoStream();
          break;
        case ops::TestkindDtoH:
          TestkindDtoH();
          break;
        case ops::TestkindHtoH:
          TestkindHtoH();
          break;
        case ops::TestkindDtoD:
          TestkindDtoD();
          break;
        case ops::TestOnMultiGPUwithOneStream:
          TestOnMultiGPUwithOneStream();
          break;
        case ops::TestkindDefault:
          TestkindDefault();
          break;
        case ops::TestkindDefaultForDtoD:
          TestkindDefaultForDtoD();
          break;
        case ops::TestDtoDonSameDevice:
          TestDtoDonSameDevice();
          break;
        default:{}
      }
    }});
  }
}


int main() {
  HipMemcpyWithStreamMultiThreadtests tests;
  for (int op = static_cast<int>(ops::TestwithOnestream);
           op < static_cast<int>(ops::END_OF_LIST); ++op) {
    tests.TestwithMultiThreaded(static_cast<ops>(op));
    switch ( static_cast<ops>(op) ) {
      case ops::TestwithOnestream:
        test_passed(HipMemcpyWithStreamMultiThreadtests
                    ::TestwithOnestream);
        break;
      case ops::TestwithTwoStream:
        test_passed(HipMemcpyWithStreamMultiThreadtests
                    ::TestwithTwoStream);
        break;
      case ops::TestkindDtoH:
        test_passed(HipMemcpyWithStreamMultiThreadtests
                    ::TestkindDtoH);
        break;
      case ops::TestkindHtoH:
        test_passed(HipMemcpyWithStreamMultiThreadtests
                    ::TestkindHtoH);
        break;
      case ops::TestkindDtoD:
        test_passed(HipMemcpyWithStreamMultiThreadtests
                    ::TestkindDtoD);
        break;
      case ops::TestOnMultiGPUwithOneStream:
        test_passed(HipMemcpyWithStreamMultiThreadtests
                    ::TestOnMultiGPUwithOneStream);
        break;
      case ops::TestkindDefault:
        test_passed(HipMemcpyWithStreamMultiThreadtests
                    ::TestkindDefault);
        break;
      case ops::TestkindDefaultForDtoD:
        test_passed(HipMemcpyWithStreamMultiThreadtests
                    ::TestkindDefaultForDtoD);
        break;
      case ops::TestDtoDonSameDevice:
        test_passed(HipMemcpyWithStreamMultiThreadtests
                    ::TestDtoDonSameDevice);
        break;
      default: { test_failed("No Operation to done with API"); }
    }
  }
}
