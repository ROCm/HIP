/*
Copyright (c) 2021-22-present Advanced Micro Devices, Inc. All rights reserved.
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
 * hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,hipMemcpyKind kind,
 * hipStream_t stream);
 */
/*
This testfile verifies the following scenarios
1. hipMemcpyWithStream with one stream
2. hipMemcpyWithStream with two streams
3. Multi GPU and single stream
4. hipMemcpyWithStream API with testkind DtoH
5. hipMemcpyWithStream API with testkind DtoD
6. hipMemcpyWithStream API with testkind HtoH
7. hipMemcpyWithStream API with testkind TestkindDefault
8. hipMemcpyWithStream API with testkind TestkindDefaultForDtoD
9. hipMemcpyWithStream API DtoD on same device
*/


#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>

#include<vector>
#include<thread>
#include<chrono>

#define LEN 64
#define SIZE LEN << 2
#define THREADS 2
#define MAX_THREADS 16

static constexpr size_t N{4 * 1024 * 1024};
static const auto MaxGPUDevices{256};
static constexpr unsigned blocksPerCU{6};  // to hide latency
static constexpr unsigned threadsPerBlock{256};

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

void TestwithOnestream(void) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpyWithStream(A_d, A_h, Nbytes,
                               hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyWithStream(B_d, B_h, Nbytes,
                               hipMemcpyHostToDevice, stream));
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, stream, static_cast<const int*>(A_d),
                     static_cast<const int*>(B_d), C_d, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipStreamDestroy(stream));
}

void TestwithTwoStream(void) {
  size_t Nbytes = N * sizeof(int);
  const int NUM_STREAMS = 2;
  int *A_d[NUM_STREAMS], *B_d[NUM_STREAMS], *C_d[NUM_STREAMS];
  int *A_h[NUM_STREAMS], *B_h[NUM_STREAMS], *C_h[NUM_STREAMS];

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  for (int i=0; i < NUM_STREAMS; ++i) {
    HipTest::initArrays(&A_d[i], &B_d[i], &C_d[i],
                        &A_h[i], &B_h[i], &C_h[i], N, false);
  }

  hipStream_t stream[NUM_STREAMS];
  for (int i=0; i < NUM_STREAMS; ++i) {
    HIP_CHECK(hipStreamCreate(&stream[i]));
  }

  for (int i=0; i < NUM_STREAMS; ++i) {
    HIP_CHECK(hipMemcpyWithStream(A_d[i], A_h[i], Nbytes,
             hipMemcpyHostToDevice, stream[i]));
    HIP_CHECK(hipMemcpyWithStream(B_d[i], B_h[i], Nbytes,
             hipMemcpyHostToDevice, stream[i]));
  }

  for (int i=0; i < NUM_STREAMS; ++i) {
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, stream[i], static_cast<const int*>(A_d[i]),
                       static_cast<const int*>(B_d[i]), C_d[i], N);
    HIP_CHECK(hipGetLastError());
  }

  for (int i=0; i < NUM_STREAMS; ++i) {
    HIP_CHECK(hipStreamSynchronize(stream[i]));
    HIP_CHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
    HipTest::checkVectorADD(A_h[i], B_h[i], C_h[i], N);
  }

  for (int i=0; i < NUM_STREAMS; ++i) {
    HipTest::freeArrays(A_d[i], B_d[i], C_d[i], A_h[i], B_h[i], C_h[i], false);
    HIP_CHECK(hipStreamDestroy(stream[i]));
  }
}

void TestDtoDonSameDevice(void) {
  size_t Nbytes = N * sizeof(int);
  const int NUM_STREAMS = 2;
  int *A_d[NUM_STREAMS], *B_d[NUM_STREAMS], *C_d[NUM_STREAMS];
  int *A_h[NUM_STREAMS], *B_h[NUM_STREAMS], *C_h[NUM_STREAMS];

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HipTest::initArrays(&A_d[0], &B_d[0], &C_d[0],
                      &A_h[0], &B_h[0], &C_h[0], N, false);


  hipStream_t stream[NUM_STREAMS];
  for (int i=0; i < NUM_STREAMS; ++i) {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamCreate(&stream[i]));
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMalloc(&A_d[1], Nbytes));
  HIP_CHECK(hipMalloc(&B_d[1], Nbytes));
  HIP_CHECK(hipMalloc(&C_d[1], Nbytes));
  C_h[1] = reinterpret_cast<int*>(malloc(Nbytes));
  HIP_ASSERT(C_h[1] != NULL);

  HIP_CHECK(hipMemcpyWithStream(A_d[0], A_h[0], Nbytes,
                               hipMemcpyHostToDevice, stream[0]));
  HIP_CHECK(hipMemcpyWithStream(B_d[0], B_h[0], Nbytes,
                               hipMemcpyHostToDevice, stream[0]));

  HIP_CHECK(hipMemcpyWithStream(A_d[1], A_d[0], Nbytes,
                               hipMemcpyDeviceToDevice, stream[1]));
  HIP_CHECK(hipMemcpyWithStream(B_d[1], B_d[0], Nbytes,
                               hipMemcpyDeviceToDevice, stream[1]));


  for (int i=0; i < NUM_STREAMS; ++i) {
    HIP_CHECK(hipSetDevice(0));
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, stream[i], static_cast<const int*>(A_d[i]),
                       static_cast<const int*>(B_d[i]), C_d[i], N);
    HIP_CHECK(hipGetLastError());
  }

  for (int i=0; i < NUM_STREAMS; ++i) {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamSynchronize(stream[i]));
    HIP_CHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
    HipTest::checkVectorADD(A_h[0], B_h[0], C_h[i], N);
  }


  HipTest::freeArrays(A_d[0], B_d[0], C_d[0], A_h[0], B_h[0], C_h[0], false);

  if (A_d[1]) {
    HIP_CHECK(hipFree(A_d[1]));
  }
  if (B_d[1]) {
    HIP_CHECK(hipFree(B_d[1]));
  }
  if (C_d[1]) {
    HIP_CHECK(hipFree(C_d[1]));
  }
  if (C_h[1]) {
    free(C_h[1]);
  }


  for (int i=0; i < NUM_STREAMS; ++i) {
    HIP_CHECK(hipStreamDestroy(stream[i]));
  }
}

void TestOnMultiGPUwithOneStream(void) {
  size_t Nbytes = N * sizeof(int);
  int NumDevices = 0;

  HIP_CHECK(hipGetDeviceCount(&NumDevices));
  // If you have single GPU machine the return
  if (NumDevices <= 1) {
    SUCCEED("NumDevices <2");
  } else {
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
    int *A_d[MaxGPUDevices], *B_d[MaxGPUDevices], *C_d[MaxGPUDevices];
    int *A_h[MaxGPUDevices], *B_h[MaxGPUDevices], *C_h[MaxGPUDevices];

    hipStream_t stream[MaxGPUDevices];
    for (int i=0; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamCreate(&stream[i]));
    }

    for (int i=0; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HipTest::initArrays(&A_d[i], &B_d[i], &C_d[i],
                          &A_h[i], &B_h[i], &C_h[i], N, false);
    }


    for (int i=0; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipMemcpyWithStream(A_d[i], A_h[i], Nbytes,
            hipMemcpyHostToDevice, stream[i]));
      HIP_CHECK(hipMemcpyWithStream(B_d[i], B_h[i], Nbytes,
            hipMemcpyHostToDevice, stream[i]));
    }

    for (int i=0; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks),
                         dim3(threadsPerBlock), 0, stream[i],
                         static_cast<const int*>(A_d[i]),
                         static_cast<const int*>(B_d[i]), C_d[i], N);
      HIP_CHECK(hipGetLastError());
    }

    for (int i=0; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamSynchronize(stream[i]));
      HIP_CHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
      HipTest::checkVectorADD(A_h[i], B_h[i], C_h[i], N);
    }

    for (int i=0; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HipTest::freeArrays(A_d[i], B_d[i], C_d[i],
                          A_h[i], B_h[i], C_h[i], false);
      HIP_CHECK(hipStreamDestroy(stream[i]));
    }
  }
}

void TestkindDtoH(void) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpyWithStream(A_d, A_h, Nbytes,
                               hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyWithStream(B_d, B_h, Nbytes,
                               hipMemcpyHostToDevice, stream));
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, stream, static_cast<const int*>(A_d),
                     static_cast<const int*>(B_d), C_d, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyWithStream(C_h, C_d, Nbytes,
                               hipMemcpyDeviceToHost, stream));
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipStreamDestroy(stream));
}

void TestkindDtoD(void) {
  size_t Nbytes = N * sizeof(int);
  int NumDevices = 0;

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGetDeviceCount(&NumDevices));
  // If you have single GPU machine the return
  if (NumDevices <= 1) {
    SUCCEED("NumDevices are less than 2");
  } else {
    int *A_d[MaxGPUDevices], *B_d[MaxGPUDevices], *C_d[MaxGPUDevices];
    int *A_h[MaxGPUDevices], *B_h[MaxGPUDevices], *C_h[MaxGPUDevices];

    hipStream_t stream[MaxGPUDevices];
    for (int i=0; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamCreate(&stream[i]));
    }

    // Initialize and create the host and device elements for first device
    HIP_CHECK(hipSetDevice(0));
    HipTest::initArrays(&A_d[0], &B_d[0], &C_d[0],
        &A_h[0], &B_h[0], &C_h[0], N, false);

    for (int i=1; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i))
        HIP_CHECK(hipMalloc(&A_d[i], Nbytes));
      HIP_CHECK(hipMalloc(&B_d[i], Nbytes));
      HIP_CHECK(hipMalloc(&C_d[i], Nbytes));
      C_h[i] = reinterpret_cast<int*>(malloc(Nbytes));
      HIP_ASSERT(C_h[i] != NULL);
    }

    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMemcpyWithStream(A_d[0], A_h[0], Nbytes,
          hipMemcpyHostToDevice, stream[0]));
    HIP_CHECK(hipMemcpyWithStream(B_d[0], B_h[0], Nbytes,
          hipMemcpyHostToDevice, stream[0]));

    // Copying device data from 1st GPU to the rest of the the GPUs that is
    // NumDevices in the setup. 1st GPU start numbering from 0,1,2..n etc.
    for (int i=1; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipMemcpyWithStream(A_d[i], A_d[0], Nbytes,
            hipMemcpyDeviceToDevice, stream[i]));
      HIP_CHECK(hipMemcpyWithStream(B_d[i], B_d[0], Nbytes,
            hipMemcpyDeviceToDevice, stream[i]));
    }


    // Launching the kernel including the 1st GPU to the no of GPUs present
    // in the setup. 1st GPU start numbering from 0,1,2..n etc.
    for (int i=0; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks),
                         dim3(threadsPerBlock),
                         0, stream[i], static_cast<const int*>(A_d[i]),
                         static_cast<const int*>(B_d[i]), C_d[i], N);
      HIP_CHECK(hipGetLastError());
    }

    for (int i=0; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamSynchronize(stream[i]));
      HIP_CHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
      HipTest::checkVectorADD(A_h[0], B_h[0], C_h[i], N);
    }

    HipTest::freeArrays(A_d[0], B_d[0], C_d[0], A_h[0], B_h[0], C_h[0], false);
    HIP_CHECK(hipStreamDestroy(stream[0]));

    for (int i=1; i < NumDevices; ++i) {
      if (A_d[i]) {
        HIP_CHECK(hipFree(A_d[i]));
      }
      if (B_d[i]) {
        HIP_CHECK(hipFree(B_d[i]));
      }
      if (C_d[i]) {
        HIP_CHECK(hipFree(C_d[i]));
      }
      if (C_h[i]) {
        free(C_h[i]);
      }
      HIP_CHECK(hipStreamDestroy(stream[i]));
    }
  }
}

void TestkindDefault(void) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpyWithStream(A_d, A_h, Nbytes, hipMemcpyDefault, stream));
  HIP_CHECK(hipMemcpyWithStream(B_d, B_h, Nbytes, hipMemcpyDefault, stream));
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, stream, static_cast<const int*>(A_d),
                     static_cast<const int*>(B_d), C_d, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyWithStream(C_h, C_d, Nbytes, hipMemcpyDefault, stream));
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipStreamDestroy(stream));
}

void TestkindDefaultForDtoD(void) {
  size_t Nbytes = N * sizeof(int);
  int NumDevices = 0;

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGetDeviceCount(&NumDevices));
  // Test case will not run on single GPU setup.
  if (NumDevices <= 1) {
    SUCCEED("No of Devices < 2");
  } else {
    int *A_d[MaxGPUDevices], *B_d[MaxGPUDevices], *C_d[MaxGPUDevices];
    int *A_h[MaxGPUDevices], *B_h[MaxGPUDevices], *C_h[MaxGPUDevices];

    // Initialize and create the host and device elements for first device
    HIP_CHECK(hipSetDevice(0));
    HipTest::initArrays(&A_d[0], &B_d[0], &C_d[0],
        &A_h[0], &B_h[0], &C_h[0], N, false);

    for (int i=1; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipMalloc(&A_d[i], Nbytes));
      HIP_CHECK(hipMalloc(&B_d[i], Nbytes));
      HIP_CHECK(hipMalloc(&C_d[i], Nbytes));
      C_h[i] = reinterpret_cast<int*>(malloc(Nbytes));
      HIP_ASSERT(C_h[i] != NULL);
    }

    hipStream_t stream[MaxGPUDevices];
    for (int i=0; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamCreate(&stream[i]));
    }

    HIP_CHECK(hipMemcpyWithStream(A_d[0], A_h[0], Nbytes,
          hipMemcpyHostToDevice, stream[0]));
    HIP_CHECK(hipMemcpyWithStream(B_d[0], B_h[0], Nbytes,
          hipMemcpyHostToDevice, stream[0]));

    // Copying device data from 1st GPU to the rest of the the GPUs
    // using hipMemcpyDefault kind  that is NumDevices in the setup.
    // 1st GPU start numbering from 0,1,2..n etc.
    for (int i=1; i < NumDevices; ++i) {
      HIP_CHECK(hipMemcpyWithStream(A_d[i], A_d[0], Nbytes,
            hipMemcpyDefault, stream[i]));
      HIP_CHECK(hipMemcpyWithStream(B_d[i], B_d[0], Nbytes,
            hipMemcpyDefault, stream[i]));
    }

    for (int i=0; i < NumDevices; ++i) {
      hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks),
                         dim3(threadsPerBlock),
                         0, stream[i], static_cast<const int*>(A_d[i]),
                         static_cast<const int*>(B_d[i]), C_d[i], N);
      HIP_CHECK(hipGetLastError());
    }

    for (int i=0; i < NumDevices; ++i) {
      HIP_CHECK(hipSetDevice(i));  // hipMemcpy will be on this device
      HIP_CHECK(hipStreamSynchronize(stream[i]));
      HIP_CHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
      // Output of each GPU is getting validated with input of 1st GPU.
      HipTest::checkVectorADD(A_h[0], B_h[0], C_h[i], N);
    }

    HipTest::freeArrays(A_d[0], B_d[0], C_d[0], A_h[0], B_h[0], C_h[0], false);
    HIP_CHECK(hipStreamDestroy(stream[0]));

    for (int i=1; i < NumDevices; ++i) {
      if (A_d[i]) {
        HIP_CHECK(hipFree(A_d[i]));
      }
      if (B_d[i]) {
        HIP_CHECK(hipFree(B_d[i]));
      }
      if (C_d[i]) {
        HIP_CHECK(hipFree(C_d[i]));
      }
      if (C_h[i]) {
        free(C_h[i]);
      }
      HIP_CHECK(hipStreamDestroy(stream[i]));
    }
  }
}

void TestkindHtoH(void) {
  size_t Nbytes = N * sizeof(int);
  int *A_h, *B_h;


  // Allocate memory to A_h and B_h
  A_h = static_cast<int*>(malloc(Nbytes));
  HIP_ASSERT(A_h != NULL);
  B_h = static_cast<int*>(malloc(Nbytes));
  HIP_ASSERT(B_h != NULL);

  for (size_t i = 0; i < N; ++i) {
    if (A_h) {
      (A_h)[i] = 3.146f + i;  // Pi
    }
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpyWithStream(B_h, A_h, Nbytes, hipMemcpyHostToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (size_t i = 0; i < N; i++) {
    HIP_ASSERT(A_h[i] == B_h[i]);
  }

  if (A_h) {
    free(A_h);
  }
  if (B_h) {
    free(B_h);
  }
  HIP_CHECK(hipStreamDestroy(stream));
}


TEST_CASE("Unit_hipMemcpyWithStream_TestWithOneStream") {
  TestwithOnestream();
}

TEST_CASE("Unit_hipMemcpyWithStream_TestwithTwoStream") {
  TestwithTwoStream();
}

TEST_CASE("Unit_hipMemcpyWithStream_TestkindDtoH") {
  TestkindDtoH();
}

TEST_CASE("Unit_hipMemcpyWithStream_TestkindHtoH") {
  TestkindHtoH();
}

TEST_CASE("Unit_hipMemcpyWithStream_TestkindDtoD") {
  TestkindDtoD();
}

TEST_CASE("Unit_hipMemcpyWithStream_TestOnMultiGPUwithOneStream") {
  TestOnMultiGPUwithOneStream();
}

TEST_CASE("Unit_hipMemcpyWithStream_TestkindDefault") {
  TestkindDefault();
}
#ifndef __HIP_PLATFORM_NVCC__
TEST_CASE("Unit_hipMemcpyWithStream_TestkindDefaultForDtoD") {
  TestkindDefaultForDtoD();
}
#endif

TEST_CASE("Unit_hipMemcpyWithStream_TestDtoDonSameDevice") {
  TestDtoDonSameDevice();
}
