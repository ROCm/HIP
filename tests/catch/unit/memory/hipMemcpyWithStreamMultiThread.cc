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


#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <vector>

#define LEN 64
#define SIZE LEN << 2
#define THREADS 2
#define MAX_THREADS 16

static constexpr auto N{1024};
static constexpr auto Nbytes{N * sizeof(int)};
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

namespace MemcpyStream {
  unsigned setNumBlocks(int blocksPerCU, int threadsPerBlock,
      size_t N) {
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    unsigned blocks = props.multiProcessorCount * blocksPerCU;
    if (blocks * threadsPerBlock > N) {
      blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    }
    return blocks;
  }
}  // namespace MemcpyStream


class HipMemcpyWithStreamMultiThreadtests {
 public:
  // Test hipMemcpyWithStream with one streams and launch kernel in
  // that stream, verify the data.
  void TestwithOnestream(bool &val_res);
  // Test hipMemcpyWithStream with two streams and launch kernels in
  // two streams, verify the data.
  void TestwithTwoStream(bool &val_res);
  // Test hipMemcpyWithStream with one stream for each gpu and launch
  // kernels in each, verify the data
  void TestOnMultiGPUwithOneStream(bool &val_res);
  // Test hipMemcpyWithStream to copy data from
  // device to host (hipMemcpyDeviceToHost).
  void TestkindDtoH(bool &val_res);
  // Test hipMemcpyWithStream with hipMemcpyDeviceToDevice on MultiGPU.
  void TestkindDtoD(bool &val_res);
  // Test hipMemcpyWithStream with hipMemcpyHostToHost.
  void TestkindHtoH(bool &val_res);
  // Test hipMemcpyWithStream with hipMemcpyDefault.
  void TestkindDefault(bool &val_res);
  // Test hipMemcpyWithStream with hipMemcpyDefault for
  // device to device transfer case.
  void TestkindDefaultForDtoD(bool &val_res);
  // Test hipMemcpyWithStream with hipMemcpyDeviceToDevice on same device.
  void TestDtoDonSameDevice(bool &val_res);
  // Allocate Memory
  void AllocateMemory(int** A_d, int** B_d,
                      int** C_d, int** A_h,
                      int** B_h,
                      int** C_h);
  // DeAllocate Memory
  void DeAllocateMemory(int* A_d, int* B_d,
                        int* C_d, int* A_h, int* B_h,
                        int* C_h);
  // Validate Result
  bool ValidateResult(int *A_h, int *B_h, int *C_h);
};

void HipMemcpyWithStreamMultiThreadtests::AllocateMemory(int** A_d, int** B_d,
                                                         int** C_d, int** A_h,
                                                         int** B_h,
                                                         int** C_h) {
  HIPCHECK(hipMalloc(A_d, Nbytes));
  HIPCHECK(hipMalloc(B_d, Nbytes));
  HIPCHECK(hipMalloc(C_d, Nbytes));
  *A_h = reinterpret_cast<int*>(malloc(Nbytes));
  *B_h = reinterpret_cast<int*>(malloc(Nbytes));
  *C_h = reinterpret_cast<int*>(malloc(Nbytes));

  for (size_t i = 0; i < N; i++) {
    if (*A_h) (*A_h)[i] = 3;
    if (*B_h) (*B_h)[i] = 4;
    if (*C_h) (*C_h)[i] = 5;
  }
}

void HipMemcpyWithStreamMultiThreadtests::DeAllocateMemory(int* A_d, int* B_d,
                                          int* C_d, int* A_h, int* B_h,
                                          int* C_h) {
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  free(A_h);
  free(B_h);
  free(C_h);
}


bool HipMemcpyWithStreamMultiThreadtests::
     ValidateResult(int *A_h, int *B_h, int *C_h) {
  bool TestPassed = true;
  for (size_t i = 0; i < N; i++) {
    if ((A_h[i] + B_h[i]) != C_h[i]) {
      TestPassed = false;
      break;
    }
  }
  return TestPassed;
}


void HipMemcpyWithStreamMultiThreadtests::TestwithOnestream(bool &val_res) {
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t Nbytes{N * sizeof(int)};
  AllocateMemory(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h);
  unsigned blocks = MemcpyStream::setNumBlocks(blocksPerCU, threadsPerBlock, N);
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
  val_res = ValidateResult(A_h, B_h, C_h);
  DeAllocateMemory(A_d, B_d, C_d, A_h, B_h, C_h);
  HIPCHECK(hipStreamDestroy(stream));
}

void HipMemcpyWithStreamMultiThreadtests::TestwithTwoStream(bool &val_res) {
  size_t Nbytes = N * sizeof(int);
  const int NoofStreams = 2;
  int *A_d[NoofStreams], *B_d[NoofStreams], *C_d[NoofStreams];
  int *A_h[NoofStreams], *B_h[NoofStreams], *C_h[NoofStreams];

  unsigned blocks = MemcpyStream::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  for (int i=0; i < NoofStreams; ++i) {
    AllocateMemory(&A_d[i], &B_d[i], &C_d[i],
                   &A_h[i], &B_h[i], &C_h[i]);
  }

  hipStream_t stream[NoofStreams];
  for (int i=0; i < NoofStreams; ++i) {
    HIPCHECK(hipStreamCreate(&stream[i]));
  }

  for (int i=0; i < NoofStreams; ++i) {
    HIPCHECK(hipMemcpyWithStream(A_d[i], A_h[i], Nbytes,
             hipMemcpyHostToDevice, stream[i]));
    HIPCHECK(hipMemcpyWithStream(B_d[i], B_h[i], Nbytes,
             hipMemcpyHostToDevice, stream[i]));
  }

  for (int i=0; i < NoofStreams; ++i) {
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, stream[i], static_cast<const int*>(A_d[i]),
                       static_cast<const int*>(B_d[i]), C_d[i], N);
  }

  for (int i=0; i < NoofStreams; ++i) {
    HIPCHECK(hipStreamSynchronize(stream[i]));
    HIPCHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
    val_res = ValidateResult(A_h[i], B_h[i], C_h[i]);
  }

  for (int i=0; i < NoofStreams; ++i) {
    DeAllocateMemory(A_d[i], B_d[i], C_d[i], A_h[i], B_h[i], C_h[i]);
    HIPCHECK(hipStreamDestroy(stream[i]));
  }
}

void HipMemcpyWithStreamMultiThreadtests::TestDtoDonSameDevice(bool &val_res) {
  size_t Nbytes = N * sizeof(int);
  const int NoofStreams = 2;
  int *A_d[NoofStreams], *B_d[NoofStreams], *C_d[NoofStreams];
  int *A_h[NoofStreams], *B_h[NoofStreams], *C_h[NoofStreams];

  unsigned blocks = MemcpyStream::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  AllocateMemory(&A_d[0], &B_d[0], &C_d[0],
                 &A_h[0], &B_h[0], &C_h[0]);


  hipStream_t stream[NoofStreams];
  for (int i=0; i < NoofStreams; ++i) {
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


  for (int i=0; i < NoofStreams; ++i) {
    HIPCHECK(hipSetDevice(0));
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, stream[i], static_cast<const int*>(A_d[i]),
                       static_cast<const int*>(B_d[i]), C_d[i], N);
  }

  for (int i=0; i < NoofStreams; ++i) {
    HIPCHECK(hipSetDevice(0));
    HIPCHECK(hipStreamSynchronize(stream[i]));
    HIPCHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
    val_res = ValidateResult(A_h[0], B_h[0], C_h[i]);
  }


  DeAllocateMemory(A_d[0], B_d[0], C_d[0], A_h[0], B_h[0], C_h[0]);

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


  for (int i=0; i < NoofStreams; ++i) {
    HIPCHECK(hipStreamDestroy(stream[i]));
  }
}

void HipMemcpyWithStreamMultiThreadtests::
     TestOnMultiGPUwithOneStream(bool &val_res) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;

  unsigned blocks = MemcpyStream::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIPCHECK(hipGetDeviceCount(&numDevices));
  // If you have single GPU machine the return
  if (numDevices <= 1) {
    return;
  }
  int *A_d[MaxGPUDevices], *B_d[MaxGPUDevices], *C_d[MaxGPUDevices];
  int *A_h[MaxGPUDevices], *B_h[MaxGPUDevices], *C_h[MaxGPUDevices];

  hipStream_t stream[MaxGPUDevices];
  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipStreamCreate(&stream[i]));
  }

  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    AllocateMemory(&A_d[i], &B_d[i], &C_d[i],
                   &A_h[i], &B_h[i], &C_h[i]);
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
    val_res = ValidateResult(A_h[i], B_h[i], C_h[i]);
  }

  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    DeAllocateMemory(A_d[i], B_d[i], C_d[i], A_h[i], B_h[i], C_h[i]);
    HIPCHECK(hipStreamDestroy(stream[i]));
  }
}

void HipMemcpyWithStreamMultiThreadtests::TestkindDtoH(bool &val_res) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  unsigned blocks = MemcpyStream::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  AllocateMemory(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h);

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
  val_res = ValidateResult(A_h, B_h, C_h);

  DeAllocateMemory(A_d, B_d, C_d, A_h, B_h, C_h);
  HIPCHECK(hipStreamDestroy(stream));
}


void HipMemcpyWithStreamMultiThreadtests::TestkindDtoD(bool &val_res) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;


  unsigned blocks = MemcpyStream::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIPCHECK(hipGetDeviceCount(&numDevices));
  // If you have single GPU machine the return
  if (numDevices <= 1) {
    return;
  }

  int *A_d[MaxGPUDevices], *B_d[MaxGPUDevices], *C_d[MaxGPUDevices];
  int *A_h[MaxGPUDevices], *B_h[MaxGPUDevices], *C_h[MaxGPUDevices];

  hipStream_t stream[MaxGPUDevices];
  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipStreamCreate(&stream[i]));
  }

  // Initialize and create the host and device elements for first device
  HIPCHECK(hipSetDevice(0));
  AllocateMemory(&A_d[0], &B_d[0], &C_d[0],
                 &A_h[0], &B_h[0], &C_h[0]);

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
    val_res = ValidateResult(A_h[0], B_h[0], C_h[i]);
  }

  DeAllocateMemory(A_d[0], B_d[0], C_d[0], A_h[0], B_h[0], C_h[0]);
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

void HipMemcpyWithStreamMultiThreadtests::
     TestkindDefault(bool &val_res) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;


  unsigned blocks = MemcpyStream::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  AllocateMemory(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h);

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  HIPCHECK(hipMemcpyWithStream(A_d, A_h, Nbytes, hipMemcpyDefault, stream));
  HIPCHECK(hipMemcpyWithStream(B_d, B_h, Nbytes, hipMemcpyDefault, stream));
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, stream, static_cast<const int*>(A_d),
                     static_cast<const int*>(B_d), C_d, N);
  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipMemcpyWithStream(C_h, C_d, Nbytes, hipMemcpyDefault, stream));
  val_res = ValidateResult(A_h, B_h, C_h);

  DeAllocateMemory(A_d, B_d, C_d, A_h, B_h, C_h);
  HIPCHECK(hipStreamDestroy(stream));
}

void HipMemcpyWithStreamMultiThreadtests::
     TestkindDefaultForDtoD(bool &val_res) {
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;


  unsigned blocks = MemcpyStream::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIPCHECK(hipGetDeviceCount(&numDevices));
  // Test case will not run on single GPU setup.
  if (numDevices <= 1) {
    return;
  }

  int *A_d[MaxGPUDevices], *B_d[MaxGPUDevices], *C_d[MaxGPUDevices];
  int *A_h[MaxGPUDevices], *B_h[MaxGPUDevices], *C_h[MaxGPUDevices];

  // Initialize and create the host and device elements for first device
  HIPCHECK(hipSetDevice(0));
  AllocateMemory(&A_d[0], &B_d[0], &C_d[0],
                 &A_h[0], &B_h[0], &C_h[0]);

  for (int i=1; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMalloc(&A_d[i], Nbytes));
    HIPCHECK(hipMalloc(&B_d[i], Nbytes));
    HIPCHECK(hipMalloc(&C_d[i], Nbytes));
    C_h[i] = reinterpret_cast<int*>(malloc(Nbytes));
    HIPASSERT(C_h[i] != NULL);
  }

  hipStream_t stream[MaxGPUDevices];
  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipStreamCreate(&stream[i]));
  }

  HIPCHECK(hipMemcpyWithStream(A_d[0], A_h[0], Nbytes,
           hipMemcpyHostToDevice, stream[0]));
  HIPCHECK(hipMemcpyWithStream(B_d[0], B_h[0], Nbytes,
           hipMemcpyHostToDevice, stream[0]));

  // Copying device data from 1st GPU to the rest of the the GPUs
  // using hipMemcpyDefault kind  that is numDevices in the setup.
  // 1st GPU start numbering from 0,1,2..n etc.
  for (int i=1; i < numDevices; ++i) {
    HIPCHECK(hipMemcpyWithStream(A_d[i], A_d[0], Nbytes,
             hipMemcpyDefault, stream[i]));
    HIPCHECK(hipMemcpyWithStream(B_d[i], B_d[0], Nbytes,
             hipMemcpyDefault, stream[i]));
  }

  for (int i=0; i < numDevices; ++i) {
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, stream[i], static_cast<const int*>(A_d[i]),
                       static_cast<const int*>(B_d[i]), C_d[i], N);
  }

  for (int i=0; i < numDevices; ++i) {
    HIPCHECK(hipSetDevice(i));  // hipMemcpy will be on this device
    HIPCHECK(hipStreamSynchronize(stream[i]));
    HIPCHECK(hipMemcpy(C_h[i], C_d[i], Nbytes, hipMemcpyDeviceToHost));
    // Output of each GPU is getting validated with input of 1st GPU.
    val_res = ValidateResult(A_h[0], B_h[0], C_h[i]);
  }

  DeAllocateMemory(A_d[0], B_d[0], C_d[0], A_h[0], B_h[0], C_h[0]);
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

void HipMemcpyWithStreamMultiThreadtests::TestkindHtoH(bool &val_res) {
  size_t Nbytes = N * sizeof(int);
  int *A_h, *B_h;

  // Allocate memory to A_h and B_h
  A_h = static_cast<int*>(malloc(Nbytes));
  B_h = static_cast<int*>(malloc(Nbytes));

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
    if ((A_h[i] != B_h[i])) {
       val_res = false;
       break;
    }
  }

  if (A_h) {
    free(A_h);
  }
  if (B_h) {
    free(B_h);
  }
  HIPCHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipMemcpyWithStream_NewMultiThread") {
  const auto Threadcount{100};
  bool ret_val[Threadcount];
  std::thread th[Threadcount];
  for (int op = static_cast<int>(ops::TestwithOnestream);
      op < static_cast<int>(ops::END_OF_LIST); ++op) {
    HipMemcpyWithStreamMultiThreadtests tests;
    for (uint32_t i = 0; i < Threadcount; i++) {
      switch ( static_cast<ops>(op) ) {
        case ops::TestwithOnestream:
          th[i] = std::thread(&HipMemcpyWithStreamMultiThreadtests::
                              TestwithOnestream,
                              &tests, std::ref(ret_val[i]));
          break;
        case ops::TestwithTwoStream:
          th[i] = std::thread(&HipMemcpyWithStreamMultiThreadtests::
                              TestwithTwoStream,
                              &tests, std::ref(ret_val[i]));
          break;
        case ops::TestkindDtoH:
          th[i] = std::thread(&HipMemcpyWithStreamMultiThreadtests::
                              TestkindDtoH,
                              &tests, std::ref(ret_val[i]));
          break;
        case ops::TestkindHtoH:
          th[i] = std::thread(&HipMemcpyWithStreamMultiThreadtests::
                              TestkindHtoH,
                              &tests, std::ref(ret_val[i]));
          break;
        case ops::TestkindDtoD:
          th[i] = std::thread(&HipMemcpyWithStreamMultiThreadtests::
                              TestkindDtoD,
                              &tests,  std::ref(ret_val[i]));
          break;
        case ops::TestOnMultiGPUwithOneStream:
          th[i] = std::thread(&HipMemcpyWithStreamMultiThreadtests::
                              TestOnMultiGPUwithOneStream,
                              &tests, std::ref(ret_val[i]));
          break;
        case ops::TestkindDefault:
          th[i] = std::thread(&HipMemcpyWithStreamMultiThreadtests::
                              TestkindDefault,
                              &tests, std::ref(ret_val[i]));
          break;
        case ops::TestkindDefaultForDtoD:
          th[i] = std::thread(&HipMemcpyWithStreamMultiThreadtests::
                              TestkindDefaultForDtoD,
                              &tests, std::ref(ret_val[i]));
          break;
        case ops::TestDtoDonSameDevice:
          th[i] = std::thread(&HipMemcpyWithStreamMultiThreadtests::
                              TestDtoDonSameDevice,
                              &tests, std::ref(ret_val[i]));
          break;
        default: {}
      }
    }

    for (uint32_t i = 0; i < Threadcount; i++) {
      th[i].join();
    }

    for (uint32_t i = 0; i < Threadcount; i++) {
      REQUIRE(ret_val[i] == true);
    }
  }
}
