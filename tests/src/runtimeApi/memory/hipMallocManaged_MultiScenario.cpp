/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

/* Test 6 is disabled */
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST_NAMED: %t hipMallocManaged1 --tests 1
 * TEST_NAMED: %t hipMallocManaged2 --tests 2
 * TEST_NAMED: %t hipMallocManagedNegativeTests --tests 3
 * TEST_NAMED: %t hipMallocManagedMultiChunkSingleDevice --tests 4
 * TEST_NAMED: %t hipMallocManagedMultiChunkMultiDevice --tests 5 EXCLUDE_HIP_PLATFORM nvidia
 * TEST_NAMED: %t hipMallocManagedOversubscription --tests 6 EXCLUDE_HIP_PLATFORM nvidia EXCLUDE_HIP_RUNTIME rocclr
 * HIT_END
 */

#include <atomic>
#include "test_common.h"
#define N 1048576  // equals to (1024*1024)
#define INIT_VAL 123

/*
 * Kernel function to perform addition operation.
 */
template <typename T>
__global__ void
vector_sum(T *Ad1, T *Ad2, size_t NUM_ELMTS) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < NUM_ELMTS; i += stride) {
        Ad2[i] = Ad1[i] + Ad1[i];
    }
}

// The following Test case tests the following scenario:
// A large chunk of hipMallocManaged() memory(Hmm) is created
// Equal parts of Hmm is accessed on available gpus and
// kernel is launched on acessed chunk of hmm memory
// and checks if there are any inconsistencies or access issues
bool MultiChunkMultiDevice(int NumDevices) {
  std::atomic<int> DataMismatch{0};
  bool IfTestPassed = true;
  int Counter = 0;
  unsigned int NUM_ELMS = (1024 * 1024);
  float *Ad[NumDevices], *Hmm = NULL, *Ah = new float[NUM_ELMS];
  hipStream_t stream[NumDevices];
  for (int Oloop = 0; Oloop < NumDevices; ++Oloop) {
    HIPCHECK(hipSetDevice(Oloop));
    HIPCHECK(hipMalloc(&Ad[Oloop], NUM_ELMS * sizeof(float)));
    HIPCHECK(hipMemset(Ad[Oloop], 0, NUM_ELMS * sizeof(float)));
    HIPCHECK(hipStreamCreate(&stream[Oloop]));
  }
  HIPCHECK(hipMallocManaged(&Hmm, (NumDevices * NUM_ELMS * sizeof(float))));
  for (int i = 0; i < NumDevices; ++i) {
    for (; Counter < ((i + 1) * NUM_ELMS); ++Counter) {
      Hmm[Counter] = INIT_VAL + i;
    }
  }
  const unsigned threadsPerBlock = 256;
  const unsigned blocks = (NUM_ELMS + 255)/256;
  for (int Klaunch = 0; Klaunch < NumDevices; ++Klaunch) {
    vector_sum<float> <<<blocks, threadsPerBlock, 0, stream[Klaunch]>>>
                      (&Hmm[Klaunch * NUM_ELMS], Ad[Klaunch], NUM_ELMS);
  }
  HIPCHECK(hipDeviceSynchronize());
  for (int m = 0; m < NumDevices; ++m) {
    HIPCHECK(hipMemcpy(Ah, Ad[m], NUM_ELMS * sizeof(float),
                       hipMemcpyDeviceToHost));
    for (int n = 0; n < NUM_ELMS; ++n) {
      if (Ah[n] != ((INIT_VAL + m) * 2)) {
        DataMismatch++;
      }
    }
    memset(reinterpret_cast<void*>(Ah), 0, NUM_ELMS * sizeof(float));
  }
  if (DataMismatch.load() != 0) {
    printf("MultiChunkMultiDevice: Mismatch observed!\n");
    IfTestPassed = false;
  }
  for (int i = 0; i < NumDevices; ++i) {
    HIPCHECK(hipFree(Ad[i]));
    HIPCHECK(hipStreamDestroy(stream[i]));
  }
  HIPCHECK(hipFree(Hmm));
  free(Ah);
  return IfTestPassed;
}

// The following Test case tests the following scenario:
// A large chunk of hipMallocManaged() memory(Hmm) is created
// Equal parts of Hmm is accessed and
// kernel is launched on acessed chunk of hmm memory
// and checks if there are any inconsistencies or access issues

bool MultiChunkSingleDevice(int NumDevices) {
  std::atomic<int> DataMismatch{0};
  int Chunks = 4, Counter = 0;
  bool IfTestPassed = true;
  unsigned int NUM_ELMS = (1024 * 1024);
  float *Ad[Chunks], *Hmm = NULL, *Ah = new float[NUM_ELMS];
  hipStream_t stream[Chunks];
  for (int i = 0; i < Chunks; ++i) {
    HIPCHECK(hipMalloc(&Ad[i], NUM_ELMS * sizeof(float)));
    HIPCHECK(hipMemset(Ad[i], 0, NUM_ELMS * sizeof(float)));
    HIPCHECK(hipStreamCreate(&stream[i]));
  }
  HIPCHECK(hipMallocManaged(&Hmm, (Chunks * NUM_ELMS * sizeof(float))));
  for (int i = 0; i < Chunks; ++i) {
    for (; Counter < ((i + 1) * NUM_ELMS); ++Counter) {
      Hmm[Counter] = (INIT_VAL + i);
    }
  }
  const unsigned threadsPerBlock = 256;
  const unsigned blocks = (NUM_ELMS + 255)/256;
  for (int k = 0; k < Chunks; ++k) {
    vector_sum<float> <<<blocks, threadsPerBlock, 0, stream[k]>>>
                      (&Hmm[k * NUM_ELMS], Ad[k], NUM_ELMS);
  }
  HIPCHECK(hipDeviceSynchronize());
  for (int m = 0; m < Chunks; ++m) {
    HIPCHECK(hipMemcpy(Ah, Ad[m], NUM_ELMS * sizeof(float),
                       hipMemcpyDeviceToHost));
    for (int n = 0; n < NUM_ELMS; ++n) {
      if (Ah[n] != ((INIT_VAL + m) * 2)) {
        DataMismatch++;
      }
    }
  }
  if (DataMismatch.load() != 0) {
    printf("MultiChunkSingleDevice: Mismatch observed!\n");
    IfTestPassed = false;
  }
  for (int i = 0; i < Chunks; ++i) {
    HIPCHECK(hipFree(Ad[i]));
    HIPCHECK(hipStreamDestroy(stream[i]));
  }
  HIPCHECK(hipFree(Hmm));
  free(Ah);
  return IfTestPassed;
}

// The following tests oversubscription hipMallocManaged() api
// Currently disabled.
bool TestOversubscriptionMallocManaged(int NumDevices) {
  bool IfTestPassed = true;
  hipError_t err;
  void *A = NULL;
  size_t total = 0, free = 0;
  HIPCHECK(hipMemGetInfo(&free, &total));
  // ToDo: In case of HMM, memory over-subscription is allowed.  Hence, relook
  // into how out of memory can be tested.
  // Demanding more mem size than available
  err = hipMallocManaged(&A, (free +1), hipMemAttachGlobal);
  if (hipErrorOutOfMemory != err) {
    printf("hipMallocManaged: Returned %s for size value > device memory\n",
           hipGetErrorString(err));
    IfTestPassed = false;
  }

  return IfTestPassed;
}

// The following test does negative testing of hipMallocManaged() api
// by passing invalid values and check if the behavior is as expected
bool NegativeTestsMallocManaged(int NumDevices) {
  bool IfTestPassed = true;
  hipError_t err;
  void *A = NULL;
  size_t total = 0, free = 0;
  HIPCHECK(hipMemGetInfo(&free, &total));

  err = hipMallocManaged(NULL, 1024, hipMemAttachGlobal);
  if (hipErrorInvalidValue != err) {
    printf("hipMallocManaged: Returned %s when devPtr is null\n",
           hipGetErrorString(err));
    IfTestPassed = false;
  }

  err = hipMallocManaged(&A, 0, hipMemAttachGlobal);
  if (hipErrorInvalidValue != err) {
    printf("hipMallocManaged: Returned %s when size is 0\n",
           hipGetErrorString(err));
    IfTestPassed = false;
  }

  err = hipMallocManaged(NULL, 0, hipMemAttachGlobal);
  if (hipErrorInvalidValue != err) {
    printf("hipMallocManaged: Returned %s when devPtr & size is null & 0\n",
           hipGetErrorString(err));
    IfTestPassed = false;
  }

#ifdef __HIP_PLATFORM_AMD__
  // The flag hipMemAttachHost is currently not supported therefore
  // api should return "hipErrorInvalidValue" for now
  err = hipMallocManaged(&A, 1024, hipMemAttachHost);
  if (hipErrorInvalidValue != err) {
    printf("hipMallocManaged: Returned %s for 'hipMemAttachHost' flag\n",
           hipGetErrorString(err));
    IfTestPassed = false;
  }
#endif  // __HIP_PLATFORM_AMD__

  err = hipMallocManaged(NULL, 0, 0);
  if (hipErrorInvalidValue != err) {
    printf("hipMallocManaged: Returned %s when params are null, 0, 0\n",
           hipGetErrorString(err));
    IfTestPassed = false;
  }

  err = hipMallocManaged(&A, 1024, 145);
  if (hipErrorInvalidValue != err) {
    printf("hipMallocManaged: Returned %s when flag param is numerical 145\n",
           hipGetErrorString(err));
    IfTestPassed = false;
  }

  err = hipMallocManaged(&A, -10, hipMemAttachGlobal);
  if (hipErrorOutOfMemory != err) {
    printf("hipMallocManaged: Returned %s for negative size value.\n",
           hipGetErrorString(err));
    IfTestPassed = false;
  }

  return IfTestPassed;
}


// Allocate two pointers using hipMallocManaged(), initialize,
// then launch kernel using these pointers directly and
// later validate the content without using any Memcpy.
template <typename T>
bool TestMallocManaged2(int NumDevices) {
  bool IfTestPassed = true;
  T *Hmm1 = NULL, *Hmm2 = NULL;

  for (int i = 0; i < NumDevices; ++i) {
    HIPCHECK(hipSetDevice(i));
    std::atomic<int> DataMismatch{0};
    HIPCHECK(hipMallocManaged(&Hmm1, N * sizeof(T)));
    HIPCHECK(hipMallocManaged(&Hmm2, N * sizeof(T)));
    for (int m = 0; m < N; ++m) {
      Hmm1[m] = m;
      Hmm2[m] = 0;
    }
    const unsigned threadsPerBlock = 256;
    const unsigned blocks = (N + 255)/256;
    // Kernel launch
    vector_sum <<<blocks, threadsPerBlock>>> (Hmm1, Hmm2, N);
    HIPCHECK(hipDeviceSynchronize());
    for (int v = 0; v < N; ++v) {
      if (Hmm2[v] != (v + v)) {
        DataMismatch++;
      }
    }
    if (DataMismatch.load() != 0) {
      IfTestPassed = false;
    }
    HIPCHECK(hipFree(Hmm1));
    HIPCHECK(hipFree(Hmm2));
  }
  return IfTestPassed;
}

// In the following test, a memory is created using hipMallocManaged() by
// setting a device and verified if it is accessible when the context is set
// to all other devices. This include verification and Device two Device
// transfers and kernel launch o discover if there any access issues.

template <typename T>
bool TestMallocManaged1(int NumDevices) {
  std::atomic<unsigned int> DataMismatch;
  bool TestPassed = true;
  T *Ah1 = new T[N], *Ah2 = new T[N], *Ad = NULL, *Hmm = NULL;

  for (int i =0; i < N; ++i) {
    Ah1[i] = INIT_VAL;
    Ah2[i] = 0;
  }
  for (int Oloop = 0; Oloop < NumDevices; ++Oloop) {
    DataMismatch = 0;
    HIPCHECK(hipSetDevice(Oloop));
    HIPCHECK(hipMallocManaged(&Hmm, N * sizeof(T)));
    for (int Iloop = 0; Iloop < NumDevices; ++Iloop) {
      HIPCHECK(hipSetDevice(Iloop));
      HIPCHECK(hipMalloc(&Ad, N * sizeof(T)));
      // Copy data from host to hipMallocMananged memory and verify
      HIPCHECK(hipMemcpy(Hmm, Ah1, N * sizeof(T), hipMemcpyHostToDevice));
      for (int v = 0; v < N; ++v) {
        if (Hmm[v] != INIT_VAL) {
          DataMismatch++;
        }
      }
      if (DataMismatch.load() != 0) {
        printf("Mismatch is observed with host data at device %d", Iloop);
        printf(" while hipMallocManaged memory set to the device %d\n", Oloop);
        TestPassed = false;
        DataMismatch = 0;
      }
      // Executing D2D transfer with hipMallocManaged memory and verify
      HIPCHECK(hipMemcpy(Ad, Hmm, N * sizeof(T), hipMemcpyDeviceToDevice));
      HIPCHECK(hipMemcpy(Ah2, Ad, N * sizeof(T), hipMemcpyDeviceToHost));
      for (int k = 0; k < N; ++k) {
        if (Ah2[k] != INIT_VAL) {
          DataMismatch++;
        }
      }
      if (DataMismatch.load() != 0) {
        printf("Mismatch is observed with D2D transfer at device %d\n", Iloop);
        printf(" while hipMallocManaged memory set to the device %d\n", Oloop);
        TestPassed = false;
        DataMismatch = 0;
      }
      HIPCHECK(hipMemset(Ad, 0, N * sizeof(T)));
      const unsigned threadsPerBlock = 256;
      const unsigned blocks = (N + 255)/256;
      // Launching the kernel to check if there is any access issue with
      // hipMallocManaged memory and local device's memory
      vector_sum <<<blocks, threadsPerBlock>>> (Hmm, Ad, N);
      hipDeviceSynchronize();
      HIPCHECK(hipMemcpy(Ah2, Ad, N * sizeof(T), hipMemcpyDeviceToHost));
      for (int m = 0; m < N; ++m) {
        if (Ah2[m] != 246) {
          DataMismatch++;
        }
      }
      if (DataMismatch.load() != 0) {
        printf("Data Mismatch observed after kernel lch device %d\n", Iloop);
        TestPassed = false;
        DataMismatch = 0;
      }
      HIPCHECK(hipFree(Ad));
    }
    HIPCHECK(hipFree(Hmm));
  }
  free(Ah1);
  free(Ah2);
  return TestPassed;
}

int main(int argc, char* argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);

  if ((p_tests <= 0) || (p_tests > 5)) {
    failed("Valid arguments are from 1 to 5");
  }

  int NumDevices = 0;
  HIPCHECK(hipGetDeviceCount(&NumDevices));
  bool TestStatus = true, OverAllStatus = true;
  if (p_tests == 1) {
    TestStatus = TestMallocManaged1<float>(NumDevices);
    if (!TestStatus) {
      printf("Test Failed with float datatype.\n");
      OverAllStatus = false;
    }
    TestStatus = TestMallocManaged1<int>(NumDevices);
    if (!TestStatus) {
      printf("Test Failed with int datatype.\n");
      OverAllStatus = false;
    }
    TestStatus = TestMallocManaged1<unsigned char>(NumDevices);
    if (!TestStatus) {
      printf("Test Failed with unsigned char datatype.\n");
      OverAllStatus = false;
    }
    TestStatus = TestMallocManaged1<double>(NumDevices);
    if (!TestStatus) {
      printf("Test Failed with double datatype.\n");
      OverAllStatus = false;
    }
    if (!OverAllStatus) {
      failed("");
    }
  }
  if (p_tests == 2) {
    TestStatus = TestMallocManaged2<float>(NumDevices);
    if (!TestStatus) {
      failed("Test Failed with float datatype.");
    }
  }
  if (p_tests == 3) {
    TestStatus = NegativeTestsMallocManaged(NumDevices);
    if (!TestStatus) {
      failed("Negative Tests with hipMallocManaged() failed!.");
    }
  }
  if (p_tests == 4) {
    TestStatus = MultiChunkSingleDevice(NumDevices);
    if (!TestStatus) {
      failed("hipMallocManaged: MultiChunkSingleDevice test failed!");
    }
  }
  if (p_tests == 5) {
    TestStatus = MultiChunkMultiDevice(NumDevices);
    if (!TestStatus) {
      failed("hipMallocManaged: MultiChunkMultiDevice test failed!");
    }
  }
  if (p_tests == 6) {
    TestStatus = TestOversubscriptionMallocManaged(NumDevices);
    if (!TestStatus) {
      failed("hipMallocManaged: TestOversubscriptionMallocManaged failed!");
    }
  }
  passed();
}
