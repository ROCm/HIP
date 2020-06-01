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
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Testcase Description: This test case achieves two scenarios
// 1) Verifies the working of Memcpy apis for range of Memory sizes from
// smallest one unit transfer to maxmem available.
// 2) Launches NUM_THREADS threads. Each thread in turn tests the working
// of 8 hipmemcpy apis

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * TEST_NAMED: %t hipMemcpyNegativeMThrdMSize_Negative_tests  --tests 1
 * TEST_NAMED: %t hipMemcpyNegativeMThrdMSize_MultiThread_tests --tests 2
 * TEST_NAMED: %t hipMemcpyNegativeMThrdMSize_MultiSize_singleType --tests 3 --memcpyPeersOnly 0 --testAllTypes 0
 * HIT_END
 */

#include <unistd.h>
#include <atomic>
#include <vector>
#include "test_common.h"

#define NUM_THREADS 10
#define NUM_ELM 1024*1024
#define HIPTEST_TRUE    1

int memcpyPeersOnly = 1;
int testAllTypes = 0;
int Available_Gpus = 0;
std::atomic<size_t> failureCount{0};

enum apiToTest {TEST_MEMCPY, TEST_MEMCPYH2D, TEST_MEMCPYD2H, TEST_MEMCPYD2D,
          TEST_MEMCPYASYNC, TEST_MEMCPYH2DASYNC, TEST_MEMCPYD2HASYNC,
          TEST_MEMCPYD2DASYNC, TEST_MAX};
std::vector<std::string> apiNameToTest = { "hipMemcpy", "hipMemcpyH2D",
          "hipMemcpyD2H", "hipMemcpyD2D", "hipMemcpyAsync",
          "hipMemcpyH2DAsync", "hipMemcpyD2HAsync", "hipMemcpyD2DAsync" };

// If memcpyPeersOnly is true, then checks if given gpus are peers and returns
// true if they are peers, else false
// If memcpyPeersOnly is false, then returns true always
bool gpusIsPeer(int gpu0, int gpu1) {
  bool bRet = true;
  if (HIPTEST_TRUE == memcpyPeersOnly) {
    int CanAccessPeer1 = 0, CanAccessPeer2 = 0;
    HIPCHECK(hipDeviceCanAccessPeer(&CanAccessPeer1, gpu0, gpu1));
    HIPCHECK(hipDeviceCanAccessPeer(&CanAccessPeer2, gpu1, gpu0));
    if ((CanAccessPeer1 * CanAccessPeer2) == 0) {
      bRet = false;
    }
  }

  return bRet;
}

template <typename T>
class memcpyTests {
 public:
    T *A_h, *B_h;
    apiToTest api;
    size_t  NUM_ELMTS = 0;
    memcpyTests(apiToTest val, size_t num_elmts);
    bool Memcpy_And_verify();
    ~memcpyTests();
};

class Memcpy_Negative_Tests {
  float *A_h = nullptr, *B_h = nullptr, *A_d = nullptr, *A_d1 = nullptr,
        *C_d = nullptr, *C_h = nullptr;
  hipStream_t stream;
 public:
  void AllocateMemory();
  void DeAllocateMemory();
  // The following function will test negative scenarios with hipMemcpy()
  bool Test_Memcpy(void);
  bool Test_MemcpyAsync(void);
  bool Test_MemcpyHtoD(void);
  bool Test_MemcpyHtoDAsync(void);
  bool Test_MemcpyDtoH(void);
  bool Test_MemcpyDtoHAsync(void);
  bool Test_MemcpyDtoD(void);
  bool Test_MemcpyDtoDAsync(void);
};

void  Memcpy_Negative_Tests::AllocateMemory() {
  A_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  B_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  C_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  if ((A_h == nullptr) || (B_h == nullptr) || (C_h == nullptr)) {
    failed("Malloc call failed!");
  }

  HIPCHECK(hipMalloc(&A_d, (NUM_ELM*sizeof(float))));
  HIPCHECK(hipMalloc(&A_d1, (NUM_ELM*sizeof(float))));
  HIPCHECK(hipMalloc(&C_d, (NUM_ELM*sizeof(float))));

  for ( int i = 0; i < NUM_ELM; ++i ) {
    A_h[i] = 123;
    B_h[i] = 0;
    C_h[i] = 1;
  }
  HIPCHECK(hipStreamCreate(&stream));
}

void Memcpy_Negative_Tests::DeAllocateMemory() {
  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(A_d1));
  HIPCHECK(hipFree(C_d));
  free(A_h);
  free(B_h);
  free(C_h);
  HIPCHECK(hipStreamDestroy(stream));
}
bool Memcpy_Negative_Tests::Test_Memcpy(void) {
  bool IfTestPassed = true;
  hipError_t hipReturn = hipSuccess;
  AllocateMemory();
  // Copying only half the memory on device side from host
  HIPCHECK(hipMemcpy(A_d, A_h, (NUM_ELM/2) * sizeof(float), hipMemcpyDefault));
  // Copying device memory to host to verify if the content is consistent
  HIPCHECK(hipMemcpy(B_h, A_d, NUM_ELM * sizeof(float), hipMemcpyDefault));
  // Verifying the host content copied in the above step for consistency.
  int Data_mismatch = 0;

  for (int i = 0; i < (NUM_ELM/2); ++i) {
    if (B_h[i] != 123) {
      Data_mismatch++;
      break;
    }
  }

  if (Data_mismatch != 0) {
    printf("Data Mismatch for negative test\n");
    IfTestPassed = false;
  }
  // Passing 0 to size and it should return Success
  // Validating it with the initial value which is A_h 123
  HIPCHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyDefault));
  hipReturn = hipMemcpy(C_d, B_h, 0, hipMemcpyDefault);
  if (hipReturn != hipSuccess) {
    printf("Failed for hipMemcpy with size 0.\n");
    IfTestPassed = false;
  } else {
    HIPCHECK(hipMemcpy(C_h, C_d, NUM_ELM * sizeof(float),
                       hipMemcpyDeviceToHost));
    for (int i =0; i < NUM_ELM; i++) {
      if (C_h[i] != A_h[0]) {
        printf("Failed for size 0 and data modified \n");
        IfTestPassed = false;
        break;
      }
    }
  }

  hipReturn = hipMemcpy(nullptr, A_d, (NUM_ELM/2) * sizeof(float),
      hipMemcpyDefault);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpy with nullptr for destination parameter.\n");
    IfTestPassed = false;
  }

  hipReturn = hipMemcpy(A_h, nullptr, (NUM_ELM/2) * sizeof(float),
      hipMemcpyDefault);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpy with nullptr for source\n");
    IfTestPassed = false;
  }

  hipReturn = hipMemcpy(nullptr, nullptr, (NUM_ELM/2) * sizeof(float),
      hipMemcpyDefault);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpy with nullptr for source and destination\n");
    IfTestPassed = false;
  }

  // To check the behaviour if both the ptrs provided are same
  HIPCHECK(hipMemcpy(A_d, A_d, (NUM_ELM/2) * sizeof(float), hipMemcpyDefault));
  HIPCHECK(hipMemcpy(A_h, A_h, (NUM_ELM/2) * sizeof(float), hipMemcpyDefault));

  // To check the consistency of the data
  HIPCHECK(hipMemcpy(B_h, A_d, (NUM_ELM/2) * sizeof(float), hipMemcpyDefault));
  Data_mismatch = 0;

  for (int i = 0; i < (NUM_ELM/2); ++i) {
    if (B_h[i] != 123) {
      Data_mismatch++;
      break;
    }
  }

  if (Data_mismatch != 0) {
    printf("Data Mismatch after memcpy of same src and destination\n");
    IfTestPassed = false;
  }

  DeAllocateMemory();
  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyAsync(void) {
  bool IfTestPassed = true;
  hipError_t hipReturn = hipSuccess;
  AllocateMemory();
  // Copying host data into the device.
  HIPCHECK(hipMemcpyAsync(A_d1, A_h, NUM_ELM * sizeof(float),
                          hipMemcpyDefault, stream));

  // Passing null pointer: seg fault observed with the following.
  hipReturn = hipMemcpyAsync(nullptr, A_h, NUM_ELM * sizeof(float),
                             hipMemcpyDefault, stream);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyAsync with nullptr for destination\n");
    IfTestPassed = false;
  }
  // Passing 0 to size and it should return Success
  // Validating it with the initial value which is A_h 123
  HIPCHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyDefault));
  hipReturn = hipMemcpyAsync(C_d, B_h, 0, hipMemcpyDefault, stream);
  if (hipReturn != hipSuccess) {
    printf("Failed for hipMemcpyAsync with size 0.\n");
    IfTestPassed = false;
  } else {
    HIPCHECK(hipMemcpy(C_h, C_d, NUM_ELM * sizeof(float),
                       hipMemcpyDeviceToHost));
    for (int i =0; i < NUM_ELM; i++) {
      if (C_h[i] != A_h[0]) {
        printf("Failed for hipMemcpyAsync size 0 and data modified \n");
        IfTestPassed = false;
        break;
      }
    }
  }
  hipReturn = hipMemcpyAsync(A_d, nullptr, NUM_ELM * sizeof(float),
                             hipMemcpyDefault, stream);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyAsync with nullptr for source\n");
    IfTestPassed = false;
  }

  hipReturn = hipMemcpyAsync(nullptr, nullptr,
                             NUM_ELM * sizeof(float),
                             hipMemcpyDefault, stream);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyAsync nullptr for source and destination\n");
    IfTestPassed = false;
  }

  // Passing default stream just for sanity kind of check
  HIPCHECK(hipMemcpyAsync(A_d, A_h, NUM_ELM * sizeof(float), hipMemcpyDefault,
                          0));

  // Passing stream object belong to destination gpu
  // which is against the suggested  usage.
  HIPCHECK(hipMemcpyAsync(A_d, A_d1, NUM_ELM * sizeof(float),
                          hipMemcpyDefault, stream));

  // Passing incorrect memcpy kind is not allowed hence those scenarios
  // are not included

  // Copying only half the memory on device side from host
  HIPCHECK(hipMemcpyAsync(A_d, A_h, (NUM_ELM/2) * sizeof(float),
                          hipMemcpyDefault, stream));
  // Copying device memory to host to verify the content is consistent.
  HIPCHECK(hipMemcpy(B_h, A_d, (NUM_ELM/2) * sizeof(float), hipMemcpyDefault));

  // Verifying the host content copied in the above step for consistency.
  int Data_mismatch = 0;
  for (int i = 0; i < (NUM_ELM/2); ++i) {
    if (B_h[i] != 123) {
      Data_mismatch++;
      break;
    }
  }

  if (Data_mismatch != 0) {
    printf("Data Mismatch after half the size memcpyAsync\n");
    IfTestPassed = false;
  }

  // To check the behaviour if both the ptrs provided are same
  HIPCHECK(hipMemcpyAsync(A_d, A_d, (NUM_ELM/2) * sizeof(float),
                          hipMemcpyDefault, stream));
  HIPCHECK(hipMemcpyAsync(A_h, A_h, (NUM_ELM/2) * sizeof(float),
                          hipMemcpyDefault, stream));
  // To check the consistency of the data
  HIPCHECK(hipMemcpy(B_h, A_d, (NUM_ELM) * sizeof(float), hipMemcpyDefault));
  Data_mismatch = 0;
  for (int i = 0; i < (NUM_ELM); ++i) {
    if (B_h[i] != 123) {
      Data_mismatch++;
      break;
    }
  }

  if (Data_mismatch != 0) {
    printf("Data Mismatch after memcpyAsync of same src and destination\n");
    IfTestPassed = false;
  }

  HIPCHECK(hipStreamSynchronize(stream));
  DeAllocateMemory();
  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyHtoD(void) {
  bool IfTestPassed = true;
  hipError_t hipReturn = hipSuccess;
  AllocateMemory();
  // Passing null ptr to check the API behavior.
  // Expectation: It should not crash and exit gracefully.
  hipReturn = hipMemcpyHtoD(hipDeviceptr_t(nullptr), A_h,
                            NUM_ELM * sizeof(float));
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyHtoD with nullptr for destination\n");
    IfTestPassed = false;
  }
  // Passing 0 to size and it should return Success
  // Validating it with the initial value which is A_h 123
  HIPCHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyDefault));
  hipReturn = hipMemcpyHtoD(hipDeviceptr_t(C_d), B_h, 0);
  if (hipReturn != hipSuccess) {
    printf("Failed for hipMemcpyHtoD with size 0.\n");
    IfTestPassed = false;
  } else {
    HIPCHECK(hipMemcpy(C_h, C_d, NUM_ELM * sizeof(float),
                       hipMemcpyDeviceToHost));
    for (int i =0; i < NUM_ELM; i++) {
      if (C_h[i] != A_h[0]) {
        printf("Failed for hipMemcpyHtoD size 0 and data modified \n");
        IfTestPassed = false;
        break;
      }
    }
  }
  hipReturn = hipMemcpyHtoD(hipDeviceptr_t(A_d), nullptr,
                            NUM_ELM * sizeof(float));
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyHtoD with nullptr for source\n");
    IfTestPassed = false;
  }
  hipReturn = hipMemcpyHtoD(hipDeviceptr_t(nullptr), nullptr,
                            NUM_ELM * sizeof(float));
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyHtoD nullptr for source and destination\n");
    IfTestPassed = false;
  }
  // Copy half of the allocated memory
  HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h,
  NUM_ELM * sizeof(float) / 2));
  // copying back to host to verify
  HIPCHECK(hipMemcpy(B_h, A_d,
  NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));

  int Data_mismatch = 0;
  for (int i = 0; i < (NUM_ELM / 2); ++i)
    if (B_h[i] != 123)
      Data_mismatch++;

  if (Data_mismatch != 0) {
    printf("Data Mismatch after hipMemcpyHtoD with half size\n");
    IfTestPassed = false;
  }
  DeAllocateMemory();
  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyHtoDAsync(void) {
  bool IfTestPassed = true;
  hipError_t hipReturn = hipSuccess;
  AllocateMemory();
  // Passing null ptr to check the API behavior.
  // Expectation: It should not crash and exit gracefully.
  hipReturn = hipMemcpyHtoDAsync(hipDeviceptr_t(nullptr), A_h,
                                 NUM_ELM * sizeof(float),
                                 stream);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyHtoDAsync with nullptr for destination\n");
    IfTestPassed = false;
  }
  // Passing 0 to size and it should return Success
  // Validating it with the initial value which is A_h 123
  HIPCHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyDefault));
  hipReturn = hipMemcpyHtoDAsync(hipDeviceptr_t(C_d), B_h, 0, stream);
  HIPCHECK(hipMemcpy(C_h, C_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
  if (hipReturn != hipSuccess) {
    printf("Failed for hipMemcpyHtoDAsync with size 0.\n");
    IfTestPassed = false;
  } else {
    for (int i =0; i < NUM_ELM; i++) {
      if (C_h[i] != A_h[0]) {
        printf("Failed for hipMemcpyH2DAsync size 0 and data modified \n");
        IfTestPassed = false;
        break;
      }
    }
  }
  hipReturn = hipMemcpyHtoDAsync(hipDeviceptr_t(A_d), nullptr,
                                 NUM_ELM * sizeof(float),
                                 stream);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyHtoDAsync with nullptr for source\n");
    IfTestPassed = false;
  }
  hipReturn = hipMemcpyHtoDAsync(hipDeviceptr_t(nullptr), nullptr,
                                 NUM_ELM * sizeof(float),
                                 stream);
  if (hipReturn == hipSuccess) {
    printf("Failed MemcpyHtoDAsync nullptr for source and destination\n");
    IfTestPassed = false;
  }

  // Copy half of the allocated memory
  HIPCHECK(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d), A_h,
  NUM_ELM * sizeof(float)/2, stream));
  // copying back to host to verify
  HIPCHECK(hipMemcpyDtoH(B_h, hipDeviceptr_t(A_d),
  NUM_ELM * sizeof(float)));
  int Data_mismatch = 0;
  for (int i = 0; i < (NUM_ELM/2); ++i)
    if (B_h[i] != 123)
      Data_mismatch++;
  if (Data_mismatch != 0) {
    printf("Data Mismatch after hipMemcpyHtoDAsync with half size\n");
    IfTestPassed = false;
  }

  DeAllocateMemory();
  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyDtoH(void) {
  bool IfTestPassed = true;
  hipError_t hipReturn = hipSuccess;
  AllocateMemory();
  // Copying data from host to device for further operations
  HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h, NUM_ELM * sizeof(float)));

  // Passing null ptr to check the API behavior.
  // Expectation: It should not crash and exit gracefully.
  hipReturn = hipMemcpyDtoH(nullptr, hipDeviceptr_t(A_d),
                            NUM_ELM * sizeof(float));
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyDtoH with nullptr for destination\n");
    IfTestPassed = false;
  }
  // Passing 0 to size and it should return Success
  // Validating it with the initial value which is 1
  HIPCHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyDefault));
  hipReturn = hipMemcpyDtoH(C_h, hipDeviceptr_t(C_d), 0);
  if (hipReturn != hipSuccess) {
    printf("Failed for hipMemcpyDtoH with size 0.\n");
    IfTestPassed = false;
  } else {
    for (int i =0; i < NUM_ELM; i++) {
      if (C_h[i] != 1) {
        printf("Failed for hipMemcpyDtoH size 0 and data modified \n");
        IfTestPassed = false;
        break;
      }
    }
  }
  hipReturn = hipMemcpyDtoH(A_h, hipDeviceptr_t(nullptr),
                            NUM_ELM * sizeof(float));
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyDtoH with nullptr for source\n");
    IfTestPassed = false;
  }
  hipReturn = hipMemcpyDtoH(nullptr, hipDeviceptr_t(nullptr),
                            NUM_ELM * sizeof(float));
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyDtoH nullptr for source and destination\n");
    IfTestPassed = false;
  }
  // Copy half of the allocated memory
  HIPCHECK(hipMemcpyDtoH(B_h, hipDeviceptr_t(A_d), NUM_ELM * sizeof(float)/2));

  int Data_mismatch = 0;
  for (int i = 0; i < (NUM_ELM/2); ++i)
    if (B_h[i] != 123)
      Data_mismatch++;

  if (Data_mismatch != 0) {
    printf("Data Mismatch after hipMemcpyDtoH with half size\n");
    IfTestPassed = false;
  }

  DeAllocateMemory();
  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyDtoHAsync(void) {
  bool IfTestPassed = true;
  hipError_t hipReturn = hipSuccess;
  AllocateMemory();

  // Copying data from host to device for further operations
  HIPCHECK(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d), A_h,
  NUM_ELM * sizeof(float), stream));

  // Passing null ptr to check the API behavior.
  // Expectation: It should not crash and exit gracefully.
  hipReturn = hipMemcpyDtoHAsync(nullptr, hipDeviceptr_t(A_d),
                                 NUM_ELM * sizeof(float),
                                 stream);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyDtoHAsync with nullptr for destination\n");
    IfTestPassed = false;
  }
  // Passing 0 to size and it should return Success
  // Validating it with the initial value which is C_h initial value 1
  HIPCHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyDefault));
  hipReturn = hipMemcpyDtoHAsync(C_h, hipDeviceptr_t(C_d), 0, stream);
  if (hipReturn != hipSuccess) {
    printf("Failed for hipMemcpyDtoHAsync with size 0.\n");
    IfTestPassed = false;
  } else {
    for (int i =0; i < NUM_ELM; i++) {
      if (C_h[i] != 1) {
        printf("Failed for hipMemcpyD2HAsync size 0 and data modified \n");
        IfTestPassed = false;
        break;
      }
    }
  }
  hipReturn = hipMemcpyDtoHAsync(A_h, hipDeviceptr_t(nullptr),
                                 NUM_ELM * sizeof(float),
                                 stream);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyDtoHAsync with nullptr for source\n");
    IfTestPassed = false;
  }
  hipReturn = hipMemcpyDtoHAsync(nullptr, hipDeviceptr_t(nullptr),
                                 NUM_ELM * sizeof(float),
                                 stream);
  if (hipReturn == hipSuccess) {
    printf("Failed hipMemcpyDtoHAsync nullptr for source and destination\n");
    IfTestPassed = false;
  }

  // Copy half of the allocated memory
  HIPCHECK(hipMemcpyDtoHAsync(B_h, hipDeviceptr_t(A_d),
  NUM_ELM * sizeof(float)/2, stream));
  HIPCHECK(hipStreamSynchronize(stream));
  int Data_mismatch = 0;
  for (int i = 0; i < (NUM_ELM/2); ++i)
    if (B_h[i] != 123)
      Data_mismatch++;

  if (Data_mismatch != 0) {
    printf("Data Mismatch after hipMemcpyDtoHAsync with half size\n");
    IfTestPassed = false;
  }
  // Checking the api with default stream
  HIPCHECK(hipMemcpyDtoHAsync(B_h, hipDeviceptr_t(A_d),
  NUM_ELM * sizeof(float), 0));
  // Setting device memory to zero

  DeAllocateMemory();
  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyDtoD(void) {
  bool IfTestPassed = true;
  hipError_t hipReturn = hipSuccess;
  AllocateMemory();
  float *A_d2 = nullptr, *Ad1 = nullptr;
  HIPCHECK(hipMalloc(&Ad1, (NUM_ELM * sizeof(float))));
  HIPCHECK(hipMemset(A_d1, 0, NUM_ELM * sizeof(float)));
  if (Available_Gpus > 1) {
    HIPCHECK(hipSetDevice(1));
    HIPCHECK(hipMalloc(&A_d2, (NUM_ELM * sizeof(float))));
    HIPCHECK(hipMemset(A_d2, 1, NUM_ELM * sizeof(float)));
  }
  // Passing null pointers to check the behaviour::
  hipReturn = hipMemcpyDtoD(hipDeviceptr_t(&A_d1),
                            hipDeviceptr_t(nullptr), NUM_ELM * sizeof(float));
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyDtoD with nullptr for source\n");
    IfTestPassed = false;
  }
  // Passing 0 to size and it should return Success
  // Validating it with the initial value which is A_h 123
  HIPCHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyDefault));
  hipReturn = hipMemcpyDtoD(hipDeviceptr_t(&C_d), hipDeviceptr_t(&A_d2), 0);
  if (hipReturn != hipSuccess) {
    printf("Failed for hipMemcpyDtoD with size 0.\n");
    IfTestPassed = false;
  } else {
    HIPCHECK(hipMemcpy(C_h, C_d, NUM_ELM * sizeof(float),
                       hipMemcpyDeviceToHost));
    for (int i =0; i < NUM_ELM; i++) {
      if (C_h[i] != A_h[0]) {
        printf("Failed for hipMemcpyDtoD size 0 and data modified \n");
        IfTestPassed = false;
        break;
      }
    }
  }
  hipReturn = hipMemcpyDtoD(hipDeviceptr_t(nullptr),
                            hipDeviceptr_t(&A_d2), NUM_ELM * sizeof(float));
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyDtoD with nullptr for destination\n");
    IfTestPassed = false;
  }
  hipReturn = hipMemcpyDtoD(hipDeviceptr_t(nullptr),
                            hipDeviceptr_t(nullptr), NUM_ELM * sizeof(float));
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyDtoD nullptr for source and destination\n");
    IfTestPassed = false;
  }

  // Pass real but host ptr:: The below two scenarios gives seg fault.
  // Behaviour is as expected
  // HIPCHECK(hipMemcpyDtoD(&A_d1, &A_h, NUM_ELM * sizeof(float)));
  // HIPCHECK(hipMemcpyDtoD(&A_h, &A_d1, NUM_ELM * sizeof(float)));
  int Data_mismatch = 0;
  // Copying half of actually allocated memory
  HIPCHECK(hipSetDevice(0));
  if (Available_Gpus > 1) {
    HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d1), A_h, NUM_ELM * sizeof(float)));
    if (true == gpusIsPeer(0, 1)) {
      HIPCHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d2), hipDeviceptr_t(A_d1),
               NUM_ELM * sizeof(float)/2));
      HIPCHECK(hipMemcpyDtoH(B_h, hipDeviceptr_t(A_d2),
               NUM_ELM * sizeof(float)));
      for (int i = 0; i < NUM_ELM/2; ++i) {
        if (B_h[i] != 123)
          Data_mismatch++;
      }
      if (Data_mismatch != 0) {
        printf("Data mismatch hipMemcpyDtoD between devices\n");
        IfTestPassed = false;
      }
    }
  }

  // Passing same pointers for source and destination
  HIPCHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d1),
                         hipDeviceptr_t(A_d1),
                         NUM_ELM * sizeof(float)));
  if (Available_Gpus > 1) {
    HIPCHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d2),
                           hipDeviceptr_t(A_d2),
                           NUM_ELM * sizeof(float)));
  }

  DeAllocateMemory();
  HIPCHECK(hipFree(Ad1));
  if (Available_Gpus > 1)
    HIPCHECK(hipFree(A_d2));

  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyDtoDAsync(void) {
  bool IfTestPassed = true;
  hipError_t hipReturn = hipSuccess;
  AllocateMemory();
  float *A_d2 = nullptr, *Ad1 = nullptr;
  HIPCHECK(hipMalloc(&Ad1, (NUM_ELM * sizeof(float))));
  HIPCHECK(hipMemset(A_d1, 0, NUM_ELM * sizeof(float)));
  if (Available_Gpus > 1) {
    HIPCHECK(hipSetDevice(1));
    HIPCHECK(hipMalloc(&A_d2, (NUM_ELM * sizeof(float))));
    HIPCHECK(hipMemset(A_d2, 1, NUM_ELM * sizeof(float)));
  }
  // Passing null pointers to check the behaviour::
  hipReturn = hipMemcpyDtoDAsync(hipDeviceptr_t(&A_d1),
                                 hipDeviceptr_t(nullptr),
                                 NUM_ELM * sizeof(float), stream);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyDtoDAsync with nullptr for source\n");
    IfTestPassed = false;
  }
  // Passing 0 to size and it should return Success
  // Validating it with the initial value which is A_h 123
  HIPCHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyDefault));
  hipReturn = hipMemcpyDtoDAsync(hipDeviceptr_t(&C_d),
                                 hipDeviceptr_t(&A_d2), 0, stream);
  if (hipReturn != hipSuccess) {
    printf("Failed for hipMemcpyDtoDAsync with size 0.\n");
    IfTestPassed = false;
  } else {
    HIPCHECK(hipMemcpy(C_h, C_d, NUM_ELM * sizeof(float),
                       hipMemcpyDeviceToHost));
    for (int i =0; i < NUM_ELM; i++) {
      if (C_h[i] != A_h[0]) {
        printf("Failed for hipMemcpyDtoDAsync size 0 and data modified \n");
        IfTestPassed = false;
        break;
      }
    }
  }
  hipReturn = hipMemcpyDtoDAsync(hipDeviceptr_t(nullptr),
                                 hipDeviceptr_t(&A_d2),
                                 NUM_ELM * sizeof(float), stream);
  if (hipReturn == hipSuccess) {
    printf("Failed for hipMemcpyDtoDAsync with nullptr for destination\n");
    IfTestPassed = false;
  }
  hipReturn = hipMemcpyDtoDAsync(hipDeviceptr_t(nullptr),
                                 hipDeviceptr_t(nullptr),
                                 NUM_ELM * sizeof(float), stream);
  if (hipReturn == hipSuccess) {
    printf("Failed MemcpyDtoDAsync with nullptr for source and destination\n");
    IfTestPassed = false;
  }

  int Data_mismatch = 0;
  // Copying half of actually allocated memory
  HIPCHECK(hipSetDevice(0));
  if (Available_Gpus > 1) {
    HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d1), A_h, NUM_ELM * sizeof(float)));
    if (true == gpusIsPeer(0, 1)) {
      HIPCHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d2),
               hipDeviceptr_t(A_d1), NUM_ELM * sizeof(float)/2, stream));
      HIPCHECK(hipMemcpyDtoH(B_h, hipDeviceptr_t(A_d2),
               NUM_ELM * sizeof(float)));
      for (int i = 0; i < NUM_ELM/2; ++i) {
        if (B_h[i] != 123)
          Data_mismatch++;
      }
      if (Data_mismatch != 0) {
        printf("Data mismatch hipMemcpyDtoDAsync between devices\n");
        IfTestPassed = false;
      }
    }
  }

  // Testing hipMemcpyDtoDAsync between two devices.
  if (Available_Gpus > 1) {
    if (true == gpusIsPeer(0, 1)) {
      HIPCHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d2),
               hipDeviceptr_t(A_d1), NUM_ELM * sizeof(float), 0));
    }
  }
  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipFree(Ad1));
  if (Available_Gpus > 1)
    HIPCHECK(hipFree(A_d2));

  return IfTestPassed;
}

template <typename T>
memcpyTests<T>::memcpyTests(apiToTest val, size_t num_elmts) {
  api = val;
  NUM_ELMTS = num_elmts;
  A_h = reinterpret_cast<T*>(malloc(NUM_ELMTS * sizeof(T)));
  B_h = reinterpret_cast<T*>(malloc(NUM_ELMTS * sizeof(T)));
  if ((A_h == nullptr) || (B_h == nullptr)) {
    exit(1);
  }

  for (size_t i = 0; i < NUM_ELMTS; ++i) {
    A_h[i] = 123;
    B_h[i] = 0;
  }
}

template <typename T>
bool memcpyTests<T>::Memcpy_And_verify() {
  bool bFail = false;
  std::atomic<size_t> Data_mismatch{0};
  T *A_d[Available_Gpus];
  hipStream_t stream[Available_Gpus];
  for (int i = 0; i < Available_Gpus; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMalloc(&A_d[i], NUM_ELMTS * sizeof(T)));
    if (api >= TEST_MEMCPYD2D) {
      HIPCHECK(hipStreamCreate(&stream[i]));
    }
  }
  HIPCHECK(hipSetDevice(0));

  switch (api) {
    case TEST_MEMCPY:  // To test hipMemcpy()
      // Copying data from host to individual devices followed by copying
      // back to host and verifying the data consistency.
      for (int i = 0; i < Available_Gpus; ++i) {
        Data_mismatch = 0;
        HIPCHECK(hipMemcpy(A_d[i], A_h, NUM_ELMTS * sizeof(T),
              hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELMTS * sizeof(T),
              hipMemcpyDeviceToHost));
        for (int j = 0; j < NUM_ELMTS; ++j) {
          if (A_h[j] != B_h[j]) {
            Data_mismatch++;
          }
        }

        if (Data_mismatch.load() != 0) {
          printf("hipMemcpy: Failed for GPU: %d\n", i);
          bFail = true;
        }
      }
      // Device to Device copying for all combinations
      for (int i = 0; i < Available_Gpus; ++i) {
        for (int j = i+1; j < Available_Gpus; ++j) {
          if (true == gpusIsPeer(i, j)) {
            Data_mismatch = 0;
            HIPCHECK(hipMemcpy(A_d[j], A_d[i], NUM_ELMTS * sizeof(T),
                  hipMemcpyDefault));
            // Copying in direction reverse of above to check if bidirectional
            // access is happening without any error
            HIPCHECK(hipMemcpy(A_d[i], A_d[j], NUM_ELMTS * sizeof(T),
                  hipMemcpyDefault));
            // Copying data to host to verify the content
            HIPCHECK(hipMemcpy(B_h, A_d[j], NUM_ELMTS * sizeof(T),
                  hipMemcpyDefault));
            for (int k = 0; k < NUM_ELMTS; ++k) {
              if (A_h[k] != B_h[k])
                Data_mismatch++;
            }

            if (Data_mismatch.load() != 0) {
              printf("hipMemcpy: Failed between GPU: %d and %d\n", i, j);
              bFail = true;
            }
          }
        }
      }
      break;
    case TEST_MEMCPYH2D:  // To test hipMemcpyHtoD()
      for (int i = 0; i < Available_Gpus; ++i) {
        Data_mismatch = 0;
        HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[i]),
              A_h, NUM_ELMTS * sizeof(T)));
        // Copying data from device to host to check data consistency
        HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELMTS * sizeof(T),
              hipMemcpyDeviceToHost));
        for (size_t j = 0; j < NUM_ELMTS; ++j) {
          if (A_h[j] != B_h[j])
            Data_mismatch++;
        }
        if (Data_mismatch.load() != 0) {
          printf("hipMemcpyHtoD: failed for GPU %d \n", i);
          bFail = true;
        }
      }
      break;
    case TEST_MEMCPYD2H:  // To test hipMemcpyDtoH()--done
      for (int i = 0; i < Available_Gpus; ++i) {
        Data_mismatch = 0;
        HIPCHECK(hipMemcpy(A_d[i], A_h, NUM_ELMTS * sizeof(T),
              hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpyDtoH(B_h, hipDeviceptr_t(A_d[i]),
              NUM_ELMTS * sizeof(T)));
        for (size_t j = 0; j < NUM_ELMTS; ++j) {
          if (A_h[j] != B_h[j])
            Data_mismatch++;
        }
        if (Data_mismatch.load() != 0) {
          printf("hipMemcpyDtoH: failed for GPU %d \n", i);
          bFail = true;
        }
      }
      break;
    case TEST_MEMCPYD2D:  // To test hipMemcpyDtoD()
      if (Available_Gpus > 1) {
        // First copy data from H to D and then from D to D followed by D to H
        // HIPCHECK(hipMemcpyHtoD(A_d[0], A_h, NUM_ELMTS * sizeof(T)));
        for (int i = 0; i < Available_Gpus; ++i) {
          for (int j = i+1; j < Available_Gpus; ++j) {
            if (true == gpusIsPeer(i, j)) {
              Data_mismatch = 0;
              HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[i]),
                    A_h, NUM_ELMTS * sizeof(T)));
              HIPCHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d[j]),
                    hipDeviceptr_t(A_d[i]), NUM_ELMTS * sizeof(T)));
              // Copying in direction reverse of above to check if bidirectional
              // access is happening without any error
              HIPCHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d[i]),
                    hipDeviceptr_t(A_d[j]), NUM_ELMTS * sizeof(T)));
              HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELMTS * sizeof(T),
                    hipMemcpyDeviceToHost));
              for (size_t k = 0; k < NUM_ELMTS; ++k) {
                if (A_h[k] != B_h[k])
                  Data_mismatch++;
              }
              if (Data_mismatch.load() != 0) {
                printf("hipMemcpyDtoD: failed between GPU: %d and %d\n", i, j);
                bFail = true;
              }
            }
          }
        }
      } else {
        // As DtoD is not possible we will transfer data from HtH(A_h to B_h)
        // so as to get through verification step
        HIPCHECK(hipMemcpy(B_h, A_h, NUM_ELMTS * sizeof(T),
              hipMemcpyHostToHost));
        for (size_t i = 0; i < NUM_ELMTS; ++i) {
          if (A_h[i] != B_h[i])
            Data_mismatch++;
        }
        if (Data_mismatch.load() != 0) {
          printf("hipMemcpy (Host to Host): failed\n");
          bFail = true;
        }
      }
      break;
    case TEST_MEMCPYASYNC:  // To test hipMemcpyAsync()
      // Copying data from host to individual devices followed by copying
      // back to host and verifying the data consistency.
      for (int i = 0; i < Available_Gpus; ++i) {
        Data_mismatch = 0;
        HIPCHECK(hipMemcpyAsync(A_d[i], A_h, NUM_ELMTS * sizeof(T),
              hipMemcpyHostToDevice, stream[i]));
        HIPCHECK(hipMemcpyAsync(B_h, A_d[i], NUM_ELMTS * sizeof(T),
              hipMemcpyDeviceToHost, stream[i]));
        HIPCHECK(hipStreamSynchronize(stream[i]));
        for (size_t k = 0; k < NUM_ELMTS; ++k) {
          if (A_h[k] != B_h[k])
            Data_mismatch++;
        }

        if (Data_mismatch.load() != 0) {
          printf("hipMemcpyAsync: failed for GPU %d\n", i);
          bFail = true;
        }
      }
      // Device to Device copying for all combinations
      for (int i = 0; i < Available_Gpus; ++i) {
        for (int j = i+1; j < Available_Gpus; ++j) {
          if (true == gpusIsPeer(i, j)) {
            Data_mismatch = 0;
            HIPCHECK(hipMemcpyAsync(A_d[j], A_d[i], NUM_ELMTS * sizeof(T),
                  hipMemcpyDefault, stream[i]));
            // Copying in direction reverse of above to check if bidirectional
            // access is happening without any error
            HIPCHECK(hipMemcpyAsync(A_d[i], A_d[j], NUM_ELMTS * sizeof(T),
                  hipMemcpyDefault, stream[i]));
            HIPCHECK(hipStreamSynchronize(stream[i]));
            HIPCHECK(hipMemcpy(B_h, A_d[j], NUM_ELMTS * sizeof(T),
                  hipMemcpyDefault));
            for (size_t k = 0; k < NUM_ELMTS; ++k) {
              if (A_h[k] != B_h[k])
                Data_mismatch++;
            }

            if (Data_mismatch.load() != 0) {
              printf("hipMemcpyAsync: Failed between GPU: %d and %d\n", i, j);
              bFail = true;
            }
          }
        }
      }
      break;
    case TEST_MEMCPYH2DASYNC:  // To test hipMemcpyHtoDAsync()
      for (int i = 0; i < Available_Gpus; ++i) {
        Data_mismatch = 0;
        HIPCHECK(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d[i]), A_h,
              NUM_ELMTS * sizeof(T), stream[i]));
        HIPCHECK(hipStreamSynchronize(stream[i]));
        // Copying data from device to host to check data consistency
        HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELMTS * sizeof(T),
              hipMemcpyDeviceToHost));
        for (size_t k = 0; k < NUM_ELMTS; ++k) {
          if (A_h[k] != B_h[k])
            Data_mismatch++;
        }
        if (Data_mismatch.load() != 0) {
          printf("hipMemcpyHtoDAsync: failed for GPU %d \n", i);
          bFail = true;
        }
      }
      break;
    case TEST_MEMCPYD2HASYNC:  // To test hipMemcpyDtoHAsync()
      for (int i = 0; i < Available_Gpus; ++i) {
        Data_mismatch = 0;
        HIPCHECK(hipMemcpy(A_d[i], A_h, NUM_ELMTS * sizeof(T),
              hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpyDtoHAsync(B_h, hipDeviceptr_t(A_d[i]),
              NUM_ELMTS * sizeof(T), stream[i]));
        HIPCHECK(hipStreamSynchronize(stream[i]));
        for (size_t j = 0; j < NUM_ELMTS; ++j) {
          if (A_h[j] != B_h[j])
            Data_mismatch++;
        }
        if (Data_mismatch.load() != 0) {
          printf("hipMemcpyDtoHAsync: failed %d \n", i);
          bFail = true;
        }
      }
      break;
    case TEST_MEMCPYD2DASYNC:  // To test hipMemcpyDtoDAsync()
      if (Available_Gpus > 1) {
        // First copy data from H to D and then from D to D followed by D to H
        HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[0]),
              A_h, NUM_ELMTS * sizeof(T)));
        for (int i = 0; i < Available_Gpus; ++i) {
          for (int j = i+1; j < Available_Gpus; ++j) {
            Data_mismatch = 0;
            if (true == gpusIsPeer(i, j)) {
              HIPCHECK(hipSetDevice(j));
              HIPCHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d[j]),
                    hipDeviceptr_t(A_d[i]), NUM_ELMTS * sizeof(T), stream[i]));
              // Copying in direction reverse of above to check if bidirectional
              // access is happening without any error
              HIPCHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d[i]),
                    hipDeviceptr_t(A_d[j]), NUM_ELMTS * sizeof(T), stream[i]));
              HIPCHECK(hipStreamSynchronize(stream[i]));
              HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELMTS * sizeof(T),
                    hipMemcpyDeviceToHost));
              for (size_t k = 0; k < NUM_ELMTS; ++k) {
                if (A_h[k] != B_h[k])
                  Data_mismatch++;
              }
              if (Data_mismatch.load() != 0) {
                printf("hipMemcpyDtoDAsync: failed GPU: %d and %d\n", i, j);
                bFail = true;
              }
            }
          }
        }
      } else {
        // As DtoD is not possible we will transfer data from HtH(A_h to B_h)
        // so as to get through verification step
        Data_mismatch = 0;
        HIPCHECK(hipMemcpy(B_h, A_h, NUM_ELMTS * sizeof(T),
              hipMemcpyHostToHost));
        for (size_t i = 0; i < NUM_ELMTS; ++i) {
          if (A_h[i] != B_h[i])
            Data_mismatch++;
        }
        if (Data_mismatch.load() != 0) {
          printf("hipMemcpy (Host to Host): failed\n");
          bFail = true;
        }
      }
      break;
    default:
      printf("Did not receive valid option!\n");
      break;
  }

  for (int i = 0; i < Available_Gpus; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipFree((A_d[i])));
    if (api >= TEST_MEMCPYD2D) {
      HIPCHECK(hipStreamDestroy(stream[i]));
    }
  }

  // Return true if test is success
  if (bFail == true) {
    return false;
  } else {
    return true;
  }
}

template <typename T>
memcpyTests<T>::~memcpyTests() {
  free(A_h);
  free(B_h);
}

void Thread_func(int Threadid) {
  for (apiToTest api = TEST_MEMCPY; api < TEST_MAX; api = apiToTest(api + 1)) {
    memcpyTests<int> obj(api, 1024);
    if (false == obj.Memcpy_And_verify()) {
      failureCount++;
    }
  }
}

int parseExtraArguments(int argc, char* argv[]) {
  int i = 0;
  for (i = 1; i < argc; i++) {
    const char* arg = argv[i];
    if (!strcmp(arg, " ")) {
      // skip nullptr args.
    } else if (!strcmp(arg, "--memcpyPeersOnly")) {
        if (++i >= argc || !HipTest::parseInt(argv[i], &memcpyPeersOnly)) {
          failed("Bad memcpyPeersOnly argument");
        }
    } else if (!strcmp(arg, "--testAllTypes")) {
        if (++i >= argc || !HipTest::parseInt(argv[i], &testAllTypes)) {
          failed("Bad testAllTypes argument");
        }
    } else {
        failed("Bad argument");
    }
  }
  return i;
}


int main(int argc, char* argv[]) {
  bool TestPassed = true;
  int extraArgs = 0;
  HIPCHECK(hipGetDeviceCount(&Available_Gpus));
  extraArgs = HipTest::parseStandardArguments(argc, argv, false);
  parseExtraArguments(extraArgs, argv);

  if (p_tests == 1) {
    Memcpy_Negative_Tests test;
    TestPassed = test.Test_Memcpy();
    TestPassed &= test.Test_MemcpyAsync();
    TestPassed &= test.Test_MemcpyHtoD();
    TestPassed &= test.Test_MemcpyHtoDAsync();
    TestPassed &= test.Test_MemcpyDtoD();
    TestPassed &= test.Test_MemcpyDtoDAsync();
    TestPassed &= test.Test_MemcpyDtoH();
    TestPassed &= test.Test_MemcpyDtoHAsync();
    if (TestPassed) {
      passed();
    } else {
      failed("Test Failed!");
    }
  } else if (p_tests == 2) {
    failureCount = 0;
    std::thread Thrd[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
    Thrd[i] = std::thread(Thread_func, i);

    // Thread join is being called separately so as to allow the
    // threads run parallely
    for (int i = 0; i < NUM_THREADS; i++)
      Thrd[i].join();
    if (failureCount.load() != 0) {
      failed("Failed");
    } else {
      passed();
    }
  } else if (p_tests == 3) {
    size_t free = 0, total = 0;
    HIPCHECK(hipMemGetInfo(&free, &total));
    failureCount = 0;
    // Need to see if allocating all of available free memory will result in
    // any issues in windows system before adding the same
    std::vector<size_t> NUM_ELMTS{1, 5, 10, 100, 1024, 10*1024, 100*1024,
                                  1024*1024, 10*1024*1024, 100*1024*1024,
                                  1024*1024*1024};

    for (apiToTest api = TEST_MEMCPY; api < TEST_MAX; api = apiToTest(api+1)) {
      for (size_t x : NUM_ELMTS) {
        if ((x * sizeof(char)) <= free) {
          memcpyTests<char> obj(api, x);
          TestPassed &= obj.Memcpy_And_verify();
          HIPCHECK(hipDeviceSynchronize());
        }

        if (HIPTEST_TRUE == testAllTypes) {
          // Testing memcpy with various data types
          if ((x * sizeof(int)) <= free) {
            memcpyTests<int> obj(api, x);
            TestPassed &= obj.Memcpy_And_verify();
            HIPCHECK(hipDeviceSynchronize());
          }
          if ((x * sizeof(size_t)) <= free) {
            memcpyTests<size_t> obj(api, x);
            TestPassed &= obj.Memcpy_And_verify();
            HIPCHECK(hipDeviceSynchronize());
          }
          if ((x * sizeof(long double)) <= free) {
            memcpyTests<long double> obj(api, x);
            TestPassed &= obj.Memcpy_And_verify();
            HIPCHECK(hipDeviceSynchronize());
          }
        }
      }
    }
    if (TestPassed) {
      passed();
    } else {
      failed("Test Failed!");
    }
  } else {
    failed("Didnt receive any valid option\n");
  }
}
