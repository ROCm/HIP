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
#include "hip/hip_runtime.h"
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
    hipStream_t stream;
    memcpyTests(apiToTest val, size_t num_elmts);
    bool Memcpy_And_verify();
    ~memcpyTests();
};

class Memcpy_Negative_Tests {
 public:
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


bool Memcpy_Negative_Tests::Test_Memcpy(void) {
  bool IfTestPassed = true;
  std::string str_out, str_err = "hipErrorInvalidValue";
  float *A_h = NULL, *B_h = NULL, *A_d = NULL, *A_d1 = NULL;
  A_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  B_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));

  if ((A_h == NULL) || (B_h == NULL)) {
    failed("Malloc call failed!");
  }

  HIPCHECK(hipMalloc(&A_d, (NUM_ELM*sizeof(float))));
  HIPCHECK(hipMalloc(&A_d1, (NUM_ELM*sizeof(float))));

  for ( int i = 0; i < NUM_ELM; ++i ) {
    A_h[i] = 123;
    B_h[i] = 0;
  }

  // Copying only half the memory on device side from host
  HIPCHECK(hipMemcpy(A_d, A_h, (NUM_ELM/2) * sizeof(float), hipMemcpyDefault));
  // Copying device memory to host to verify if the content is consistent
  HIPCHECK(hipMemcpy(B_h, A_d, NUM_ELM * sizeof(float), hipMemcpyDefault));
  // Verifying the host content copied in the above step for consistency.
  int Data_mismatch = 0;

  for (int i = 0; i < (NUM_ELM/2); ++i) {
    if (B_h[i] != 123) {
      Data_mismatch++;
    }
  }

  if (Data_mismatch != 0) {
    printf("Data Mismatch for negative test\n");
    IfTestPassed = false;
  }

  str_out = hipGetErrorString(hipMemcpy(NULL, A_d, (NUM_ELM/2) * sizeof(float),
                                        hipMemcpyDefault));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpy with NULL for destination parameter.\n");
    printf("Error: %s\n", str_out.c_str());
    IfTestPassed = false;
  }

  str_out = hipGetErrorString(hipMemcpy(A_h, NULL, (NUM_ELM/2) * sizeof(float),
                                        hipMemcpyDefault));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpy with NULL for source\n");
    IfTestPassed = false;
  }

  str_out = hipGetErrorString(hipMemcpy(NULL, NULL, (NUM_ELM/2) * sizeof(float),
                                        hipMemcpyDefault));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpy with NULL for source and destination\n");
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
    }
  }

  if (Data_mismatch != 0) {
    printf("Data Mismatch after memcpy of same src and destination\n");
    IfTestPassed = false;
  }

  // Memory copy on same device with two different regions
  HIPCHECK(hipMemcpy(A_d1, A_d, (NUM_ELM) * sizeof(float), hipMemcpyDefault));

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(A_d1));
  free(A_h);
  free(B_h);

  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyAsync(void) {
  bool IfTestPassed = true;
  float *A_h = NULL, *B_h = NULL, *A_d = NULL, *A_d1 = NULL;
  std::string str_out, str_err = "hipErrorInvalidValue";
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));
  A_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  B_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));

  if ((A_h == nullptr) || (B_h == nullptr)) {
    failed("Malloc call failed!");
  }

  HIPCHECK(hipMalloc(&A_d, (NUM_ELM * sizeof(float))));
  HIPCHECK(hipMalloc(&A_d1, (NUM_ELM * sizeof(float))));
  for (int i = 0; i < NUM_ELM; ++i) {
    A_h[i] = 123;
    B_h[i] = 0;
  }

  // Copying host data into the device.
  HIPCHECK(hipMemcpyAsync(A_d1, A_h, NUM_ELM * sizeof(float),
                          hipMemcpyDefault, stream));

  // Passing null pointer: seg fault observed with the following.
  str_out = hipGetErrorString(hipMemcpyAsync(NULL, A_h, NUM_ELM * sizeof(float),
                              hipMemcpyDefault, stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyAsync with NULL for destination\n");
    IfTestPassed = false;
  }

  str_out = hipGetErrorString(hipMemcpyAsync(A_d, NULL, NUM_ELM * sizeof(float),
                                             hipMemcpyDefault, stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyAsync with NULL for source\n");
    IfTestPassed = false;
  }

  str_out = hipGetErrorString(hipMemcpyAsync(NULL, NULL,
                                             NUM_ELM * sizeof(float),
                                             hipMemcpyDefault, stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyAsync with NULL for source and destination\n");
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
    }
  }

  if (Data_mismatch != 0) {
    printf("Data Mismatch after memcpyAsync of same src and destination\n");
    IfTestPassed = false;
  }

  // Memory copy on same device with two different regions
  HIPCHECK(hipMemcpyAsync(A_d1, A_d, (NUM_ELM) * sizeof(float),
                          hipMemcpyDefault, stream));

  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(A_d1));
  free(A_h);
  free(B_h);

  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyHtoD(void) {
  bool IfTestPassed = true;
  float *A_h = NULL, *B_h = NULL, *A_d = NULL, *A_d1 = NULL;
  std::string str_out, str_err = "hipErrorInvalidValue";
  A_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  B_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  if ((A_h == nullptr) || (B_h == nullptr)) {
    failed("Malloc call failed!");
  }
  HIPCHECK(hipMalloc(&A_d, (NUM_ELM * sizeof(float))));
  HIPCHECK(hipMalloc(&A_d1, (NUM_ELM * sizeof(float))));
  for (int i = 0; i < NUM_ELM; ++i) {
    A_h[i] = 123;
    B_h[i] = 0;
  }

  // Passing null ptr to check the API behavior.
  // Expectation: It should not crash and exit gracefully.
  str_out = hipGetErrorString(hipMemcpyHtoD(NULL, A_h,
                              NUM_ELM * sizeof(float)));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyHtoD with NULL for destination\n");
    IfTestPassed = false;
  }
  str_out = hipGetErrorString(hipMemcpyHtoD(A_d, NULL,
                                            NUM_ELM * sizeof(float)));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyHtoD with NULL for source\n");
    IfTestPassed = false;
  }
  str_out = hipGetErrorString(hipMemcpyHtoD(NULL, NULL,
                                            NUM_ELM * sizeof(float)));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyHtoD with NULL for source and destination\n");
    IfTestPassed = false;
  }
  // Copy half of the allocated memory
  HIPCHECK(hipMemcpyHtoD(A_d, A_h, NUM_ELM * sizeof(float) / 2));
  // copying back to host to verify
  HIPCHECK(hipMemcpyDtoH(B_h, A_d, NUM_ELM * sizeof(float)));

  int Data_mismatch = 0;
  for (int i = 0; i < (NUM_ELM / 2); ++i)
    if (B_h[i] != 123)
      Data_mismatch++;

  if (Data_mismatch != 0) {
    printf("Data Mismatch after hipMemcpyHtoD with half size\n");
    IfTestPassed = false;
  }

  // Setting device memory to zero
  HIPCHECK(hipMemset(A_d, 0, NUM_ELM * sizeof(float)));
  // Swap source and destination pointer
  HIPCHECK(hipMemcpyHtoD(A_h, A_d, NUM_ELM * sizeof(float)));

  // Pass same pointers in source and destination params
  HIPCHECK(hipMemcpyHtoD(A_h, A_h, NUM_ELM * sizeof(float)));
  HIPCHECK(hipMemcpyHtoD(A_d, A_d, NUM_ELM * sizeof(float)));

  // Mem copy on same device with two different regions
  HIPCHECK(hipMemcpyHtoD(A_d1, A_d, NUM_ELM * sizeof(float)));

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(A_d1));
  free(A_h);
  free(B_h);

  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyHtoDAsync(void) {
  bool IfTestPassed = true;
  float *A_h = NULL, *B_h = NULL, *A_d = NULL, *A_d1 = NULL;
  std::string str_out, str_err = "hipErrorInvalidValue";
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));
  A_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  B_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  if ((A_h == nullptr) || (B_h == nullptr)) {
    failed("Malloc call failed!");
  }
  HIPCHECK(hipMalloc(&A_d, (NUM_ELM * sizeof(float))));
  HIPCHECK(hipMalloc(&A_d1, (NUM_ELM * sizeof(float))));
  for (int i = 0; i < NUM_ELM; ++i) {
    A_h[i] = 123;
    B_h[i] = 0;
  }

  // Passing null ptr to check the API behavior.
  // Expectation: It should not crash and exit gracefully.
  str_out = hipGetErrorString(hipMemcpyHtoDAsync(NULL, A_h,
                                                 NUM_ELM * sizeof(float),
                                                 stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyHtoDAsync with NULL for destination\n");
    IfTestPassed = false;
  }

  str_out = hipGetErrorString(hipMemcpyHtoDAsync(A_d, NULL,
                                                 NUM_ELM * sizeof(float),
                                                 stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyHtoDAsync with NULL for source\n");
    IfTestPassed = false;
  }
  str_out = hipGetErrorString(hipMemcpyHtoDAsync(NULL, NULL,
                                                 NUM_ELM * sizeof(float),
                                                 stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed MemcpyHtoDAsync with NULL for source and destination\n");
    IfTestPassed = false;
  }

  // Copy half of the allocated memory
  HIPCHECK(hipMemcpyHtoDAsync(A_d, A_h, NUM_ELM * sizeof(float)/2, stream));
  // copying back to host to verify
  HIPCHECK(hipMemcpyDtoH(B_h, A_d, NUM_ELM * sizeof(float)));
  int Data_mismatch = 0;
  for (int i = 0; i < (NUM_ELM/2); ++i)
    if (B_h[i] != 123)
      Data_mismatch++;
  if (Data_mismatch != 0) {
    printf("Data Mismatch after hipMemcpyHtoDAsync with half size\n");
    IfTestPassed = false;
  }

  // Setting device memory to zero
  HIPCHECK(hipMemset(A_d, 0, NUM_ELM * sizeof(float)));
  // Swap source and destination pointer
  HIPCHECK(hipMemcpyHtoDAsync(B_h, A_d, NUM_ELM * sizeof(float), stream));
  HIPCHECK(hipStreamSynchronize(stream));
  if (B_h[0] != 0) {
    printf("Data Mismatch after hipMemcpyHtoDAsync with memset to 0\n");
    IfTestPassed = false;
  }

  // Pass same pointers in source and destination params
  HIPCHECK(hipMemcpyHtoDAsync(A_h, A_h, NUM_ELM * sizeof(float), stream));
  HIPCHECK(hipMemcpyHtoDAsync(A_d, A_d, NUM_ELM * sizeof(float), stream));

  // Mem copy on same device with two different regions
  HIPCHECK(hipMemcpyHtoDAsync(A_d1, A_d, NUM_ELM * sizeof(float), stream));
  HIPCHECK(hipStreamSynchronize(stream));
  // Checking the api with null stream
  HIPCHECK(hipMemcpyHtoDAsync(A_d1, A_d, NUM_ELM * sizeof(float), 0));
  HIPCHECK(hipStreamSynchronize(stream));

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(A_d1));
  free(A_h);
  free(B_h);

  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyDtoH(void) {
  bool IfTestPassed = true;
  float *A_h = NULL, *B_h = NULL, *A_d = NULL, *A_d1 = NULL;
  std::string str_out, str_err = "hipErrorInvalidValue";
  A_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  B_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  if ((A_h == nullptr) || (B_h == nullptr)) {
    failed("Malloc call failed!");
  }
  HIPCHECK(hipMalloc(&A_d, (NUM_ELM * sizeof(float))));
  HIPCHECK(hipMalloc(&A_d1, (NUM_ELM * sizeof(float))));
  for (int i = 0; i < NUM_ELM; ++i) {
    A_h[i] = 123;
    B_h[i] = 0;
  }

  // Copying data from host to device for further operations
  HIPCHECK(hipMemcpyHtoD(A_d, A_h, NUM_ELM * sizeof(float)));

  // Passing null ptr to check the API behavior.
  // Expectation: It should not crash and exit gracefully.
  str_out = hipGetErrorString(hipMemcpyDtoH(NULL, A_d,
                                            NUM_ELM * sizeof(float)));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyDtoH with NULL for destination\n");
    IfTestPassed = false;
  }
  str_out = hipGetErrorString(hipMemcpyDtoH(A_d, NULL,
                                            NUM_ELM * sizeof(float)));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyDtoH with NULL for source\n");
    IfTestPassed = false;
  }
  str_out = hipGetErrorString(hipMemcpyDtoH(NULL, NULL,
                                            NUM_ELM * sizeof(float)));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyDtoH with NULL for source and destination\n");
    IfTestPassed = false;
  }
  // Copy half of the allocated memory
  HIPCHECK(hipMemcpyDtoH(B_h, A_d, NUM_ELM * sizeof(float)/2));

  int Data_mismatch = 0;
  for (int i = 0; i < (NUM_ELM/2); ++i)
    if (B_h[i] != 123)
      Data_mismatch++;

  if (Data_mismatch != 0) {
    printf("Data Mismatch after hipMemcpyDtoH with half size\n");
    IfTestPassed = false;
  }

  // Setting device memory to zero
  HIPCHECK(hipMemset(A_d, 0, NUM_ELM * sizeof(float)));
  // Swap source and destination pointer
  HIPCHECK(hipMemcpyDtoH(A_d, A_h, NUM_ELM * sizeof(float)));

  // Pass same pointers in source and destination params
  HIPCHECK(hipMemcpyDtoH(A_h, A_h, NUM_ELM * sizeof(float)));
  HIPCHECK(hipMemcpyDtoH(A_d, A_d, NUM_ELM * sizeof(float)));

  // Mem copy on same device with two diffeent regions
  HIPCHECK(hipMemcpyDtoH(A_d1, A_d, NUM_ELM * sizeof(float)));

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(A_d1));
  free(A_h);
  free(B_h);

  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyDtoHAsync(void) {
  bool IfTestPassed = true;
  float *A_h = NULL, *B_h = NULL, *A_d = NULL, *A_d1 = NULL;
  std::string str_out, str_err = "hipErrorInvalidValue";
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));
  A_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  B_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  if ((A_h == nullptr) || (B_h == nullptr)) {
    failed("Malloc call failed!");
  }
  HIPCHECK(hipMalloc(&A_d, (NUM_ELM * sizeof(float))));
  HIPCHECK(hipMalloc(&A_d1, (NUM_ELM * sizeof(float))));
  for (int i = 0; i < NUM_ELM; ++i) {
    A_h[i] = 123;
    B_h[i] = 0;
  }

  // Copying data from host to device for further operations
  HIPCHECK(hipMemcpyHtoDAsync(A_d, A_h, NUM_ELM * sizeof(float), stream));

  // Passing null ptr to check the API behavior.
  // Expectation: It should not crash and exit gracefully.
  str_out = hipGetErrorString(hipMemcpyDtoHAsync(NULL, A_d,
                                                 NUM_ELM * sizeof(float),
                                                 stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyDtoHAsync with NULL for destination\n");
    IfTestPassed = false;
  }
  str_out = hipGetErrorString(hipMemcpyDtoHAsync(A_d, NULL,
                                                 NUM_ELM * sizeof(float),
                                                 stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyDtoHAsync with NULL for source\n");
    IfTestPassed = false;
  }
  str_out = hipGetErrorString(hipMemcpyDtoHAsync(NULL, NULL,
                                                 NUM_ELM * sizeof(float),
                                                 stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed hipMemcpyDtoHAsync with NULL for source and destination\n");
    IfTestPassed = false;
  }

  // Copy half of the allocated memory
  HIPCHECK(hipMemcpyDtoHAsync(B_h, A_d, NUM_ELM * sizeof(float)/2, stream));
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
  HIPCHECK(hipMemcpyDtoHAsync(B_h, A_d, NUM_ELM * sizeof(float), 0));
  // Setting device memory to zero
  HIPCHECK(hipMemset(A_d, 0, NUM_ELM * sizeof(float)));
  // Swap source and destination pointer
  HIPCHECK(hipMemcpyDtoHAsync(A_d, A_h, NUM_ELM * sizeof(float), stream));

  // Pass same pointers in source and destination params
  HIPCHECK(hipMemcpyDtoHAsync(A_h, A_h, NUM_ELM * sizeof(float), stream));
  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipMemcpyDtoHAsync(A_d, A_d, NUM_ELM * sizeof(float), stream));
  HIPCHECK(hipStreamSynchronize(stream));
  // Mem copy on same device with two different regions
  HIPCHECK(hipMemcpyDtoHAsync(A_d1, A_d, NUM_ELM * sizeof(float), stream));
  HIPCHECK(hipStreamSynchronize(stream));

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(A_d1));
  free(A_h);
  free(B_h);

  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyDtoD(void) {
  bool IfTestPassed = true;
  float *A_h = NULL, *B_h = NULL, *A_d1 = NULL, *A_d2 = NULL, *Ad1 = NULL;
  std::string str_out, str_err = "hipErrorInvalidValue";
  A_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  B_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  if ((A_h == nullptr) || (B_h == nullptr)) {
    failed("Malloc call failed!");
  }
  HIPCHECK(hipMalloc(&A_d1, (NUM_ELM * sizeof(float))));
  HIPCHECK(hipMalloc(&Ad1, (NUM_ELM * sizeof(float))));
  HIPCHECK(hipMemset(A_d1, 0, NUM_ELM * sizeof(float)));
  if (Available_Gpus > 1) {
    HIPCHECK(hipSetDevice(1));
    HIPCHECK(hipMalloc(&A_d2, (NUM_ELM * sizeof(float))));
    HIPCHECK(hipMemset(A_d2, 1, NUM_ELM * sizeof(float)));
  }
  for (int i = 0; i < NUM_ELM; ++i) {
    A_h[i] = 123;
    B_h[i] = 0;
  }
  // Passing null pointers to check the behaviour::
  str_out = hipGetErrorString(hipMemcpyDtoD(&A_d1, NULL,
                                            NUM_ELM * sizeof(float)));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyDtoD with NULL for source\n");
    IfTestPassed = false;
  }
  str_out = hipGetErrorString(hipMemcpyDtoD(NULL, &A_d2,
                                            NUM_ELM * sizeof(float)));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyDtoD with NULL for destination\n");
    IfTestPassed = false;
  }
  str_out = hipGetErrorString(hipMemcpyDtoD(NULL, NULL,
                                            NUM_ELM * sizeof(float)));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyDtoD with NULL for source and destination\n");
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
    HIPCHECK(hipMemcpyHtoD(A_d1, A_h, NUM_ELM * sizeof(float)));
    if (true == gpusIsPeer(0, 1)) {
      HIPCHECK(hipMemcpyDtoD(A_d2, A_d1, NUM_ELM * sizeof(float)/2));
      HIPCHECK(hipMemcpyDtoH(B_h, A_d2, NUM_ELM * sizeof(float)));
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
  HIPCHECK(hipMemcpyDtoD(A_d1, A_d1, NUM_ELM * sizeof(float)));
  if (Available_Gpus > 1) {
    HIPCHECK(hipMemcpyDtoD(A_d2, A_d2, NUM_ELM * sizeof(float)));
  }
  // Memcpy on same device with two different regions
  HIPCHECK(hipMemcpyDtoD(Ad1, A_d1, NUM_ELM * sizeof(float)));

  HIPCHECK(hipFree(A_d1));
  HIPCHECK(hipFree(Ad1));
  if (Available_Gpus > 1)
    HIPCHECK(hipFree(A_d2));
  free(A_h);
  free(B_h);

  return IfTestPassed;
}

bool Memcpy_Negative_Tests::Test_MemcpyDtoDAsync(void) {
  bool IfTestPassed = true;
  float *A_h = NULL, *B_h = NULL, *A_d1 = NULL, *A_d2 = NULL, *Ad1 = NULL;
  std::string str_out, str_err = "hipErrorInvalidValue";
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));
  A_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  B_h = reinterpret_cast<float*>(malloc(NUM_ELM * sizeof(float)));
  if ((A_h == nullptr) || (B_h == nullptr)) {
    failed("Malloc call failed!");
  }
  HIPCHECK(hipMalloc(&A_d1, (NUM_ELM * sizeof(float))));
  HIPCHECK(hipMalloc(&Ad1, (NUM_ELM * sizeof(float))));
  HIPCHECK(hipMemset(A_d1, 0, NUM_ELM * sizeof(float)));
  if (Available_Gpus > 1) {
    HIPCHECK(hipSetDevice(1));
    HIPCHECK(hipMalloc(&A_d2, (NUM_ELM * sizeof(float))));
    HIPCHECK(hipMemset(A_d2, 1, NUM_ELM * sizeof(float)));
  }
  for (int i = 0; i < NUM_ELM; ++i) {
    A_h[i] = 123;
    B_h[i] = 0;
  }
  // Passing null pointers to check the behaviour::
  str_out =  hipGetErrorString(hipMemcpyDtoDAsync(&A_d1, NULL,
                                                  NUM_ELM * sizeof(float),
                                                  stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyDtoDAsync with NULL for source\n");
    IfTestPassed = false;
  }
  str_out =  hipGetErrorString(hipMemcpyDtoDAsync(NULL, &A_d2,
                                                  NUM_ELM * sizeof(float),
                                                  stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed for hipMemcpyDtoDAsync with NULL for destination\n");
    IfTestPassed = false;
  }
  str_out =  hipGetErrorString(hipMemcpyDtoDAsync(NULL, NULL,
                                                  NUM_ELM * sizeof(float),
                                                  stream));
  if ((str_err.compare(str_out)) != 0) {
    printf("Failed MemcpyDtoDAsync with NULL for source and destination\n");
    IfTestPassed = false;
  }

  int Data_mismatch = 0;
  // Copying half of actually allocated memory
  HIPCHECK(hipSetDevice(0));
  if (Available_Gpus > 1) {
    HIPCHECK(hipMemcpyHtoD(A_d1, A_h, NUM_ELM * sizeof(float)));
    if (true == gpusIsPeer(0, 1)) {
      HIPCHECK(hipMemcpyDtoDAsync(A_d2, A_d1, NUM_ELM * sizeof(float)/2,
                                  stream));
      HIPCHECK(hipMemcpyDtoH(B_h, A_d2, NUM_ELM * sizeof(float)));
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

  // Memcpy on same device with two different regions
  HIPCHECK(hipMemcpyDtoDAsync(Ad1, A_d1, NUM_ELM * sizeof(float) , stream));
  // Testing hipMemcpyDtoDAsync between two devices.
  if (Available_Gpus > 1) {
    if (true == gpusIsPeer(0, 1)) {
      HIPCHECK(hipMemcpyDtoDAsync(A_d2, A_d1, NUM_ELM * sizeof(float), 0));
    }
  }
  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipFree(A_d1));
  HIPCHECK(hipFree(Ad1));
  if (Available_Gpus > 1)
    HIPCHECK(hipFree(A_d2));
  free(A_h);
  free(B_h);

  return IfTestPassed;
}

template <typename T>
memcpyTests<T>::memcpyTests(apiToTest val, size_t num_elmts) {
  api = val;
  NUM_ELMTS = num_elmts;
  printf("%zu ", NUM_ELMTS * sizeof(T));
  fflush(stdout);
  A_h = reinterpret_cast<T*>(malloc(NUM_ELMTS * sizeof(T)));
  B_h = reinterpret_cast<T*>(malloc(NUM_ELMTS * sizeof(T)));
  if ((A_h == NULL) || (B_h == NULL)) {
    exit(1);
  }

  if (api >= TEST_MEMCPYD2D) {
    HIPCHECK(hipStreamCreate(&stream));
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
  for (int i = 0; i < Available_Gpus; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMalloc(&A_d[i], NUM_ELMTS * sizeof(T)));
  }
  HIPCHECK(hipSetDevice(0));

  switch (api) {
    case TEST_MEMCPY:  // To test hipMemcpy()
      // Copying data from host to individual devices followed by copying
      // back to host and verifying the data consistency.
      for (int i = 0; i < Available_Gpus; ++i) {
        // HIPCHECK(hipSetDevice(i));
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
        for (int j = 1; j < Available_Gpus; ++j) {
          if (true == gpusIsPeer(i, j)) {
            HIPCHECK(hipMemcpy(A_d[j], A_d[i], NUM_ELMTS * sizeof(T),
                               hipMemcpyDefault));
            // Copying in direction reverse of above to check if bidirectional
            // access is happening without any error
            HIPCHECK(hipMemcpy(A_d[i], A_d[j], NUM_ELMTS * sizeof(T),
                               hipMemcpyDefault));
            // Copying data to host to verify the content
            HIPCHECK(hipMemcpy(B_h, A_d[j], NUM_ELMTS * sizeof(T),
                               hipMemcpyDefault));
            for (int i = 0; i < NUM_ELMTS; ++i) {
            if (A_h[i] != B_h[i])
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
        HIPCHECK(hipMemcpyHtoD(A_d[i], A_h, NUM_ELMTS * sizeof(T)));
        // Copying data from device to host to check data consistency
        HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELMTS * sizeof(T),
                 hipMemcpyDeviceToHost));
        for (size_t i = 0; i < NUM_ELMTS; ++i) {
          if (A_h[i] != B_h[i])
            Data_mismatch++;
        }
        if (Data_mismatch.load() != 0) {
          printf("hipMemcpyHtoD: failed\n");
          bFail = true;
        }
      }
      break;
    case TEST_MEMCPYD2H:  // To test hipMemcpyDtoH()--done
      for (int i = 0; i < Available_Gpus; ++i) {
        HIPCHECK(hipMemcpy(A_d[i], A_h, NUM_ELMTS * sizeof(T),
               hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpyDtoH(B_h, A_d[i], NUM_ELMTS * sizeof(T)));
        for (size_t i = 0; i < NUM_ELMTS; ++i) {
          if (A_h[i] != B_h[i])
            Data_mismatch++;
        }
        if (Data_mismatch.load() != 0) {
          printf("hipMemcpyDtoH: failed\n");
          bFail = true;
        }
      }
      break;
    case TEST_MEMCPYD2D:  // To test hipMemcpyDtoD()
      if (Available_Gpus > 1) {
        // First copy data from H to D and then from D to D followed by D to H
        // HIPCHECK(hipMemcpyHtoD(A_d[0], A_h, NUM_ELMTS * sizeof(T)));
        for (int i = 0; i < Available_Gpus; ++i) {
          for (int j = 1; j < Available_Gpus; ++j) {
            if (true == gpusIsPeer(i, j)) {
              HIPCHECK(hipMemcpyHtoD(A_d[i], A_h, NUM_ELMTS * sizeof(T)));
              HIPCHECK(hipMemcpyDtoD(A_d[j], A_d[i], NUM_ELMTS * sizeof(T)));
              // Copying in direction reverse of above to check if bidirectional
              // access is happening without any error
              HIPCHECK(hipMemcpyDtoD(A_d[i], A_d[j], NUM_ELMTS * sizeof(T)));
              HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELMTS * sizeof(T),
                                 hipMemcpyDeviceToHost));
              for (size_t i = 0; i < NUM_ELMTS; ++i) {
                if (A_h[i] != B_h[i])
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
        HIPCHECK(hipMemcpyAsync(A_d[i], A_h, NUM_ELMTS * sizeof(T),
                                hipMemcpyHostToDevice, stream));
        HIPCHECK(hipMemcpyAsync(B_h, A_d[i], NUM_ELMTS * sizeof(T),
                                hipMemcpyDeviceToHost, stream));
        for (size_t i = 0; i < NUM_ELMTS; ++i) {
          if (A_h[i] != B_h[i])
            Data_mismatch++;
        }

        if (Data_mismatch.load() != 0) {
          printf("hipMemcpyAsync: failed for GPU %d\n", i);
          bFail = true;
        }
      }
       // Device to Device copying for all combinations
      for (int i = 0; i < Available_Gpus; ++i) {
        for (int j = 1; j < Available_Gpus; ++j) {
          if (true == gpusIsPeer(i, j)) {
            HIPCHECK(hipMemcpyAsync(A_d[j], A_d[i], NUM_ELMTS * sizeof(T),
                                    hipMemcpyDefault, stream));
            // Copying in direction reverse of above to check if bidirectional
            // access is happening without any error
            HIPCHECK(hipMemcpyAsync(A_d[i], A_d[j], NUM_ELMTS * sizeof(T),
                                   hipMemcpyDefault, stream));
            HIPCHECK(hipMemcpy(B_h, A_d[j], NUM_ELMTS * sizeof(T),
                               hipMemcpyDefault));
            for (size_t i = 0; i < NUM_ELMTS; ++i) {
             if (A_h[i] != B_h[i])
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
        HIPCHECK(hipMemcpyHtoDAsync(A_d[i], A_h, NUM_ELMTS * sizeof(T),
                                    stream));
        // Copying data from device to host to check data consistency
        HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELMTS * sizeof(T),
                 hipMemcpyDeviceToHost));
        for (size_t i = 0; i < NUM_ELMTS; ++i) {
          if (A_h[i] != B_h[i])
             Data_mismatch++;
        }
        if (Data_mismatch.load() != 0) {
          printf("hipMemcpyHtoDAsync: failed\n");
          bFail = true;
        }
      }
      break;
    case TEST_MEMCPYD2HASYNC:  // To test hipMemcpyDtoHAsync()
      for (int i = 0; i < Available_Gpus; ++i) {
        HIPCHECK(hipMemcpy(A_d[i], A_h, NUM_ELMTS * sizeof(T),
                           hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpyDtoHAsync(B_h, A_d[i], NUM_ELMTS * sizeof(T),
                                    stream));
        for (size_t i = 0; i < NUM_ELMTS; ++i) {
          if (A_h[i] != B_h[i])
            Data_mismatch++;
        }
        if (Data_mismatch.load() != 0) {
          printf("hipMemcpyDtoHAsync: failed\n");
          bFail = true;
        }
      }
      break;
    case TEST_MEMCPYD2DASYNC:  // To test hipMemcpyDtoDAsync()
      if (Available_Gpus > 1) {
        // First copy data from H to D and then from D to D followed by D to H
        HIPCHECK(hipMemcpyHtoD(A_d[0], A_h, NUM_ELMTS * sizeof(T)));
        for (int i = 0; i < Available_Gpus; ++i) {
          for (int j = 1; j < Available_Gpus; ++j) {
            if (true == gpusIsPeer(i, j)) {
              HIPCHECK(hipSetDevice(j));
              HIPCHECK(hipMemcpyDtoDAsync(A_d[j], A_d[i], NUM_ELMTS * sizeof(T),
                                          stream));
              // Copying in direction reverse of above to check if bidirectional
              // access is happening without any error
              HIPCHECK(hipMemcpyDtoDAsync(A_d[i], A_d[j], NUM_ELMTS * sizeof(T),
                                          stream));
              HIPCHECK(hipDeviceSynchronize());
              HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELMTS * sizeof(T),
                                 hipMemcpyDeviceToHost));
              for (size_t i = 0; i < NUM_ELMTS; ++i) {
                if (A_h[i] != B_h[i])
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
    HIPCHECK(hipFree((A_d[i])));
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
  if (api >= TEST_MEMCPYD2D) {
    HIPCHECK(hipStreamDestroy(stream));
  }
}

void Thread_func(int Threadid) {
  for (apiToTest api = TEST_MEMCPY; api < TEST_MAX; api = apiToTest(api + 1)) {
    memcpyTests<int> obj(api, 1024*1024);
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
      // skip NULL args.
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
      printf("\nTesting %s for size: ", apiNameToTest[api].c_str());

      // Check for 0 size
      memcpyTests<char> obj(api, 0);
      obj.Memcpy_And_verify();
      HIPCHECK(hipDeviceSynchronize());

      for (size_t x : NUM_ELMTS) {
        if ((x * sizeof(char)) <= free) {
          memcpyTests<char> obj(api, x);
          obj.Memcpy_And_verify();
          HIPCHECK(hipDeviceSynchronize());
        }

        if (HIPTEST_TRUE == testAllTypes) {
          // Testing memcpy with various data types
          if ((x * sizeof(int)) <= free) {
            memcpyTests<int> obj(api, x);
            obj.Memcpy_And_verify();
            HIPCHECK(hipDeviceSynchronize());
          }
          if ((x * sizeof(size_t)) <= free) {
            memcpyTests<size_t> obj(api, x);
            obj.Memcpy_And_verify();
            HIPCHECK(hipDeviceSynchronize());
          }
          if ((x * sizeof(long double)) <= free) {
            memcpyTests<long double> obj(api, x);
            obj.Memcpy_And_verify();
            HIPCHECK(hipDeviceSynchronize());
          }
        }
      }
    }
    printf("\n");
    passed();
  } else {
    failed("Didnt receive any valid option\n");
  }
}
