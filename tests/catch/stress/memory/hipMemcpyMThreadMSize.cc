/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <utility>
#include <vector>
/*
This testfile verifies the following scenarios of all hipMemcpy API
1. Multi thread
2. Multi size
*/

static auto Available_Gpus{0};
static constexpr auto MAX_GPU{256};

enum apiToTest {TEST_MEMCPY, TEST_MEMCPYH2D, TEST_MEMCPYD2H, TEST_MEMCPYD2D,
                TEST_MEMCPYASYNC, TEST_MEMCPYH2DASYNC, TEST_MEMCPYD2HASYNC,
                TEST_MEMCPYD2DASYNC};

template<typename TestType>
void Memcpy_And_verify(int NUM_ELM) {
  TestType *A_h, *B_h;
  for (apiToTest api = TEST_MEMCPY; api <= TEST_MEMCPYD2DASYNC;
      api = apiToTest(api + 1)) {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
        &A_h, &B_h, nullptr,
        NUM_ELM);
    HIP_CHECK(hipGetDeviceCount(&Available_Gpus));
    TestType *A_d[MAX_GPU];
    hipStream_t stream[MAX_GPU];
    for (int i = 0; i < Available_Gpus; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipMalloc(&A_d[i], NUM_ELM * sizeof(TestType)));
      if (api >= TEST_MEMCPYD2D) {
        HIP_CHECK(hipStreamCreate(&stream[i]));
      }
    }
    HIP_CHECK(hipSetDevice(0));
    int canAccessPeer = 0;
    switch (api) {
      case TEST_MEMCPY:
        {
          // To test hipMemcpy()
          // Copying data from host to individual devices followed by copying
          // back to host and verifying the data consistency.
          for (int i = 0; i < Available_Gpus; ++i) {
            HIP_CHECK(hipMemcpy(A_d[i], A_h, NUM_ELM * sizeof(TestType),
                  hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(TestType),
                  hipMemcpyDeviceToHost));
            HipTest::checkTest(A_h, B_h, NUM_ELM);
          }
          // Device to Device copying for all combinations
          for (int i = 0; i < Available_Gpus; ++i) {
            for (int j = i+1; j < Available_Gpus; ++j) {
              canAccessPeer = 0;
              hipDeviceCanAccessPeer(&canAccessPeer, i, j);
              if (canAccessPeer) {
                HIP_CHECK(hipMemcpy(A_d[j], A_d[i], NUM_ELM * sizeof(TestType),
                      hipMemcpyDefault));
                // Copying in reverse dir of above to check if bidirectional
                // access is happening without any error
                HIP_CHECK(hipMemcpy(A_d[i], A_d[j], NUM_ELM * sizeof(TestType),
                      hipMemcpyDefault));
                // Copying data to host to verify the content
                HIP_CHECK(hipMemcpy(B_h, A_d[j], NUM_ELM * sizeof(TestType),
                      hipMemcpyDefault));
                HipTest::checkTest(A_h, B_h, NUM_ELM);
              }
            }
          }
          break;
        }
      case TEST_MEMCPYH2D:  // To test hipMemcpyHtoD()
        {
          for (int i = 0; i < Available_Gpus; ++i) {
            HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[i]),
                  A_h, NUM_ELM * sizeof(TestType)));
            // Copying data from device to host to check data consistency
            HIP_CHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(TestType),
                  hipMemcpyDeviceToHost));
            HipTest::checkTest(A_h, B_h, NUM_ELM);
          }
          break;
        }
      case TEST_MEMCPYD2H:  // To test hipMemcpyDtoH()--done
        {
          for (int i = 0; i < Available_Gpus; ++i) {
            HIP_CHECK(hipMemcpy(A_d[i], A_h, NUM_ELM * sizeof(TestType),
                  hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpyDtoH(B_h, hipDeviceptr_t(A_d[i]),
                  NUM_ELM * sizeof(TestType)));
            HipTest::checkTest(A_h, B_h, NUM_ELM);
          }
          break;
        }
      case TEST_MEMCPYD2D:  // To test hipMemcpyDtoD()
        {
          if (Available_Gpus > 1) {
            // First copy data from H to D and then
            // from D to D followed by D to H
            // HIP_CHECK(hipMemcpyHtoD(A_d[0], A_h,
            // NUM_ELM * sizeof(TestType)));
            int canAccessPeer = 0;
            for (int i = 0; i < Available_Gpus; ++i) {
              for (int j = i+1; j < Available_Gpus; ++j) {
                hipDeviceCanAccessPeer(&canAccessPeer, i, j);
                if (canAccessPeer) {
                  HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[i]),
                        A_h, NUM_ELM * sizeof(TestType)));
                  HIP_CHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d[j]),
                        hipDeviceptr_t(A_d[i]), NUM_ELM * sizeof(TestType)));
                  // Copying in direction reverse of above to check if
                  // bidirectional
                  // access is happening without any error
                  HIP_CHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d[i]),
                        hipDeviceptr_t(A_d[j]), NUM_ELM * sizeof(TestType)));
                  HIP_CHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(TestType),
                        hipMemcpyDeviceToHost));
                  HipTest::checkTest(A_h, B_h, NUM_ELM);
                }
              }
            }
          } else {
            // As DtoD is not possible transfer data from HtH(A_h to B_h)
            // so as to get through verification step
            HIP_CHECK(hipMemcpy(B_h, A_h, NUM_ELM * sizeof(TestType),
                  hipMemcpyHostToHost));
            HipTest::checkTest(A_h, B_h, NUM_ELM);
          }
          break;
        }
      case TEST_MEMCPYASYNC:
        {
          // To test hipMemcpyAsync()
          // Copying data from host to individual devices followed by copying
          // back to host and verifying the data consistency.
          for (int i = 0; i < Available_Gpus; ++i) {
            HIP_CHECK(hipMemcpyAsync(A_d[i], A_h, NUM_ELM * sizeof(TestType),
                  hipMemcpyHostToDevice, stream[i]));
            HIP_CHECK(hipMemcpyAsync(B_h, A_d[i], NUM_ELM * sizeof(TestType),
                  hipMemcpyDeviceToHost, stream[i]));
            HIP_CHECK(hipStreamSynchronize(stream[i]));
            HipTest::checkTest(A_h, B_h, NUM_ELM);
          }
          // Device to Device copying for all combinations
          for (int i = 0; i < Available_Gpus; ++i) {
            for (int j = i+1; j < Available_Gpus; ++j) {
              canAccessPeer = 0;
              hipDeviceCanAccessPeer(&canAccessPeer, i, j);
              if (canAccessPeer) {
                HIP_CHECK(hipMemcpyAsync(A_d[j], A_d[i],
                      NUM_ELM * sizeof(TestType),
                      hipMemcpyDefault, stream[i]));
                // Copying in direction reverse of above to
                // check if bidirectional
                // access is happening without any error
                HIP_CHECK(hipMemcpyAsync(A_d[i], A_d[j],
                      NUM_ELM * sizeof(TestType),
                      hipMemcpyDefault, stream[i]));
                HIP_CHECK(hipStreamSynchronize(stream[i]));
                HIP_CHECK(hipMemcpy(B_h, A_d[j], NUM_ELM * sizeof(TestType),
                      hipMemcpyDefault));
                HipTest::checkTest(A_h, B_h, NUM_ELM);
              }
            }
          }
          break;
        }
      case TEST_MEMCPYH2DASYNC:  // To test hipMemcpyHtoDAsync()
        {
          for (int i = 0; i < Available_Gpus; ++i) {
            HIP_CHECK(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d[i]), A_h,
                  NUM_ELM * sizeof(TestType), stream[i]));
            HIP_CHECK(hipStreamSynchronize(stream[i]));
            // Copying data from device to host to check data consistency
            HIP_CHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(TestType),
                  hipMemcpyDeviceToHost));
            HipTest::checkTest(A_h, B_h, NUM_ELM);
          }
          break;
        }
      case TEST_MEMCPYD2HASYNC:  // To test hipMemcpyDtoHAsync()
        {
          for (int i = 0; i < Available_Gpus; ++i) {
            HIP_CHECK(hipMemcpy(A_d[i], A_h, NUM_ELM * sizeof(TestType),
                  hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpyDtoHAsync(B_h, hipDeviceptr_t(A_d[i]),
                  NUM_ELM * sizeof(TestType), stream[i]));
            HIP_CHECK(hipStreamSynchronize(stream[i]));
            HipTest::checkTest(A_h, B_h, NUM_ELM);
          }
          break;
        }
      case TEST_MEMCPYD2DASYNC:  // To test hipMemcpyDtoDAsync()
        {
          if (Available_Gpus > 1) {
            // First copy data from H to D and then from D to D followed by D2H
            HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[0]),
                  A_h, NUM_ELM * sizeof(TestType)));
            for (int i = 0; i < Available_Gpus; ++i) {
              for (int j = i+1; j < Available_Gpus; ++j) {
                canAccessPeer = 0;
                hipDeviceCanAccessPeer(&canAccessPeer, i, j);
                if (canAccessPeer) {
                  HIP_CHECK(hipSetDevice(j));
                  HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d[j]),
                        hipDeviceptr_t(A_d[i]), NUM_ELM * sizeof(TestType),
                        stream[i]));
                  // Copying in direction reverse of above to check if
                  // bidirectional
                  // access is happening without any error
                  HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d[i]),
                        hipDeviceptr_t(A_d[j]), NUM_ELM * sizeof(TestType),
                        stream[i]));
                  HIP_CHECK(hipStreamSynchronize(stream[i]));
                  HIP_CHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(TestType),
                        hipMemcpyDeviceToHost));
                  HipTest::checkTest(A_h, B_h, NUM_ELM);
                }
              }
            }
          } else {
            // As DtoD is not possible we will transfer data
            // from HtH(A_h to B_h)
            // so as to get through verification step
            HIP_CHECK(hipMemcpy(B_h, A_h, NUM_ELM * sizeof(TestType),
                  hipMemcpyHostToHost));
            HipTest::checkTest(A_h, B_h, NUM_ELM);
          }
          break;
        }
    }
    for (int i = 0; i < Available_Gpus; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipFree((A_d[i])));
      if (api >= TEST_MEMCPYD2D) {
        HIP_CHECK(hipStreamDestroy(stream[i]));
      }
    }
    HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
        A_h, B_h, nullptr, false);
  }
}

TEMPLATE_TEST_CASE("Stress_hipMemcpy_multiDevice-AllAPIs", "",
                   char, int, size_t, long double) {
  auto diff_size = GENERATE(1, 5, 10, 100, 1024, 10*1024, 100*1024,
                            1024*1024, 10*1024*1024, 100*1024*1024,
                            1024*1024*1024);
  size_t free = 0, total = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  if ((diff_size * sizeof(TestType)) <= free) {
    Memcpy_And_verify<TestType>(diff_size);
    HIP_CHECK(hipDeviceSynchronize());
  }
}
