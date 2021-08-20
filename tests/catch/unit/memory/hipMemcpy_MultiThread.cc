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
*/
static constexpr auto NUM_ELM{1024};
static constexpr auto NUM_THREADS{5};
static auto Available_Gpus{0};
static constexpr auto MAX_GPU{256};

enum apiToTest {TEST_MEMCPY, TEST_MEMCPYH2D, TEST_MEMCPYD2H, TEST_MEMCPYD2D,
                TEST_MEMCPYASYNC, TEST_MEMCPYH2DASYNC, TEST_MEMCPYD2HASYNC,
                TEST_MEMCPYD2DASYNC};


template <typename T>
class memcpyTests {
 public:
    T *A_h, *B_h;
    apiToTest api;
    explicit memcpyTests(apiToTest val);
    memcpyTests() = delete;
    void Memcpy_And_verify(bool *ret_val);
    bool CheckTests(T* A_h, T* B_h, int NUM_ELEMENTS);
    ~memcpyTests();
};

template <typename T>
bool memcpyTests<T>::CheckTests(T *A_h, T *B_h, int NUM_ELEMENTS) {
  for (auto i =0; i < NUM_ELEMENTS; i++) {
    if (A_h[i] != B_h[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
memcpyTests<T>::memcpyTests(apiToTest val) {
  api = val;
  A_h = reinterpret_cast<T*>(malloc(NUM_ELM * sizeof(T)));
  B_h = reinterpret_cast<T*>(malloc(NUM_ELM * sizeof(T)));
  if ((A_h == nullptr) || (B_h == nullptr)) {
    exit(1);
  }

  for (size_t i = 0; i < NUM_ELM; ++i) {
    A_h[i] = 123;
    B_h[i] = 0;
  }
}


template <typename T>
void memcpyTests<T>::Memcpy_And_verify(bool *ret_val) {
  HIPCHECK(hipGetDeviceCount(&Available_Gpus));
  T *A_d[MAX_GPU];
  hipStream_t stream[MAX_GPU];
  for (int i = 0; i < Available_Gpus; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMalloc(&A_d[i], NUM_ELM * sizeof(T)));
    if (api >= TEST_MEMCPYD2D) {
      HIPCHECK(hipStreamCreate(&stream[i]));
    }
  }
  HIPCHECK(hipSetDevice(0));
  int canAccessPeer = 0;
  switch (api) {
    case TEST_MEMCPY:
      {
        // To test hipMemcpy()
        // Copying data from host to individual devices followed by copying
        // back to host and verifying the data consistency.
        for (int i = 0; i < Available_Gpus; ++i) {
          HIPCHECK(hipMemcpy(A_d[i], A_h, NUM_ELM * sizeof(T),
                hipMemcpyHostToDevice));
          HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(T),
                hipMemcpyDeviceToHost));
          *ret_val = CheckTests(A_h, B_h, NUM_ELM);
        }
        // Device to Device copying for all combinations
        for (int i = 0; i < Available_Gpus; ++i) {
          for (int j = i+1; j < Available_Gpus; ++j) {
            canAccessPeer = 0;
            hipDeviceCanAccessPeer(&canAccessPeer, i, j);
            if (canAccessPeer) {
              HIPCHECK(hipMemcpy(A_d[j], A_d[i], NUM_ELM * sizeof(T),
                    hipMemcpyDefault));
              // Copying in reverse dir of above to check if bidirectional
              // access is happening without any error
              HIPCHECK(hipMemcpy(A_d[i], A_d[j], NUM_ELM * sizeof(T),
                    hipMemcpyDefault));
              // Copying data to host to verify the content
              HIPCHECK(hipMemcpy(B_h, A_d[j], NUM_ELM * sizeof(T),
                    hipMemcpyDefault));
              *ret_val &= CheckTests(A_h, B_h, NUM_ELM);
            }
          }
        }
        break;
      }
    case TEST_MEMCPYH2D:  // To test hipMemcpyHtoD()
      {
        for (int i = 0; i < Available_Gpus; ++i) {
          HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[i]),
                A_h, NUM_ELM * sizeof(T)));
          // Copying data from device to host to check data consistency
          HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(T),
                hipMemcpyDeviceToHost));
          *ret_val &= CheckTests(A_h, B_h, NUM_ELM);
        }
        break;
      }
    case TEST_MEMCPYD2H:  // To test hipMemcpyDtoH()--done
      {
        for (int i = 0; i < Available_Gpus; ++i) {
          HIPCHECK(hipMemcpy(A_d[i], A_h, NUM_ELM * sizeof(T),
                hipMemcpyHostToDevice));
          HIPCHECK(hipMemcpyDtoH(B_h, hipDeviceptr_t(A_d[i]),
                NUM_ELM * sizeof(T)));
          *ret_val &= CheckTests(A_h, B_h, NUM_ELM);
        }
        break;
      }
    case TEST_MEMCPYD2D:  // To test hipMemcpyDtoD()
      {
        if (Available_Gpus > 1) {
          // First copy data from H to D and then
          // from D to D followed by D to H
          // HIPCHECK(hipMemcpyHtoD(A_d[0], A_h,
          // NUM_ELM * sizeof(T)));
          int canAccessPeer = 0;
          for (int i = 0; i < Available_Gpus; ++i) {
            for (int j = i+1; j < Available_Gpus; ++j) {
              hipDeviceCanAccessPeer(&canAccessPeer, i, j);
              if (canAccessPeer) {
                HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[i]),
                      A_h, NUM_ELM * sizeof(T)));
                HIPCHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d[j]),
                      hipDeviceptr_t(A_d[i]), NUM_ELM * sizeof(T)));
                // Copying in direction reverse of above to check if
                // bidirectional
                // access is happening without any error
                HIPCHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d[i]),
                      hipDeviceptr_t(A_d[j]), NUM_ELM * sizeof(T)));
                HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(T),
                      hipMemcpyDeviceToHost));
                *ret_val &= CheckTests(A_h, B_h, NUM_ELM);
              }
            }
          }
        } else {
          // As DtoD is not possible transfer data from HtH(A_h to B_h)
          // so as to get through verification step
          HIPCHECK(hipMemcpy(B_h, A_h, NUM_ELM * sizeof(T),
                hipMemcpyHostToHost));
          *ret_val &= CheckTests(A_h, B_h, NUM_ELM);
        }
        break;
      }
    case TEST_MEMCPYASYNC:
      {
        // To test hipMemcpyAsync()
        // Copying data from host to individual devices followed by copying
        // back to host and verifying the data consistency.
        for (int i = 0; i < Available_Gpus; ++i) {
          HIPCHECK(hipMemcpyAsync(A_d[i], A_h, NUM_ELM * sizeof(T),
                hipMemcpyHostToDevice, stream[i]));
          HIPCHECK(hipMemcpyAsync(B_h, A_d[i], NUM_ELM * sizeof(T),
                hipMemcpyDeviceToHost, stream[i]));
          HIPCHECK(hipStreamSynchronize(stream[i]));
          *ret_val &= CheckTests(A_h, B_h, NUM_ELM);
        }
        // Device to Device copying for all combinations
        for (int i = 0; i < Available_Gpus; ++i) {
          for (int j = i+1; j < Available_Gpus; ++j) {
            canAccessPeer = 0;
            hipDeviceCanAccessPeer(&canAccessPeer, i, j);
            if (canAccessPeer) {
              HIPCHECK(hipMemcpyAsync(A_d[j], A_d[i],
                    NUM_ELM * sizeof(T),
                    hipMemcpyDefault, stream[i]));
              // Copying in direction reverse of above to
              // check if bidirectional
              // access is happening without any error
              HIPCHECK(hipMemcpyAsync(A_d[i], A_d[j],
                    NUM_ELM * sizeof(T),
                    hipMemcpyDefault, stream[i]));
              HIPCHECK(hipStreamSynchronize(stream[i]));
              HIPCHECK(hipMemcpy(B_h, A_d[j], NUM_ELM * sizeof(T),
                    hipMemcpyDefault));
              *ret_val &= CheckTests(A_h, B_h, NUM_ELM);
            }
          }
        }
        break;
      }
    case TEST_MEMCPYH2DASYNC:  // To test hipMemcpyHtoDAsync()
      {
        for (int i = 0; i < Available_Gpus; ++i) {
          HIPCHECK(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d[i]), A_h,
                NUM_ELM * sizeof(T), stream[i]));
          HIPCHECK(hipStreamSynchronize(stream[i]));
          // Copying data from device to host to check data consistency
          HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(T),
                hipMemcpyDeviceToHost));
          *ret_val &= CheckTests(A_h, B_h, NUM_ELM);
        }
        break;
      }
    case TEST_MEMCPYD2HASYNC:  // To test hipMemcpyDtoHAsync()
      {
        for (int i = 0; i < Available_Gpus; ++i) {
          HIPCHECK(hipMemcpy(A_d[i], A_h, NUM_ELM * sizeof(T),
                hipMemcpyHostToDevice));
          HIPCHECK(hipMemcpyDtoHAsync(B_h, hipDeviceptr_t(A_d[i]),
                NUM_ELM * sizeof(T), stream[i]));
          HIPCHECK(hipStreamSynchronize(stream[i]));
          *ret_val &= CheckTests(A_h, B_h, NUM_ELM);
        }
        break;
      }
    case TEST_MEMCPYD2DASYNC:  // To test hipMemcpyDtoDAsync()
      {
        if (Available_Gpus > 1) {
          // First copy data from H to D and then from D to D followed by D2H
          HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[0]),
                A_h, NUM_ELM * sizeof(T)));
          for (int i = 0; i < Available_Gpus; ++i) {
            for (int j = i+1; j < Available_Gpus; ++j) {
              canAccessPeer = 0;
              hipDeviceCanAccessPeer(&canAccessPeer, i, j);
              if (canAccessPeer) {
                HIPCHECK(hipSetDevice(j));
                HIPCHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d[j]),
                      hipDeviceptr_t(A_d[i]), NUM_ELM * sizeof(T),
                      stream[i]));
                // Copying in direction reverse of above to check if
                // bidirectional
                // access is happening without any error
                HIPCHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d[i]),
                      hipDeviceptr_t(A_d[j]), NUM_ELM * sizeof(T),
                      stream[i]));
                HIPCHECK(hipStreamSynchronize(stream[i]));
                HIPCHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(T),
                      hipMemcpyDeviceToHost));
                *ret_val &= CheckTests(A_h, B_h, NUM_ELM);
              }
            }
          }
        } else {
          // As DtoD is not possible we will transfer data
          // from HtH(A_h to B_h)
          // so as to get through verification step
          HIPCHECK(hipMemcpy(B_h, A_h, NUM_ELM * sizeof(T),
                hipMemcpyHostToHost));
          *ret_val &= CheckTests(A_h, B_h, NUM_ELM);
        }
        break;
      }
  }
  for (int i = 0; i < Available_Gpus; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipFree((A_d[i])));
    if (api >= TEST_MEMCPYD2D) {
      HIPCHECK(hipStreamDestroy(stream[i]));
    }
  }
}
template <typename T>
memcpyTests<T>::~memcpyTests() {
  free(A_h);
  free(B_h);
}

void Thread_func(bool &ret_val) {
  for (apiToTest api = TEST_MEMCPY; api <= TEST_MEMCPYD2DASYNC;
       api = apiToTest(api + 1)) {
    memcpyTests<int> obj(api);
    obj.Memcpy_And_verify(&ret_val);
  }
}



TEST_CASE("Unit_hipMemcpy_MultiThread-AllAPIs") {
  std::thread Thrd[NUM_THREADS];
  bool ret_val[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++)
    Thrd[i] = std::thread(Thread_func, std::ref(ret_val[i]));

  // Thread join is being called separately so as to allow the
  // threads run parallely
  for (int i = 0; i < NUM_THREADS; i++)
    Thrd[i].join();

  for (int i = 0; i < NUM_THREADS; i++)
    REQUIRE(ret_val[i] == true);
}
