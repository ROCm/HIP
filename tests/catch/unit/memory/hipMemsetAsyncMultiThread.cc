/*
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
 * Test that validates functionality of hipmemsetAsync apis over multi threads
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>


#define NUM_THREADS 20
#define ITER 10
#define N (4*1024*1024)


template <typename T>
class MemSetAsyncMthreadTest {
 public:
  T *A_h, *A_d, *B_h;
  T memSetVal;
  size_t Nbytes;
  bool testResult = true;
  int validateCount = 0;
  hipStream_t stream;

  void memAllocate(T memSetValue) {
    memSetVal = memSetValue;
    Nbytes = N * sizeof(T);

    A_h = reinterpret_cast<T*>(malloc(Nbytes));
    HIP_ASSERT(A_h != nullptr);

    HIP_CHECK(hipMalloc(&A_d, Nbytes));
    B_h = reinterpret_cast<T*>(malloc(Nbytes));
    HIP_ASSERT(B_h != nullptr);

    HIP_CHECK(hipStreamCreate(&stream));
  }

  void threadCompleteStatus() {
    for (int k = 0 ; k < N ; k++) {
      if ((A_h[k] == memSetVal) && (B_h[k] == memSetVal)) {
        validateCount+= 1;
      }
    }
  }

  bool resultAfterAllIterations() {
    memDeallocate();
    testResult = (validateCount == (ITER * N)) ? true: false;
    return testResult;
  }

  void memDeallocate() {
    HIP_CHECK(hipFree(A_d));
    free(A_h);
    free(B_h);
    HIP_CHECK(hipStreamDestroy(stream));
  }
};

template <typename T>
void queueJobsForhipMemsetAsync(T* A_d, T* A_h, T memSetVal, size_t Nbytes,
                                hipStream_t stream) {
  HIPCHECK(hipMemsetAsync(A_d, memSetVal, N, stream));
  HIPCHECK(hipMemcpyAsync(A_h, A_d, Nbytes, hipMemcpyDeviceToHost, stream));
}

template <typename T>
void queueJobsForhipMemsetD32Async(T* A_d, T* A_h, T memSetVal, size_t Nbytes,
                                   hipStream_t stream) {
  HIPCHECK(hipMemsetD32Async((hipDeviceptr_t)A_d, memSetVal, N, stream));
  HIPCHECK(hipMemcpyAsync(A_h, A_d, Nbytes, hipMemcpyDeviceToHost, stream));
}

template <typename T>
void queueJobsForhipMemsetD16Async(T* A_d, T* A_h, T memSetVal, size_t Nbytes,
                                   hipStream_t stream) {
  HIPCHECK(hipMemsetD16Async((hipDeviceptr_t)A_d, memSetVal, N, stream));
  HIPCHECK(hipMemcpyAsync(A_h, A_d, Nbytes, hipMemcpyDeviceToHost, stream));
}

template <typename T>
void queueJobsForhipMemsetD8Async(T* A_d, T* A_h, T memSetVal, size_t Nbytes,
                                  hipStream_t stream) {
  HIPCHECK(hipMemsetD8Async((hipDeviceptr_t)A_d, memSetVal, N, stream));
  HIPCHECK(hipMemcpyAsync(A_h, A_d, Nbytes, hipMemcpyDeviceToHost, stream));
}

/* Queue hipMemsetAsync jobs on multiple threads and verify they all
 * finished on all threads successfully
 */
bool testhipMemsetAsyncWithMultiThread() {
  MemSetAsyncMthreadTest <char> obj;
  constexpr char memsetval = 0x42;
  obj.memAllocate(memsetval);
  std::thread t[NUM_THREADS];

  for (int i = 0 ; i < ITER ; i++) {
    for (int k = 0 ; k < NUM_THREADS ; k++) {
      if (k%2) {
        t[k] = std::thread(queueJobsForhipMemsetAsync<char>, obj.A_d, obj.A_h,
                           obj.memSetVal, obj.Nbytes, obj.stream);
      } else {
        t[k] = std::thread(queueJobsForhipMemsetAsync<char>, obj.A_d, obj.B_h,
                             obj.memSetVal, obj.Nbytes, obj.stream);
      }
    }

    for (int j = 0 ; j < NUM_THREADS ; j++) {
      t[j].join();
    }

    HIP_CHECK(hipStreamSynchronize(obj.stream));
    obj.threadCompleteStatus();
  }
  return obj.resultAfterAllIterations();
}

bool testhipMemsetD32AsyncWithMultiThread() {
  MemSetAsyncMthreadTest <int32_t> obj;
  constexpr int memsetD32val = 0xDEADBEEF;
  obj.memAllocate(memsetD32val);
  std::thread t[NUM_THREADS];

  for (int i = 0 ; i < ITER ; i++) {
    for (int k = 0 ; k < NUM_THREADS ; k++) {
      if (k%2) {
        t[k] = std::thread(queueJobsForhipMemsetD32Async<int32_t>, obj.A_d,
                           obj.A_h, obj.memSetVal, obj.Nbytes, obj.stream);
      } else {
        t[k] = std::thread(queueJobsForhipMemsetD32Async<int32_t>, obj.A_d,
                           obj.B_h, obj.memSetVal, obj.Nbytes, obj.stream);
      }
    }

    for (int j = 0 ; j < NUM_THREADS ; j++) {
      t[j].join();
    }

    HIP_CHECK(hipStreamSynchronize(obj.stream));
    obj.threadCompleteStatus();
  }
  return obj.resultAfterAllIterations();
}

bool testhipMemsetD16AsyncWithMultiThread() {
  MemSetAsyncMthreadTest <int16_t> obj;
  constexpr int16_t memsetD16val = 0xDEAD;
  obj.memAllocate(memsetD16val);
  std::thread t[NUM_THREADS];

  for (int i = 0 ; i < ITER ; i++) {
    for (int k = 0 ; k < NUM_THREADS ; k++) {
      if (k%2) {
        t[k] = std::thread(queueJobsForhipMemsetD16Async<int16_t>, obj.A_d,
                           obj.A_h, obj.memSetVal, obj.Nbytes, obj.stream);
      } else {
        t[k] = std::thread(queueJobsForhipMemsetD16Async<int16_t>, obj.A_d,
                           obj.B_h, obj.memSetVal, obj.Nbytes, obj.stream);
      }
    }

    for (int j = 0 ; j < NUM_THREADS ; j++) {
      t[j].join();
    }

    HIP_CHECK(hipStreamSynchronize(obj.stream));
    obj.threadCompleteStatus();
  }
  return obj.resultAfterAllIterations();
}

bool testhipMemsetD8AsyncWithMultiThread() {
  MemSetAsyncMthreadTest <char> obj;
  constexpr char memsetD8val = 0xDE;
  obj.memAllocate(memsetD8val);
  std::thread t[NUM_THREADS];

  for (int i = 0 ; i < ITER ; i++) {
    for (int k = 0 ; k < NUM_THREADS ; k++) {
      if (k%2) {
        t[k] = std::thread(queueJobsForhipMemsetD8Async<char>, obj.A_d,
                           obj.A_h, obj.memSetVal, obj.Nbytes, obj.stream);
      } else {
        t[k] = std::thread(queueJobsForhipMemsetD8Async<char>, obj.A_d,
                           obj.B_h, obj.memSetVal, obj.Nbytes, obj.stream);
      }
    }
    for (int j = 0 ; j < NUM_THREADS ; j++) {
      t[j].join();
    }

    HIP_CHECK(hipStreamSynchronize(obj.stream));
    obj.threadCompleteStatus();
  }
  return obj.resultAfterAllIterations();
}


/*
 * Test that validates functionality of hipmemsetAsync apis over multi threads
 */
TEST_CASE("Unit_hipMemsetAsync_QueueJobsMultithreaded") {
  bool ret;

  SECTION("hipMemsetAsync With MultiThread") {
    ret = testhipMemsetAsyncWithMultiThread();
    REQUIRE(ret == true);
  }

  SECTION("hipMemsetD32Async With MultiThread") {
    ret = testhipMemsetD32AsyncWithMultiThread();
    REQUIRE(ret == true);
  }

  SECTION("hipMemsetD16Async With MultiThread") {
    ret = testhipMemsetD16AsyncWithMultiThread();
    REQUIRE(ret == true);
  }

  SECTION("hipMemsetD8Async With MultiThread") {
    ret = testhipMemsetD8AsyncWithMultiThread();
    REQUIRE(ret == true);
  }
}
