/*
 * Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#define NUM_THREADS 50
#define ITER 50

template <typename T>
class MemSetTest {
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
    HIPASSERT(A_h != NULL);

    HIPCHECK(hipMalloc(&A_d, Nbytes));
    B_h = reinterpret_cast<T*>(malloc(Nbytes));
    HIPASSERT(B_h != NULL);

    HIPCHECK(hipStreamCreate(&stream));
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
    HIPCHECK(hipFree(A_d));
    free(A_h);
    free(B_h);
    HIPCHECK(hipStreamDestroy(stream));
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
  HIPCHECK(hipMemsetD32Async(A_d, memSetVal, N, stream));
  HIPCHECK(hipMemcpyAsync(A_h, A_d, Nbytes, hipMemcpyDeviceToHost, stream));
}

template <typename T>
void queueJobsForhipMemsetD16Async(T* A_d, T* A_h, T memSetVal, size_t Nbytes,
                                   hipStream_t stream) {
  HIPCHECK(hipMemsetD16Async(A_d, memSetVal, N, stream));
  HIPCHECK(hipMemcpyAsync(A_h, A_d, Nbytes, hipMemcpyDeviceToHost, stream));
}

template <typename T>
void queueJobsForhipMemsetD8Async(T* A_d, T* A_h, T memSetVal, size_t Nbytes,
                                  hipStream_t stream) {
  HIPCHECK(hipMemsetD8Async(A_d, memSetVal, N, stream));
  HIPCHECK(hipMemcpyAsync(A_h, A_d, Nbytes, hipMemcpyDeviceToHost, stream));
}

/* Queue hipMemsetAsync jobs on multiple threads and verify they all
 * finished on all threads successfully
 */

bool testhipMemsetAsyncWithMultiThread() {
  MemSetTest <char> obj;
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

    HIPCHECK(hipStreamSynchronize(obj.stream));
    obj.threadCompleteStatus();
  }
  return obj.resultAfterAllIterations();
}

bool testhipMemsetD32AsyncWithMultiThread() {
  MemSetTest <int32_t> obj;
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

    HIPCHECK(hipStreamSynchronize(obj.stream));
    obj.threadCompleteStatus();
  }
  return obj.resultAfterAllIterations();
}

bool testhipMemsetD16AsyncWithMultiThread() {
  MemSetTest <int16_t> obj;
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

    HIPCHECK(hipStreamSynchronize(obj.stream));
    obj.threadCompleteStatus();
  }
  return obj.resultAfterAllIterations();
}

bool testhipMemsetD8AsyncWithMultiThread() {
  MemSetTest <char> obj;
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

    HIPCHECK(hipStreamSynchronize(obj.stream));
    obj.threadCompleteStatus();
  }
  return obj.resultAfterAllIterations();
}

int main() {
  bool testResult = true;
  printf("Queueing up hipMemSetAsync jobs on multiple threads"
         "and checking results\n");

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  printf("blocks: %u\n", blocks);

  testResult &= testhipMemsetAsyncWithMultiThread();
  if (!(testResult)) {
    printf("Thread execution did not complete for hipMemsetAsync\n");
  }

  testResult &= testhipMemsetD32AsyncWithMultiThread();
  if (!(testResult)) {
    printf("Thread execution did not complete for hipMemsetD32Async\n");
  }

  testResult &= testhipMemsetD16AsyncWithMultiThread();
  if (!(testResult)) {
    printf("Thread execution did not complete for hipMemsetD16Async\n");
  }
  testResult &= testhipMemsetD8AsyncWithMultiThread();
  if (!(testResult)) {
    printf("Thread execution did not complete for hipMemsetD8Async\n");
  }

  if (testResult) {
    printf("All threads ran successfully for all hipMemsetAsync apis\n");
    passed();
  } else {
      failed("One or more tests failed\n");
  }
}
