/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "deviceAllocCommon.h"

__device__ static void* dev_mem_glob;
__device__ struct deviceAllocFunc allocfunc{&deviceAlloc, &deviceWrite,
                                            &deviceFree};
__device__ class derivedAlloc classalloc;
constexpr auto num_threads = 5;
static bool thread_results[num_threads];
__device__ static void* dev_ptr[num_threads][GRIDSIZE];

/**
 * This kernel allocates and deallocates in every thread
 * of every block.
 */
template <typename T>
static __global__ void kerTestDynamicAllocInAllThread(T *outputBuf,
                        int test_type, T value, size_t perThreadSize) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate
  size_t size = 0;
  T* ptr = nullptr;
  if (test_type == TEST_MALLOC_FREE) {
    size = perThreadSize * sizeof(T);
    ptr = reinterpret_cast<T*> (malloc(size));
  } else {
    size = perThreadSize;
    ptr = new T[perThreadSize];
  }
  if (ptr == nullptr) {
    printf("Device Allocation in thread %d Failed! \n", myId);
    return;
  }
  // Set memory
  for (size_t idx = 0; idx < perThreadSize; idx++) {
    ptr[idx] = value;
  }
  // Copy to output buffer
  for (size_t idx = 0; idx < perThreadSize; idx++) {
    outputBuf[myId*perThreadSize + idx] = ptr[idx];
  }
  // Free memory
  if (test_type == TEST_MALLOC_FREE) {
    free(ptr);
  } else {
    delete[] ptr;
  }
}

/**
 * This kernel allocates and deallocates using virtual functions in every
 * thread of every block.
 */
static __global__ void kerTestDynamicAllocVirtualFunc(int *outputBuf,
                                size_t perThreadSize) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  baseAlloc *palloc = &classalloc;
  // Allocate
  int* ptr = palloc->alloc(perThreadSize);

  if (ptr == nullptr) {
    printf("Device Allocation in thread %d Failed! \n", myId);
    return;
  }
  // Set memory
  for (size_t idx = 0; idx < perThreadSize; idx++) {
    ptr[idx] = myId;
  }
  // Copy to output buffer
  for (size_t idx = 0; idx < perThreadSize; idx++) {
    outputBuf[myId*perThreadSize + idx] = ptr[idx];
  }
  // Free memory
  palloc->free(ptr);
}

/**
 * This kernel allocates memory in one thread,
 * access/modifies it in all threads of block and copies
 * data to host and frees the memory in another thread.
 */
template <typename T>
static __global__ void kerTestAccessInAllThreadsInBlock(T *outputBuf,
                            int test_type, T value, int host_thr_idx) {
  int myThreadId = threadIdx.x, lastThreadId = (blockDim.x - 1);
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate memory in thread 0
  if (0 == myThreadId) {
    if (test_type == TEST_MALLOC_FREE) {
      dev_ptr[host_thr_idx][blockIdx.x] =
      reinterpret_cast<void*> (malloc(blockDim.x*sizeof(T)));
    } else {
      dev_ptr[host_thr_idx][blockIdx.x] =
      reinterpret_cast<void*> (new T[blockDim.x]);
    }
  }
  // All threads wait at this barrier
  __syncthreads();
  // Check allocated memory in all threads in block before access
  if (dev_ptr[host_thr_idx][blockIdx.x] == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }
  T *ptr = reinterpret_cast<T*> (dev_ptr[host_thr_idx][blockIdx.x]);
  // Copy to buffer
  ptr[myThreadId] = value;
  // All threads wait
  __syncthreads();
  // Copy memory to host and free the memory in thread <blockDim.x - 1>
  if (lastThreadId == myThreadId) {
    for (size_t idx = 0; idx < blockDim.x; idx++) {
      outputBuf[idx + blockDim.x * blockIdx.x] = ptr[idx];
    }
    if (test_type == TEST_MALLOC_FREE) {
      free(ptr);
    } else {
      delete[] ptr;
    }
  }
}

/**
 * This kernel allocates a nested structure per block in one thread,
 * access/modifies it in all threads of block and copies
 * data to host and frees the memory in another thread.
 */
static __global__ void kerTestAccessInAllThreads_CmplxStr(int test_type,
                                                          int *result) {
  int myThreadId = threadIdx.x;
  int lastThreadId = (blockDim.x - 1);
  int myBlockId = blockIdx.x;
  int myGid = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate memory in thread 0
  if (0 == myThreadId) {
    if (test_type == TEST_MALLOC_FREE) {
      dev_ptr[0][blockIdx.x] =
      reinterpret_cast<void*> (malloc(sizeof(struct complexStructure)));
    } else {
      dev_ptr[0][blockIdx.x] =
      reinterpret_cast<void*> (new struct complexStructure);
    }
    struct complexStructure *ptr =
    reinterpret_cast<struct complexStructure*> (dev_ptr[0][blockIdx.x]);
    ptr->alloc_internal_members(test_type, BLOCKSIZE);
  }
  // All threads wait at this barrier
  __syncthreads();
  // Check allocated memory in all threads in block before access
  if (dev_ptr[0][blockIdx.x] == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myGid);
    return;
  }
  struct complexStructure *ptr =
  reinterpret_cast<struct complexStructure*> (dev_ptr[0][blockIdx.x]);
  if (ptr->sthreadInfo == nullptr) {
    printf("Structure Allocation Failed in thread = %d \n", myGid);
    return;
  }
  // Copy to buffer
  ptr->sthreadInfo[myThreadId].threadid = myThreadId;
  ptr->sthreadInfo[myThreadId].blockid = myBlockId;
  ptr->sthreadInfo[myThreadId].ival = INT_MAX;
  ptr->sthreadInfo[myThreadId].dval = DBL_MAX;
  ptr->sthreadInfo[myThreadId].fval = FLT_MAX;
  ptr->sthreadInfo[myThreadId].sval = SHRT_MAX;
  ptr->sthreadInfo[myThreadId].cval = SCHAR_MAX;
  // All threads wait
  __syncthreads();
  // Copy memory to host and free the memory in thread <blockDim.x - 1>
  if (lastThreadId == myThreadId) {
    int match = 1;
    for (int idx = 0; idx < BLOCKSIZE; idx++) {
      if ((ptr->sthreadInfo[idx].threadid != idx) ||
          (ptr->sthreadInfo[idx].blockid != myBlockId) ||
          (ptr->sthreadInfo[idx].ival != INT_MAX) ||
          (ptr->sthreadInfo[idx].dval != DBL_MAX) ||
          (ptr->sthreadInfo[idx].fval != FLT_MAX) ||
          (ptr->sthreadInfo[idx].sval != SHRT_MAX) ||
          (ptr->sthreadInfo[idx].cval != SCHAR_MAX)) {
        match = 0;
        break;
      }
    }
    result[blockIdx.x] = match;
    ptr->free_internal_members(test_type);
    if (test_type == TEST_MALLOC_FREE) {
      free(ptr);
    } else {
      delete ptr;
    }
  }
}

/**
 * This kernel allocates a union per block in one thread,
 * access/modifies it in all threads of block and copies
 * data to host and frees the memory in another thread.
 */
static __global__ void kerTestAccessInAllThreadsForUnion(
                            testInfoUnion *outputBuf, int test_type) {
  int myThreadId = threadIdx.x, lastThreadId = (blockDim.x - 1);
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate memory in thread 0
  if (0 == myThreadId) {
    if (test_type == TEST_MALLOC_FREE) {
      dev_ptr[0][blockIdx.x] =
      reinterpret_cast<void*> (malloc(blockDim.x*sizeof(testInfoUnion)));
    } else {
      dev_ptr[0][blockIdx.x] =
      reinterpret_cast<void*> (new testInfoUnion[blockDim.x]);
    }
  }
  // All threads wait at this barrier
  __syncthreads();
  // Check allocated memory in all threads in block before access
  if (dev_ptr[0][blockIdx.x] == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }
  testInfoUnion *ptr =
  reinterpret_cast<testInfoUnion*> (dev_ptr[0][blockIdx.x]);
  // Copy to buffer
  switch (myId % 5) {
    case 0: ptr[myThreadId].ival = INT_MAX; break;
    case 1: ptr[myThreadId].dval = DBL_MAX; break;
    case 2: ptr[myThreadId].fval = FLT_MAX; break;
    case 3: ptr[myThreadId].sval = SHRT_MAX; break;
    case 4: ptr[myThreadId].cval = SCHAR_MAX; break;
  }
  // All threads wait
  __syncthreads();
  // Copy memory to host and free the memory in thread <blockDim.x - 1>
  if (lastThreadId == myThreadId) {
    for (size_t idx = 0; idx < blockDim.x; idx++) {
      outputBuf[idx + blockDim.x * blockIdx.x] = ptr[idx];
    }
    if (test_type == TEST_MALLOC_FREE) {
      free(ptr);
    } else {
      delete[] ptr;
    }
  }
}

/**
 * This kernel allocates memory in one thread.
 */
template <typename T>
static __global__ void kerAlloc(int test_type) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate memory in thread 0 of block 0
  if (0 == myId) {
    if (test_type == TEST_MALLOC_FREE) {
      dev_mem_glob =
      reinterpret_cast<void*> (malloc(blockDim.x*gridDim.x*sizeof(T)));
    } else {
      dev_mem_glob =
      reinterpret_cast<void*> (new T[blockDim.x*gridDim.x]);
    }
  }
}

/**
 * This kernel writes to memory allocated in <kerAlloc>.
 */
template <typename T>
static __global__ void kerWrite(T value) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Check allocated memory in all threads in block before access
  if (dev_mem_glob == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }
  T *ptr = reinterpret_cast<T*> (dev_mem_glob);
  // Copy to buffer
  ptr[myId] = value;
}

/**
 * This kernel copies the contents of memory allocated in <kerAlloc>
 * to host and deletes the memory from thread 0.
 */
template <typename T>
static __global__ void kerFree(T *outputBuf, int test_type) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Check allocated memory in all threads in block before access
  if (dev_mem_glob == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }

  T *ptr = reinterpret_cast<T*> (dev_mem_glob);
  if (0 == myId) {
    for (size_t idx = 0; idx < (blockDim.x*gridDim.x); idx++) {
      outputBuf[idx] = ptr[idx];
    }
    if (test_type == TEST_MALLOC_FREE) {
      free(ptr);
    } else {
      delete[] ptr;
    }
  }
}

/**
 * This device function allocates memory in one thread.
 */
static __device__ int* deviceAlloc(int test_type) {
  int *ptr = nullptr;
  if (test_type == TEST_MALLOC_FREE) {
    ptr =
    reinterpret_cast<int*> (malloc(INTERNAL_BUFFER_SIZE*sizeof(int)));
  } else {
    ptr =
    reinterpret_cast<int*> (new int[INTERNAL_BUFFER_SIZE]);
  }
  return ptr;
}

/**
 * This device function writes to memory allocated in deviceAlloc().
 */
static __device__ void deviceWrite(int myId, int *devmem) {
  // Check allocated memory in all threads in block before access
  if (devmem == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }
  // Copy to buffer
  for (size_t idx = 0; idx < INTERNAL_BUFFER_SIZE; idx++) {
    devmem[idx] = myId;
  }
}

/**
 * This device function copies the contents of memory allocated
 * in deviceAlloc() to host and deletes the memory from thread 0.
 */
static __device__ void deviceFree(int *outputBuf, int *devmem,
                                  int test_type, int myId) {
  // Check allocated memory in all threads in block before access
  if (devmem == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }
  for (size_t idx = 0; idx < INTERNAL_BUFFER_SIZE; idx++) {
    outputBuf[myId*INTERNAL_BUFFER_SIZE + idx] = devmem[idx];
  }
  if (test_type == TEST_MALLOC_FREE) {
    free(devmem);
  } else {
    delete[] devmem;
  }
}

/**
 * This kernel invokes __device__ allocation functions via pointers.
 */
static __global__ void kerTestAllocationUsingDevFunc(int *outputBuf,
                                    int test_type) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  struct deviceAllocFunc *func = &allocfunc;
  int *dev_ptr = nullptr;
  dev_ptr = func->alloc(test_type);
  func->write(myId, dev_ptr);
  func->free(outputBuf, dev_ptr, test_type, myId);
}

/**
 * Local function: Allocate local and device memory from host,
 * launches kerTestDynamicAllocInAllThread<<<>>> and copies data back
 * to host to validate.
 */
template <typename T>
static bool TestAllocInAllThread(int test_type,
                    T value, size_t sizeBufferPerThread) {
  T *outputVec_d{nullptr}, *outputVec_h{nullptr};
  size_t arraysize = (sizeBufferPerThread * BLOCKSIZE * GRIDSIZE);
  outputVec_h = reinterpret_cast<T*> (malloc(sizeof(T) * arraysize));
  REQUIRE(outputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&outputVec_d, (sizeof(T) * arraysize)));
  // Launch Test Kernel
  kerTestDynamicAllocInAllThread<T><<<GRIDSIZE, BLOCKSIZE>>>(
                    outputVec_d, test_type, value, sizeBufferPerThread);
  HIP_CHECK(hipDeviceSynchronize());
  // Copy to host buffer
  HIP_CHECK(hipMemcpy(outputVec_h, outputVec_d, sizeof(T) * arraysize,
                    hipMemcpyDefault));
  bool bPassed = true;
  for (size_t idx = 0; idx < arraysize; idx++) {
    if (outputVec_h[idx] != value) {
      bPassed = false;
      break;
    }
  }
  hipFree(outputVec_d);
  free(outputVec_h);
  return bPassed;
}

/**
 * Local function: Allocate local and device memory from host,
 * launches kerTestAccessInAllThreadsInBlock<<<>>> and copies data back
 * to host to validate.
 */
template <typename T>
static bool TestMemoryAccessInAllThread(int test_type, int thread_idx) {
  T *outputVec_d{nullptr}, *outputVec_h{nullptr};
  size_t arraysize = (BLOCKSIZE * GRIDSIZE);
  T data_value = std::numeric_limits<T>::max();
  outputVec_h = reinterpret_cast<T*> (malloc(sizeof(T) * arraysize));
  REQUIRE(outputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&outputVec_d, (sizeof(T) * arraysize)));
  // Launch Test Kernel
  kerTestAccessInAllThreadsInBlock<T><<<GRIDSIZE, BLOCKSIZE>>>(outputVec_d,
                                        test_type, data_value, thread_idx);
  HIP_CHECK(hipDeviceSynchronize());
  // Copy to host buffer
  HIP_CHECK(hipMemcpy(outputVec_h, outputVec_d, sizeof(T) * arraysize,
                    hipMemcpyDefault));
  bool bPassed = true;
  for (size_t idx = 0; idx < arraysize; idx++) {
    if (outputVec_h[idx] != data_value) {
      bPassed = false;
      break;
    }
  }
  hipFree(outputVec_d);
  free(outputVec_h);
  return bPassed;
}

/**
 * Local function: Launch kerAlloc<<<>>>
 */
template <typename T>
static void runTestMemoryAccessInAllThread(int test_type, int thread_idx) {
  thread_results[thread_idx] = TestMemoryAccessInAllThread<T>(test_type,
                               thread_idx);
}

/**
 * Local function: Launch kerAlloc<<<>>>, kerWrite<<<>>> and kerFree<<<>>>
 * to test kernel allocated memory access across multiple kernels and multiple
 * streams.
 */
template <typename T>
static bool TestMemoryAcrossMulKernels(int test_type,
                                       bool multistream = false) {
  T *outputVec_d{nullptr}, *outputVec_h{nullptr};
  size_t arraysize = (BLOCKSIZE * GRIDSIZE);
  T data_value = std::numeric_limits<T>::max();
  outputVec_h = reinterpret_cast<T*> (malloc(sizeof(T) * arraysize));
  REQUIRE(outputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&outputVec_d, (sizeof(T) * arraysize)));
  // Launch Test Kernel
  if (multistream) {
    hipStream_t stream1, stream2, stream3;
    HIP_CHECK(hipStreamCreate(&stream1));
    HIP_CHECK(hipStreamCreate(&stream2));
    HIP_CHECK(hipStreamCreate(&stream3));
    kerAlloc<T><<<GRIDSIZE, BLOCKSIZE, 0, stream1>>>(test_type);
    HIP_CHECK(hipStreamSynchronize(stream1));
    kerWrite<T><<<GRIDSIZE, BLOCKSIZE, 0, stream2>>>(data_value);
    HIP_CHECK(hipStreamSynchronize(stream2));
    kerFree<T><<<GRIDSIZE, BLOCKSIZE, 0, stream3>>>(outputVec_d, test_type);
    HIP_CHECK(hipStreamSynchronize(stream3));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipStreamDestroy(stream2));
    HIP_CHECK(hipStreamDestroy(stream3));
  } else {
    kerAlloc<T><<<GRIDSIZE, BLOCKSIZE>>>(test_type);
    kerWrite<T><<<GRIDSIZE, BLOCKSIZE>>>(data_value);
    kerFree<T><<<GRIDSIZE, BLOCKSIZE>>>(outputVec_d, test_type);
    HIP_CHECK(hipDeviceSynchronize());
  }
  // Copy to host buffer
  HIP_CHECK(hipMemcpy(outputVec_h, outputVec_d, sizeof(T) * arraysize,
                      hipMemcpyDefault));
  bool bPassed = true;
  for (size_t idx = 0; idx < arraysize; idx++) {
    if (outputVec_h[idx] != data_value) {
      bPassed = false;
      break;
    }
  }
  hipFree(outputVec_d);
  free(outputVec_h);
  return bPassed;
}

/**
 * Local function: Launch kerAlloc<<<>>>
 */
template <typename T>
static void runKerAlloc(int test_type) {
  kerAlloc<T><<<GRIDSIZE, BLOCKSIZE>>>(test_type);
}

/**
 * Local function: Launch kerWrite<<<>>>
 */
template <typename T>
static void runKerWrite(T data_value) {
  kerWrite<T><<<GRIDSIZE, BLOCKSIZE>>>(data_value);
}

/**
 * Local function: Launch kerFree<<<>>>
 */
template <typename T>
static void runKerFree(T *outputVec_d, int test_type) {
  kerFree<T><<<GRIDSIZE, BLOCKSIZE>>>(outputVec_d, test_type);
}

/**
 * Local function: Launch kerAlloc<<<>>>, kerWrite<<<>>> and kerFree<<<>>>
 * across multiple threads.
 */
template <typename T>
static bool TestDevMemAllocMulKerMulThrd(int test_type) {
  T *outputVec_d{nullptr}, *outputVec_h{nullptr};
  size_t arraysize = (BLOCKSIZE * GRIDSIZE);
  T data_value = std::numeric_limits<T>::max();
  outputVec_h = reinterpret_cast<T*> (malloc(sizeof(T) * arraysize));
  REQUIRE(outputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&outputVec_d, (sizeof(T) * arraysize)));
  // Launch all Test Kernel threads
  std::thread threadAlloc(runKerAlloc<T>, test_type);
  threadAlloc.join();
  std::thread threadWrite(runKerWrite<T>, data_value);
  threadWrite.join();
  std::thread threadFree(runKerFree<T>, outputVec_d, test_type);
  threadFree.join();
  // Wait for all kernels in device
  HIP_CHECK(hipDeviceSynchronize());
  // Copy to host buffer
  HIP_CHECK(hipMemcpy(outputVec_h, outputVec_d, sizeof(T) * arraysize,
                      hipMemcpyDefault));
  bool bPassed = true;
  for (size_t idx = 0; idx < arraysize; idx++) {
    if (outputVec_h[idx] != data_value) {
      bPassed = false;
      break;
    }
  }
  hipFree(outputVec_d);
  free(outputVec_h);
  return bPassed;
}
/**
 * Local function: Allocate local and device memory from host,
 * launches kerTestAccessInAllThreads_CmplxStr<<<>>> and copies data back
 * to host to validate.
 */
static bool TestMemoryAccessInAllThread_CmplxStr(int test_type) {
  int *result_d{nullptr}, *result_h{nullptr};
  size_t arraysize = BLOCKSIZE;
  result_h = reinterpret_cast<int*> (malloc(sizeof(int) * arraysize));
  REQUIRE(result_h != nullptr);
  HIP_CHECK(hipMalloc(&result_d, (sizeof(int) * arraysize)));
  HIP_CHECK(hipMemset(result_d, 0, (sizeof(int) * arraysize)));
  // Launch Test Kernel
  kerTestAccessInAllThreads_CmplxStr<<<GRIDSIZE, BLOCKSIZE>>>(
                                        test_type, result_d);
  HIP_CHECK(hipDeviceSynchronize());
  // Copy to host buffer
  HIP_CHECK(hipMemcpy(result_h, result_d, sizeof(int) * arraysize,
                    hipMemcpyDefault));
  bool bPassed = true;
  for (size_t idx = 0; idx < GRIDSIZE; idx++) {
    if (result_h[idx] != 1) {
      bPassed = false;
      break;
    }
  }
  hipFree(result_d);
  free(result_h);
  return bPassed;
}

/**
 * Local function: Allocate host and device memory of type union,
 * launches kerTestAccessInAllThreadsForUnion<<<>>> and copies data back
 * to host to validate.
 */
static bool TestMemoryAccessInAllThread_Union(int test_type) {
  testInfoUnion *outputVec_d{nullptr}, *outputVec_h{nullptr};
  size_t arraysize = (BLOCKSIZE * GRIDSIZE);
  outputVec_h = reinterpret_cast<testInfoUnion*>
                (malloc(sizeof(testInfoUnion) * arraysize));
  REQUIRE(outputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&outputVec_d,
           (sizeof(testInfoUnion) * arraysize)));
  // Launch Test Kernel
  kerTestAccessInAllThreadsForUnion<<<GRIDSIZE, BLOCKSIZE>>>(outputVec_d,
                                                             test_type);
  HIP_CHECK(hipDeviceSynchronize());
  // Copy to host buffer
  HIP_CHECK(hipMemcpy(outputVec_h, outputVec_d,
            sizeof(testInfoUnion) * arraysize, hipMemcpyDefault));
  bool bPassed = true;
  for (size_t idx = 0; idx < arraysize; idx++) {
    switch (idx % 5) {
      case 0:
        if (outputVec_h[idx].ival != INT_MAX) {
          bPassed = false;
        }
        break;
      case 1:
        if (outputVec_h[idx].dval != DBL_MAX) {
          bPassed = false;
        }
        break;
      case 2:
        if (outputVec_h[idx].fval != FLT_MAX) {
          bPassed = false;
        }
        break;
      case 3:
        if (outputVec_h[idx].sval != SHRT_MAX) {
          bPassed = false;
        }
        break;
      case 4:
        if (outputVec_h[idx].cval != SCHAR_MAX) {
          bPassed = false;
        }
        break;
    }
    if (bPassed == false) break;
  }
  hipFree(outputVec_d);
  free(outputVec_h);
  return bPassed;
}

/**
 * Local function: Allocate local and device memory from host,
 * launches ker_TestDynamicAllocInAllThreads_CodeObj<<<>>> and
 * copies data back to host to validate.
 */
static bool TestAlloc_Load_SingleKer_AllocFree(int test_type,
                    int value, size_t sizeBufferPerThread) {
  int *outputVec_d{nullptr}, *outputVec_h{nullptr};
  size_t arraysize = (sizeBufferPerThread * BLOCKSIZE * GRIDSIZE);
  outputVec_h = reinterpret_cast<int*> (malloc(sizeof(int) * arraysize));
  REQUIRE(outputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&outputVec_d, (sizeof(int) * arraysize)));
  // Launch Test Kernel
  hipModule_t Module;
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, DEV_ALLOC_SINGKER_COBJ));
  HIP_CHECK(hipModuleGetFunction(&Function, Module,
                                DEV_ALLOC_SINGKER_COBJ_FUNC));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  struct {
    void* _Output_d;
    int _test_type;
    int _value;
    size_t _size;
  } args;
  args._Output_d = reinterpret_cast<void*>(outputVec_d);
  args._test_type = test_type;
  args._value = value;
  args._size = sizeBufferPerThread;
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(Function, GRIDSIZE, 1, 1,
                                 BLOCKSIZE, 1, 1, 0,
                                 stream, NULL,
                                 reinterpret_cast<void**>(&config)));
  HIP_CHECK(hipDeviceSynchronize());
  // Copy to host buffer
  HIP_CHECK(hipMemcpy(outputVec_h, outputVec_d, sizeof(int) * arraysize,
                    hipMemcpyDefault));
  bool bPassed = true;
  for (size_t idx = 0; idx < arraysize; idx++) {
    if (outputVec_h[idx] != value) {
      bPassed = false;
      break;
    }
  }
  HIP_CHECK(hipModuleUnload(Module));
  HIP_CHECK(hipStreamDestroy(stream));
  hipFree(outputVec_d);
  free(outputVec_h);
  return bPassed;
}

/**
 * Local function: Allocate local and device memory from host,
 * launches ker_Alloc_MultCodeObj<<<>>>, ker_Write_MultCodeObj<<<>>> and
 * ker_Free_MultCodeObj<<<>>> copies data back to host to validate.
 */
static bool TestAlloc_Load_MultKernels(int test_type,
                                int value) {
  int *outputVec_d{nullptr}, *outputVec_h{nullptr};
  int **dev_addr{nullptr};
  size_t arraysize = (BLOCKSIZE * GRIDSIZE);
  outputVec_h = reinterpret_cast<int*> (malloc(sizeof(int) * arraysize));
  REQUIRE(outputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&outputVec_d, (sizeof(int) * arraysize)));
  HIP_CHECK(hipMalloc(&dev_addr, (sizeof(int*))));
  // Launch Test Kernel
  hipModule_t ModuleAlloc, ModuleWrite, ModuleFree;
  hipFunction_t FunctionAlloc, FunctionAcess, FunctionFree;
  // Load ker_Alloc_MultCodeObj
  HIP_CHECK(hipModuleLoad(&ModuleAlloc, DEV_ALLOC_MULCOBJ));
  HIP_CHECK(hipModuleLoad(&ModuleWrite, DEV_WRITE_MULCOBJ));
  HIP_CHECK(hipModuleLoad(&ModuleFree, DEV_FREE_MULCOBJ));
  HIP_CHECK(hipModuleGetFunction(&FunctionAlloc, ModuleAlloc,
                                DEV_ALLOC_MULCODEOBJ_ALLOC));
  // Load ker_Write_MultCodeObj
  HIP_CHECK(hipModuleGetFunction(&FunctionAcess, ModuleWrite,
                                DEV_ALLOC_MULCODEOBJ_WRITE));
  // Load ker_Free_MultCodeObj
  HIP_CHECK(hipModuleGetFunction(&FunctionFree, ModuleFree,
                                DEV_ALLOC_MULCODEOBJ_FREE));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  struct {
    void **__dev_addr;
    int _test_type;
  } args1;
  args1.__dev_addr = reinterpret_cast<void**>(dev_addr);
  args1._test_type = test_type;
  size_t size1 = sizeof(args1);

  void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                    HIP_LAUNCH_PARAM_END};
  struct {
    void **__dev_addr;
    int _value;
  } args2;
  args2.__dev_addr = reinterpret_cast<void**>(dev_addr);
  args2._value = value;
  size_t size2 = sizeof(args2);

  void* config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
                    HIP_LAUNCH_PARAM_END};
  struct {
    void* _output;
    void **__dev_addr;
    int _test_type;
  } args3;
  args3._output = reinterpret_cast<void*>(outputVec_d);
  args3.__dev_addr = reinterpret_cast<void**>(dev_addr);
  args3._test_type = test_type;
  size_t size3 = sizeof(args3);

  void* config3[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args3,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size3,
                    HIP_LAUNCH_PARAM_END};
  // Launch ker_Alloc_MultCodeObj
  HIP_CHECK(hipModuleLaunchKernel(FunctionAlloc, GRIDSIZE, 1, 1,
                                 BLOCKSIZE, 1, 1, 0,
                                 stream, NULL,
                                 reinterpret_cast<void**>(&config1)));
  // Launch ker_Write_MultCodeObj
  HIP_CHECK(hipModuleLaunchKernel(FunctionAcess, GRIDSIZE, 1, 1,
                                 BLOCKSIZE, 1, 1, 0,
                                 stream, NULL,
                                 reinterpret_cast<void**>(&config2)));
  // Launch ker_Free_MultCodeObj
  HIP_CHECK(hipModuleLaunchKernel(FunctionFree, GRIDSIZE, 1, 1,
                                 BLOCKSIZE, 1, 1, 0,
                                 stream, NULL,
                                 reinterpret_cast<void**>(&config3)));
  HIP_CHECK(hipDeviceSynchronize());
  // Copy to host buffer
  HIP_CHECK(hipMemcpy(outputVec_h, outputVec_d, sizeof(int) * arraysize,
                    hipMemcpyDefault));
  bool bPassed = true;
  for (size_t idx = 0; idx < arraysize; idx++) {
    if (outputVec_h[idx] != value) {
      bPassed = false;
      break;
    }
  }
  HIP_CHECK(hipModuleUnload(ModuleAlloc));
  HIP_CHECK(hipModuleUnload(ModuleWrite));
  HIP_CHECK(hipModuleUnload(ModuleFree));
  HIP_CHECK(hipStreamDestroy(stream));
  hipFree(dev_addr);
  hipFree(outputVec_d);
  free(outputVec_h);
  return bPassed;
}

/**
 * Local function: Launch kerAlloc<<<>>>, kerWrite<<<>>> and kerFree<<<>>>
 * to test kernel allocated memory access across multiple kernels using
 * hipGraph.
 */
template <typename T>
static bool TestMemoryAcrossMulKernelsUsingGraph(int test_type) {
  T *outputVec_d{nullptr}, *outputVec_h{nullptr};
  size_t arraysize = (BLOCKSIZE * GRIDSIZE);
  T data_value = std::numeric_limits<T>::max();
  outputVec_h = reinterpret_cast<T*> (malloc(sizeof(T) * arraysize));
  REQUIRE(outputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&outputVec_d, (sizeof(T) * arraysize)));
  // Launch Test Kernels using graph
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create Allocation Kernel Node
  hipGraphNode_t kernelnode_1;
  hipKernelNodeParams kernelNodeParams1{};
  void* kernelArgs1[] = {reinterpret_cast<void *>(&test_type)};
  kernelNodeParams1.func = reinterpret_cast<void *>(kerAlloc<T>);
  kernelNodeParams1.gridDim = dim3(GRIDSIZE);
  kernelNodeParams1.blockDim = dim3(BLOCKSIZE);
  kernelNodeParams1.sharedMemBytes = 0;
  kernelNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams1.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelnode_1, graph, nullptr, 0,
                                  &kernelNodeParams1));
  // Create Write Kernel Node
  hipGraphNode_t kernelnode_2;
  hipKernelNodeParams kernelNodeParams2{};
  void* kernelArgs2[] = {reinterpret_cast<void *>(&data_value)};
  kernelNodeParams2.func = reinterpret_cast<void *>(kerWrite<T>);
  kernelNodeParams2.gridDim = dim3(GRIDSIZE);
  kernelNodeParams2.blockDim = dim3(BLOCKSIZE);
  kernelNodeParams2.sharedMemBytes = 0;
  kernelNodeParams2.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams2.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelnode_2, graph, nullptr, 0,
                                  &kernelNodeParams2));
  // Create Free Kernel Node
  hipGraphNode_t kernelnode_3;
  hipKernelNodeParams kernelNodeParams3{};
  void* kernelArgs3[] =
  {&outputVec_d, reinterpret_cast<void *>(&test_type)};
  kernelNodeParams3.func = reinterpret_cast<void *>(kerFree<T>);
  kernelNodeParams3.gridDim = dim3(GRIDSIZE);
  kernelNodeParams3.blockDim = dim3(BLOCKSIZE);
  kernelNodeParams3.sharedMemBytes = 0;
  kernelNodeParams3.kernelParams = reinterpret_cast<void**>(kernelArgs3);
  kernelNodeParams3.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelnode_3, graph, nullptr, 0,
                                  &kernelNodeParams3));
  // Create Memcpy Node
  hipGraphNode_t memcpyD2H;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H, graph, nullptr, 0,
            outputVec_h, outputVec_d, (sizeof(T) * arraysize),
            hipMemcpyDeviceToHost));
  // Create dependencies for graph
  HIP_CHECK(hipGraphAddDependencies(graph, &kernelnode_1,
                                    &kernelnode_2, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernelnode_2,
                                    &kernelnode_3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernelnode_3,
                                    &memcpyD2H, 1));
  // Instantiate and launch the graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  bool bPassed = true;
  for (size_t idx = 0; idx < arraysize; idx++) {
    if (outputVec_h[idx] != data_value) {
      bPassed = false;
      break;
    }
  }
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  hipFree(outputVec_d);
  free(outputVec_h);
  return bPassed;
}
/**
 * Local function: Allocate local and device memory from host,
 * launches kerTestAllocationUsingDevFunc<<<>>> and copies data back
 * to host to validate.
 */
static bool TestAllocInDeviceFunc(int test_type) {
  int *outputVec_d{nullptr}, *outputVec_h{nullptr};
  size_t arraysize = (INTERNAL_BUFFER_SIZE * BLOCKSIZE * GRIDSIZE);
  outputVec_h = reinterpret_cast<int*> (malloc(sizeof(int) * arraysize));
  REQUIRE(outputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&outputVec_d, (sizeof(int) * arraysize)));
  // Launch Test Kernel
  kerTestAllocationUsingDevFunc<<<GRIDSIZE, BLOCKSIZE>>>(outputVec_d,
                                                        test_type);
  HIP_CHECK(hipDeviceSynchronize());
  // Copy to host buffer
  HIP_CHECK(hipMemcpy(outputVec_h, outputVec_d, sizeof(int) * arraysize,
                    hipMemcpyDefault));
  bool bPassed = true;
  for (size_t idx = 0; idx < arraysize; idx++) {
    if (outputVec_h[idx] != (idx / INTERNAL_BUFFER_SIZE)) {
      bPassed = false;
      break;
    }
  }
  hipFree(outputVec_d);
  free(outputVec_h);
  return bPassed;
}

/**
 * Scenario: This test validates device allocation and deallocation
 * using malloc/free in every gpu thread and block for primitive data
 * types like char, short, int etc.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_PerThread_PrimitiveDataType") {
  constexpr size_t sizePerThread = 128;

  // malloc()/free() tests
  SECTION("Test char datatype allocation with malloc") {
    REQUIRE(true == TestAllocInAllThread<char>(TEST_MALLOC_FREE,
                            SCHAR_MAX, sizePerThread));
  }

  SECTION("Test short datatype allocation with malloc") {
    REQUIRE(true == TestAllocInAllThread<int16_t>(TEST_MALLOC_FREE,
                            SHRT_MAX, sizePerThread));
  }

  SECTION("Test int datatype allocation with malloc") {
    REQUIRE(true == TestAllocInAllThread<int32_t>(TEST_MALLOC_FREE,
                            INT_MAX, sizePerThread));
  }

  SECTION("Test float datatype allocation with malloc") {
    REQUIRE(true == TestAllocInAllThread<float>(TEST_MALLOC_FREE,
                            FLT_MAX, sizePerThread));
  }

  SECTION("Test double datatype allocation with malloc") {
    REQUIRE(true == TestAllocInAllThread<double>(TEST_MALLOC_FREE,
                            DBL_MAX, sizePerThread));
  }
}

/**
 * Scenario: This test validates device allocation and deallocation
 * using new/delete in every gpu thread and block for primitive data
 * types like char, short, int etc.
 */
TEST_CASE("Unit_deviceAllocation_New_PerThread_PrimitiveDataType") {
  constexpr size_t sizePerThread = 128;

  // new/delete tests
  SECTION("Test char datatype allocation with new") {
    REQUIRE(true == TestAllocInAllThread<char>(TEST_NEW_DELETE,
                            SCHAR_MAX, sizePerThread));
  }

  SECTION("Test short datatype allocation with new") {
    REQUIRE(true == TestAllocInAllThread<int16_t>(TEST_NEW_DELETE,
                            SHRT_MAX, sizePerThread));
  }

  SECTION("Test int datatype allocation with new") {
    REQUIRE(true == TestAllocInAllThread<int32_t>(TEST_NEW_DELETE,
                            INT_MAX, sizePerThread));
  }

  SECTION("Test float datatype allocation with new") {
    REQUIRE(true == TestAllocInAllThread<float>(TEST_NEW_DELETE,
                            FLT_MAX, sizePerThread));
  }

  SECTION("Test double datatype allocation with new") {
    REQUIRE(true == TestAllocInAllThread<double>(TEST_NEW_DELETE,
                            DBL_MAX, sizePerThread));
  }
}

/**
 * Scenario: This test validates device allocation and deallocation
 * using malloc/free in every gpu thread and block for structure.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_PerThread_StructDataType") {
  constexpr size_t sizePerThread = 64;
  struct simpleStruct sampleStr{INT_MAX, DBL_MAX, FLT_MAX, SHRT_MAX,
                    SCHAR_MAX, {1, 2, 3, 4, 5, 6, 7, 8}};
  REQUIRE(true == TestAllocInAllThread<struct simpleStruct>(TEST_MALLOC_FREE,
                                sampleStr, sizePerThread));
}

/**
 * Scenario: This test validates device allocation and deallocation
 * using new/delete in every gpu thread and block for structure.
 */
TEST_CASE("Unit_deviceAllocation_New_PerThread_StructDataType") {
  constexpr size_t sizePerThread = 64;
  struct simpleStruct sampleStr{INT_MAX, DBL_MAX, FLT_MAX, SHRT_MAX,
                    SCHAR_MAX, {1, 2, 3, 4, 5, 6, 7, 8}};
  REQUIRE(true == TestAllocInAllThread<struct simpleStruct>(TEST_NEW_DELETE,
                                sampleStr, sizePerThread));
}

/**
 * Scenario: This test validates device memory allocation and free
 * in 1 thread and access in block for different primitive types like
 * char, short, int etc.
 */
TEST_CASE("Unit_deviceAllocation_InOneThread_AccessInAllThreads") {
  // malloc()/free() tests
  SECTION("Test char datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAccessInAllThread<char>(TEST_MALLOC_FREE, 0));
  }

  SECTION("Test short datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAccessInAllThread<int16_t>(TEST_MALLOC_FREE, 0));
  }

  SECTION("Test int datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAccessInAllThread<int32_t>(TEST_MALLOC_FREE, 0));
  }

  SECTION("Test float datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAccessInAllThread<float>(TEST_MALLOC_FREE, 0));
  }

  SECTION("Test double datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAccessInAllThread<double>(TEST_MALLOC_FREE, 0));
  }

  // new/delete tests
  SECTION("Test char datatype allocation with new") {
    REQUIRE(true == TestMemoryAccessInAllThread<char>(TEST_NEW_DELETE, 0));
  }

  SECTION("Test short datatype allocation with new") {
    REQUIRE(true == TestMemoryAccessInAllThread<int16_t>(TEST_NEW_DELETE, 0));
  }

  SECTION("Test int datatype allocation with new") {
    REQUIRE(true == TestMemoryAccessInAllThread<int32_t>(TEST_NEW_DELETE, 0));
  }

  SECTION("Test float datatype allocation with new") {
    REQUIRE(true == TestMemoryAccessInAllThread<float>(TEST_NEW_DELETE, 0));
  }

  SECTION("Test double datatype allocation with new") {
    REQUIRE(true == TestMemoryAccessInAllThread<double>(TEST_NEW_DELETE, 0));
  }
}

/**
 * Scenario: This test validates device allocation malloc, access and free
 * across multiple kernels for different primitive types like char, short,
 * int etc.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_AcrossKernels") {
  // malloc()/free() tests
  SECTION("Test char datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAcrossMulKernels<char>(TEST_MALLOC_FREE));
  }

  SECTION("Test short datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAcrossMulKernels<int16_t>(TEST_MALLOC_FREE));
  }

  SECTION("Test int datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAcrossMulKernels<int32_t>(TEST_MALLOC_FREE));
  }

  SECTION("Test float datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAcrossMulKernels<float>(TEST_MALLOC_FREE));
  }

  SECTION("Test double datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAcrossMulKernels<double>(TEST_MALLOC_FREE));
  }
}

/**
 * Scenario: This test validates device new, access and delete
 * across multiple kernels for different primitive types like char, short,
 * int etc.
 */
TEST_CASE("Unit_deviceAllocation_New_AcrossKernels") {
  // new/delete tests
  SECTION("Test char datatype allocation with new") {
    REQUIRE(true == TestMemoryAcrossMulKernels<char>(TEST_NEW_DELETE));
  }

  SECTION("Test short datatype allocation with new") {
    REQUIRE(true == TestMemoryAcrossMulKernels<int16_t>(TEST_NEW_DELETE));
  }

  SECTION("Test int datatype allocation with new") {
    REQUIRE(true == TestMemoryAcrossMulKernels<int32_t>(TEST_NEW_DELETE));
  }

  SECTION("Test float datatype allocation with new") {
    REQUIRE(true == TestMemoryAcrossMulKernels<float>(TEST_NEW_DELETE));
  }

  SECTION("Test double datatype allocation with new") {
    REQUIRE(true == TestMemoryAcrossMulKernels<double>(TEST_NEW_DELETE));
  }
}

/**
 * Scenarios:
 * A) This test validates device allocation malloc, access and free
 * across multiple kernels for nested structure.
 * B) This test also validates memory allocation and deallocation through
 * __device__ functions.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_ComplexDataType") {
  // malloc()/free() tests
  REQUIRE(true == TestMemoryAccessInAllThread_CmplxStr(TEST_MALLOC_FREE));
}

/**
 * Scenario:
 * A) This test validates device allocation malloc, access and free
 * across multiple kernels for nested structure.
 * B) This test also validates memory allocation and deallocation through
 * __device__ functions.
 */
TEST_CASE("Unit_deviceAllocation_New_ComplexDataType") {
  // new/delete tests
  REQUIRE(true == TestMemoryAccessInAllThread_CmplxStr(TEST_NEW_DELETE));
}

/**
 * Scenario: This test validates device allocation malloc, access and free
 * across multiple kernels for Union data type.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_UnionType") {
  // malloc()/free() tests
  REQUIRE(true == TestMemoryAccessInAllThread_Union(TEST_MALLOC_FREE));
}

/**
 * Scenario: This test validates device allocation new, access and delete
 * across multiple kernels for Union data type.
 */
TEST_CASE("Unit_deviceAllocation_New_UnionType") {
  // new/delete tests
  REQUIRE(true == TestMemoryAccessInAllThread_Union(TEST_NEW_DELETE));
}

/**
 * Scenario: This test validates device allocation and deallocation
 * using malloc/free in every gpu thread and block using Single
 * Code Object kernel.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_SingleCodeObj") {
  constexpr size_t sizePerThread = 128;

  REQUIRE(true == TestAlloc_Load_SingleKer_AllocFree(TEST_MALLOC_FREE,
                                INT_MAX, sizePerThread));
}

/**
 * Scenario: This test validates device allocation and deallocation
 * using new/delete in every gpu thread and block using Single
 * Code Object kernel.
 */
TEST_CASE("Unit_deviceAllocation_New_SingleCodeObj") {
  constexpr size_t sizePerThread = 128;

  REQUIRE(true == TestAlloc_Load_SingleKer_AllocFree(TEST_NEW_DELETE,
                            INT_MAX, sizePerThread));
}

#if HT_NVIDIA
/**
 * Scenario: This test validates device allocation and deallocation
 * using malloc/free in multikernel and multistream environment.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_PerThread_MultKerMultStrm") {
  // malloc()/free() tests
  SECTION("Test char datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAcrossMulKernels<char>(TEST_MALLOC_FREE,
                                                    true));
  }

  SECTION("Test short datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAcrossMulKernels<int16_t>(TEST_MALLOC_FREE,
                                                        true));
  }

  SECTION("Test int datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAcrossMulKernels<int32_t>(TEST_MALLOC_FREE,
                                                        true));
  }

  SECTION("Test float datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAcrossMulKernels<float>(TEST_MALLOC_FREE,
                                                    true));
  }

  SECTION("Test double datatype allocation with malloc") {
    REQUIRE(true == TestMemoryAcrossMulKernels<double>(TEST_MALLOC_FREE,
                                                       true));
  }
}

/**
 * Scenario: This test validates device allocation and deallocation
 * using new/delete in multikernel and multistream environment.
 */
TEST_CASE("Unit_deviceAllocation_New_PerThread_MultKerMultStrm") {
  // new/delete tests
  SECTION("Test char datatype allocation with new") {
    REQUIRE(true == TestMemoryAcrossMulKernels<char>(TEST_NEW_DELETE,
                                                    true));
  }

  SECTION("Test short datatype allocation with new") {
    REQUIRE(true == TestMemoryAcrossMulKernels<int16_t>(TEST_NEW_DELETE,
                                                        true));
  }

  SECTION("Test int datatype allocation with new") {
    REQUIRE(true == TestMemoryAcrossMulKernels<int32_t>(TEST_NEW_DELETE,
                                                        true));
  }

  SECTION("Test float datatype allocation with new") {
    REQUIRE(true == TestMemoryAcrossMulKernels<float>(TEST_NEW_DELETE,
                                                    true));
  }

  SECTION("Test double datatype allocation with new") {
    REQUIRE(true == TestMemoryAcrossMulKernels<double>(TEST_NEW_DELETE,
                                                    true));
  }
}
#endif

/**
 * Scenario: This test validates device allocation and deallocation
 * using malloc/free in graph.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_PerThread_Graph") {
  // malloc()/free() tests
  SECTION("Test char datatype allocation with malloc") {
    REQUIRE(true ==
            TestMemoryAcrossMulKernelsUsingGraph<char>(TEST_MALLOC_FREE));
  }

  SECTION("Test short datatype allocation with malloc") {
    REQUIRE(true ==
            TestMemoryAcrossMulKernelsUsingGraph<int16_t>(TEST_MALLOC_FREE));
  }

  SECTION("Test int datatype allocation with malloc") {
    REQUIRE(true ==
            TestMemoryAcrossMulKernelsUsingGraph<int32_t>(TEST_MALLOC_FREE));
  }

  SECTION("Test float datatype allocation with malloc") {
    REQUIRE(true ==
            TestMemoryAcrossMulKernelsUsingGraph<float>(TEST_MALLOC_FREE));
  }

  SECTION("Test double datatype allocation with malloc") {
    REQUIRE(true ==
            TestMemoryAcrossMulKernelsUsingGraph<double>(TEST_MALLOC_FREE));
  }
}

/**
 * Scenario: This test validates device allocation and deallocation
 * using new/delete in graph.
 */
TEST_CASE("Unit_deviceAllocation_New_PerThread_Graph") {
  // new/delete tests
  SECTION("Test char datatype allocation with new") {
    REQUIRE(true ==
            TestMemoryAcrossMulKernelsUsingGraph<char>(TEST_NEW_DELETE));
  }

  SECTION("Test short datatype allocation with new") {
    REQUIRE(true ==
            TestMemoryAcrossMulKernelsUsingGraph<int16_t>(TEST_NEW_DELETE));
  }

  SECTION("Test int datatype allocation with new") {
    REQUIRE(true ==
            TestMemoryAcrossMulKernelsUsingGraph<int32_t>(TEST_NEW_DELETE));
  }

  SECTION("Test float datatype allocation with new") {
    REQUIRE(true ==
            TestMemoryAcrossMulKernelsUsingGraph<float>(TEST_NEW_DELETE));
  }

  SECTION("Test double datatype allocation with new") {
    REQUIRE(true ==
            TestMemoryAcrossMulKernelsUsingGraph<double>(TEST_NEW_DELETE));
  }
}

/**
 * Scenario: This test validates device allocation malloc, access and free
 * using pointers to device functions.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_DeviceFunc") {
  // malloc/free tests
  REQUIRE(true == TestAllocInDeviceFunc(TEST_MALLOC_FREE));
}

/**
 * Scenario: This test validates device allocation new, access and delete
 * using pointers to device functions.
 */
TEST_CASE("Unit_deviceAllocation_New_DeviceFunc") {
  // new/delete tests
  REQUIRE(true == TestAllocInDeviceFunc(TEST_NEW_DELETE));
}

/**
 * Scenario: This test validates device allocation using vitual functions
 */
TEST_CASE("Unit_deviceAllocation_VirtualFunction") {
  int *outputVec_d{nullptr}, *outputVec_h{nullptr};
  constexpr size_t sizeBufferPerThread = 8;
  size_t arraysize = (sizeBufferPerThread * BLOCKSIZE * GRIDSIZE);
  outputVec_h = reinterpret_cast<int*> (malloc(sizeof(int) * arraysize));
  REQUIRE(outputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&outputVec_d, (sizeof(int) * arraysize)));
  // Launch Test Kernel
  kerTestDynamicAllocVirtualFunc<<<GRIDSIZE, BLOCKSIZE>>>(
                    outputVec_d, sizeBufferPerThread);
  HIP_CHECK(hipDeviceSynchronize());
  // Copy to host buffer
  HIP_CHECK(hipMemcpy(outputVec_h, outputVec_d, sizeof(int) * arraysize,
                    hipMemcpyDefault));
  bool bPassed = true;
  for (size_t idx = 0; idx < arraysize; idx++) {
    if (outputVec_h[idx] != (idx / sizeBufferPerThread)) {
      bPassed = false;
      break;
    }
  }
  REQUIRE(true == bPassed);
  hipFree(outputVec_d);
  free(outputVec_h);
}

/**
 * Scenario: This test validates device allocation malloc, access and free
 * across multiple kernels launched using threads.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_MulKernels_MulThreads") {
  // malloc()/free() tests
  SECTION("Test char datatype allocation with malloc") {
    REQUIRE(true == TestDevMemAllocMulKerMulThrd<char>(TEST_MALLOC_FREE));
  }

  SECTION("Test short datatype allocation with malloc") {
    REQUIRE(true == TestDevMemAllocMulKerMulThrd<int16_t>(TEST_MALLOC_FREE));
  }

  SECTION("Test int datatype allocation with malloc") {
    REQUIRE(true == TestDevMemAllocMulKerMulThrd<int32_t>(TEST_MALLOC_FREE));
  }

  SECTION("Test float datatype allocation with malloc") {
    REQUIRE(true == TestDevMemAllocMulKerMulThrd<float>(TEST_MALLOC_FREE));
  }

  SECTION("Test double datatype allocation with malloc") {
    REQUIRE(true == TestDevMemAllocMulKerMulThrd<double>(TEST_MALLOC_FREE));
  }
}

/**
 * Scenario: This test validates device new, access and delete
 * across multiple kernels launched using threads.
 */
TEST_CASE("Unit_deviceAllocation_New_MulKernels_MulThreads") {
  // new/delete tests
  SECTION("Test char datatype allocation with new") {
    REQUIRE(true == TestDevMemAllocMulKerMulThrd<char>(TEST_NEW_DELETE));
  }

  SECTION("Test short datatype allocation with new") {
    REQUIRE(true == TestDevMemAllocMulKerMulThrd<int16_t>(TEST_NEW_DELETE));
  }

  SECTION("Test int datatype allocation with new") {
    REQUIRE(true == TestDevMemAllocMulKerMulThrd<int32_t>(TEST_NEW_DELETE));
  }

  SECTION("Test float datatype allocation with new") {
    REQUIRE(true == TestDevMemAllocMulKerMulThrd<float>(TEST_NEW_DELETE));
  }

  SECTION("Test double datatype allocation with new") {
    REQUIRE(true == TestDevMemAllocMulKerMulThrd<double>(TEST_NEW_DELETE));
  }
}

#if HT_AMD
// Scenarios Unit_deviceAllocation_Malloc_SingKernels_MulThreads and
// are failing on NVIDIA platform.
/**
 * Scenario: This test validates device allocation malloc, access and free
 * in a single kernel launched using threads.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_SingKernels_MulThreads") {
  // malloc()/free() tests
  std::vector<std::thread> tests;
  // Spawn the test threads
  for (int idx = 0; idx < num_threads; idx++) {
    thread_results[idx] = false;
    tests.push_back(std::thread(runTestMemoryAccessInAllThread<int32_t>,
                                TEST_MALLOC_FREE, idx));
  }
  // Wait for all threads to complete
  for (std::thread &t : tests) {
    t.join();
  }
  // Verify All Results
  for (int idx = 0; idx < num_threads; idx++) {
    REQUIRE(thread_results[idx]);
  }
}

/**
 * Scenario: This test validates device new, access and delete
 * in a single kernel launched using threads.
 */
TEST_CASE("Unit_deviceAllocation_New_SingKernels_MulThreads") {
  // new/delete tests
  std::vector<std::thread> tests;
  // Spawn the test threads
  for (int idx = 0; idx < num_threads; idx++) {
    thread_results[idx] = false;
    tests.push_back(std::thread(runTestMemoryAccessInAllThread<int32_t>,
                                TEST_NEW_DELETE, idx));
  }
  // Wait for all threads to complete
  for (std::thread &t : tests) {
    t.join();
  }
  // Verify All Results
  for (int idx = 0; idx < num_threads; idx++) {
    REQUIRE(thread_results[idx]);
  }
}
#endif

/**
 * Scenario: This test validates Allocation and Deallocation in multiple
 * code object kernels defined in different source files.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_MulCodeObj") {
  REQUIRE(true == TestAlloc_Load_MultKernels(TEST_MALLOC_FREE,
                                INT_MAX));
}

/**
 * Scenario: This test validates Allocation and Deallocation in multiple
 * code object kernels defined in different source files.
 */
TEST_CASE("Unit_deviceAllocation_New_MulCodeObj") {
  REQUIRE(true == TestAlloc_Load_MultKernels(TEST_NEW_DELETE,
                                INT_MAX));
}
