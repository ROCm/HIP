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
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <unistd.h>
// Size Macros
#define MEMORY_CHUNK_SIZE (1024*1024)
#define MEMORY_CHUNK_SIZE_ODD (1025*1025)
#define MAXIMUM_CHUNKS (256*1024)
// Subtest Macros
#define NO_ALLOCATION_ONHOST 0
#define ALLOCATE_ONHOST_HIPMALLOCMANAGED 1
#define ALLOCATE_ONHOST_HIPMALLOC 2
// Test Type Macros
#define TEST_MALLOC_FREE 1
#define TEST_NEW_DELETE 2
// GPU threads
#define BLOCKSIZE 512
#define GRIDSIZE 512
// Test parameters
// Two different loops
#define NUM_OF_LOOP_SINGLE_KER 100000
#define NUM_OF_LOOP_MULTIPLE_KER 20000

// The following flag is defined for platforms (nvidia)
// which honors device memory limit. For AMD this flag
// is disabled and defect is raised.
#if HT_NVIDIA
#define HT_HONORS_DEVICEMEMORY_LIMIT
#endif

#ifdef HT_HONORS_DEVICEMEMORY_LIMIT
__device__ static char* dev_mem_glob[MAXIMUM_CHUNKS];
#endif
__device__ static int* dev_mem[GRIDSIZE];
__device__ static int* dev_common_ptr;

#ifdef HT_HONORS_DEVICEMEMORY_LIMIT
/**
 * This kernel checks kernel allocation of size more than available
 * memory.
 */
static __global__ void kerTestDynamicAllocNeg(int test_type,
                                           size_t perThreadSize,
                                           int *ret) {
  // Allocate
  char* ptr = nullptr;
  printf("Memory to allocate in GPU = %zu \n", perThreadSize);
  if (test_type == TEST_MALLOC_FREE) {
    ptr = reinterpret_cast<char*> (malloc(perThreadSize));
  } else {
    ptr = new char[perThreadSize];
  }
  printf("Allocation Done \n");
  if (ptr == nullptr) {
    printf("Allocation Failed. PASSED! \n");
    *ret = 0;
    return;
  } else {
    // Free memory
    if (test_type == TEST_MALLOC_FREE) {
      free(ptr);
    } else {
      delete[] ptr;
    }
    *ret = -1;
  }
}

/**
 * This kernel allocates memory till nullptr is returned.
 */
static __global__ void kerAllocTillExhaust(int test_type,
                        size_t *total_allocated_mem,
                        size_t mem_chunk_size) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate memory in thread 0 of block 0
  if (0 == myId) {
    for (int idx = 0; idx < MAXIMUM_CHUNKS; idx++) {
      dev_mem_glob[idx] = nullptr;
    }
    int idx = 0;
    if (test_type == TEST_MALLOC_FREE) {
      do {
        dev_mem_glob[idx] =
        reinterpret_cast<char*> (malloc(mem_chunk_size));
        if (idx >= MAXIMUM_CHUNKS) {
          break;
        }
      } while (dev_mem_glob[idx++] != nullptr);
    } else {
      do {
        dev_mem_glob[idx] =
        reinterpret_cast<char*> (new char[mem_chunk_size]);
        if (idx >= MAXIMUM_CHUNKS) {
          break;
        }
      } while (dev_mem_glob[idx++] != nullptr);
    }
    idx = 0;
    *total_allocated_mem = 0;
    while ((dev_mem_glob[idx] != nullptr) &&
           (idx < MAXIMUM_CHUNKS)) {
      *total_allocated_mem = *total_allocated_mem + mem_chunk_size;
      idx++;
    }
  }
}

/**
 * This kernel deletes the memory.
 */
static __global__ void kerFreeAll(int test_type) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (0 == myId) {
    if (test_type == TEST_MALLOC_FREE) {
      int idx = 0;
      while (dev_mem_glob[idx] != nullptr) {
        free(dev_mem_glob[idx++]);
        if (idx >= MAXIMUM_CHUNKS) {
          break;
        }
      }
    } else {
      int idx = 0;
      while (dev_mem_glob[idx] != nullptr) {
        delete[] (dev_mem_glob[idx++]);
        if (idx >= MAXIMUM_CHUNKS) {
          break;
        }
      }
    }
  }
}
#endif
/**
 * This kernel allocates memory once in thread 0 of each block and
 * access this memory in all threads of the block. The memory is
 * finally deleted in last thread of each block.
 */
static __global__ void kerBlockLevelMemoryAllocation(int *outputBuf,
                                                     int test_type) {
  int myThreadId = threadIdx.x, lastThreadId = (blockDim.x - 1);
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate memory in thread 0
  if (0 == myThreadId) {
    if (test_type == TEST_MALLOC_FREE) {
      dev_mem[blockIdx.x] =
      reinterpret_cast<int*> (malloc(blockDim.x*sizeof(int)));
    } else {
      dev_mem[blockIdx.x] =
      reinterpret_cast<int*> (new int[blockDim.x]);
    }
  }
  // All threads wait at this barrier
  __syncthreads();
  // Check allocated memory in all threads in block before access
  if (dev_mem[blockIdx.x] == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }
  int *ptr = reinterpret_cast<int*> (dev_mem[blockIdx.x]);
  // Copy to buffer
  ptr[myThreadId] = myId;
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
static __global__ void kerAlloc(int test_type) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate memory in thread 0 of block 0
  if (0 == myId) {
    if (test_type == TEST_MALLOC_FREE) {
      dev_common_ptr =
      reinterpret_cast<int*> (malloc(blockDim.x*gridDim.x*sizeof(int)));
    } else {
      dev_common_ptr =
      reinterpret_cast<int*> (new int[blockDim.x*gridDim.x]);
    }
  }
}

/**
 * This kernel writes to memory allocated in <kerAlloc>.
 */
static __global__ void kerWrite() {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Check allocated memory in all threads in block before access
  if (dev_common_ptr == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }
  // Copy to buffer
  dev_common_ptr[myId] = myId;
}

/**
 * This kernel copies the contents of memory allocated in <kerAlloc>
 * to host and deletes the memory from thread 0.
 */
static __global__ void kerFree(int *outputBuf, int test_type) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Check allocated memory in all threads in block before access
  if (dev_common_ptr == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }
  if (0 == myId) {
    for (size_t idx = 0; idx < (blockDim.x*gridDim.x); idx++) {
      outputBuf[idx] = dev_common_ptr[idx];
    }
    if (test_type == TEST_MALLOC_FREE) {
      free(dev_common_ptr);
    } else {
      delete[] dev_common_ptr;
    }
  }
}

#ifdef HT_HONORS_DEVICEMEMORY_LIMIT
/**
 * Local function: Launch kerAllocTillExhaust<<<>>> and
 * kerFreeAll<<<>>> to test memory allocation till all device
 * memory is exhausted.
 */
static bool TestAllocationOfAllAvailableMemory(int test_type,
                            int category, size_t mem_chunk_size) {
  size_t avail1 = 0, avail2 = 0, tot = 0;
  constexpr size_t host_alloc = 2147483648;  // 2 GB
  HIP_CHECK(hipMemGetInfo(&avail1, &tot));
#if HT_NVIDIA
  HIP_CHECK(hipDeviceSetLimit(hipLimitMallocHeapSize, avail1));
#endif
  size_t *tot_alloc_mem_d = nullptr, *tot_alloc_mem_h = nullptr;
  tot_alloc_mem_h =
  reinterpret_cast<size_t*> (malloc(sizeof(size_t)));
  REQUIRE(nullptr != tot_alloc_mem_h);
  HIP_CHECK(hipMalloc(&tot_alloc_mem_d, sizeof(size_t)));
  REQUIRE(nullptr != tot_alloc_mem_d);
  char *devptrHost = nullptr;
  if (category == ALLOCATE_ONHOST_HIPMALLOCMANAGED) {
    HIP_CHECK(hipMallocManaged(&devptrHost, host_alloc));
  } else if (category == ALLOCATE_ONHOST_HIPMALLOC) {
    HIP_CHECK(hipMalloc(&devptrHost, host_alloc));
  }
  HIP_CHECK(hipMemGetInfo(&avail1, &tot));
  INFO("Total available memory " << tot);
  INFO("Available memory before allocation " << avail1);
  // Launch Test Kernel
  kerAllocTillExhaust<<<1, 1>>>(test_type, tot_alloc_mem_d,
                                mem_chunk_size);
  HIP_CHECK(hipDeviceSynchronize());
  // Copy to host buffer
  HIP_CHECK(hipMemcpy(tot_alloc_mem_h, tot_alloc_mem_d,
                      sizeof(size_t), hipMemcpyDefault));
  HIP_CHECK(hipMemGetInfo(&avail2, &tot));
  kerFreeAll<<<1, 1>>>(test_type);
  HIP_CHECK(hipDeviceSynchronize());
  // Copy to host buffer
  bool bPassed = false;
  INFO("Available memory after allocation " << avail2);
  if (category == NO_ALLOCATION_ONHOST) {
    size_t allocated_dev_mem = (tot - avail2);
    if (allocated_dev_mem >= *tot_alloc_mem_h) {
      bPassed = true;
    }
  } else if ((category == ALLOCATE_ONHOST_HIPMALLOCMANAGED) ||
             (category == ALLOCATE_ONHOST_HIPMALLOC)) {
    size_t allocated_dev_mem = (tot - avail2 - host_alloc);
    if (allocated_dev_mem >= *tot_alloc_mem_h) {
      bPassed = true;
    }
    hipFree(devptrHost);
  }
  hipFree(tot_alloc_mem_d);
  free(tot_alloc_mem_h);
  return bPassed;
}
#endif
/**
 * Local function: Launch kerBlockLevelMemoryAllocation<<<>>>
 * in a loop to stress test allocation and deallocation.
 */
static bool TestMemoryAllocationInLoop(int test_type,
                                       bool isMultikernel = false) {
  int *outputVec_d{nullptr}, *outputVec_h{nullptr};
  int arraysize = (BLOCKSIZE * GRIDSIZE);
  outputVec_h = reinterpret_cast<int*> (malloc(sizeof(int) * arraysize));
  REQUIRE(outputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&outputVec_d, (sizeof(int) * arraysize)));
  bool bPassed = true;
  // Launch Test Kernel
  int max_index = 0;
  if (isMultikernel) {
    max_index = NUM_OF_LOOP_MULTIPLE_KER;
  } else {
    max_index = NUM_OF_LOOP_SINGLE_KER;
  }
  for (int idx = 0; idx < max_index; idx++) {
    if (isMultikernel) {
      kerAlloc<<<GRIDSIZE, BLOCKSIZE>>>(test_type);
      kerWrite<<<GRIDSIZE, BLOCKSIZE>>>();
      kerFree<<<GRIDSIZE, BLOCKSIZE>>>(outputVec_d, test_type);
    } else {
      kerBlockLevelMemoryAllocation<<<GRIDSIZE, BLOCKSIZE>>>(outputVec_d,
                                                   test_type);
    }
    HIP_CHECK(hipDeviceSynchronize());
    // Copy to host buffer
    HIP_CHECK(hipMemcpy(outputVec_h, outputVec_d, sizeof(int) * arraysize,
                        hipMemcpyDefault));
    bPassed = true;
    for (int idx = 0; idx < arraysize; idx++) {
      if (outputVec_h[idx] != idx) {
        bPassed = false;
        break;
      }
    }
    if (!bPassed) break;
  }
  hipFree(outputVec_d);
  free(outputVec_h);
  return bPassed;
}

#ifdef HT_HONORS_DEVICEMEMORY_LIMIT
/**
 * Scenario: Test malloc till nullptr is returned using even chunksize.
 */
TEST_CASE("Stress_deviceAllocation_malloc_Even") {
  REQUIRE(true == TestAllocationOfAllAvailableMemory(TEST_MALLOC_FREE,
                NO_ALLOCATION_ONHOST, MEMORY_CHUNK_SIZE));
}

/**
 * Scenario: Test malloc till nullptr is returned using odd chunksize.
 */
TEST_CASE("Stress_deviceAllocation_malloc_Odd") {
  REQUIRE(true == TestAllocationOfAllAvailableMemory(TEST_MALLOC_FREE,
                NO_ALLOCATION_ONHOST, MEMORY_CHUNK_SIZE_ODD));
}

/**
 * Scenario: Test new till nullptr is returned using even chunksize.
 */
TEST_CASE("Stress_deviceAllocation_new_Even") {
  REQUIRE(true == TestAllocationOfAllAvailableMemory(TEST_NEW_DELETE,
                NO_ALLOCATION_ONHOST, MEMORY_CHUNK_SIZE));
}

/**
 * Scenario: Test new till nullptr is returned using odd chunksize.
 */
TEST_CASE("Stress_deviceAllocation_new_Odd") {
  REQUIRE(true == TestAllocationOfAllAvailableMemory(TEST_NEW_DELETE,
                NO_ALLOCATION_ONHOST, MEMORY_CHUNK_SIZE_ODD));
}

/**
 * Scenario: This test checks device allocation using malloc till nullptr
 * is returned. Device memory is also allocated using hipmallocmanaged
 * from host.
 */
TEST_CASE("Stress_deviceAllocation_malloc_hipmallocmanaged") {
  REQUIRE(true == TestAllocationOfAllAvailableMemory(TEST_MALLOC_FREE,
                ALLOCATE_ONHOST_HIPMALLOCMANAGED, MEMORY_CHUNK_SIZE));
}

/**
 * Scenario: This test checks device allocation using new till nullptr
 * is returned. Device memory is also allocated using hipmallocmanaged
 * from host.
 */
TEST_CASE("Stress_deviceAllocation_new_hipmallocmanaged") {
  REQUIRE(true == TestAllocationOfAllAvailableMemory(TEST_NEW_DELETE,
                ALLOCATE_ONHOST_HIPMALLOCMANAGED, MEMORY_CHUNK_SIZE));
}

/**
 * Scenario: This test checks device allocation using malloc till nullptr
 * is returned. Device memory is also allocated using hipmalloc from host.
 */
TEST_CASE("Stress_deviceAllocation_malloc_hipmalloc") {
  REQUIRE(true == TestAllocationOfAllAvailableMemory(TEST_MALLOC_FREE,
                ALLOCATE_ONHOST_HIPMALLOC, MEMORY_CHUNK_SIZE));
}

/**
 * Scenario: This test checks device allocation using new till nullptr
 * is returned. Device memory is also allocated using hipmalloc from host.
 */
TEST_CASE("Stress_deviceAllocation_new_hipmalloc") {
  REQUIRE(true == TestAllocationOfAllAvailableMemory(TEST_NEW_DELETE,
                ALLOCATE_ONHOST_HIPMALLOC, MEMORY_CHUNK_SIZE));
}

/**
 * Scenario: This test validates device allocation negative scenario
 * when size > available memory.
 */
TEST_CASE("Stress_deviceAllocation_Negative") {
  int *ret_d{nullptr}, *ret_h{nullptr};
  size_t avail = 0, tot = 0;
  HIP_CHECK(hipMemGetInfo(&avail, &tot));
  printf("Available Memory in GPU = %zu \n", avail);
  ret_h = reinterpret_cast<int*> (malloc(sizeof(int)));
  REQUIRE(ret_h != nullptr);
  HIP_CHECK(hipMalloc(&ret_d, (sizeof(int))));
  SECTION("Test allocation with malloc") {
    kerTestDynamicAllocNeg<<<1, 1>>>(TEST_MALLOC_FREE, (avail + 1), ret_d);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(ret_h, ret_d, sizeof(int), hipMemcpyDefault));
    REQUIRE(0 == *ret_h);
  }

  SECTION("Test allocation with new") {
    kerTestDynamicAllocNeg<<<1, 1>>>(TEST_NEW_DELETE, (avail + 1), ret_d);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(ret_h, ret_d, sizeof(int), hipMemcpyDefault));
    REQUIRE(0 == *ret_h);
  }
  hipFree(ret_d);
  free(ret_h);
}
#endif
/**
 * Scenario: This test performs stress test of malloc/free in a loop
 * using single kernel.
 */
TEST_CASE("Stress_deviceAllocation_malloc_loop_singlekernel") {
  REQUIRE(true == TestMemoryAllocationInLoop(TEST_MALLOC_FREE, false));
}

/**
 * Scenario: This test performs stress test of new/delete in a loop
 * using single kernel.
 */
TEST_CASE("Stress_deviceAllocation_new_loop_singlekernel") {
  REQUIRE(true == TestMemoryAllocationInLoop(TEST_NEW_DELETE, false));
}

/**
 * Scenario: This test performs stress test of malloc/free in a loop
 * using multiple kernel.
 */
TEST_CASE("Stress_deviceAllocation_malloc_loop_multkernel") {
  REQUIRE(true == TestMemoryAllocationInLoop(TEST_MALLOC_FREE, true));
}

/**
 * Scenario: This test performs stress test of new/delete in a loop
 * using multiple kernel.
 */
TEST_CASE("Stress_deviceAllocation_new_loop_multkernel") {
  REQUIRE(true == TestMemoryAllocationInLoop(TEST_NEW_DELETE, true));
}
