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
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#ifdef __linux__
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <dlfcn.h>
#endif

#define SIZE 2097152
// GPU threads
#define BLOCKSIZE 512
#define GRIDSIZE 256

__device__ static char* dev_common_ptr = nullptr;

/**
 * This kernel allocates a memory chunk using malloc().
 */
static __global__ void kerTestDeviceMalloc(size_t size) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate
  if (myId == 0) {
    dev_common_ptr = reinterpret_cast<char*> (malloc(size));
    if (dev_common_ptr == nullptr) {
      printf("Device Allocation Failed! \n");
      return;
    }
  }
}

/**
 * This kernel writes to the memory location allocated in kernel
 * kerTestDeviceMalloc or kerTestDeviceNew.
 */
static __global__ void kerTestDeviceWrite() {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate
  if (dev_common_ptr == nullptr) {
    printf("Device Allocation Failed! \n");
    return;
  }
  *(dev_common_ptr + myId) = SCHAR_MAX;
}

/**
 * This kernel frees the memory chunk allocated in kernel
 * kerTestDeviceMalloc using free().
 */
static __global__ void kerTestDeviceFree(int *result) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate
  if (myId == 0) {
    if (dev_common_ptr != nullptr) {
      *result = 1;
      for (int idx = 0; idx < (BLOCKSIZE*GRIDSIZE); idx++) {
        if (*(dev_common_ptr + myId) != SCHAR_MAX) {
          *result = 0;
          break;
        }
      }
      free(dev_common_ptr);
    } else {
      *result = 0;
    }
  }
}

/**
 * This kernel allocates a memory chunk using new operator.
 */
static __global__ void kerTestDeviceNew(size_t size) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate
  if (myId == 0) {
    dev_common_ptr = new char[size];
    if (dev_common_ptr == nullptr) {
      printf("Device Allocation Failed! \n");
      return;
    }
  }
}

/**
 * This kernel frees the memory chunk allocated in kernel
 * kerTestDeviceNew using delete operator.
 */
static __global__ void kerTestDeviceDelete(int *result) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate
  if (myId == 0) {
    if (dev_common_ptr != nullptr) {
      *result = 1;
      for (int idx = 0; idx < (BLOCKSIZE*GRIDSIZE); idx++) {
        if (*(dev_common_ptr + myId) != SCHAR_MAX) {
          *result = 0;
          break;
        }
      }
      delete[] dev_common_ptr;
    } else {
      *result = 0;
    }
  }
}

/**
 * Test device malloc()/new in both Parent and Child Process.
 * Allocate SIZE bytes in both parent and child process. Verify
 * the allocated size in both parent and child process.
 */
static bool testDeviceAllocMulProc(bool testmalloc) {
  int fd[2];
  pid_t childpid;
  bool testResult = false;
  size_t avail = 0, tot = 0;
  // create pipe descriptors
  pipe(fd);
  // fork process
  childpid = fork();
  if (childpid > 0) {  // Parent
    close(fd[1]);
    // Allocate in parent
    if (testmalloc) {
      kerTestDeviceMalloc<<<1, 1>>>(SIZE);
    } else {
      kerTestDeviceNew<<<1, 1>>>(SIZE);
    }
    HIP_CHECK(hipDeviceSynchronize());
    // Check allocated memory size
    HIP_CHECK(hipMemGetInfo(&avail, &tot));
    if ((tot - avail) < SIZE) {
      return false;
    }
    // parent will wait to read the device cnt
    read(fd[0], &testResult, sizeof(testResult));
    // close the read-descriptor
    close(fd[0]);
    // wait for child exit
    wait(NULL);
    // At this point the child process exits.
    // Ensure that device memory allocated from child is freed.
    HIP_CHECK(hipMemGetInfo(&avail, &tot));
    if ((tot - avail) < SIZE) {
      testResult = false;
    }
  } else if (!childpid) {  // Child
    // Wait for hipDeviceSetLimit() completion in parent.
    close(fd[0]);
    // Allocate in child
    if (testmalloc) {
      kerTestDeviceMalloc<<<1, 1>>>(SIZE);
    } else {
      kerTestDeviceNew<<<1, 1>>>(SIZE);
    }
    HIP_CHECK(hipDeviceSynchronize());
    // Check allocated memory size
    HIP_CHECK(hipMemGetInfo(&avail, &tot));
    if ((tot - avail) < SIZE) {
      testResult = false;
    } else {
      testResult = true;
    }
    // send the value on the write-descriptor:
    write(fd[1], &testResult, sizeof(testResult));
    // close the write descriptor:
    close(fd[1]);
    exit(0);
  }
  return testResult;
}

/**
 * Test device malloc()/new, write and free()/delete[]
 * from both Parent and Child Process. From both Parent and
 * Child Process invoke the kernel to allocate memory, the
 * kernel to write to the allocated memory and a third kernel
 * to verify the memory contents and free it.
 */
static bool testDeviceMemMulProc(bool testmalloc) {
  int fd[2];
  bool testResult = false;
  pid_t childpid;
  int testResultChild = 0;
  size_t size = BLOCKSIZE*GRIDSIZE;
  // create pipe descriptors
  pipe(fd);
  // fork process
  childpid = fork();
  if (childpid > 0) {  // Parent
    close(fd[1]);
    int *result_d{nullptr}, *result_h{nullptr};
    HIP_CHECK(hipMalloc(&result_d, sizeof(int)));
    result_h = reinterpret_cast<int*> (malloc(sizeof(int)));
    REQUIRE(result_h != nullptr);
    // Allocate in parent
    if (testmalloc) {
      kerTestDeviceMalloc<<<1, 1>>>(size);
    } else {
      kerTestDeviceNew<<<1, 1>>>(size);
    }
    // Write
    kerTestDeviceWrite<<<GRIDSIZE, BLOCKSIZE>>>();
    // Free
    if (testmalloc) {
      kerTestDeviceFree<<<1, 1>>>(result_d);
    } else {
      kerTestDeviceDelete<<<1, 1>>>(result_d);
    }
    HIP_CHECK(hipDeviceSynchronize());
    *result_h = 0;
    HIP_CHECK(hipMemcpy(result_h, result_d, sizeof(int),
              hipMemcpyDefault));
    if (*result_h == 0) {
      testResult = false;
    } else {
      testResult = true;
    }
    // parent will wait to read the device cnt
    read(fd[0], &testResultChild, sizeof(int));
    if (testResultChild == 0) {
      testResult &= false;
    } else {
      testResult &= true;
    }
    // close the read-descriptor
    close(fd[0]);
    hipFree(result_d);
    free(result_h);
    // wait for child exit
    wait(NULL);
  } else if (!childpid) {  // Child
    // Wait for hipDeviceSetLimit() completion in parent.
    close(fd[0]);
    int *result_d{nullptr}, *result_h{nullptr};
    HIP_CHECK(hipMalloc(&result_d, sizeof(int)));
    result_h = reinterpret_cast<int*> (malloc(sizeof(int)));
    REQUIRE(result_h != nullptr);
    // Allocate in child
    if (testmalloc) {
      kerTestDeviceMalloc<<<1, 1>>>(size);
    } else {
      kerTestDeviceNew<<<1, 1>>>(size);
    }
    // Write
    kerTestDeviceWrite<<<GRIDSIZE, BLOCKSIZE>>>();
    // Free
    if (testmalloc) {
      kerTestDeviceFree<<<1, 1>>>(result_d);
    } else {
      kerTestDeviceDelete<<<1, 1>>>(result_d);
    }
    HIP_CHECK(hipDeviceSynchronize());
    *result_h = 0;
    HIP_CHECK(hipMemcpy(result_h, result_d, sizeof(int),
              hipMemcpyDefault));
    // send the value on the write-descriptor:
    write(fd[1], result_h, sizeof(int));
    // close the write descriptor:
    close(fd[1]);
    hipFree(result_d);
    free(result_h);
    exit(0);
  }
  return testResult;
}

/**
 * Multiprocess device side malloc test.
 */
TEST_CASE("Unit_deviceAllocation_Malloc_MultProcess") {
  auto res = testDeviceAllocMulProc(true);
  REQUIRE(res == true);
}

/**
 * Multiprocess device side new test.
 */
TEST_CASE("Unit_deviceAllocation_New_MultProcess") {
  auto res = testDeviceAllocMulProc(false);
  REQUIRE(res == true);
}

/**
 * Multiprocess device side malloc, write and free test.
 */
TEST_CASE("Unit_deviceAllocation_MallocFree_MultProcess") {
  auto res = testDeviceMemMulProc(true);
  REQUIRE(res == true);
}

/**
 * Multiprocess device side new, write and delete test.
 */
TEST_CASE("Unit_deviceAllocation_NewDelete_MultProcess") {
  auto res = testDeviceMemMulProc(false);
  REQUIRE(res == true);
}
