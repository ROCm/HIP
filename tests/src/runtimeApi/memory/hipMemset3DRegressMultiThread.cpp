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

/**
Testcase Scenarios :

 (TestCase 1)::
 1) Validate Async behavior of hipMemset3DAsync with commands queued
 concurrently from multiple threads.
 2) Validate hipMemset3DAsync behavior when api is queued along with kernel
 function operating on same memory.

 (TestCase 2)::
 3) Perform regression of hipMemset3D api in loop with device memory allocated
 on different gpus.
 4) Perform regression of hipMemset3DAsync api in loop with device memory
 allocated on different gpus.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * TEST: %t --tests 1
 * HIT_END
 */

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include "test_common.h"


/*
 * Defines
 */
#define MAX_REGRESS_ITERS 20

/**
 * kernel function sets device memory with value passed
 */
__global__ void func_set_value(hipPitchedPtr devicePitchedPointer,
                               hipExtent extent,
                               unsigned char val) {
  // Index Calculation
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;

  // Get attributes from device pitched pointer
  char *devicePointer = reinterpret_cast<char *>(devicePitchedPointer.ptr);
  size_t pitch = devicePitchedPointer.pitch;
  size_t slicePitch = pitch * extent.height;

  // Loop over the device buffer
  if (z < extent.depth) {
    char *current_slice_index = devicePointer + z * slicePitch;
    if (y < extent.height) {
      // Get data array containing all elements from the current row
      char *current_row = reinterpret_cast<char *>(current_slice_index
                                                   + y * pitch);
      if (x < extent.width) {
        current_row[x] = val;
      }
    }
  }
}

/**
 * Fetches Gpu device count
 */
void getDeviceCount(int *pdevCnt) {
#ifdef __linux__
  int fd[2], val = 0;
  pid_t childpid;

  // create pipe descriptors
  pipe(fd);

  // disable visible_devices env from shell
  unsetenv("ROCR_VISIBLE_DEVICES");
  unsetenv("HIP_VISIBLE_DEVICES");

  childpid = fork();

  if (childpid > 0) {  // Parent
    close(fd[1]);
    // parent will wait to read the device cnt
    read(fd[0], &val, sizeof(val));

    // close the read-descriptor
    close(fd[0]);

    // wait for child exit
    wait(NULL);

    *pdevCnt = val;
  } else if (!childpid) {  // Child
    int devCnt = 1;
    // writing only, no need for read-descriptor
    close(fd[0]);

    HIPCHECK(hipGetDeviceCount(&devCnt));
    // send the value on the write-descriptor:
    write(fd[1], &devCnt, sizeof(devCnt));

    // close the write descriptor:
    close(fd[1]);
    exit(0);
  } else {  // failure
    *pdevCnt = 1;
    return;
  }

#else
  HIPCHECK(hipGetDeviceCount(pdevCnt));
#endif
}

/**
 * Performs api regression in loop
 */
bool loopRegression(bool bAsync) {
  bool testPassed = true;
  char *A_h;
  int memsetval = 1, numGpu = 0, hasPeerAccess = 0;
  size_t numH = 256, numW = 100, depth = 10;
  size_t width = numW * sizeof(char);
  hipExtent extent = make_hipExtent(width, numH, depth);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW* numH* depth;
  std::vector<hipPitchedPtr> devPitchedPtrlist;
  hipPitchedPtr pitchedPtr, devpPtr;

  A_h = reinterpret_cast<char *>(malloc(sizeElements));
  HIPASSERT(A_h != NULL);
  memset(A_h, 0, sizeElements);

  // Populate hipMemcpy3D parameters
  hipMemcpy3DParms myparms = {0};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.extent = extent;
#ifdef __HIP_PLATFORM_NVCC__
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif

  getDeviceCount(&numGpu);

  // Alloc 3D arrays in all GPUs
  for (int j = 0; j < numGpu; j++) {
    HIPCHECK(hipSetDevice(j));
    HIPCHECK(hipMalloc3D(&pitchedPtr, extent));
    devPitchedPtrlist.push_back(pitchedPtr);
  }

  for (int itern = 0; itern < MAX_REGRESS_ITERS; itern++) {
    // Validate hipMemset3D data consistency in multiple iters
    for (int i = 0; i < numGpu; i++) {
      for (int j = 0; j < numGpu; j++) {
        HIPCHECK(hipDeviceCanAccessPeer(&hasPeerAccess, i, j));
        if (!hasPeerAccess) {
            // Skip and continue if no peer access
            continue;
        }
        HIPCHECK(hipSetDevice(i));
        devpPtr = devPitchedPtrlist[j];
        HIPCHECK(hipMemset3D(devpPtr, 0, extent));

        if (bAsync) {
          hipStream_t stream;
          HIPCHECK(hipStreamCreate(&stream));
          HIPCHECK(hipMemset3DAsync(devpPtr, memsetval, extent, stream));
          HIPCHECK(hipStreamSynchronize(stream));
          HIPCHECK(hipStreamDestroy(stream));
        } else {
          HIPCHECK(hipMemset3D(devpPtr, memsetval, extent));
        }

        myparms.srcPtr = devpPtr;
        memset(A_h, 0, sizeElements);
        HIPCHECK(hipMemcpy3D(&myparms));

        for (int indx = 0; indx < elements; indx++) {
          if (A_h[indx] != memsetval) {
            testPassed = false;
            printf("RegressIter : mismatch at index:%d computed:%02x, "
            "memsetval:%02x\n", indx, static_cast<int>(A_h[indx]),
            static_cast<int>(memsetval));
            break;
          }
        }
      }
    }
  }

  for (int j = 0; j < numGpu; j++) {
    HIPCHECK(hipFree(devPitchedPtrlist[j].ptr));
  }

  free(A_h);
  return testPassed;
}


/**
 * Thread function queues kernel function and memset cmds
 */
void threadFunc(hipStream_t stream, hipPitchedPtr devpPtr, int memsetval,
                int testval, hipExtent extent, hipMemcpy3DParms myparms) {
  // Kernel Launch Configuration
  dim3 threadsPerBlock = dim3(8, 8, 8);
  dim3 blocks;
  blocks = dim3((extent.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (extent.height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                (extent.depth + threadsPerBlock.z - 1) / threadsPerBlock.z);

  hipLaunchKernelGGL(func_set_value, dim3(blocks), dim3(threadsPerBlock), 0,
                     stream, devpPtr, extent, memsetval);
  HIPCHECK(hipMemset3DAsync(devpPtr, testval, extent, stream));
  HIPCHECK(hipMemcpy3DAsync(&myparms, stream));
}

/**
 * Async commands queued concurrently and executed
 */
bool validateAsyncConcurrencyMthread() {
  bool testPassed = true;
  char *A_h;
  int memsetval = 1, numGpu = 0, testval = 2;
  size_t numH = 256, numW = 100, depth = 10;
  size_t width = numW * sizeof(char);
  hipExtent extent = make_hipExtent(width, numH, depth);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW* numH* depth;
  hipPitchedPtr devpPtr;
  hipStream_t stream;

  HIPCHECK(hipStreamCreate(&stream));
  HIPCHECK(hipMalloc3D(&devpPtr, extent));

  A_h = reinterpret_cast<char *>(malloc(sizeElements));
  HIPASSERT(A_h != NULL);
  memset(A_h, 0, sizeElements);

  // Populate hipMemcpy3D parameters
  hipMemcpy3DParms myparms = {0};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = devpPtr;
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.extent = extent;
#ifdef __HIP_PLATFORM_NVCC__
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif

  std::vector<std::thread> threadlist;

  // Queue cmds concurrently from multiple threads on same stream
  for (int i = 0; i < MAX_THREADS; i++) {
    threadlist.push_back(std::thread(threadFunc, stream, devpPtr, memsetval,
                                     testval, extent, myparms));
  }

  for (auto &t : threadlist) {
    t.join();
  }

  HIPCHECK(hipStreamSynchronize(stream));

  for (int k = 0 ; k < elements ; k++) {
    if (A_h[k] != testval) {
      printf("validateAsyncConcurrencyMthread: Test failed\n");
      testPassed = false;
      break;
    }
  }

  HIPCHECK(hipStreamDestroy(stream));
  free(A_h);
  HIPCHECK(hipFree(devpPtr.ptr));
  return testPassed;
}


int main(int argc, char *argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  bool TestPassed = true;

  if (p_tests == 1) {
    TestPassed = validateAsyncConcurrencyMthread();
  } else if (p_tests == 2) {
    /* TODO : Loop regression test auto execution in HIT is currently disabled.
       To be enabled back after HIP API fix */
    TestPassed &= loopRegression(0);
    TestPassed &= loopRegression(1);
  } else {
    printf("Didnt receive any valid option. Try options 1 to 2\n");
    TestPassed = false;
  }

  if (TestPassed) {
    passed();
  } else {
    failed("hipMemset3DRegressMultiThread() validation Failed!");
  }
}
