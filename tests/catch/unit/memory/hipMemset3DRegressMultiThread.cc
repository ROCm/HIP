/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
 1) Validate Async behavior of hipMemset3DAsync with commands queued
 concurrently from multiple threads.
 2) Validate hipMemset3DAsync behavior when api is queued along with kernel
 function operating on same memory.
 3) Perform regression of hipMemset3D api in loop with device memory allocated
 on different gpus.
 4) Perform regression of hipMemset3DAsync api in loop with device memory
 allocated on different gpus.
*/

#include <hip_test_common.hh>


/*
 * Defines
 */
#define MAX_REGRESS_ITERS 2
#define MAX_THREADS 10

/**
 * kernel function sets device memory with value passed
 */
static __global__ void func_set_value(hipPitchedPtr devicePitchedPointer,
                               hipExtent extent,
                               unsigned char val) {
  // Index Calculation
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t z = threadIdx.z + blockDim.z * blockIdx.z;

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
 * Thread function queues kernel function and memset cmds
 */
static void threadFunc(hipStream_t stream, hipPitchedPtr devpPtr,
      int memsetval, int testval, hipExtent extent, hipMemcpy3DParms myparms) {
  // Kernel Launch Configuration
  constexpr auto size = 8;
  dim3 threadsPerBlock = dim3(size, size, size);
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
 * Performs api regression in loop
 */
bool loopRegression(bool bAsync) {
  bool testPassed = true;
  char *A_h;
  constexpr int memsetval = 1;
  constexpr size_t numH = 256, numW = 100, depth = 10;
  int numGpu = 0, hasPeerAccess = 0;
  size_t width = numW * sizeof(char);
  hipExtent extent = make_hipExtent(width, numH, depth);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW* numH* depth;
  std::vector<hipPitchedPtr> devPitchedPtrlist;
  hipPitchedPtr pitchedPtr, devpPtr;

  A_h = reinterpret_cast<char *>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);
  memset(A_h, 0, sizeElements);

  // Populate hipMemcpy3D parameters
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.extent = extent;

#if HT_NVIDIA
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif

  HIP_CHECK(hipGetDeviceCount(&numGpu));
  REQUIRE(numGpu > 0);

  // Alloc 3D arrays in all GPUs
  for (int j = 0; j < numGpu; j++) {
    HIP_CHECK(hipSetDevice(j));
    HIP_CHECK(hipMalloc3D(&pitchedPtr, extent));
    devPitchedPtrlist.push_back(pitchedPtr);
  }

  for (int itern = 0; itern < MAX_REGRESS_ITERS; itern++) {
    // Validate hipMemset3D data consistency in multiple iters
    for (int i = 0; i < numGpu; i++) {
      for (int j = 0; j < numGpu; j++) {
        HIP_CHECK(hipDeviceCanAccessPeer(&hasPeerAccess, i, j));
        if (!hasPeerAccess) {
            // Skip and continue if no peer access
            continue;
        }

        HIP_CHECK(hipSetDevice(i));
        devpPtr = devPitchedPtrlist[j];
        HIP_CHECK(hipDeviceEnablePeerAccess(j, 0));
        HIP_CHECK(hipMemset3D(devpPtr, 0, extent));

        if (bAsync) {
          hipStream_t stream;
          HIP_CHECK(hipStreamCreate(&stream));
          HIP_CHECK(hipMemset3DAsync(devpPtr, memsetval, extent, stream));
          HIP_CHECK(hipStreamSynchronize(stream));
          HIP_CHECK(hipStreamDestroy(stream));
        } else {
          HIP_CHECK(hipMemset3D(devpPtr, memsetval, extent));
        }

        myparms.srcPtr = devpPtr;
        memset(A_h, 0, sizeElements);
        HIP_CHECK(hipMemcpy3D(&myparms));

        for (size_t indx = 0; indx < elements; indx++) {
          if (A_h[indx] != memsetval) {
            testPassed = false;
            printf("RegressIter : mismatch at index:%d computed:%02x, "
            "memsetval:%02x\n", static_cast<int>(indx),
            static_cast<int>(A_h[indx]), static_cast<int>(memsetval));
            break;
          }
        }
      }
    }
  }

  for (int j = 0; j < numGpu; j++) {
    HIP_CHECK(hipFree(devPitchedPtrlist[j].ptr));
  }

  free(A_h);
  return testPassed;
}

/**
 * Perform regression of hipMemset3D api with device memory allocated
 * on different gpus.
 */
TEST_CASE("Unit_hipMemset3D_RegressInLoop") {
  bool TestPassed = false;

  TestPassed = loopRegression(0);
  REQUIRE(TestPassed == true);
}

/**
 * Perform regression of hipMemset3DAsync api with device memory allocated
 * on different gpus.
 */
TEST_CASE("Unit_hipMemset3DAsync_RegressInLoop") {
  bool TestPassed = false;

  TestPassed = loopRegression(1);
  REQUIRE(TestPassed == true);
}

/**
 * Async commands queued concurrently and executed
 */
TEST_CASE("Unit_hipMemset3DAsync_ConcurrencyMthread") {
  char *A_h;
  constexpr int memsetval = 1, testval = 2;
  constexpr size_t numH = 256, numW = 100, depth = 10;
  size_t width = numW * sizeof(char);
  hipExtent extent = make_hipExtent(width, numH, depth);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW* numH* depth;
  hipPitchedPtr devpPtr;
  hipStream_t stream;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc3D(&devpPtr, extent));

  A_h = reinterpret_cast<char *>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);
  memset(A_h, 0, sizeElements);

  // Populate hipMemcpy3D parameters
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = devpPtr;
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.extent = extent;

#if HT_NVIDIA
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

  HIP_CHECK(hipStreamSynchronize(stream));

  for (size_t k = 0 ; k < elements ; k++) {
    if (A_h[k] != testval) {
      CAPTURE(A_h[k], testval, k);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  free(A_h);
  HIP_CHECK(hipFree(devpPtr.ptr));
}
