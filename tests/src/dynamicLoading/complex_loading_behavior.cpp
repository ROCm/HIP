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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* Test for loading device kernels from a library created with extern "C" function
 */

/* HIT_START
 * BUILD_CMD: libLazyLoad_amd %hc %S/%s -o liblazyLoad.so -I%S/.. -fPIC -lpthread -shared -DTEST_SHARED_LIBRARY EXCLUDE_HIP_PLATFORM nvidia EXCLUDE_HIP_LIB_TYPE static
 * BUILD_CMD: libLazyLoad_nvidia %hc %S/%s --std=c++11 -o liblazyLoad.so -I%S/.. -Xcompiler -fPIC -lpthread -shared -DTEST_SHARED_LIBRARY EXCLUDE_HIP_PLATFORM amd
 * BUILD_CMD: %t %hc %S/%s --std=c++11 -o %T/%t -I%S/.. -ldl EXCLUDE_HIP_LIB_TYPE static
 * TEST: %t
 * HIT_END
 */

#if !defined(TEST_SHARED_LIBRARY)

#include <dlfcn.h>
#include <iostream>
#include "test_common.h"

__global__ void vector_add(float* C, float* A, float* B, size_t N) {
  size_t offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  size_t stride = hipBlockDim_x * hipGridDim_x;
  for (size_t i = offset; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

bool launch_local_kernel() {
  bool testResult = true;
  float *A_d, *B_d, *C_d;
  float *A_h, *B_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(float);
  static int device = 0;

  HIPCHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, device /*deviceID*/));

  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(A_h == nullptr ? hipErrorOutOfMemory : hipSuccess);
  B_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(B_h == nullptr ? hipErrorOutOfMemory : hipSuccess);
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(C_h == nullptr ? hipErrorOutOfMemory : hipSuccess);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1.618f + i;
    B_h[i] = 1.618f + i;
  }

  HIPCHECK(hipMalloc(&A_d, Nbytes));
  HIPCHECK(hipMalloc(&B_d, Nbytes));
  HIPCHECK(hipMalloc(&C_d, Nbytes));
  HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  const unsigned blocks = 512;
  const unsigned threadsPerBlock = 256;
  hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threadsPerBlock),
                     0, 0, C_d, A_d, B_d, N);
  HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  for (size_t i=0; i < N ; i++) {
    if (C_h[i] != (A_h[i] + B_h[i])) {
      printf("data mismatch. Local kernel failed");
      testResult = false;
      break;
    }
  }

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(B_d));
  HIPCHECK(hipFree(C_d));

  free(A_h);
  free(B_h);
  free(C_h);

  std::cout << "Local kernel executed successfully\n";
  return testResult;
}

bool launch_dynamically_loaded_kernel() {
  bool testResult = true;
  int ret = 1;

  void* handle = dlopen("./liblazyLoad.so", RTLD_LAZY);

  if (!handle) {
    std::cout << dlerror() << "\n";
    testResult = false;
    return testResult;
  }

  std::cout << "loaded liblazyLoad.so\n";

  void* sym = dlsym(handle, "lazyLoad");
  if (!sym) {
    std::cout << "unable to locate lazyLoad within lazyLoad.so\n";
    std::cout << dlerror() << "\n";
    dlclose(handle);
    testResult = false;
    return testResult;
  }

  int(*fp)() = reinterpret_cast<int(*)()>(sym);

  ret = fp();

  if (ret == 0) {
    std::cout << "dynamic launch failed\n";
    testResult = false;
  } else {
    std::cout << "dynamic launch succeeded\n";
  }

  dlclose(handle);
  return testResult;
}

int main() {
  bool testResult = true;

  testResult &= launch_local_kernel();
  testResult &= launch_dynamically_loaded_kernel();

  if (testResult == true) {
    passed();
  } else {
    failed("One or more tests failed");
  }
}

#else   // !defined(TEST_SHARED_LIBRARY)

#include <iostream>
#include "test_common.h"

__global__ void vAdd(float* C, float* A, float* B, size_t N) {
  size_t offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  size_t stride = hipBlockDim_x * hipGridDim_x;

  for (size_t i = offset; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

int vectorAddKernelTest() {
  int testResult = 1;
  float *A_d, *B_d, *C_d;
  float *A_h, *B_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(float);
  static int device = 0;

  HIPCHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(A_h == nullptr ? hipErrorOutOfMemory : hipSuccess);
  B_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(B_h == nullptr ? hipErrorOutOfMemory : hipSuccess);
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(C_h == nullptr ? hipErrorOutOfMemory : hipSuccess);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1.618f + i;
    B_h[i] = 1.618f + i;
  }

  HIPCHECK(hipMalloc(&A_d, Nbytes));
  HIPCHECK(hipMalloc(&B_d, Nbytes));
  HIPCHECK(hipMalloc(&C_d, Nbytes));
  HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  const unsigned blocks = 512;
  const unsigned threadsPerBlock = 256;

  std::cout << "info: Launching vAdd kernel\n";
  hipLaunchKernelGGL(vAdd, dim3(blocks), dim3(threadsPerBlock),
                     0, 0, C_d, A_d, B_d, N);
  HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  for (size_t i=0; i < N ; i++) {
    if (C_h[i] != (A_h[i] + B_h[i])) {
      printf("info: data mismatch. vAdd kernel failed");
      testResult = 0;
      break;
    }
  }

  if (testResult) {
    std::cout << "info: vAdd kernel executed fine\n";
  }

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(B_d));
  HIPCHECK(hipFree(C_d));

  free(A_h);
  free(B_h);
  free(C_h);
  return testResult;
}

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip/hip_cooperative_groups.h"

namespace cg = cooperative_groups;

static const uint BufferSizeInDwords = 448 * 1024 * 1024;

__global__ void test_gws(uint* buf, uint bufSize,
                         long* tmpBuf, long* result) {
  extern __shared__ long tmp[];
  uint offset = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = gridDim.x  * blockDim.x;
  cg::grid_group gg = cg::this_grid();

  long sum = 0;

  for (uint i = offset; i < bufSize; i += stride) {
      sum += buf[i];
  }

  tmp[threadIdx.x] = sum;
  __syncthreads();

  if (threadIdx.x == 0) {
    sum = 0;
    for (uint i = 0; i < blockDim.x; i++) {
      sum += tmp[i];
    }
    tmpBuf[blockIdx.x] = sum;
  }

  gg.sync();

  if (offset == 0) {
    for (uint i = 1; i < gridDim.x; ++i) {
      sum += tmpBuf[i];
    }
    *result = sum;
  }
}

int cooperativeKernelTest() {
  int testResult = 1;
  uint* dA;
  long* dB;
  long* dC;
  long* Ah;

  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);

  if (!deviceProp.cooperativeLaunch) {
    std::cout << "info: Device doesn't support cooperative launch!"
                 "skipping the test!\n";
    return testResult;
  }

  uint32_t* init = new uint32_t[BufferSizeInDwords];

  for (uint32_t i = 0; i < BufferSizeInDwords; ++i) {
    init[i] = i;
  }

  std::cout << "info: Launch kernel to test hipLaunchCooperativeKernel api\n";
  std::cout << "info: running on bus 0x" << deviceProp.pciBusID << " " <<
               deviceProp.name << "\n";

  size_t SIZE = BufferSizeInDwords * sizeof(uint);

  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dA), SIZE));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dC), sizeof(long)));
  HIPCHECK(hipMemcpy(dA, init, SIZE, hipMemcpyHostToDevice));
  Ah = reinterpret_cast<long*>(malloc(sizeof(long)));

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  dim3 dimBlock = dim3(1);
  dim3 dimGrid  = dim3(1);

  int numBlocks = 0;
  uint workgroups[4] = {32, 64, 128, 256};

  for (uint i = 0; i < 4; ++i) {
    dimBlock.x = workgroups[i];
    /* Calculate the device occupancy to know how many blocks can be
       run concurrently */
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
    test_gws, dimBlock.x * dimBlock.y * dimBlock.z, dimBlock.x * sizeof(long));
    dimGrid.x = deviceProp.multiProcessorCount * std::min(numBlocks, 32);
    HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dB),
                       dimGrid.x * sizeof(long)));

    void *params[4];
    params[0] = reinterpret_cast<void*>(&dA);
    params[1] = (void*)&BufferSizeInDwords;
    params[2] = reinterpret_cast<void*>(&dB);
    params[3] = reinterpret_cast<void*>(&dC);

    std::cout << "Testing with grid size = " << dimGrid.x <<
                 " and block size = " << dimBlock.x << "\n";

    HIPCHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_gws),
                                        dimGrid, dimBlock, params,
                                        dimBlock.x * sizeof(long), stream));

    HIPCHECK(hipMemcpy(Ah, dC, sizeof(long), hipMemcpyDeviceToHost));

    if (*Ah != (((long)(BufferSizeInDwords) * (BufferSizeInDwords - 1)) / 2)) {
      std::cout << "Data validation failed for grid size = " << dimGrid.x <<
                   " and block size = " << dimBlock.x << "\n";
      HIPCHECK(hipFree(dB));
      std::cout << "Test failed! \n";
      testResult = 0;
      break;

    } else {
      std::cout << "info: data validated!\n";
      HIPCHECK(hipFree(dB));
    }
  }

  if (testResult) {
    std::cout <<"info: hipLaunchCooperativeKernel api executed fine\n";
  }

  HIPCHECK(hipStreamDestroy(stream));
  HIPCHECK(hipFree(dC));
  HIPCHECK(hipFree(dA));
  delete [] init;
  free(Ah);
  return testResult;
}

extern "C" int lazyLoad() {
  return vectorAddKernelTest() & cooperativeKernelTest();
}

#endif   // !defined(TEST_SHARED_LIBRARY)
