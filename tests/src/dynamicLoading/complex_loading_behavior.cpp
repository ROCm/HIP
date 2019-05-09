/*
Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

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

/* HIT_START
 * BUILD_CMD: libfoo %hc %S/%s -o libfoo.so -fPIC -lpthread -shared -DTEST_SHARED_LIBRARY
 * BUILD_CMD: %t %hc %S/%s -o %T/%t -ldl
 * TEST: %t
 * HIT_END
 */

#if !defined(TEST_SHARED_LIBRARY)

#include <dlfcn.h>
#include <iostream>
#include <hip/hip_runtime.h>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            return (EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

__global__ void vector_add(float* C, float* A, float* B, size_t N) {
  size_t offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  size_t stride = hipBlockDim_x * hipGridDim_x;
  for (size_t i = offset; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

int launch_local_kernel() {
    float *A_d, *B_d, *C_d;
    float *A_h, *B_h, *C_h;
    size_t N = 1000000;
    size_t Nbytes = N * sizeof(float);
    static int device = 0;
    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    B_h = (float*)malloc(Nbytes);
    CHECK(B_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    C_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    // Fill with Phi + i
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.618f + i;
        B_h[i] = 1.618f + i;
    }

    CHECK(hipMalloc(&A_d, Nbytes));
    CHECK(hipMalloc(&B_d, Nbytes));
    CHECK(hipMalloc(&C_d, Nbytes));
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, B_d, N);
    CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    CHECK(hipFree(A_d));
    CHECK(hipFree(B_d));
    CHECK(hipFree(C_d));

    free(A_h);
    free(B_h);
    free(C_h);

    std::cout << "PASSED!\n";
    return 0;
}

int launch_dynamically_loaded_kernel() {
  void* handle = dlopen("./libfoo.so", RTLD_LAZY);
  if (!handle) {
    std::cout << dlerror() << "\n";
    return -1;
  }
  std::cout << "loaded libfoo.so\n";

  void* sym = dlsym(handle, "foo");
  if (!sym) {
    std::cout << "unable to locate foo within libfoo.so\n";
    std::cout << dlerror() << "\n";
    dlclose(handle);
    return -1;
  }

  int(*fp)() = reinterpret_cast<int(*)()>(sym);

  int ret = fp();
  if (ret) {
    std::cout << "dynamic launch failed\n";
  } else {
    std::cout << "dynamic launch succeeded\n";
  }

  dlclose(handle);
  return ret;
}

int main() {
  int ret = 0;
  ret = launch_local_kernel();
  if (ret) {
    return ret;
  }

  ret = launch_dynamically_loaded_kernel();
  if (ret) {
    return ret;
  }

  return 0;
}

#else // !defined(TEST_SHARED_LIBRARY)

#include <dlfcn.h>
#include <iostream>
#include <hip/hip_runtime.h>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            return (EXIT_FAILURE);                                                                 \
        }                                                                                          \
    }

__global__ void vadd(float* C, float* A, float* B, size_t N) {
  size_t offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  size_t stride = hipBlockDim_x * hipGridDim_x;
  for (size_t i = offset; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

extern "C" int foo() {
    float *A_d, *B_d, *C_d;
    float *A_h, *B_h, *C_h;
    size_t N = 1000000;
    size_t Nbytes = N * sizeof(float);
    static int device = 0;
    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    B_h = (float*)malloc(Nbytes);
    CHECK(B_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    C_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    // Fill with Phi + i
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.618f + i;
        B_h[i] = 1.618f + i;
    }

    CHECK(hipMalloc(&A_d, Nbytes));
    CHECK(hipMalloc(&B_d, Nbytes));
    CHECK(hipMalloc(&C_d, Nbytes));
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    std::cout << "Launch vadd\n";
    hipLaunchKernelGGL(vadd, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, B_d, N);
    CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    CHECK(hipFree(A_d));
    CHECK(hipFree(B_d));
    CHECK(hipFree(C_d));

    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}

#endif // !defined(TEST_SHARED_LIBRARY)
