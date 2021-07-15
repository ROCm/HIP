/*
Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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
// Simple test for Fine Grained CPU-GPU coherency.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp HIPCC_OPTIONS -std=c++11 -lpthread EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"
#include <stdio.h>
#include <thread>

typedef _Atomic(unsigned int) atomic_uint;

// Helper function to spin on address until address equals value.
// If the address holds the value of -1, abort because the other thread failed.
__device__ void
gpu_spin_loop_or_abort_on_negative_one(unsigned int* address,
                                       unsigned int value) {
  unsigned int compare;
  bool check = false;
  do {
    compare = value;
    check = __opencl_atomic_compare_exchange_strong(
      (atomic_uint*)address, /*expected=*/ &compare, /*desired=*/ value,
      __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
      /*scope=*/ __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
    if (compare == -1)
      abort();
  } while(!check);
}

// This kernel requires a single block, single thread dispatch.
__global__ void
gpu_kernel(int *A, int *B, int *X, int *Y, size_t N,
           unsigned int *AA1, unsigned int *AA2,
           unsigned int *BA1, unsigned int *BA2) {
  for (size_t i = 0; i < N; i++) {
    // Store data into A, system fence, and atomically mark flag.
    // This guarantees this global write is visible by device 1.
    A[i] = X[i];
    __opencl_atomic_fetch_add((atomic_uint*)AA1, 1, __ATOMIC_RELEASE,
                              __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
    // Wait on device 1's global write to B.
    gpu_spin_loop_or_abort_on_negative_one(BA1, i+1);

    // Check device 1 properly stored Y into B.
    bool stored_data_matches = (B[i] == Y[i]);
    if(!stored_data_matches) {
      // If the data does not match, alert other thread and abort.
      printf("FAIL: at i=%lu, B[i]=%d, which does not match Y[i]=%d.\n",
             i, B[i], Y[i]);
      __opencl_atomic_exchange((atomic_uint*)AA2, -1, __ATOMIC_RELEASE,
                               __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
      abort();
    }
    // Otherwise tell the other thread to continue.
    __opencl_atomic_fetch_add((atomic_uint*)AA2, 1, __ATOMIC_RELEASE,
                              __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
    // Wait on kernel gpu_cache1 to finish checking X is stored in A.
    gpu_spin_loop_or_abort_on_negative_one(BA2, i+1);
  }
}

__host__ void
cpu_spin_loop_or_abort_on_negative_one(unsigned int* address,
                                       unsigned int value) {
  unsigned int compare;
  bool check = false;
  do {
    compare = value;
    check = __atomic_compare_exchange_n(
      address, /*expected=*/ &compare, /*desired=*/ value,
      /*weak=*/ false, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE);
    if (compare == -1)
      abort();
  } while(!check);
}

// This host thread runs only on a single CPU thread.
__host__ void
cpu_thread(int *A, int *B, int *X, int *Y, size_t N,
           unsigned int *AA1, unsigned int *AA2,
           unsigned int *BA1, unsigned int *BA2) {
  for (size_t i = 0; i < N; i++) {
    B[i] = Y[i];
    __atomic_fetch_add(BA1, 1, __ATOMIC_RELEASE);
    cpu_spin_loop_or_abort_on_negative_one(AA1, i+1);

    bool stored_data_matches = (A[i] == X[i]);
    if(!stored_data_matches) {
      printf("FAIL: at i=%lu, A[i]=%d, which does not match X[i]=%d.\n",
             i, A[i], X[i]);
      __atomic_exchange_n(BA2, -1, __ATOMIC_RELEASE);
      abort();
    }
    __atomic_fetch_add(BA2, 1, __ATOMIC_RELEASE);
    cpu_spin_loop_or_abort_on_negative_one(AA2, i+1);
  }
}

// This test runs on devices where XGMI enables fine-grained communication
// between GPUs. This performs a message passing test.
// Array A is allocated on Device 0, and remotely on host.
// Device 0 also increments atomic ints AA1 and AA2.
// Array B is allocated on host, and remotely on Device 0.
// Host also increments atomic ints BA1 and BA2.
// Kernel will launch on Device 0, and store array X into array A.
// Host Thread will store array Y into array B.
// Kernel will validate that the correct values of array Y are stored in B.
// Host Thread will validate that the correct values of array X are stored in A.

bool cpu_to_gpu_coherency() {
  int *A_d, *B_d, *X_d, *Y_d;
  int *A_res, *A_h, *B_h, *X_h, *Y_h;
  size_t N = 1024;
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;

  HIPCHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 1) {
    printf("info: no GPU detected.\n");
    return false;
  }

  // Skip this test if feature is not supported.
  static int device0 = 0;
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, device0));
  if (strncmp(props.gcnArchName, "gfx90a", 6) != 0 &&
      strncmp(props.gcnArchName, "gfx940", 6) != 0) {
    printf("info: skipping test on devices other than gfx90a and gfx940.\n");
    return true;
  }

  // Allocate Host Side Memory. Coherent Fine-grained Memory for array B.
  printf("info: allocate host mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
  HIPCHECK(hipHostMalloc(&B_h, Nbytes, (hipHostMallocCoherent | hipHostMallocMapped)));
  HIPCHECK(hipHostGetDevicePointer((void**)&B_d, B_h, 0));
  X_h = (int*)malloc(Nbytes); HIPCHECK(X_h == 0 ? hipErrorOutOfMemory : hipSuccess );
  Y_h = (int*)malloc(Nbytes); HIPCHECK(Y_h == 0 ? hipErrorOutOfMemory : hipSuccess );

  // Initialize the arrays and atomic variables.
  for (size_t i = 0; i < N; i++)
  {
    X_h[i] = 100000000 + i;
    Y_h[i] = 300000000 + i;
  }

  // Initialize shared atomic flags between CPU and GPU.
  unsigned int *AA1_h, *AA2_h, *BA1_h, *BA2_h;
  unsigned int *AA1_d, *AA2_d, *BA1_d, *BA2_d;
  HIPCHECK(hipHostMalloc(&AA1_h, sizeof(unsigned int), hipHostMallocCoherent));
  HIPCHECK(hipHostGetDevicePointer((void**)&AA1_d, AA1_h, 0)); *AA1_h = 0;
  HIPCHECK(hipHostMalloc(&AA2_h, sizeof(unsigned int), hipHostMallocCoherent));
  HIPCHECK(hipHostGetDevicePointer((void**)&AA2_d, AA2_h, 0)); *AA2_h = 0;
  HIPCHECK(hipHostMalloc(&BA1_h, sizeof(unsigned int), hipHostMallocCoherent));
  HIPCHECK(hipHostGetDevicePointer((void**)&BA1_d, BA1_h, 0)); *BA1_h = 0;
  HIPCHECK(hipHostMalloc(&BA2_h, sizeof(unsigned int), hipHostMallocCoherent));
  HIPCHECK(hipHostGetDevicePointer((void**)&BA2_d, BA2_h, 0)); *BA2_h = 0;

  // Skip the first stream, ensure stream is non-blocking.
  hipStream_t stream[2];
  HIPCHECK(hipStreamCreate(&stream[0]));
  HIPCHECK(hipSetDevice(0));
  HIPCHECK(hipStreamCreateWithFlags(&stream[1], hipStreamNonBlocking));

  // Allocate Device Side Memory. Coherent Fine-grained Memory for array A.
  printf("info: allocate device 0 mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
  hipError_t status = hipExtMallocWithFlags((void**)&A_d, Nbytes, hipDeviceMallocFinegrained);
  if (status == hipErrorOutOfMemory || A_d == 0 ){
    printf("info: device fine-grained memory not supported on this config\n");
    printf("info: skipping this CPU-GPU coherency test\n");
    return true;
  } else if (status != hipSuccess) {
    printf("error: failed to allocate device fine-grain memory\n");
    return false;
  }
  // SVM memory - host pointer is the same as device pointer to array A.
  A_h = A_d;
  HIPCHECK(hipMalloc(&X_d, Nbytes));
  HIPCHECK(hipMalloc(&Y_d, Nbytes));

  printf("info: copy Host2Device\n");
  HIPCHECK(hipMemcpy(X_d, X_h, Nbytes, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(Y_d, Y_h, Nbytes, hipMemcpyHostToDevice));

  // Launch the GPU kernel.
  printf("info: prepare args and launch params for device 0\n");
  const unsigned blocks = 1;
  const unsigned threadsPerBlock = 1;
  printf("info: launch gpu_kernel\n");
  hipLaunchKernelGGL(gpu_kernel, dim3(blocks), dim3(threadsPerBlock),
                     0, stream[1],
                     A_d, B_d, X_d, Y_d, N,
                     AA1_d, AA2_d, BA1_d, BA2_d);
  // Check if launch failed.
  HIPCHECK(hipGetLastError());

  // Do not sync the launched stream, instead run the cpu_thread.
  printf("info: start cpu_thread\n");
  std::thread host_thread(cpu_thread,
                          A_h, B_h, X_h, Y_h, N,
                          AA1_h, AA2_h, BA1_h, BA2_h);
  host_thread.detach();

  // Wait for Device side to finish.
  HIPCHECK(hipStreamSynchronize(stream[1]));

  // Evaluate the resultant arrays A and B.
  printf("info: copy Device2Host\n");
  A_res = (int*)malloc(Nbytes); HIPCHECK(A_res == 0 ? hipErrorOutOfMemory : hipSuccess );
  HIPCHECK(hipMemcpy(A_res, A_d, Nbytes, hipMemcpyDeviceToHost));
  printf("info: check result\n");
  for (size_t i = 0; i < N; i++)  {
    assert(A_res[i] == (100000000 + i));
    assert(B_h[i] == (300000000 + i));
  }

  // Free all the device and host memory allocated.
  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(X_d));
  HIPCHECK(hipFree(Y_d));
  HIPCHECK(hipHostFree(AA1_h));
  HIPCHECK(hipHostFree(AA2_h));
  HIPCHECK(hipHostFree(BA1_h));
  HIPCHECK(hipHostFree(BA2_h));
  HIPCHECK(hipHostFree(B_h));
  free(X_h);
  free(Y_h);
  free(A_res);

  printf("info: finished CPU-GPU test!\n");
  return true;
}

int main(int argc, char *argv[]) {
  bool passed = true;

  // Coherency between CPU and GPU sharing host and device memory.
  passed = passed & cpu_to_gpu_coherency();

  if (passed)
    passed();
  return passed;
}
