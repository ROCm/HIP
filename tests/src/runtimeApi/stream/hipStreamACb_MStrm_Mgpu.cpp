/*
  Copyright (c) 2019-present Advanced Micro Devices, Inc. All rights reserved.
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

// Testcase Description: Streams are launched in individual GPUs with different
// kernel. Verify that all the kernels queued are executed before the callback.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11 EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include <stdio.h>
#include <unistd.h>
#include <thread>
#include <chrono>
#include "hip/hip_runtime.h"
#include "test_common.h"

#ifdef __HIP_PLATFORM_HCC__
#define HIPRT_CB
#endif


size_t N_ELMTS = 4096;

// Data structure for holding and validating data
struct gpu_data {
  int *int_ptr = NULL;
  int gpu;
  int acknowledge;
};

enum {
  SUCCESS = 0,
  KERNEL_EXECUTION_MISMATCH,
  KERNEL_COMPUTATION_MISMATCH
};

__global__ void Add_Data(int* A_d, size_t N_ELMTS) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = offset; i < N_ELMTS; i += stride) {
    // Increment the value of A_d[i] by 1
    A_d[i] = A_d[i] + 1;
  }
}

// below kernel is just to load the gpu with multiple jobs
__global__ void Square_plus_one(int* A_d, int* C_d, size_t N_ELMTS) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = offset; i < N_ELMTS; i += stride) {
    C_d[i] = A_d[i]*A_d[i] + 1;
  }
}

static void HIPRT_CB Stream_Callback(hipStream_t stream, hipError_t status,
                                     void* userData) {
  gpu_data *ptr = reinterpret_cast<gpu_data *>(userData);

  // int_ptr in the passed userData will contain the data copied from device to
  // host.  Expected data in this field is the gpu ordinal.
  if (*((*ptr).int_ptr) != (*ptr).gpu + 1) {
    (*ptr).acknowledge = 100;   // Assign unexpected value to indicate fail
  } else {
    (*ptr).acknowledge = (*ptr).gpu;  // Assign the gpu ordinal received
  }
}

void launch_gpu(int gpu_ordinal) {
  HIPCHECK(hipSetDevice(gpu_ordinal));
  int *A_d, *A_h, *C_h, *C_d;
  size_t Nbytes = N_ELMTS * sizeof(int), Data_mismatch = 0;
  bool cb = false;
  A_h = (int *)malloc(Nbytes);
  HIPCHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  C_h = (int *)malloc(Nbytes);
  HIPCHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);

  // Fill with 0
  for (size_t i = 0; i < N_ELMTS; i++) {
    A_h[i] = 0;
  }

  // setting gpu value in the struct object
  gpu_data *ptr = new gpu_data;
  ptr->int_ptr = C_h;
  ptr->gpu = gpu_ordinal;
  ptr->acknowledge = 100;

  HIPCHECK(hipMalloc(&A_d, Nbytes));
  HIPCHECK(hipMalloc(&C_d, Nbytes));

  hipStream_t mystream;
  HIPCHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));

  HIPCHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mystream));

  const unsigned threadsPerBlock = 256;
  const unsigned blocks = (N_ELMTS + 255)/threadsPerBlock;

  // A_d is initialized to 0.  Add_Data kernel does A_d = A_d + 1
  // The Add_data kernel is called 1 time for gpu0, 2 times for gpu1 etc.
  // At the end of the loop, A_d should have the gpu_ordinal number
  for (int i = 0; i < gpu_ordinal + 1; i++) {
    hipLaunchKernelGGL(Add_Data, dim3(blocks), dim3(threadsPerBlock), 0,
                       mystream, A_d, N_ELMTS);
    hipLaunchKernelGGL(Square_plus_one, 1, 1, 0, mystream, A_d, C_d, N_ELMTS);
  }
  HIPCHECK(hipMemcpyAsync(C_h, A_d, Nbytes, hipMemcpyDeviceToHost, mystream));

  // Pass the ptr as user data which contains the gpu_ordinal, default value
  // for ack and the data that is copied to host
  HIPCHECK(hipStreamAddCallback(mystream, Stream_Callback,
                                reinterpret_cast<void *>(ptr), 0));
  HIPCHECK(hipStreamSynchronize(mystream));

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(C_d));
  HIPCHECK(hipStreamDestroy(mystream));

  int result = SUCCESS;
  if (C_h[0] != gpu_ordinal + 1) {
    result = KERNEL_EXECUTION_MISMATCH;
  }

  if (ptr->gpu != ptr->acknowledge) {
    result = KERNEL_COMPUTATION_MISMATCH;
  }

  free(A_h);
  free(C_h);
  free(ptr);

  if (result == KERNEL_EXECUTION_MISMATCH) {
    failed("Number of kernels expected to be executed does not match");
  } else if (result == KERNEL_COMPUTATION_MISMATCH) {
    failed("Mismatch found in the result of the computation!");
  }
}


int main() {
  int gpu_cnt = 0;

  HIPCHECK(hipGetDeviceCount(&gpu_cnt));
  if (gpu_cnt < 2) {
    printf("Minimum of 2 gpus are needed for this test, skipping the test\n");
    passed();
  }

  std::thread T[gpu_cnt];

  // Launching threads for each GPU
  for (int i = 0; i < gpu_cnt; i++) {
    T[i] = std::thread(launch_gpu, i);
  }

  for (int i=0; i < gpu_cnt; i++) {
    T[i].join();
  }
  passed();
}
