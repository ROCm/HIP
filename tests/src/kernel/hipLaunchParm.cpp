/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

static const int  STRUCT_SIZE = 1024;

// This test is to verify Struct with variables to check the hipLaunchKernel() support, read and write into the same struct
typedef struct hipLaunchKernelStruct1 {
  int li;  // local int
  float lf;  // local float
  bool result;  // default is false, will be set to true if the condition is met
} hipLaunchKernelStruct_t1;

// This test is to verify struct with padding, read and write into the same struct
typedef struct hipLaunchKernelStruct2 {
  char c1;  // local char
  long l1;  // local long
  char c2;  // local char
  long l2;  // local long
  bool result;  // default is false, will be set to true if the condition is met
} hipLaunchKernelStruct_t2;

typedef struct hipLaunchKernelStruct3 {
  char bf1;
  char bf2;
  long l1;
  char bf3;
  bool result;  // default is false, will be set to true if the condition is met
} hipLaunchKernelStruct_t3;


// Passing struct to a hipLaunchKernel(), read and write into the same struct
__global__ void hipLaunchKernelStructFunc1(hipLaunchParm lp, hipLaunchKernelStruct_t1* hipLaunchKernelStruct_) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    hipLaunchKernelStruct_[x].result =  ((hipLaunchKernelStruct_[x].li == 1) && (hipLaunchKernelStruct_[x].lf == 1.0)) ? true : false;
}

// Passing struct to a hipLaunchKernel(), checks padding, read and write into the same struct
__global__ void hipLaunchKernelStructFunc2(hipLaunchParm lp, hipLaunchKernelStruct_t2* hipLaunchKernelStruct_) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    hipLaunchKernelStruct_[x].result =  ((hipLaunchKernelStruct_[x].c1 == 'a') && (hipLaunchKernelStruct_[x].l1 == 1.0)
                                          && (hipLaunchKernelStruct_[x].c2 == 'b') && (hipLaunchKernelStruct_[x].l2 == 2.0) ) ? true : false;
}

// Passing struct to a hipLaunchKernel(), checks padding, read and write into the same struct
__global__ void hipLaunchKernelStructFunc3(hipLaunchParm lp, hipLaunchKernelStruct_t3* hipLaunchKernelStruct_) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    hipLaunchKernelStruct_[x].result =  ((hipLaunchKernelStruct_[x].bf1 == 1) && (hipLaunchKernelStruct_[x].bf2 == 1)
                                           && (hipLaunchKernelStruct_[x].l1 == 1.0) && (hipLaunchKernelStruct_[x].bf3 == 1) ) ? true : false;
}

__global__ void vAdd(hipLaunchParm lp, float* a) {}

//---
// Some wrapper macro for testing:
#define WRAP(...) __VA_ARGS__

#include <sys/time.h>
#define GPU_PRINT_TIME(cmd, elapsed, quiet)                                                        \
    do {                                                                                           \
        struct timeval start, stop;                                                                \
        float elapsed;                                                                             \
        gettimeofday(&start, NULL);                                                                \
        hipDeviceSynchronize();                                                                    \
        cmd;                                                                                       \
        hipDeviceSynchronize();                                                                    \
        gettimeofday(&stop, NULL);                                                                 \
    } while (0);


#define MY_LAUNCH(command, doTrace, msg)                                                           \
    {                                                                                              \
        if (doTrace) printf("TRACE: %s %s\n", msg, #command);                                      \
        command;                                                                                   \
    }


#define MY_LAUNCH_WITH_PAREN(command, doTrace, msg)                                                \
    {                                                                                              \
        if (doTrace) printf("TRACE: %s %s\n", msg, #command);                                      \
        (command);                                                                                 \
    }


int main() {
    float* Ad;

    hipMalloc((void**)&Ad, 1024);

    // Struct type,  check access from device.
    hipLaunchKernelStruct_t1 *hipLaunchKernelStruct_d1, *hipLaunchKernelStruct_h1;
    hipMalloc((void**)&hipLaunchKernelStruct_d1, STRUCT_SIZE*sizeof(hipLaunchKernelStruct_t1));
    hipHostMalloc((void**)&hipLaunchKernelStruct_h1, STRUCT_SIZE*sizeof(hipLaunchKernelStruct_t1));
    for (int k = 0; k < STRUCT_SIZE; ++k) {
        hipLaunchKernelStruct_d1[k].li = 1;
        hipLaunchKernelStruct_d1[k].lf = 1.0;
        hipLaunchKernelStruct_d1[k].result = false;  // This will be set to true if the the condition is satisfied, from device side
    }

    // Struct type, checks padding
    hipLaunchKernelStruct_t2 *hipLaunchKernelStruct_d2, *hipLaunchKernelStruct_h2;
    hipMalloc((void**)&hipLaunchKernelStruct_d2, STRUCT_SIZE*sizeof(hipLaunchKernelStruct_t2));
    hipHostMalloc((void**)&hipLaunchKernelStruct_h2, STRUCT_SIZE*sizeof(hipLaunchKernelStruct_t2));
    for (int k = 0; k < STRUCT_SIZE; ++k) {
        hipLaunchKernelStruct_d2[k].c1 = 'a';
        hipLaunchKernelStruct_d2[k].l1 = 1.0;
        hipLaunchKernelStruct_d2[k].c2 = 'b';
        hipLaunchKernelStruct_d2[k].l2 = 2.0;
        hipLaunchKernelStruct_d2[k].result = false;  // This will be set to true if the the condition is satisfied, from device side
    }

    // Struct type, checks padding, assigning integer to a char
    hipLaunchKernelStruct_t3 *hipLaunchKernelStruct_d3, *hipLaunchKernelStruct_h3;
    hipMalloc((void**)&hipLaunchKernelStruct_d3, STRUCT_SIZE*sizeof(hipLaunchKernelStruct_t3));
    hipHostMalloc((void**)&hipLaunchKernelStruct_h3, STRUCT_SIZE*sizeof(hipLaunchKernelStruct_t3));
    for (int k = 0; k < STRUCT_SIZE; ++k) {
      hipLaunchKernelStruct_d3[k].bf1 = 1;
      hipLaunchKernelStruct_d3[k].bf2 = 1;
      hipLaunchKernelStruct_d3[k].l1 = 1.0;
      hipLaunchKernelStruct_d3[k].bf3 = 1;
      hipLaunchKernelStruct_d3[k].result = false;  // This will be set to true if the the condition is satisfied, from device side
    }

    // Test the different hipLaunchParm options:
    hipLaunchKernel(vAdd, size_t(1024), 1, 0, 0, Ad);
    hipLaunchKernel(vAdd, 1024, dim3(1), 0, 0, Ad);
    hipLaunchKernel(vAdd, dim3(1024), 1, 0, 0, Ad);
    hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad);
    hipLaunchKernel(hipLaunchKernelStructFunc1, dim3(STRUCT_SIZE), dim3(1), 0, 0, hipLaunchKernelStruct_d1);
    hipLaunchKernel(hipLaunchKernelStructFunc2, dim3(STRUCT_SIZE), dim3(1), 0, 0, hipLaunchKernelStruct_d2);
    hipLaunchKernel(hipLaunchKernelStructFunc3, dim3(STRUCT_SIZE), dim3(1), 0, 0, hipLaunchKernelStruct_d3);

    // Validation part of the struct
    hipMemcpy(hipLaunchKernelStruct_h1, hipLaunchKernelStruct_d1, STRUCT_SIZE*sizeof(hipLaunchKernelStruct_t1), hipMemcpyDeviceToHost);
    for (int k = 0; k < STRUCT_SIZE; ++k)
      HIPASSERT(hipLaunchKernelStruct_h1[k].result == true);

    // Validation part of the struct
    hipMemcpy(hipLaunchKernelStruct_h2, hipLaunchKernelStruct_d2, STRUCT_SIZE*sizeof(hipLaunchKernelStruct_t2), hipMemcpyDeviceToHost);
    for (int k = 0; k < STRUCT_SIZE; ++k)
    HIPASSERT(hipLaunchKernelStruct_h2[k].result == true);

    // Validation part of the struct
    hipMemcpy(hipLaunchKernelStruct_h3, hipLaunchKernelStruct_d3, STRUCT_SIZE*sizeof(hipLaunchKernelStruct_t3), hipMemcpyDeviceToHost);
    for (int k = 0; k < STRUCT_SIZE; ++k)
     HIPASSERT(hipLaunchKernelStruct_h3[k].result == true);

    // Test case with hipLaunchKernel inside another macro:
    float e0;
    GPU_PRINT_TIME(hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), e0, j);
    GPU_PRINT_TIME(WRAP(hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad)), e0, j);

#ifdef EXTRA_PARENS_1
    // Don't wrap hipLaunchKernel in extra set of parens:
    GPU_PRINT_TIME((hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad)), e0, j);
#endif

    MY_LAUNCH(hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), true, "firstCall");

    float* A;
    float e1;
    MY_LAUNCH_WITH_PAREN(hipMalloc(&A, 100), true, "launch2");

#ifdef EXTRA_PARENS_2
    // MY_LAUNCH_WITH_PAREN wraps cmd in () which can cause issues.
    MY_LAUNCH_WITH_PAREN(hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), true, "firstCall");
#endif

    passed();
}
