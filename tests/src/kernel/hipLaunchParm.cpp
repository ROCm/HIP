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

#define ENABLE_ALIGNMENT_TEST 0

static const int  BLOCK_DIM_SIZE = 1024;

// This test is to verify Struct with variables
// support, read from device.
typedef struct hipLaunchKernelStruct1 {
  int li;  // local int
  float lf;  // local float
  bool result;  // local bool
} hipLaunchKernelStruct_t1;

// This test is to verify struct with padding, read from device
typedef struct hipLaunchKernelStruct2 {
  char c1;
  long l1;
  char c2;
  long l2;
  bool result;
} hipLaunchKernelStruct_t2;

// This test is to verify struct with padding, read from device
typedef struct hipLaunchKernelStruct3 {
  char bf1;
  char bf2;
  long l1;
  char bf3;
  bool result;
} hipLaunchKernelStruct_t3;

// This test is to verify empty struct
typedef struct hipLaunchKernelStruct4 {
  // empty struct, size will be verified from device side,size 1Byte
} hipLaunchKernelStruct_t4;

// This test is to verify struct with pointer member variable.
typedef struct hipLaunchKernelStruct5 {
  char c1;
  char* cp;  // char pointer
} hipLaunchKernelStruct_t5;


// This test is to verify struct with aligned(8),
// right now it's broken on hcc & hip-clang
typedef struct hipLaunchKernelStruct6 {
  char c1;
  short int si;
} __attribute__((aligned(8)))  hipLaunchKernelStruct_t6;

// This test is to verify struct with aligned(16),
// right now it's brokenon hcc & hip-clang
typedef struct hipLaunchKernelStruct7 {
  char c1;
  short int si;
} __attribute__((aligned(16)))  hipLaunchKernelStruct_t7;

// This test is to verify struct with packed & aligned,
// size should be 4Bytes right now it's broken on hcc & hip-clang
typedef struct hipLaunchKernelStruct8 {
  char c1;
  short int si;
  bool b;
}__attribute__((packed, aligned(4)))  hipLaunchKernelStruct_t8;

// Passing struct to a hipLaunchKernel(),
// read and write into the same struct
__global__ void hipLaunchKernelStructFunc1(
                    hipLaunchParm lp,
                    hipLaunchKernelStruct_t1 hipLaunchKernelStruct_,
                    bool* result_d1) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d1[x] =  ((hipLaunchKernelStruct_.li == 1)
                      && (hipLaunchKernelStruct_.lf == 1.0)
                      && (hipLaunchKernelStruct_.result == false));
}

// Passing struct to a hipLaunchKernel(), checks padding,
// read and write into the same struct
__global__ void hipLaunchKernelStructFunc2(
                    hipLaunchParm lp,
                    hipLaunchKernelStruct_t2 hipLaunchKernelStruct_,
                    bool* result_d2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d2[x] =  ((hipLaunchKernelStruct_.c1 == 'a')
                      && (hipLaunchKernelStruct_.l1 == 1.0)
                      && (hipLaunchKernelStruct_.c2 == 'b')
                      && (hipLaunchKernelStruct_.l2 == 2.0) );
}

// Passing struct to a hipLaunchKernel(), checks padding,
// read and write into the same struct
__global__ void hipLaunchKernelStructFunc3(
                    hipLaunchParm lp,
                    hipLaunchKernelStruct_t3 hipLaunchKernelStruct_,
                    bool* result_d3) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d3[x] =  ((hipLaunchKernelStruct_.bf1 == 1)
                     && (hipLaunchKernelStruct_.bf2 == 1)
                     && (hipLaunchKernelStruct_.l1 == 1.0)
                     && (hipLaunchKernelStruct_.bf3 == 1) );
}

// Passing empty struct to a hipLaunchKernel(),
// check the size of 1Byte, set  result_d4 to true if condition met
__global__ void hipLaunchKernelStructFunc4(
                    hipLaunchParm lp,
                    hipLaunchKernelStruct_t4 hipLaunchKernelStruct_,
                    bool* result_d4) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d4[x] =  (sizeof(hipLaunchKernelStruct_) == 1);
}

// Passing struct with pointer object to a hipLaunchKernel()
__global__ void hipLaunchKernelStructFunc5(
                    hipLaunchParm lp,
                    hipLaunchKernelStruct_t5 hipLaunchKernelStruct_,
                    bool* result_d5) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d5[x] =  ((hipLaunchKernelStruct_.c1 == 'c')
                      && (*hipLaunchKernelStruct_.cp == 'p'));
}

// Passing struct which is aligned to 8Byte to a hipLaunchKernel(),
// set the result_d6 to true if condition met
__global__ void hipLaunchKernelStructFunc6(
                    hipLaunchParm lp,
                    hipLaunchKernelStruct_t6 hipLaunchKernelStruct_,
                    bool* result_d6) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    // get the address of the struct
    // size_t(p)%8 will be 0 if aligned to 8Byte address space
    int *p = (int*)(&hipLaunchKernelStruct_);
    result_d6[x] =  ((hipLaunchKernelStruct_.c1 == 'c')
                      && (hipLaunchKernelStruct_.si == 1)
                      && ((size_t(p))%8 ==0));
}

// Passing struct which is aligned to 16Byte,
// set the result_d7 to true if condition met
__global__ void hipLaunchKernelStructFunc7(
                    hipLaunchParm lp,
                    hipLaunchKernelStruct_t7 hipLaunchKernelStruct_,
                    bool* result_d7) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    // get the address of the struct
    // size_t(p)%16 will be 0 if aligned to 16Byte address space
    int *p = (int*)(&hipLaunchKernelStruct_);
    result_d7[x] =  ((hipLaunchKernelStruct_.c1 == 'c')
                      && (hipLaunchKernelStruct_.si == 1)
                      && ((size_t(p))%16 ==0) );
}

// Passing struct which is packed & aligned to 4Byte,
// set the result_d8 to true if condition met
__global__ void hipLaunchKernelStructFunc8(
                    hipLaunchParm lp,
                    hipLaunchKernelStruct_t8 hipLaunchKernelStruct_,
                    bool* result_d8) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    // get the address of the xth element, struct[x],
    // size_t(p)%4 will be 0 if aligned to 4Byte address space
    int *p = (int*)(&hipLaunchKernelStruct_);
    result_d8[x] =  ((hipLaunchKernelStruct_.c1 == 'c')
                      && (hipLaunchKernelStruct_.si == 1)
                      && ((size_t(p))%4 ==0) );
}

__global__ void vAdd(hipLaunchParm lp, float* a) {}

//---
// Some wrapper macro for testing:
#define WRAP(...) __VA_ARGS__

#include <sys/time.h>
#define GPU_PRINT_TIME(cmd, elapsed, quiet)                         \
    do {                                                            \
        struct timeval start, stop;                                 \
        float elapsed;                                              \
        gettimeofday(&start, NULL);                                 \
        hipDeviceSynchronize();                                     \
        cmd;                                                        \
        hipDeviceSynchronize();                                     \
        gettimeofday(&stop, NULL);                                  \
    } while (0);


#define MY_LAUNCH(command, doTrace, msg)                            \
    {                                                               \
        if (doTrace) printf("TRACE: %s %s\n", msg, #command);       \
        command;                                                    \
    }


#define MY_LAUNCH_WITH_PAREN(command, doTrace, msg)                 \
    {                                                               \
        if (doTrace) printf("TRACE: %s %s\n", msg, #command);       \
        (command);                                                  \
    }


int main() {
    float* Ad;

    hipMalloc((void**)&Ad, 1024);

    // Struct type,  check access from device.
    hipLaunchKernelStruct_t1 hipLaunchKernelStruct_h1;
    bool *result_d1, *result_h1;
    hipMalloc((void**)&result_d1, BLOCK_DIM_SIZE*sizeof(bool));
    hipHostMalloc((void**)&result_h1, BLOCK_DIM_SIZE*sizeof(bool));
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
      result_d1[k] = false;
                // initialize to false, will be set to
                // true if the struct size is 1Byte, from device size
    }
    hipLaunchKernelStruct_h1.li = 1;
    hipLaunchKernelStruct_h1.lf = 1.0;
    hipLaunchKernelStruct_h1.result = false;

    // Struct type, checks padding
    hipLaunchKernelStruct_t2 hipLaunchKernelStruct_h2;
    bool *result_d2, *result_h2;
    hipMalloc((void**)&result_d2, BLOCK_DIM_SIZE*sizeof(bool));
    hipHostMalloc((void**)&result_h2, BLOCK_DIM_SIZE*sizeof(bool));
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
      result_d2[k] = false;  // initialize to false, will be set to
                             // true if the struct is accessible from device.
    }
    hipLaunchKernelStruct_h2.c1 = 'a';
    hipLaunchKernelStruct_h2.l1 = 1.0;
    hipLaunchKernelStruct_h2.c2 = 'b';
    hipLaunchKernelStruct_h2.l2 = 2.0;
    hipLaunchKernelStruct_h2.result = false;

    // Struct type, checks padding, assigning integer to a char
    hipLaunchKernelStruct_t3 hipLaunchKernelStruct_h3;
    bool *result_d3, *result_h3;
    hipMalloc((void**)&result_d3, BLOCK_DIM_SIZE*sizeof(bool));
    hipHostMalloc((void**)&result_h3, BLOCK_DIM_SIZE*sizeof(bool));
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
      result_d2[k] = false;
                // initialize to false, will be set to
                // true if the struct size is 1Byte, from device size
    }
    hipLaunchKernelStruct_h3.bf1 = 1;
    hipLaunchKernelStruct_h3.bf2 = 1;
    hipLaunchKernelStruct_h3.l1 = 1.0;
    hipLaunchKernelStruct_h3.bf3 = 1;
    hipLaunchKernelStruct_h3.result = false; 
                // initialize to false, will be set to
                // true if the struct size is 1Byte, from device size

    // empty struct
    hipLaunchKernelStruct_t4 hipLaunchKernelStruct_h4;
    bool *result_d4, *result_h4;
    hipMalloc((void**)&result_d4, BLOCK_DIM_SIZE*sizeof(bool));
    hipHostMalloc((void**)&result_h4, BLOCK_DIM_SIZE*sizeof(bool));
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
      result_d4[k] = false;
                // initialize to false, will be set to
                // true if the struct size is 1Byte, from device size
    }

    // Passing struct with pointer object to a hipLaunchKernel()
    hipLaunchKernelStruct_t5 hipLaunchKernelStruct_h5;
    // This is passed as pointer to struct member, struct.cp = &cp_d5
    char* cp_d5;
    bool *result_d5, *result_h5;
    hipMalloc((void**)&result_d5, BLOCK_DIM_SIZE*sizeof(bool));
    // allocating memory for char pointer on device
    hipMalloc((void**)&cp_d5, sizeof(char));
    hipHostMalloc((void**)&result_h5, BLOCK_DIM_SIZE*sizeof(bool));
    *cp_d5 = 'p';  // initializing memory to 'p'
    hipLaunchKernelStruct_h5.c1 = 'c';
    hipLaunchKernelStruct_h5.cp = cp_d5;
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
      result_d5[k] = false;
                // initialize to false, will be set to
                // true if the struct size is 1Byte, from device size
    }

    // Passing struct with aligned(8)
    hipLaunchKernelStruct_t6 hipLaunchKernelStruct_h6;
    bool *result_d6, *result_h6;
    hipMalloc((void**)&result_d6, BLOCK_DIM_SIZE*sizeof(bool));
    hipHostMalloc((void**)&result_h6, BLOCK_DIM_SIZE*sizeof(bool));
    hipLaunchKernelStruct_h6.c1 = 'c';
    hipLaunchKernelStruct_h6.si = 1;
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
      result_d6[k] = false;
                // initialize to false, will be set to
                // true if the struct size is 1Byte, from device size
    }

    // Passing struct with aligned(16)
    hipLaunchKernelStruct_t7 hipLaunchKernelStruct_h7;
    bool *result_d7, *result_h7;
    hipMalloc((void**)&result_d7, BLOCK_DIM_SIZE*sizeof(bool));
    hipHostMalloc((void**)&result_h7, BLOCK_DIM_SIZE*sizeof(bool));
    hipLaunchKernelStruct_h7.c1 = 'c';
    hipLaunchKernelStruct_h7.si = 1;
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
      result_d7[k] = false;  
                // initialize to false, will be set to
                // true if the struct size is 1Byte, from device size
    }
    // Passing struct with packed aligned to 6Bytes
    hipLaunchKernelStruct_t8 hipLaunchKernelStruct_h8;
    bool *result_d8, *result_h8;
    hipMalloc((void**)&result_d8, BLOCK_DIM_SIZE*sizeof(bool));
    hipHostMalloc((void**)&result_h8, BLOCK_DIM_SIZE*sizeof(bool));
    hipLaunchKernelStruct_h8.c1 = 'c';
    hipLaunchKernelStruct_h8.si = 1;
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
      result_d8[k] = false;
                // initialize to false, will be set to
                // true if the struct size is 1Byte, from device size
    }

    // Test the different hipLaunchParm options:
    hipLaunchKernel(vAdd, size_t(1024), 1, 0, 0, Ad);
    hipLaunchKernel(vAdd, 1024, dim3(1), 0, 0, Ad);
    hipLaunchKernel(vAdd, dim3(1024), 1, 0, 0, Ad);
    hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad);
    hipLaunchKernel(hipLaunchKernelStructFunc1, dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h1,
                    result_d1);
    hipLaunchKernel(hipLaunchKernelStructFunc2, dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h2,
                    result_d2);
    hipLaunchKernel(hipLaunchKernelStructFunc3, dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h3,
                    result_d3);
    hipLaunchKernel(hipLaunchKernelStructFunc4, dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h4,
                    result_d4);
    hipLaunchKernel(hipLaunchKernelStructFunc5, dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h5,
                    result_d5);
    hipLaunchKernel(hipLaunchKernelStructFunc6, dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h6,
                    result_d6);
    hipLaunchKernel(hipLaunchKernelStructFunc7, dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h7,
                    result_d7);
    hipLaunchKernel(hipLaunchKernelStructFunc8, dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h8,
                    result_d8);

    // Validation part of the struct, hipLaunchKernelStructFunc1
    hipMemcpy(result_h1, result_d1, BLOCK_DIM_SIZE*sizeof(bool),
              hipMemcpyDeviceToHost);
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
      HIPASSERT(result_h1[k] == true);

    // Validation part of the struct, hipLaunchKernelStructFunc2
    hipMemcpy(result_h2, result_d2, BLOCK_DIM_SIZE*sizeof(bool),
              hipMemcpyDeviceToHost);
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
    HIPASSERT(result_h2[k] == true);

    // Validation part of the struct, hipLaunchKernelStructFunc3
    hipMemcpy(result_h3, result_d3, BLOCK_DIM_SIZE*sizeof(bool),
              hipMemcpyDeviceToHost);
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
     HIPASSERT(result_h3[k] == true);

    // Validation part of the struct, hipLaunchKernelStructFunc4
    hipMemcpy(result_h4, result_d4, BLOCK_DIM_SIZE*sizeof(bool),
              hipMemcpyDeviceToHost);
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
      HIPASSERT(result_h4[k] == true);

    // Validation part of the struct, hipLaunchKernelStructFunc5
    hipMemcpy(result_h5, result_d5, BLOCK_DIM_SIZE*sizeof(bool),
              hipMemcpyDeviceToHost);
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
      HIPASSERT(result_h5[k] == true);

    // alignment is broken hence disabled the validation part
    #if ENABLE_ALIGNMENT_TEST
    // Validation part of the struct, hipLaunchKernelStructFunc6
    hipMemcpy(result_h6, result_d6, BLOCK_DIM_SIZE*sizeof(bool),
              hipMemcpyDeviceToHost);
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
      HIPASSERT(result_h6[k] == true);

    // Validation part of the struct, hipLaunchKernelStructFunc7
    hipMemcpy(result_h7, result_d7, BLOCK_DIM_SIZE*sizeof(bool),
              hipMemcpyDeviceToHost);
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
      HIPASSERT(result_h7[k] == true);

    // Validation part of the struct, hipLaunchKernelStructFunc7
    hipMemcpy(result_h8, result_d8, BLOCK_DIM_SIZE*sizeof(bool),
              hipMemcpyDeviceToHost);
    for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
      HIPASSERT(result_h8[k] == true);
     #endif

    // Test case with hipLaunchKernel inside another macro:
    float e0;
    GPU_PRINT_TIME(hipLaunchKernel(vAdd, dim3(1024),
                   dim3(1), 0, 0, Ad), e0, j);
    GPU_PRINT_TIME(WRAP(hipLaunchKernel(vAdd, dim3(1024),
                   dim3(1), 0, 0, Ad)), e0, j);

#ifdef EXTRA_PARENS_1
    // Don't wrap hipLaunchKernel in extra set of parens:
    GPU_PRINT_TIME((hipLaunchKernel(vAdd, dim3(1024),
                    dim3(1), 0, 0, Ad)), e0, j);
#endif

    MY_LAUNCH(hipLaunchKernel(vAdd, dim3(1024), dim3(1),
              0, 0, Ad), true, "firstCall");

    float* A;
    float e1;
    MY_LAUNCH_WITH_PAREN(hipMalloc(&A, 100), true, "launch2");

#ifdef EXTRA_PARENS_2
    // MY_LAUNCH_WITH_PAREN wraps cmd in () which can cause issues.
    MY_LAUNCH_WITH_PAREN(hipLaunchKernel(vAdd, dim3(1024),
                         dim3(1), 0, 0, Ad), true, "firstCall");
#endif

    hipFree((void **)&result_h1);
    hipFree((void **)&result_d1);
    hipFree((void **)&result_h2);
    hipFree((void **)&result_d2);
    hipFree((void **)&result_h3);
    hipFree((void **)&result_d3);
    hipFree((void **)&result_h4);
    hipFree((void **)&result_d4);
    hipFree((void **)&result_h5);
    hipFree((void **)&result_d5);
    hipFree((void **)&result_h6);
    hipFree((void **)&result_d6);
    hipFree((void **)&result_h7);
    hipFree((void **)&result_d7);
    hipFree((void **)&result_h8);
    hipFree((void **)&result_d8);
    passed();
}
