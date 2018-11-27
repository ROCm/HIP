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
 * BUILD: %t %s ../test_common.cpp NVCC_OPTIONS -std=c++11
 * RUN: %t
 * HIT_END
 */

#include <cstdint>
#include "hip/hip_runtime.h"
#include "test_common.h"

// Memory alignment is broken
// Update: with latest changes the aligment is working fine, hence enabled
#define ENABLE_ALIGNMENT_TEST_SMALL_BAR 1

// Packed member atribute broken
#define ENABLE_PACKED_TEST 0

// Update: with latest changes struct class object
// from device is working fine, hence enabled
#define ENABLE_CLASS_OBJ_ACCESS 1

// accessing dynamic/heap memory from device is broken
#define ENABLE_HEAP_MEMORY_ACCESS 0

// Update: with latest changes it's working hence enabled
#define ENABLE_USER_STL 1

// Update: with latest changes it's working hence enabled
#define ENABLE_OUT_OF_ORDER_INITIALIZATION 1

// Direct initialization of struct broken,
// ip_d9 is a pointer, uint_t*, hipLaunchKernelStruct_h9 = {'c', ip_d9};
#define ENABLE_DECLARE_INITIALIZATION_POINTER 0

// Bit fields are broken
#define ENABLE_BIT_FIELDS 0

static const int  BLOCK_DIM_SIZE = 512;

// allocate memory on device and host for result validation
static bool *result_d, *result_h;
static hipError_t hipMallocError = hipMalloc((void**)&result_d,
                                              BLOCK_DIM_SIZE*sizeof(bool));
static hipError_t hipHostMallocError = hipHostMalloc((void**)&result_h,
                                              BLOCK_DIM_SIZE*sizeof(bool));
static hipError_t hipMemsetError = hipMemset(result_d,
                                              false, BLOCK_DIM_SIZE);

static void ResultValidation() {
    hipMemcpy(result_h, result_d, BLOCK_DIM_SIZE*sizeof(bool),
              hipMemcpyDeviceToHost);

    for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
      HIPASSERT(result_h[k] == true);
    }
    return;
}

// Segregating the reset part as it was causing a problem when i put inside
// ResultValidation() function, the memory was not reset correctly for the
// tests which were disabled.
static void ResetValidationMem() {
    // reset the memory to false to reuse it.
    hipMemset(result_d, false, BLOCK_DIM_SIZE);
    hipMemset(result_h, false, BLOCK_DIM_SIZE);
    return;
}

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

// This test is to verify struct with packed, no alignment as Sam suggested
// size should be 4Bytes, right now it's broken on hcc & hip-clang
typedef struct hipLaunchKernelStruct8A {
  char c1;
  short int si;
  bool b;
}__attribute__((packed))  hipLaunchKernelStruct_t8A;

// This test is to verify struct with alignment, no packing as Sam suggested
// size should be 8Bytes as no packing, right now it's broken on hcc & hip-clang
typedef struct hipLaunchKernelStruct8B {
  char c1;
  short int si;
  bool b;
}__attribute__((aligned(8)))  hipLaunchKernelStruct_t8B;

// This test is to verify const struct object
typedef struct hipLaunchKernelStruct9 {
  char c1;
  uint32_t* ip;  // uint pointer
} hipLaunchKernelStruct_t9;

// This test is to verify struct with stdint types, uintN_t
typedef struct hipLaunchKernelStruct10 {
  uint64_t u64;
  uint32_t u32;
  uint8_t u8;
} hipLaunchKernelStruct_t10;

// This test is to verify struct with volatile member
typedef struct hipLaunchKernelStruct11 {
  int i1;
  volatile unsigned int vint;
} hipLaunchKernelStruct_t11;

// This test is to verify struct with simple class object
class base {
 public:
    int i = 0;
    base() {}
};
typedef struct hipLaunchKernelStruct12 {
  base b;
  char c1;
} hipLaunchKernelStruct_t12;

// This test is to verify struct with __device__ func() attribute
typedef struct hipLaunchKernelStruct13 {
  int i1;
  __device__ int getvalue() { return i1; }
} hipLaunchKernelStruct_t13;

// This test is to verify struct with array variable,
// write to from device
typedef struct hipLaunchKernelStruct14 {
  int readint;
  int writeint[BLOCK_DIM_SIZE];  // will write to this from device
} hipLaunchKernelStruct_t14;

// This test is to verify struct with dynamic memory, new int
// the heap memory will be accessed from device
typedef struct hipLaunchKernelStruct15 {
  char c1;
  int* heapmem;  // allocated using hipMalloc()
} hipLaunchKernelStruct_t15;

// This test is to verify simple template struct
template<typename T>
struct hipLaunchKernelStruct_t16 {
  T t1;
};

// This test is to verify simple explicity template struct
template<typename T> struct hipLaunchKernelStruct_t17 {};
template<>  // explicit template
struct hipLaunchKernelStruct_t17<int> {
  int t1;
};

// This test is to verity write to struct memory using __device__ func()
typedef struct hipLaunchKernelStruct18 {
  char c1;
  __device__ void setChar(char c) { c1 = c; }
  __device__ int getChar() { return c1; }
} hipLaunchKernelStruct_t18;

// This test is to verity user defined STL, simple stack implementation
typedef struct stackNode {
    int data;
    stackNode* nextNode = NULL;
} stackNode_t;
typedef struct hipLaunchKernelStruct19 {
  stackNode_t* stack = NULL;
  unsigned int size_ = 0;
  void pushMe(int value) {  // not a device function, setting from host
    stackNode_t* newNode;
    hipMalloc((void**)&newNode, sizeof(stackNode_t));
    hipMemset(&newNode->data, value, sizeof(stackNode_t));
    //newNode->data = value;
    ++size_;
    if (stack == NULL) {
      stack = newNode;
      return;
    }
    stackNode_t* currentHead = stack;
    stack = newNode;
    stack->nextNode = currentHead;
    return;
  }
  __device__ void popMe() {
    stackNode_t* currentHead = stack;
    stack = stack->nextNode;
    --size_;
    // delete currentHead;  // no idea why delete not working
    return;
  }
  int stackSize() {
    return size_;
  }
} hipLaunchKernelStruct_t19;

// This test is to verify out of order initalizer of struct elements
// and access in-order, from device.
typedef struct hipLaunchKernelStruct20 {
  char name;
  int age;
  int rank;
} hipLaunchKernelStruct_t20;

// This test is to verify bit fields operations
// the size should be 1Bytes
typedef struct hipLaunchKernelStruct21 {
  int i : 3;  // limiting bits to 3
  int j : 2;  // limiting bits to 2
} hipLaunchKernelStruct_t21;

// Passing struct to a hipLaunchKernelGGL(),
// read and write into the same struct
__global__ void hipLaunchKernelStructFunc1(
                    hipLaunchKernelStruct_t1 hipLaunchKernelStruct_,
                    bool* result_d1) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d1[x] =  ((hipLaunchKernelStruct_.li == 1)
                      && (hipLaunchKernelStruct_.lf == 1.0)
                      && (hipLaunchKernelStruct_.result == false));
}

// Passing struct to a hipLaunchKernelGGL(), checks padding,
// read and write into the same struct
__global__ void hipLaunchKernelStructFunc2(
                    hipLaunchKernelStruct_t2 hipLaunchKernelStruct_,
                    bool* result_d2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d2[x] =  ((hipLaunchKernelStruct_.c1 == 'a')
                      && (hipLaunchKernelStruct_.l1 == 1.0)
                      && (hipLaunchKernelStruct_.c2 == 'b')
                      && (hipLaunchKernelStruct_.l2 == 2.0) );
}

// Passing struct to a hipLaunchKernelGGL(), checks padding,
// read and write into the same struct
__global__ void hipLaunchKernelStructFunc3(
                    hipLaunchKernelStruct_t3 hipLaunchKernelStruct_,
                    bool* result_d3) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d3[x] =  ((hipLaunchKernelStruct_.bf1 == 1)
                     && (hipLaunchKernelStruct_.bf2 == 1)
                     && (hipLaunchKernelStruct_.l1 == 1.0)
                     && (hipLaunchKernelStruct_.bf3 == 1) );
}

// Passing empty struct to a hipLaunchKernelGGL(),
// check the size of 1Byte, set  result_d4 to true if condition met
__global__ void hipLaunchKernelStructFunc4(
                    hipLaunchKernelStruct_t4 hipLaunchKernelStruct_,
                    bool* result_d4) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d4[x] =  (sizeof(hipLaunchKernelStruct_) == 1);
}

// Passing struct with pointer object to a hipLaunchKernelGGL()
__global__ void hipLaunchKernelStructFunc5(
                    hipLaunchKernelStruct_t5 hipLaunchKernelStruct_,
                    bool* result_d5) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d5[x] =  ((hipLaunchKernelStruct_.c1 == 'c')
                      && (*hipLaunchKernelStruct_.cp == 'p'));
}

// Passing struct which is aligned to 8Byte to a hipLaunchKernelGGL(),
// set the result_d6 to true if condition met
__global__ void hipLaunchKernelStructFunc6(
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
                    hipLaunchKernelStruct_t8 hipLaunchKernelStruct_,
                    bool* result_d8) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    // get the address of the xth element, struct[x],
    // size_t(p)%4 will be 0 if aligned to 4Byte address space
    int *p = (int*)(&hipLaunchKernelStruct_);
    result_d8[x] =  ((hipLaunchKernelStruct_.c1 == 'c')
                      && (hipLaunchKernelStruct_.si == 1)
                      && ((size_t(p))%4 ==0)
                      && (sizeof(hipLaunchKernelStruct_) == 4));
}

// Passing struct which is packed only, as Sam suggested, should be 4Bytes
// set the result_d8A to true if condition met
__global__ void hipLaunchKernelStructFunc8A(
                    hipLaunchKernelStruct_t8A hipLaunchKernelStruct_,
                    bool* result_d8A) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    // this is packed struct
    // the address will not be aglined in this case hence condition removed
    // only sizeof(hipLaunchKernelStruct_) will be valided
    result_d8A[x] =  ((hipLaunchKernelStruct_.c1 == 'c')
                      && (hipLaunchKernelStruct_.si == 1)
                      && (sizeof(hipLaunchKernelStruct_) == 4));
}

// Passing struct which is aligned(4) only, as Sam suggested
// , size should be 8Bytes, set the result_d8B to true if condition met
__global__ void hipLaunchKernelStructFunc8B(
                    hipLaunchKernelStruct_t8B hipLaunchKernelStruct_,
                    bool* result_d8B) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    // get the address of the xth element, struct[x],
    // size_t(p)%4 will be 0 if aligned to 4Byte address space
    int *p = (int*)(&hipLaunchKernelStruct_);
    result_d8B[x] =  ((hipLaunchKernelStruct_.c1 == 'c')
                      && (hipLaunchKernelStruct_.si == 1)
                      && ((size_t(p))%8 == 0)
                      && (sizeof(hipLaunchKernelStruct_) == 8));
}

// Passing struct with uint pointer object to a hipLaunchKernelGGL()
__global__ void hipLaunchKernelStructFunc9(
                    const hipLaunchKernelStruct_t9 hipLaunchKernelStruct_,
                    bool* result_d9) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d9[x] =  ((hipLaunchKernelStruct_.c1 == 'c')
                      && (*hipLaunchKernelStruct_.ip == 1));
}

// Passing struct with stdint types object, uintN_t, to a hipLaunchKernelGGL()
__global__ void hipLaunchKernelStructFunc10(
                    hipLaunchKernelStruct_t10 hipLaunchKernelStruct_,
                    bool* result_d10) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d10[x] =  ((hipLaunchKernelStruct_.u64 == UINT64_MAX)
                      && (hipLaunchKernelStruct_.u32 == 1)
                      && (hipLaunchKernelStruct_.u8 == UINT8_MAX));
}

// Passing struct with volatile member, to a hipLaunchKernelGGL()
__global__ void hipLaunchKernelStructFunc11(
                    hipLaunchKernelStruct_t11 hipLaunchKernelStruct_,
                    bool* result_d11) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d11[x] =  ((hipLaunchKernelStruct_.i1 == 1)
                      && (hipLaunchKernelStruct_.vint == 0));
}

// Passing struct with simple class obj, to a hipLaunchKernelGGL()
__global__ void hipLaunchKernelStructFunc12(
                    hipLaunchKernelStruct_t12 hipLaunchKernelStruct_,
                    bool* result_d12) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d12[x] =  ((hipLaunchKernelStruct_.c1 == 'c')
                      && (hipLaunchKernelStruct_.b.i == 0));
}

// Passing struct with simple __device__ func(), to a hipLaunchKernelGGL()
__global__ void hipLaunchKernelStructFunc13(
                    hipLaunchKernelStruct_t13 hipLaunchKernelStruct_,
                    bool* result_d13) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d13[x] =  ((hipLaunchKernelStruct_.i1 == 1)
                      && (hipLaunchKernelStruct_.getvalue() == 1));
}

// Passing struct with array variable, write to from device
__global__ void hipLaunchKernelStructFunc14(
                    hipLaunchKernelStruct_t14 hipLaunchKernelStruct_,
                    bool* result_d14) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    hipLaunchKernelStruct_.writeint[x] = 1;
    // set the result to true if the condition met
    result_d14[x] =  ((hipLaunchKernelStruct_.readint == 1)
                      && (hipLaunchKernelStruct_.writeint[x] == 1));
}

// Passing struct with struct with dynamic memory, new int
// the heap memory will be accessed from device
__global__ void hipLaunchKernelStructFunc15(
                    hipLaunchKernelStruct_t15 hipLaunchKernelStruct_,
                    bool* result_d15) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d15[x] =  ((hipLaunchKernelStruct_.c1 == 'c')
                       && (hipLaunchKernelStruct_.heapmem[x] == 1));
}

// Passing simple template struct
__global__ void hipLaunchKernelStructFunc16(
                    hipLaunchKernelStruct_t16<char> hipLaunchKernelStruct_,
                    bool* result_d16) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d16[x] =  (hipLaunchKernelStruct_.t1 == 'c');
}

// Passing simple explicit template struct
__global__ void hipLaunchKernelStructFunc17(
                    hipLaunchKernelStruct_t17<int> hipLaunchKernelStruct_,
                    bool* result_d17) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // set the result to true if the condition met
    result_d17[x] =  (hipLaunchKernelStruct_.t1 == 1);
}

// Passing struct and write to struct memory using __device__ func()
__global__ void hipLaunchKernelStructFunc18(
                    hipLaunchKernelStruct_t18 hipLaunchKernelStruct_,
                    bool* result_d18) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    hipLaunchKernelStruct_.setChar('c');
    // set the result to true if the condition met
    result_d18[x] =  (hipLaunchKernelStruct_.getChar() == 'c');
}

// Passing simple user defined stack implemenration,  using __device__ func()
__global__ void hipLaunchKernelStructFunc19(
                    hipLaunchKernelStruct_t19 hipLaunchKernelStruct_) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // stack should be empty after the kernel execustion, verify on host side
    hipLaunchKernelStruct_.popMe();
}

// Passing out of order initalized struct, access in-order
__global__ void hipLaunchKernelStructFunc20(
                    hipLaunchKernelStruct_t20 hipLaunchKernelStruct_,
                    bool* result_d20) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // accessing struct members in order
    result_d20[x] = (hipLaunchKernelStruct_.name == 'A'
    // strcmp(hipLaunchKernelStruct_.name, "AMD") -> strcmp is not broken
                     && hipLaunchKernelStruct_.age == 42
                     && hipLaunchKernelStruct_.rank == 2);
}

// Passing struct with bit fields
__global__ void hipLaunchKernelStructFunc21(
                    hipLaunchKernelStruct_t21 hipLaunchKernelStruct_,
                    bool* result_d21) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // accessing struct members in order
    result_d21[x] = (hipLaunchKernelStruct_.i == 2
                     && hipLaunchKernelStruct_.j == 0
                     && (sizeof(hipLaunchKernelStruct_) == 1));
}

__global__ void vAdd(float* a) {}

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
    // Validating memory & initial value, for result_d, result_h
    HIPASSERT(hipMallocError == hipSuccess);
    HIPASSERT(hipHostMallocError == hipSuccess);
    HIPASSERT(hipMemsetError == hipSuccess);

    // Test: Passing Struct type,  check access from device.
    ResetValidationMem();
    hipLaunchKernelStruct_t1 hipLaunchKernelStruct_h1;
    hipLaunchKernelStruct_h1.li = 1;
    hipLaunchKernelStruct_h1.lf = 1.0;
    hipLaunchKernelStruct_h1.result = false;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc1),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h1,
                    result_d);
    ResultValidation();

    // Test: Passing Struct type, checks padding
    ResetValidationMem();
    hipLaunchKernelStruct_t2 hipLaunchKernelStruct_h2;
    hipLaunchKernelStruct_h2.c1 = 'a';
    hipLaunchKernelStruct_h2.l1 = 1.0;
    hipLaunchKernelStruct_h2.c2 = 'b';
    hipLaunchKernelStruct_h2.l2 = 2.0;
    hipLaunchKernelStruct_h2.result = false;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc2),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h2,
                    result_d);
    ResultValidation();

    // Test: Passing Struct type, checks padding, assigning integer to a char
    ResetValidationMem();
    hipLaunchKernelStruct_t3 hipLaunchKernelStruct_h3;
    hipLaunchKernelStruct_h3.bf1 = 1;
    hipLaunchKernelStruct_h3.bf2 = 1;
    hipLaunchKernelStruct_h3.l1 = 1.0;
    hipLaunchKernelStruct_h3.bf3 = 1;
    hipLaunchKernelStruct_h3.result = false;
                // initialize to false, will be set to
                // true if the struct size is 1Byte, from device size
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc3),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h3,
                    result_d);
    ResultValidation();

    // Test: Passing empty struct
    ResetValidationMem();
    hipLaunchKernelStruct_t4 hipLaunchKernelStruct_h4;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc4),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h4,
                    result_d);
    ResultValidation();

    // Test: Passing struct with pointer object to a hipLaunchKernelGGL()
    ResetValidationMem();
    hipLaunchKernelStruct_t5 hipLaunchKernelStruct_h5;
    char* cp_d5;  // This is passed as pointer to struct member
    // allocating memory for char pointer on device
    HIPCHECK(hipMalloc((void**)&cp_d5, sizeof(char)));
    HIPCHECK(hipMemset(cp_d5, 'p', sizeof(char)));
    hipLaunchKernelStruct_h5.c1 = 'c';
    hipLaunchKernelStruct_h5.cp = cp_d5;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc5),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h5,
                    result_d);
    ResultValidation();

    // Test: Passing struct with aligned(8)
    ResetValidationMem();
    hipLaunchKernelStruct_t6 hipLaunchKernelStruct_h6;
    hipLaunchKernelStruct_h6.c1 = 'c';
    hipLaunchKernelStruct_h6.si = 1;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc6),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h6,
                    result_d);
    // alignment is broken hence disabled the validation part
    #if ENABLE_ALIGNMENT_TEST_SMALL_BAR
    ResultValidation();
    #endif


    // Test: Passing struct with aligned(16)
    ResetValidationMem();
    hipLaunchKernelStruct_t7 hipLaunchKernelStruct_h7;
    hipLaunchKernelStruct_h7.c1 = 'c';
    hipLaunchKernelStruct_h7.si = 1;
    #if ENABLE_ALIGNMENT_TEST_SMALL_BAR  // This is broken on small bar
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc7),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h7,
                    result_d);
    ResultValidation();
    #endif

    // Test: Passing struct with packed aligned to 4Bytes
    ResetValidationMem();
    hipLaunchKernelStruct_t8 hipLaunchKernelStruct_h8;
    hipLaunchKernelStruct_h8.c1 = 'c';
    hipLaunchKernelStruct_h8.si = 1;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc8),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h8,
                    result_d);
    // packed member broken on large and small bar setup.
    #if ENABLE_PACKED_TEST
    ResultValidation();
    #endif

    // Test: Passing struct with packed to 4Bytes
    ResetValidationMem();
    hipLaunchKernelStruct_t8A hipLaunchKernelStruct_h8A;
    hipLaunchKernelStruct_h8A.c1 = 'c';
    hipLaunchKernelStruct_h8A.si = 1;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc8A),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h8A,
                    result_d);
    // packed member broken on large and small bar setup.
    #if ENABLE_PACKED_TEST
    ResultValidation();
    #endif

    // Test: Passing struct with aligned(4) to 4Bytes, size is 8Bytes
    ResetValidationMem();
    hipLaunchKernelStruct_t8B hipLaunchKernelStruct_h8B;
    hipLaunchKernelStruct_h8B.c1 = 'c';
    hipLaunchKernelStruct_h8B.si = 1;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc8B),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h8B,
                    result_d);
    // alignment is broken hence disabled the validation part
    #if ENABLE_ALIGNMENT_TEST_SMALL_BAR
    ResultValidation();
    #endif

    // Test: Passing const struct object to a hipLaunchKernelGGL()
    ResetValidationMem();
    uint32_t* ip_d9;
    // allocating memory for char pointer on device
    HIPCHECK(hipMalloc((void**)&ip_d9, sizeof(uint32_t)));
    HIPCHECK(hipMemset(ip_d9, 1, sizeof(uint32_t)));
    // ip_d9 passed as pointer to struct member, struct.ip = &ip_d9
    const hipLaunchKernelStruct_t9 hipLaunchKernelStruct_h9 = {'c', ip_d9};
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc9),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h9,
                    result_d);
    #if ENABLE_DECLARE_INITIALIZATION_POINTER
    ResultValidation();
    #endif


    // Test: Passing struct with uintN_t as member variables
    ResetValidationMem();
    hipLaunchKernelStruct_t10 hipLaunchKernelStruct_h10;
    hipLaunchKernelStruct_h10.u64 = UINT64_MAX;
    hipLaunchKernelStruct_h10.u32 = 1;
    hipLaunchKernelStruct_h10.u8 = UINT8_MAX;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc10),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h10,
                    result_d);
    ResultValidation();


    // Test: Passing struct with uintN_t as member variables
    ResetValidationMem();
    hipLaunchKernelStruct_t11 hipLaunchKernelStruct_h11;
    hipLaunchKernelStruct_h11.i1 = 1;
    hipLaunchKernelStruct_h11.vint = 0;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc11),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h11,
                    result_d);
    ResultValidation();

    // Test: Passing struct with simple class object
    ResetValidationMem();
    hipLaunchKernelStruct_t12 hipLaunchKernelStruct_h12;
    hipLaunchKernelStruct_h12.c1 = 'c';
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc12),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h12,
                    result_d);
    #if ENABLE_CLASS_OBJ_ACCESS  // access class obj from device broken
    // Validation part of the struct, hipLaunchKernelStructFunc12
    ResultValidation();
    #endif

    // Test: Passing struct with simple __device__ func()
    ResetValidationMem();
    hipLaunchKernelStruct_t13 hipLaunchKernelStruct_h13;
    hipLaunchKernelStruct_h13.i1 = 1;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc13),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h13,
                    result_d);
    ResultValidation();

    // Test: Passing struct with array variable, write to from device
    ResetValidationMem();
    hipLaunchKernelStruct_t14 hipLaunchKernelStruct_h14;
    hipLaunchKernelStruct_h14.readint = 1;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc14),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h14,
                    result_d);
    ResultValidation();


    // Test: Passing struct with heap memory, read to from device
    ResetValidationMem();
    hipLaunchKernelStruct_t15 hipLaunchKernelStruct_h15;
    hipLaunchKernelStruct_h15.c1 = 'c';

    #if ENABLE_HEAP_MEMORY_ACCESS  // causing page fault here,
                                   // on small bar set
    HIPCHECK(hipMalloc(&hipLaunchKernelStruct_h15.heapmem,
                       BLOCK_DIM_SIZE*sizeof(int)));
    HIPCHECK(hipMemset(&hipLaunchKernelStruct_h15.heapmem,
                       0, BLOCK_DIM_SIZE));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc15),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h15,
                    result_d);
    ResultValidation();
    #endif

    // Test: Passing simple template struct
    ResetValidationMem();
    hipLaunchKernelStruct_t16<char> hipLaunchKernelStruct_h16;
    hipLaunchKernelStruct_h16.t1 = 'c';
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc16),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h16,
                    result_d);
    ResultValidation();

    // Test: Passing simple explicit template struct
    ResetValidationMem();
    hipLaunchKernelStruct_t17<int> hipLaunchKernelStruct_h17;
    hipLaunchKernelStruct_h17.t1 = 1;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc17),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h17,
                    result_d);
    ResultValidation();

    // Test: Passing struct with simple __device__ func() to struct memory
    ResetValidationMem();
    hipLaunchKernelStruct_t18 hipLaunchKernelStruct_h18;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc18),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h18,
                    result_d);
    ResultValidation();

    // Test: Passing user defined stack,
    ResetValidationMem();
    hipLaunchKernelStruct_t19 hipLaunchKernelStruct_h19;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc19),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h19);
    #if ENABLE_USER_STL
    // Validation part of the struct, hipLaunchKernelStructFunc19
    HIPASSERT(hipLaunchKernelStruct_h19.stackSize() == 0);
    #endif

    // Test: Passing struct which is initiazed out of order
    // accessing same elements in order from device
    ResetValidationMem();
    hipLaunchKernelStruct_t20 hipLaunchKernelStruct_h20;
    hipLaunchKernelStruct_h20.name = 'A';
    hipLaunchKernelStruct_h20.rank = 2;
    hipLaunchKernelStruct_h20.age = 42;
    bool *result_d20, *result_h20;
    #if ENABLE_OUT_OF_ORDER_INITIALIZATION
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc20),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h20, result_d);
    ResultValidation();
    #endif

    // Test: Passing struct with bit fields operation
    // accessing same elements in order from device
    ResetValidationMem();
    hipLaunchKernelStruct_t21 hipLaunchKernelStruct_h21 =
    // out of order initalization
                     {2,0};
    bool *result_d21, *result_h21;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipLaunchKernelStructFunc21),
                    dim3(BLOCK_DIM_SIZE),
                    dim3(1), 0, 0, hipLaunchKernelStruct_h21, result_d);
    #if ENABLE_BIT_FIELDS
    ResultValidation();
    #endif

    // Test: Passing the different hipLaunchParm options:
    float* Ad;
    hipMalloc((void**)&Ad, 1024);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vAdd), size_t(1024), 1, 0, 0, Ad);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vAdd), 1024, dim3(1), 0, 0, Ad);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vAdd), dim3(1024), 1, 0, 0, Ad);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vAdd), dim3(1024), dim3(1), 0, 0, Ad);

    // Test: Passing hipLaunchKernelGGL inside another macro:
    float e0;
    GPU_PRINT_TIME(hipLaunchKernelGGL(vAdd, dim3(1024),
                   dim3(1), 0, 0, Ad), e0, j);
    GPU_PRINT_TIME(WRAP(hipLaunchKernelGGL(vAdd, dim3(1024),
                   dim3(1), 0, 0, Ad)), e0, j);

#ifdef EXTRA_PARENS_1
    // Don't wrap hipLaunchKernelGGL in extra set of parens:
    GPU_PRINT_TIME((hipLaunchKernelGGL(vAdd, dim3(1024),
                    dim3(1), 0, 0, Ad)), e0, j);
#endif

    MY_LAUNCH(hipLaunchKernelGGL(vAdd, dim3(1024), dim3(1),
              0, 0, Ad), true, "firstCall");

    float* A;
    float e1;
    MY_LAUNCH_WITH_PAREN(hipMalloc(&A, 100), true, "launch2");

#ifdef EXTRA_PARENS_2
    // MY_LAUNCH_WITH_PAREN wraps cmd in () which can cause issues.
    MY_LAUNCH_WITH_PAREN(hipLaunchKernelGGL(vAdd, dim3(1024),
                         dim3(1), 0, 0, Ad), true, "firstCall");
#endif

    HIPCHECK(hipHostFree(result_h));
    HIPCHECK(hipFree(result_d));

    passed();
}
