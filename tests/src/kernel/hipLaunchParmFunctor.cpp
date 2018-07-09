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

#include <iostream>
#include "hip/hip_runtime.h"
#include "../test_common.h"
#include "hip/hip_runtime_api.h"


#define test_passed(test_name)  printf("%s %s  PASSED!%s\n", KGRN, #test_name, KNRM);

class HipFunctorTests {
 public:
    // Test that a class functor can be passed to hiplaunchparam
    // and can be used in kernel
    void TestForSimpleClassFunctor(void);
    // Test that a templated class functor can be passed to hiplaunchparam
    // and can be used in kernel
    void TestForClassTemplateFunctor(void);
    // Test that a class functor object ptr  can be passed to hiplaunchparam
    // and can be used in kernel
    void TestForClassObjPtrFunctor(void);
    // Test that a class object containing functor can be passed to hiplaunchparam
    // and can be used in kernel
    void TestForFunctorContainInClassObj(void);
    // Test that a stuct functor can be passed to hiplaunchparam
    // and can be used in kernel
    void TestForSimpleStructFunctor(void);
    // Test that a stuct functor object ptr  can be passed to hiplaunchparam
    // and can be used in kernel
    void TestForStructObjPtrFunctor(void);
    // Test that a templated struct functor can be passed to hiplaunchparam
    // and can be used in kernel
    void TestForStructTemplateFunctor(void);
    // Test that a struct object containing functor can be passed to hiplaunchparam
    // and can be used in kernel
    void TestForFunctorContainInStructObj(void);
};

static const int  BLOCK_DIM_SIZE = 1024;

// class functor tests

// Simple doubler Functor
class DoublerFunctor{
 public:
    __device__ int operator()(int x) { return x * 2;}
};

// simple doubler functor kernel to a hipLaunchKernel(),
__global__ void DoublerFunctorKernel(
                    hipLaunchParm lp,
                    DoublerFunctor doubler_,
                    bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = doubler_(5);
  deviceResult[x] = (result == 10);
}

void HipFunctorTests::TestForSimpleClassFunctor(void) {
  DoublerFunctor doubler;
  bool *deviceResults, *hostResults;
  HIPCHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE*sizeof(bool)));
  HIPCHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE*sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    deviceResults[k] = false;
  }

  hipLaunchKernel(DoublerFunctorKernel, dim3(BLOCK_DIM_SIZE),
                  dim3(1), 0, 0, doubler, deviceResults);

  // Validation part of TestForSimpleClassFunctor
  HIPCHECK(hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE*sizeof(bool),
           hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
    HIPASSERT(hostResults[k] == true);
  HIPCHECK(hipHostFree(hostResults));
  HIPCHECK(hipFree(deviceResults));
}

// ptr functor kernel as a pointer to a hipLaunchKernel(),
__global__ void PtrDoublerFunctorKernel(
                    hipLaunchParm lp,
                    DoublerFunctor *doubler_,
                    bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = (*doubler_)(5);
  deviceResult[x] = (result == 10);
}

void HipFunctorTests::TestForClassObjPtrFunctor(void) {
  DoublerFunctor *ptrdoubler;
  bool *deviceResults, *hostResults;
  HIPCHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE*sizeof(bool)));
  HIPCHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE*sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    deviceResults[k] = false;
  }

  hipLaunchKernel(PtrDoublerFunctorKernel, dim3(BLOCK_DIM_SIZE),
                  dim3(1), 0, 0, ptrdoubler, deviceResults);

  // Validation part of TestForClassObjPtrFunctor
  HIPCHECK(hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE*sizeof(bool),
           hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
    HIPASSERT(hostResults[k] == true);
  HIPCHECK(hipHostFree(hostResults));
  HIPCHECK(hipFree(deviceResults));
  delete ptrdoubler;
}

class compare {
 public:
    template<typename T1, typename T2>
    __device__ bool operator()(const T1& v1, const T2& v2) {
       return v1 > v2;
    }
};

// template functor kernel to a hipLaunchKernel(),
__global__ void TemplateFunctorKernel(
                    hipLaunchParm lp,
                    compare compare_,
                    bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  deviceResult[x] = compare_(2.2, 2.1);
  deviceResult[x] = compare_(2, 1);
  deviceResult[x] = compare_('b', 'a');
}

void HipFunctorTests::TestForClassTemplateFunctor(void) {
  compare comparefunctor;
  bool *deviceResults, *hostResults;
  HIPCHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE*sizeof(bool)));
  HIPCHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE*sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    deviceResults[k] = false;
  }

  hipLaunchKernel(TemplateFunctorKernel, dim3(BLOCK_DIM_SIZE),
                  dim3(1), 0, 0, comparefunctor, deviceResults);

  // Validation part of TestForClassTemplateFunctor
  HIPCHECK(hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE*sizeof(bool),
           hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
    HIPASSERT(hostResults[k] == true);
  HIPCHECK(hipHostFree(hostResults));
  HIPCHECK(hipFree(deviceResults));
}


// Doubler calculator
class DoublerCaluclator {
 public:
    int a, result;
    // fucntor contained in class object
    DoublerFunctor doubler;
};

// simple doubler functor kernel to a hipLaunchKernel(),
__global__ void DoublerCalculatorFunctorKernel(
                    hipLaunchParm lp,
                    DoublerCaluclator doubler_,
                    bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = doubler_.doubler(doubler_.a);
  deviceResult[x] = (doubler_.result == result);
}

void HipFunctorTests::TestForFunctorContainInClassObj(void) {
  DoublerCaluclator Doubler;
  bool *deviceResults, *hostResults;
  HIPCHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE*sizeof(bool)));
  HIPCHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE*sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    deviceResults[k] = false;
  }

  Doubler.a = 5;
  Doubler.result = 10;
  // pass comparefunctor to  hipLaunchParm
  hipLaunchKernel(DoublerCalculatorFunctorKernel, dim3(BLOCK_DIM_SIZE),
                  dim3(1), 0, 0, Doubler, deviceResults);

  // Validation part of TestForStructTemplateFunctor
  HIPCHECK(hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE*sizeof(bool),
           hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
    HIPASSERT(hostResults[k] == true);
  HIPCHECK(hipHostFree(hostResults));
  HIPCHECK(hipFree(deviceResults));
}


// Struct functor tests

// Simple doubler Functor
struct sDoublerFunctor {
 public:
    __device__ int operator()(int x) { return x * 2;}
};

// simple doubler functor kernel to a hipLaunchKernel(),
__global__ void structDoublerFunctorKernel(
                    hipLaunchParm lp,
                    sDoublerFunctor doubler_,
                    bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = doubler_(5);
  deviceResult[x] = (result == 10);
}

void HipFunctorTests::TestForSimpleStructFunctor(void) {
  sDoublerFunctor doubler;
  bool *deviceResults, *hostResults;
  HIPCHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE*sizeof(bool)));
  HIPCHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE*sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    deviceResults[k] = false;
  }

  hipLaunchKernel(structDoublerFunctorKernel, dim3(BLOCK_DIM_SIZE),
                  dim3(1), 0, 0, doubler, deviceResults);

  // Validation part of TestForSimpleStructFunctor
  HIPCHECK(hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE*sizeof(bool),
           hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
    HIPASSERT(hostResults[k] == true);
  HIPCHECK(hipHostFree(hostResults));
  HIPCHECK(hipFree(deviceResults));
}

// ptr functor kernel as a pointer to a hipLaunchKernel(),
__global__ void structPtrDoublerFunctorKernel(
                    hipLaunchParm lp,
                    sDoublerFunctor *doubler_,
                    bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = (*doubler_)(5);
  deviceResult[x] = (result == 10);
}

void HipFunctorTests::TestForStructObjPtrFunctor(void) {
  sDoublerFunctor *ptrdoubler;
  bool *deviceResults, *hostResults;
  HIPCHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE*sizeof(bool)));
  HIPCHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE*sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    deviceResults[k] = false;
  }

  hipLaunchKernel(structPtrDoublerFunctorKernel, dim3(BLOCK_DIM_SIZE),
                  dim3(1), 0, 0, ptrdoubler, deviceResults);

  // Validation part of TestForStructObjPtrFunctor
  HIPCHECK(hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE*sizeof(bool),
           hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
    HIPASSERT(hostResults[k] == true);
  HIPCHECK(hipHostFree(hostResults));
  HIPCHECK(hipFree(deviceResults));
  delete ptrdoubler;
}

struct sCompare {
 public:
    template< typename T1, typename T2 >
    __device__ bool operator()(const T1& v1, const T2& v2) {
    return v1 > v2;
    }
};

// template functor kernel to a hipLaunchKernel(),
__global__ void structTemplateFunctorKernel(
                    hipLaunchParm lp,
                    sCompare compare_,
                    bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  deviceResult[x] = compare_(2.2, 2.1);
  deviceResult[x] = compare_(2, 1);
  deviceResult[x] = compare_('b', 'a');
}

void HipFunctorTests::TestForStructTemplateFunctor(void) {
  sCompare comparefunctor;
  bool *deviceResults, *hostResults;
  HIPCHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE*sizeof(bool)));
  HIPCHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE*sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    deviceResults[k] = false;
  }

  // pass comparefunctor to  hipLaunchParm
  hipLaunchKernel(structTemplateFunctorKernel, dim3(BLOCK_DIM_SIZE),
                  dim3(1), 0, 0, comparefunctor, deviceResults);

  // Validation part of TestForStructTemplateFunctor
  HIPCHECK(hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE*sizeof(bool),
           hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
    HIPASSERT(hostResults[k] == true);
  HIPCHECK(hipHostFree(hostResults));
  HIPCHECK(hipFree(deviceResults));
}

// Doubler calculator struct
struct sDoublerCaluclator {
 public:
    int a, result;
    // fucntor contained in class object
    DoublerFunctor doubler;
};

// simple doubler functor kernel to a hipLaunchKernel(),
__global__ void DoublerCalculatorFunctorKernel(
                    hipLaunchParm lp,
                    sDoublerCaluclator doubler_,
                    bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = doubler_.doubler(doubler_.a);
  deviceResult[x] = (doubler_.result == result);
}

void HipFunctorTests::TestForFunctorContainInStructObj(void) {
  sDoublerCaluclator Doubler;
  bool *deviceResults, *hostResults;
  HIPCHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE*sizeof(bool)));
  HIPCHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE*sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    deviceResults[k] = false;
  }

  Doubler.a = 5;
  Doubler.result = 10;
  // pass comparefunctor to  hipLaunchParm
  hipLaunchKernel(DoublerCalculatorFunctorKernel, dim3(BLOCK_DIM_SIZE),
                  dim3(1), 0, 0, Doubler, deviceResults);

  // Validation part of TestForStructTemplateFunctor
  HIPCHECK(hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE*sizeof(bool),
           hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k)
    HIPASSERT(hostResults[k] == true);
  HIPCHECK(hipHostFree(hostResults));
  HIPCHECK(hipFree(deviceResults));
}

int main() {
  HipFunctorTests FunctorTests;
  FunctorTests.TestForSimpleClassFunctor();
  test_passed(TestForSimpleClassFunctor);

  FunctorTests.TestForClassObjPtrFunctor();
  test_passed(TestForClassObjPtrFunctor);

  FunctorTests.TestForClassTemplateFunctor();
  test_passed(TestForClassTemplateFunctor);

  FunctorTests.TestForSimpleStructFunctor();
  test_passed(TestForSimpleStructFunctor);

  FunctorTests.TestForStructObjPtrFunctor();
  test_passed(TestForStructObjPtrFunctor);

  FunctorTests.TestForStructTemplateFunctor();
  test_passed(TestForStructTemplateFunctor);

  FunctorTests.TestForFunctorContainInClassObj();
  test_passed(TestForFunctorContainInClassObj);

  FunctorTests.TestForFunctorContainInStructObj();
  test_passed(TestForFunctorContainInStructObj);
}
