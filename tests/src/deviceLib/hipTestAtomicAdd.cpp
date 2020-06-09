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
 1) Execute atomicAdd in multi threaded scenario by diverging the data across
 multiple threads and validate the output at the end of all operations.
 2) Execute atomicAddNoRet in multi threaded scenario by diverging the data
 across multiple threads and validate the output at the end of all operations.

 (TestCase 2)::
 3) Execute atomicAdd API and validate the result.
 4) Execute atomicAddNoRet API and validate the result.

 (TestCase 3)::
 5) atomicadd/NoRet negative scenarios (TBD).

*/


/* HIT_START
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST_NAMED: %t hipTestAtomicnoret-manywaves --atomicnoret --tests 1
 * TEST_NAMED: %t hipTestAtomicnoret-simple --atomicnoret --tests 2
 * TEST_NAMED: %t hipTestAtomic-manywaves --tests 1
 * TEST_NAMED: %t hipTestAtomic-simple --tests 2
 * HIT_END
 */

#include <stdio.h>
#include "test_common.h"

/*
 * Defines initial and increment values
 */
#define INCREMENT_VALUE 10

#define INT_INITIAL_VALUE 10
#define FLOAT_INITIAL_VALUE 10.50
#define DOUBLE_INITIAL_VALUE 200.12
#define LONG_INITIAL_VALUE 10000
#define UNSIGNED_INITIAL_VALUE 20



/*
 * Square each element in the array A and write to array C.
 */
bool p_atomicNoRet = false;

template <typename T>
__global__ void atomicnoret_manywaves(T* C_d) {
  size_t tid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
  switch (tid % 9) {
    case 0:
      atomicAddNoRet(C_d, INCREMENT_VALUE);
      break;
    case 1:
      atomicAddNoRet(C_d, INCREMENT_VALUE);
      break;
    case 2:
      atomicAddNoRet(C_d, INCREMENT_VALUE);
      break;
    case 3:
      atomicAddNoRet(C_d, INCREMENT_VALUE);
      break;
    case 4:
      atomicAddNoRet(C_d, INCREMENT_VALUE);
      break;
    case 5:
      atomicAddNoRet(C_d, INCREMENT_VALUE);
      break;
    case 6:
      atomicAddNoRet(C_d, INCREMENT_VALUE);
      break;
    case 7:
      atomicAddNoRet(C_d, INCREMENT_VALUE);
      break;
    case 8:
      atomicAddNoRet(C_d, INCREMENT_VALUE);
      break;
  }
}



template <typename T>
__global__ void atomic_manywaves(T* C_d) {
  size_t tid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);

  switch (tid % 9) {
    case 0:
      atomicAdd(C_d, INCREMENT_VALUE);
      break;
    case 1:
      atomicAdd(C_d, INCREMENT_VALUE);
      break;
    case 2:
      atomicAdd(C_d, INCREMENT_VALUE);
      break;
    case 3:
      atomicAdd(C_d, INCREMENT_VALUE);
      break;
    case 4:
      atomicAdd(C_d, INCREMENT_VALUE);
      break;
    case 5:
      atomicAdd(C_d, INCREMENT_VALUE);
      break;
    case 6:
      atomicAdd(C_d, INCREMENT_VALUE);
      break;
    case 7:
      atomicAdd(C_d, INCREMENT_VALUE);
      break;
    case 8:
      atomicAdd(C_d, INCREMENT_VALUE);
      break;
  }
}


template <typename T>
__global__ void atomicnoret_simple(T* C_d) {
  atomicAddNoRet(C_d, INCREMENT_VALUE);
}

template <typename T>
__global__ void atomic_simple(T* C_d) {
  atomicAdd(C_d, INCREMENT_VALUE);
}


template <typename T>
bool atomictest_manywaves(const T& initial_val) {
  unsigned int ThreadsperBlock = 10;
  unsigned int numBlocks = 1;
  bool testPassed = true;
  T memSize = sizeof(T);
  T* hOData = reinterpret_cast<T*>(malloc(memSize));
  *hOData = initial_val;
  T* dOData;
  HIPCHECK(hipMalloc(&dOData, memSize));
  // copy host memory to device to initialize to zero
  HIPCHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));

  // execute the kernel
  hipLaunchKernelGGL(atomic_manywaves, dim3(numBlocks),
      dim3(ThreadsperBlock), 0, 0, dOData);

  // Copy result from device to host
  HIPCHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));
  if (hOData[0] != initial_val+(INCREMENT_VALUE*(ThreadsperBlock*numBlocks)))
    testPassed = false;

  // Cleanup memory
  free(hOData);
  hipFree(dOData);

  return testPassed;
}

template <typename T>
bool atomictestnoret_manywaves(const T& initial_val) {
  unsigned int ThreadsperBlock = 10;
  unsigned int numBlocks = 1;
  bool testPassed = true;
  T memSize = sizeof(T);
  T* hOData = reinterpret_cast<T*>(malloc(memSize));
  *hOData = initial_val;
  T* dOData;
  HIPCHECK(hipMalloc(&dOData, memSize));
  // copy host memory to device to initialize to zero
  HIPCHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));

  // execute the kernel
  hipLaunchKernelGGL(atomicnoret_manywaves, dim3(numBlocks),
      dim3(ThreadsperBlock), 0, 0, dOData);

  // Copy result from device to host
  HIPCHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));
  if (hOData[0] != initial_val+(INCREMENT_VALUE*(ThreadsperBlock*numBlocks)))
    testPassed = false;

  // Cleanup memory
  free(hOData);
  hipFree(dOData);

  return testPassed;
}

template <typename T>
bool atomictest_simple(const T& initial_val) {
  unsigned int ThreadsperBlock = 1;
  unsigned int numBlocks = 1;
  bool testPassed = true;
  T memSize = sizeof(T);
  T* hOData = reinterpret_cast<T*>(malloc(memSize));
  *hOData = initial_val;
  T* dOData;
  HIPCHECK(hipMalloc(&dOData, memSize));
  // copy host memory to device to initialize to zero
  HIPCHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));

  // execute the kernel
  hipLaunchKernelGGL(atomic_simple, dim3(numBlocks),
      dim3(ThreadsperBlock), 0, 0, dOData);

  // Copy result from device to host
  HIPCHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));
  if (hOData[0] != initial_val+INCREMENT_VALUE)
    testPassed = false;

  // Cleanup memory
  free(hOData);
  hipFree(dOData);

  return testPassed;
}


template <typename T>
bool atomictestnoret_simple(const T& initial_val) {
  unsigned int ThreadsperBlock = 1;
  unsigned int numBlocks = 1;
  bool testPassed = true;
  T memSize = sizeof(T);
  T* hOData = reinterpret_cast<T*>(malloc(memSize));
  *hOData = initial_val;
  T* dOData;
  HIPCHECK(hipMalloc(&dOData, memSize));
  // copy host memory to device to initialize to zero
  HIPCHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));

  // execute the kernel
  hipLaunchKernelGGL(atomicnoret_simple, dim3(numBlocks),
      dim3(ThreadsperBlock), 0, 0, dOData);

  // Copy result from device to host
  HIPCHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));
  if (hOData[0] != initial_val+INCREMENT_VALUE)
    testPassed = false;

  // Cleanup memory
  free(hOData);
  hipFree(dOData);

  return testPassed;
}


// Parse arguments specific to this test.
void parseMyArguments(int argc, char* argv[]) {
  int more_argc = HipTest::parseStandardArguments(argc, argv, false);

  // parse args for this test:
  for (int i = 1; i < more_argc; i++) {
    const char* arg = argv[i];
    if (!strcmp(arg, "--atomicnoret")) {
      p_atomicNoRet = true;
    } else {
      failed("Bad argument '%s'", arg);
    }
  }
}

int main(int argc, char* argv[]) {
  parseMyArguments(argc, argv);
  HIPCHECK(hipSetDevice(p_gpuDevice));
  bool TestPassed = true;

  if (p_tests == 1) {
    if (!p_atomicNoRet) {
      TestPassed &= atomictest_manywaves<int>(INT_INITIAL_VALUE);
      TestPassed &= atomictest_manywaves<unsigned int>(UNSIGNED_INITIAL_VALUE);
      TestPassed &= atomictest_manywaves<float>(FLOAT_INITIAL_VALUE);
      TestPassed &=
          atomictest_manywaves<unsigned long long>(LONG_INITIAL_VALUE);
      TestPassed &=
          atomictest_manywaves<double>(DOUBLE_INITIAL_VALUE);
    } else {
      atomictestnoret_manywaves<float>(FLOAT_INITIAL_VALUE);
    }
  } else if (p_tests == 2) {
    if (!p_atomicNoRet) {
      TestPassed &= atomictest_simple<int>(INT_INITIAL_VALUE);
      TestPassed &= atomictest_simple<unsigned int>(UNSIGNED_INITIAL_VALUE);
      TestPassed &= atomictest_simple<float>(FLOAT_INITIAL_VALUE);
      TestPassed &= atomictest_simple<unsigned long long>(LONG_INITIAL_VALUE);
      TestPassed &= atomictest_simple<double>(DOUBLE_INITIAL_VALUE);
    } else {
      TestPassed &= atomictestnoret_simple<float>(FLOAT_INITIAL_VALUE);
    }
  } else {
    printf("Didnt receive any valid option. Try options 1 or 2\n");
    TestPassed = false;
  }

  if (TestPassed) {
    passed();
  } else {
    failed("hipTestAtomicAdd TC validation Failed!");
  }
}
