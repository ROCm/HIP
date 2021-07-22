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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip/hip_runtime.h>
#ifdef __linux__
#include "printf_common.h"
#endif
#include <hip_test_common.hh>

#define BLOCK_SIZE 512
#define GRID_SIZE 512
#define CHUNK_SIZE 256
#define CONST_STR "Hello World from Device.Iam printing 55 bytes of data.\n"
#define CONST_STR1 "Hello World from Device.Iam printing from even thread.\n"
#define CONST_STR2 "Hello World from Device.This is odd thread.\n"
#define CONST_STR3 "Hello World from Device. The sum of all threadID = "

namespace hipPrintfStressTest {
struct SizeStruct {
  unsigned int block_size;
  unsigned int grid_size;
  unsigned int iteration;
};
// These values are empirically determined for kernel_divergent_str3
// Any modification to the function or CONST_STR3 will change these values
const struct SizeStruct EmpiricalValues1[12] = {
  {512, 512, 16},
  {512, 512, 32},
  {512, 512, 48},
  {512, 512, 64},
  {512, 512, 80},
  {512, 512, 96},
  {512, 512, 110},
  {512, 512, 126},
  {512, 512, 140},
  {512, 512, 156},
  {512, 512, 172},
  {512, 512, 186}
};
// These values are empirically determined for kernel_dependent_calc
// and kernel_dependent_calc_atomic.
// Any modification to the functions will change these values.
const struct SizeStruct EmpiricalValues2[12] = {
  {512, 512, 20},
  {512, 512, 40},
  {512, 512, 60},
  {512, 512, 80},
  {512, 512, 100},
  {512, 512, 120},
  {512, 512, 140},
  {512, 512, 160},
  {512, 512, 180},
  {512, 512, 200},
  {512, 512, 220},
  {512, 512, 240}
};
// Print a constant string in a kernel for 'n' iterations per thread
// using 'b' block size and 'g' grid size such that
// (total bytes per iteration)*n*b*g ≈ N GB where N is user input.
__global__ void kernel_printf_conststr(uint iterCount) {
  for (uint count = 0; count < iterCount; count++) {
    printf("%s", CONST_STR);
  }
}
// Print 2 different constant strings (using if and else conditionals)
// in a kernel for 'n' iterations per thread using 'b' block size and
// 'g' grid size such that (total bytes per iteration)*n*b*g ≈ N GB,
// where N is user input.
__global__ void kernel_printf_two_conditionalstr(uint iterCount) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  uint mod_tid = (tid % 2);
  if (0 == mod_tid) {
    for (uint count = 0; count < iterCount; count++) {
      printf("%s", CONST_STR1);
    }
  } else {
    for (uint count = 0; count < iterCount; count++) {
      printf("%s", CONST_STR2);
    }
  }
}
// Print a constant string (using only if condition) in a kernel for 'n'
// iterations per thread using 'b' block size and 'g' grid size such that
// (total bytes per iteration)*n*b*g ≈ N GB, where N is user input.
__global__ void kernel_printf_single_conditionalstr(uint iterCount) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  uint mod_tid = (tid % 2);
  if (0 == mod_tid) {
    for (uint count = 0; count < iterCount; count++) {
      printf("%s", CONST_STR1);
    }
  }
}
// Please do not nodify this function.
// Any modification to this function will fail the test case.
// Print variable size string using integer data in a kernel for 'n'
// iterations per thread using 'b' block size and 'g' grid size such
// that (total bytes per iteration)*n*b*g ≈ N GB, where N is user input.
__global__ void kernel_printf_variablestr(uint iterCount, int *ret) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  int retlocal = 0;
  const char *const_str =
  "Hello World from Device.Iam printing (threadID,number)=";
  for (int count = 0; count < (const int)iterCount; count++) {
    retlocal += printf("%s%u,%d\n", const_str, tid, count);
    retlocal += printf("%s%u,%d\n", const_str, tid, 10*count);
    retlocal += printf("%s%u,%d\n", const_str, tid, 100*count);
    retlocal += printf("%s%u,%d\n", const_str, tid, 1000*count);
  }
  ret[tid] = retlocal;
}
// Please do not nodify this function.
// Any modification to this function will fail the test case.
// Perform dependent calculations and print the result after each
// calculation in a kernel for 'n' iterations per thread using 'b' block
// size and 'g' grid size such that
// (total bytes per iteration)*n*b*g ≈ N GB, where N is user input.
__global__ void kernel_dependent_calc(uint32_t iterCount, int *ret) {
  uint32_t tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  int retlocal = 0;
  const char *const_str =
  "Hello World from Device.Iam printing number=";
  for (int count = 0; count < (const int)iterCount; count++) {
    uint32_t x = tid + count;
    retlocal += printf("%s%u\n", const_str, x);
    uint32_t y = x + tid;
    retlocal += printf("%s%u\n", const_str, y);
    uint32_t z = x*y;
    retlocal += printf("%s%u\n", const_str, z);
    uint32_t a = z/(tid + 1);
    retlocal += printf("%s%u\n", const_str, a);
  }
  ret[tid] = retlocal;
}
// Please do not nodify this function.
// Any modification to this function will fail the test case.
// Perform atomic calculations and print the result after each
// calculation in a kernel for 'n' iterations per thread using 'b' block
// size and 'g' grid size such that
// (total bytes per iteration)*n*b*g ≈ N GB, where N is user input.
__global__ void kernel_dependent_calc_atomic(uint32_t iterCount,
                                             int *ret) {
  uint32_t tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  int retlocal = 0;
  const char *const_str =
  "Hello World from Device.Iam printing number=";
  for (uint32_t count = 0; count < iterCount; count++) {
    uint32_t x = tid;
    atomicAdd(&x, count);
    retlocal += printf("%s%u\n", const_str, x);
    uint32_t y = x;
    atomicAdd(&y, tid);
    retlocal += printf("%s%u\n", const_str, y);
    uint32_t z = y;
    atomicSub(&z, count);
    retlocal += printf("%s%u\n", const_str, z);
    uint32_t a = z;
    atomicAnd(&a, 0x0000ffff);
    retlocal += printf("%s%u\n", const_str, a);
  }
  ret[tid] = retlocal;
}
// Print variable size string using floating point data of varying
// precision in a kernel for 'n' iterations per thread using 'b' block
// size and 'g' grid size such that
// (total bytes per iteration)*n*b*g ≈ N GB, where N is user input.
__device__ __host__ int printPi(int maxPrecision) {
  int printSize = 0;
  size_t expo = 1000000000000;
  double pi = 3.1415926535;
  double piScaled = pi*expo;
  const char *const_str =
  "Hello World from Device.Iam printing decimal number=";
  for (int prec = 0; prec <= maxPrecision ; prec++) {
    printSize += printf("%s%.*f %.*e\n", const_str, prec, pi,
                        prec, piScaled);
  }
  return printSize;
}

__global__ void kernel_decimal_calculation(uint iterCount,
                                           int maxPrecision) {
  for (int count = 0; count < (const int)iterCount; count++) {
    printPi(maxPrecision);
  }
}
// Print the value of shared memory variable using a stream of size 'n',
// 'b' block size and 'g' grid size such that
// (total bytes per thread)*n*b*g ≈ N GB, where N is user input.
__global__ void kernel_shared_mem() {
  __shared__ uint32_t sharedMem;
  sharedMem = 0;
  __syncthreads();
  atomicAdd(&sharedMem, hipThreadIdx_x);
  __syncthreads();
  printf("%s%u\n", CONST_STR3, sharedMem);
}
// Synchronize the prints in a block using __syncthreads. Only 1 block
// is launched in a stream of size 'n'. The size of the block is 'b'.
// (total bytes per thread)*n*b ≈ N GB. where N is user input.
__global__ void kernel_synchronized_printf() {
  printf("%s%u\n", CONST_STR3, 0);
  __syncthreads();
  printf("%s%u\n", CONST_STR3, 1);
  __syncthreads();
  printf("%s%u\n", CONST_STR3, 2);
}
#ifdef __linux__
// Launches kernel_printf_conststr to generate the printf log file
// and validates the generated file size and number of printed lines
// with the calculated file size and lines.
bool test_printf_conststr(uint32_t num_blocks, uint32_t threads_per_block,
                          uint32_t print_limit) {
  uint32_t iterCount = 0;
  uint32_t sizePrintString = (sizeof(CONST_STR)-1);  // Excluding NULL character
  // Calculate the number of iterations from print_limit.
  size_t stress_limit_bytes = ((size_t)print_limit*1024*1024*1024);
  iterCount = static_cast<uint32_t>(1 +
  stress_limit_bytes/(num_blocks*threads_per_block*sizePrintString));
  // Calculate expected lines of print and file size.
  uint32_t totalExpectedLines = num_blocks*threads_per_block*iterCount;
  size_t expectedFileSize = ((size_t)totalExpectedLines*sizePrintString);
  size_t actualFileSize = 0;
  uint32_t totalActualLinecount = 0;
  // DO NOT PUT ANY PRINTF WITHIN THIS BLOCK OF CODE
  {
    CaptureStream captured(stdout);
    hipLaunchKernelGGL(kernel_printf_conststr, dim3(num_blocks, 1, 1),
                       dim3(threads_per_block, 1, 1),
                       0, 0, iterCount);
    HIP_CHECK(hipStreamSynchronize(0));
    std::ifstream CapturedData = captured.getCapturedData();
    char *buffer = new char[CHUNK_SIZE];
    while (CapturedData.good()) {
      CapturedData.getline(buffer, CHUNK_SIZE);
      totalActualLinecount++;
    }
    delete[] buffer;
    struct stat st;
    if (stat(captured.getTempFilename(), &st)) {
      printf("Temp File not found \n");
      return false;
    }
    actualFileSize = st.st_size;
  }
  printf("totalExpectedLines = %u \n", totalExpectedLines);
  // Excluding the trailing newline
  printf("totalActualLinecount = %u \n", totalActualLinecount-1);
  printf("expectedFileSize = %zu \n", expectedFileSize);
  printf("actualFileSize = %zu \n", actualFileSize);
  if ((totalExpectedLines != (totalActualLinecount - 1))||
     (expectedFileSize != actualFileSize)) {
    return false;
  }
  return true;
}
// Launches kernel_printf_two_conditionalstr to generate the printf log file
// and validates the generated file size and number of printed lines
// with the calculated file size and lines.
bool test_printf_two_conditionalstr(uint32_t num_blocks,
                                    uint32_t threads_per_block,
                                    uint32_t print_limit) {
  uint32_t iterCount = 0;
  uint32_t sizePrintStringEven, sizePrintStringOdd, avgsizePrintString;
  sizePrintStringEven = (sizeof(CONST_STR1)-1);  // Excluding NULL character
  sizePrintStringOdd = (sizeof(CONST_STR2)-1);  // Excluding NULL character
  avgsizePrintString = (sizePrintStringEven + sizePrintStringOdd)/2;
  // Calculate the number of iterations from print_limit
  size_t stress_limit_bytes = ((size_t)print_limit*1024*1024*1024);
  iterCount = static_cast<uint32_t>(1 +
  stress_limit_bytes/(num_blocks*threads_per_block*avgsizePrintString));
  // Calculate expected lines of print and file size.
  uint32_t totalExpectedEvenLines, totalExpectedOddLines;
  // 0, 1, 2, 3
  // 0, 1, 2
  totalExpectedEvenLines = ((num_blocks*threads_per_block)%2 == 0)?
                       (num_blocks*threads_per_block*iterCount)/2 :
                       (((num_blocks*threads_per_block)/2)+ 1)*iterCount;
  totalExpectedOddLines = (num_blocks*threads_per_block*iterCount
                           - totalExpectedEvenLines);
  size_t expectedFileSize =
                    ((size_t)totalExpectedEvenLines*sizePrintStringEven +
                     (size_t)totalExpectedOddLines*sizePrintStringOdd);
  size_t actualFileSize = 0;
  uint32_t totalActualEvenLines = 0, totalActualOddLines = 0;
  // DO NOT PUT ANY PRINTF WITHIN THIS BLOCK OF CODE
  {
    CaptureStream captured(stdout);
    hipLaunchKernelGGL(kernel_printf_two_conditionalstr,
                       dim3(num_blocks, 1, 1),
                       dim3(threads_per_block, 1, 1),
                       0, 0, iterCount);
    HIP_CHECK(hipStreamSynchronize(0));
    std::ifstream CapturedData = captured.getCapturedData();
    char *buffer = new char[CHUNK_SIZE];
    while (CapturedData.good()) {
      CapturedData.getline(buffer, CHUNK_SIZE);
      uint32_t bufferlen = strlen(buffer);
      if ((sizePrintStringEven - 1) == bufferlen) {
        totalActualEvenLines++;
      } else if ((sizePrintStringOdd - 1) == bufferlen) {
        totalActualOddLines++;
      }
    }
    delete[] buffer;
    struct stat st;
    if (stat(captured.getTempFilename(), &st)) {
      printf("Temp File not found \n");
      return false;
    }
    actualFileSize = st.st_size;
  }
  printf("totalExpectedEvenLines = %u \n", totalExpectedEvenLines);
  printf("totalActualEvenLines = %u \n", totalActualEvenLines);
  printf("totalExpectedOddLines = %u \n", totalExpectedOddLines);
  printf("totalActualOddLines = %u \n", totalActualOddLines);
  printf("expectedFileSize = %zu \n", expectedFileSize);
  printf("actualFileSize = %zu \n", actualFileSize);
  if ((totalExpectedEvenLines != totalActualEvenLines)||
      (totalExpectedOddLines != totalActualOddLines)||
      (expectedFileSize != actualFileSize)) {
    return false;
  }
  return true;
}
// Launches kernel_printf_single_conditionalstr to generate the printf log
// and validates the generated file size and number of printed lines
// with the calculated file size and lines.
bool test_printf_single_conditionalstr(uint32_t num_blocks,
                                       uint32_t threads_per_block,
                                       uint32_t print_limit) {
  uint32_t iterCount = 0;
  uint32_t sizePrintStringEven = (sizeof(CONST_STR1)-1);
  // Excluding NULL character
  // Calculate the number of iterations from print_limit
  size_t stress_limit_bytes = ((size_t)print_limit*1024*1024*1024);
  iterCount = static_cast<uint32_t>((2*stress_limit_bytes)/
             (num_blocks*threads_per_block*sizePrintStringEven));
  // Calculate expected lines of print and file size.
  uint32_t totalExpectedLines;
  totalExpectedLines = ((num_blocks*threads_per_block)%2 == 0)?
                       (num_blocks*threads_per_block*iterCount)/2 :
                       (((num_blocks*threads_per_block)/2)+ 1)*iterCount;
  size_t expectedFileSize =
                    (size_t)totalExpectedLines*sizePrintStringEven;
  size_t actualFileSize = 0;
  uint32_t totalActualLines = 0;
  // DO NOT PUT ANY PRINTF WITHIN THIS BLOCK OF CODE
  {
    CaptureStream captured(stdout);
    hipLaunchKernelGGL(kernel_printf_single_conditionalstr,
                       dim3(num_blocks, 1, 1),
                       dim3(threads_per_block, 1, 1),
                       0, 0, iterCount);
    HIP_CHECK(hipStreamSynchronize(0));
    std::ifstream CapturedData = captured.getCapturedData();
    char *buffer = new char[CHUNK_SIZE];
    while (CapturedData.good()) {
      CapturedData.getline(buffer, CHUNK_SIZE);
      totalActualLines++;
    }
    delete[] buffer;
    struct stat st;
    if (stat(captured.getTempFilename(), &st)) {
      printf("Temp File not found \n");
      return false;
    }
    actualFileSize = st.st_size;
  }
  printf("totalExpectedLines = %u \n", totalExpectedLines);
  printf("totalActualLines = %u \n", totalActualLines-1);
  printf("expectedFileSize = %zu \n", expectedFileSize);
  printf("actualFileSize = %zu \n", actualFileSize);
  if ((totalExpectedLines != (totalActualLines - 1))||
      (expectedFileSize != actualFileSize)) {
    return false;
  }
  return true;
}
// Launches kernel_printf_variablestr Or kernel_dependent_calc Or
// kernel_dependent_calc_atomic to generate the printf log
// and validates the generated file size and number of printed lines
// with the calculated file size and lines.
bool test_variable_str(uint32_t print_limit,
                       void(*func)(uint32_t, int *),
                       const struct SizeStruct* table) {
  uint32_t iterCount = table[print_limit - 1].iteration;
  uint32_t num_blocks = table[print_limit - 1].grid_size;
  uint32_t threads_per_block = table[print_limit - 1].block_size;
  // Calculate expected lines of print and file size.
  size_t actualFileSize = 0;
  uint32_t totalActualLines = 0;
  uint32_t totalExpectedLines = 4*iterCount*num_blocks*threads_per_block;
  size_t expectedFileSize = 0;

  uint32_t buffsize = threads_per_block*num_blocks;
  int32_t *Ah;
  int32_t *Ad;
  Ah = new int32_t[buffsize];
  for (uint32_t i = 0; i < buffsize; i++) {
    Ah[i] = 0;
  }
  HIP_CHECK(hipMalloc(&Ad, buffsize*sizeof(int32_t)));
  HIP_CHECK(hipMemcpy(Ad, Ah, buffsize*sizeof(int32_t),
          hipMemcpyHostToDevice));
  // DO NOT PUT ANY PRINTF WITHIN THIS BLOCK OF CODE
  {
    CaptureStream captured(stdout);
    hipLaunchKernelGGL(func, dim3(num_blocks, 1, 1),
                       dim3(threads_per_block, 1, 1),
                       0, 0, iterCount, Ad);
    HIP_CHECK(hipStreamSynchronize(0));
    HIP_CHECK(hipMemcpy(Ah, Ad, buffsize*sizeof(int32_t),
             hipMemcpyDeviceToHost));
    std::ifstream CapturedData = captured.getCapturedData();
    char *buffer = new char[CHUNK_SIZE];
    while (CapturedData.good()) {
      CapturedData.getline(buffer, CHUNK_SIZE);
      totalActualLines++;
    }
    delete[] buffer;
    struct stat st;
    if (stat(captured.getTempFilename(), &st)) {
      printf("Temp File not found \n");
      return false;
    }
    actualFileSize = st.st_size;
  }
  for (uint32_t i = 0; i < buffsize; i++) {
    expectedFileSize += Ah[i];
  }
  HIP_CHECK(hipFree(Ad));
  delete[] Ah;
  printf("totalExpectedLines = %u \n", totalExpectedLines);
  printf("totalActualLines = %u \n", totalActualLines-1);
  printf("expectedFileSize = %zu \n", expectedFileSize);
  printf("actualFileSize = %zu \n", actualFileSize);
  if ((totalExpectedLines != (totalActualLines - 1))||
      (expectedFileSize != actualFileSize)) {
    return false;
  }
  return true;
}
// Launches kernel_decimal_calculation to generate the printf log file
// and validates the generated file size and number of printed lines
// with the calculated file size and lines.
bool test_decimal_str(uint32_t num_blocks, uint32_t threads_per_block,
                      uint32_t print_limit) {
  // Calculate the number of iterations from print_limit
  size_t stress_limit_bytes = ((size_t)print_limit*1024*1024*1024);
  int maxPrecision = 10;
  int totalPrintSizePerIter = printPi(maxPrecision);
  uint32_t iterCount = static_cast<uint32_t>(1+ stress_limit_bytes/
             (num_blocks*threads_per_block*totalPrintSizePerIter));
  // Calculate expected lines of print and file size.
  size_t actualFileSize = 0;
  size_t expectedFileSize =
  (size_t)num_blocks*threads_per_block*iterCount*totalPrintSizePerIter;
  uint32_t totalActualLines = 0;
  uint32_t totalExpectedLines =
       (maxPrecision + 1)*iterCount*num_blocks*threads_per_block;
  // DO NOT PUT ANY PRINTF WITHIN THIS BLOCK OF CODE
  {
    CaptureStream captured(stdout);
    hipLaunchKernelGGL(kernel_decimal_calculation, dim3(num_blocks, 1, 1),
                       dim3(threads_per_block, 1, 1),
                       0, 0, iterCount, maxPrecision);
    HIP_CHECK(hipStreamSynchronize(0));
    std::ifstream CapturedData = captured.getCapturedData();
    char *buffer = new char[CHUNK_SIZE];
    while (CapturedData.good()) {
      CapturedData.getline(buffer, CHUNK_SIZE);
      totalActualLines++;
    }
    delete[] buffer;
    struct stat st;
    if (stat(captured.getTempFilename(), &st)) {
      printf("Temp File not found \n");
      return false;
    }
    actualFileSize = st.st_size;
  }
  printf("totalExpectedLines = %u \n", totalExpectedLines);
  printf("totalActualLines = %u \n", totalActualLines-1);
  printf("expectedFileSize = %zu \n", expectedFileSize);
  printf("actualFileSize = %zu \n", actualFileSize);
  if ((totalExpectedLines != (totalActualLines - 1))||
      (expectedFileSize != actualFileSize)) {
    return false;
  }
  return true;
}
// Launches kernel_shared_mem to generate the printf log file
// and validates the generated file size and number of printed lines
// with the calculated file size and lines.
bool test_shared_mem(uint32_t num_blocks, uint32_t threads_per_block,
                     uint32_t print_limit) {
  // Calculate the number of iterations from print_limit
  size_t stress_limit_bytes = ((size_t)print_limit*1024*1024*1024);
  unsigned total_0_to_blksize = (BLOCK_SIZE - 1)*BLOCK_SIZE / 2;
  char buffer[CHUNK_SIZE];
  int totalPrintSizePerThread = snprintf(buffer, CHUNK_SIZE,
                              "%s%u\n", CONST_STR3, total_0_to_blksize);
  uint32_t iterCount = static_cast<uint32_t>(1+ stress_limit_bytes/
             (num_blocks*threads_per_block*totalPrintSizePerThread));
  // Calculate expected lines of print and file size.
  size_t actualFileSize = 0;
  size_t expectedFileSize =
  (size_t)num_blocks*threads_per_block*iterCount*totalPrintSizePerThread;
  uint32_t totalActualLines = 0;
  uint32_t totalExpectedLines = iterCount*num_blocks*threads_per_block;
  // DO NOT PUT ANY PRINTF WITHIN THIS BLOCK OF CODE
  {
    CaptureStream captured(stdout);
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    for (int count = 0; count < (const int)iterCount; count++) {
      HIP_CHECK(hipLaunchKernel((const void*)kernel_shared_mem,
               dim3(num_blocks, 1, 1), dim3(threads_per_block, 1, 1),
               NULL, 0, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
    std::ifstream CapturedData = captured.getCapturedData();
    char *buffer = new char[CHUNK_SIZE];
    while (CapturedData.good()) {
      CapturedData.getline(buffer, CHUNK_SIZE);
      totalActualLines++;
    }
    delete[] buffer;
    struct stat st;
    if (stat(captured.getTempFilename(), &st)) {
      printf("Temp File not found \n");
      return false;
    }
    actualFileSize = st.st_size;
  }
  printf("totalExpectedLines = %u \n", totalExpectedLines);
  printf("totalActualLines = %u \n", totalActualLines-1);
  printf("expectedFileSize = %zu \n", expectedFileSize);
  printf("actualFileSize = %zu \n", actualFileSize);
  if ((totalExpectedLines != (totalActualLines - 1))||
      (expectedFileSize != actualFileSize)) {
    return false;
  }
  return true;
}
// Launches kernel_synchronized_printf to generate the printf log file
// and validates the generated file size and number of printed lines
// with the calculated file size and lines.
bool test_synchronized_printf(uint32_t num_blocks,
                              uint32_t threads_per_block,
                              uint32_t print_limit) {
  // Calculate the number of iterations from print_limit
  size_t stress_limit_bytes = ((size_t)print_limit*1024*1024*1024);
  char buffer0[CHUNK_SIZE], buffer1[CHUNK_SIZE], buffer2[CHUNK_SIZE];
  int totalPrintSizePerThread = snprintf(buffer0, CHUNK_SIZE,
                              "%s%u\n", CONST_STR3, 0);
  totalPrintSizePerThread += snprintf(buffer1, CHUNK_SIZE,
                              "%s%u\n", CONST_STR3, 1);
  totalPrintSizePerThread += snprintf(buffer2, CHUNK_SIZE,
                              "%s%u\n", CONST_STR3, 2);
  uint32_t iterCount = static_cast<uint32_t>(1+ stress_limit_bytes/
             (num_blocks*threads_per_block*totalPrintSizePerThread));
  // Calculate expected lines of print and file size.
  size_t actualFileSize = 0;
  size_t expectedFileSize =
  (size_t)num_blocks*threads_per_block*iterCount*totalPrintSizePerThread;
  uint32_t totalActualLines = 0;
  uint32_t totalExpectedLines = 3*iterCount*num_blocks*threads_per_block;
  bool TestPassed = true;
  size_t len = strlen(buffer0) - 1;
  // DO NOT PUT ANY PRINTF WITHIN THIS BLOCK OF CODE
  {
    CaptureStream captured(stdout);
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    for (int count = 0; count < (const int)iterCount; count++) {
      HIP_CHECK(hipLaunchKernel((const void*)kernel_synchronized_printf,
               dim3(num_blocks, 1, 1), dim3(threads_per_block, 1, 1),
               NULL, 0, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
    std::ifstream CapturedData = captured.getCapturedData();
    char *buffer = new char[CHUNK_SIZE];
    while (CapturedData.good()) {
      CapturedData.getline(buffer, CHUNK_SIZE);
      if (!strcmp(buffer, "")) {
        break;
      }
      if (0 == ((totalActualLines / threads_per_block) % 3)) {
         if (strncmp(buffer, buffer0, len)) {
           TestPassed = false;
           break;
         }
      } else if (1 == ((totalActualLines / threads_per_block) % 3)) {
         if (strncmp(buffer, buffer1, len)) {
           TestPassed = false;
           break;
         }
      } else if (2 == ((totalActualLines / threads_per_block) % 3)) {
         if (strncmp(buffer, buffer2, len)) {
           TestPassed = false;
           break;
         }
      }
      totalActualLines++;
    }
    delete[] buffer;
    struct stat st;
    if (stat(captured.getTempFilename(), &st)) {
      printf("Temp File not found");
      return false;
    }
    actualFileSize = st.st_size;
  }
  printf("totalExpectedLines = %u \n", totalExpectedLines);
  printf("totalActualLines = %u \n", totalActualLines);
  printf("expectedFileSize = %zu \n", expectedFileSize);
  printf("actualFileSize = %zu \n", actualFileSize);
  if ((TestPassed == false)||
      (expectedFileSize != actualFileSize)) {
    return false;
  }
  return true;
}
#endif
}  // namespace hipPrintfStressTest

TEST_CASE("Stress_printf_ConstStr") {
#ifdef __linux__
  printf("Test: Stress_printf_ConstStr\n");
  bool TestPassed = true;
  uint threads_per_block = BLOCK_SIZE;
  uint num_blocks = GRID_SIZE;
  // N provide the print limit
  unsigned int print_limit = 1;  // = 1 GB
  TestPassed =
  hipPrintfStressTest::test_printf_conststr(num_blocks, threads_per_block,
                                           print_limit);
  REQUIRE(TestPassed);
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}

TEST_CASE("Stress_printf_IfElseConditionalStr") {
#ifdef __linux__
  printf("Test: Stress_printf_IfElseConditionalStr\n");
  bool TestPassed = true;
  uint threads_per_block = BLOCK_SIZE;
  uint num_blocks = GRID_SIZE;
  // N provide the print limit
  unsigned int print_limit = 1;  // = 1 GB
  TestPassed =
  hipPrintfStressTest::test_printf_two_conditionalstr(num_blocks,
                                 threads_per_block, print_limit);
  REQUIRE(TestPassed);
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}

TEST_CASE("Stress_printf_IfConditionalStr") {
#ifdef __linux__
  printf("Test: Stress_printf_IfConditionalStr\n");
  bool TestPassed = true;
  uint threads_per_block = BLOCK_SIZE;
  uint num_blocks = GRID_SIZE;
  // N provide the print limit
  unsigned int print_limit = 1;  // = 1 GB
  TestPassed =
  hipPrintfStressTest::test_printf_single_conditionalstr(num_blocks,
                                   threads_per_block, print_limit);
  REQUIRE(TestPassed);
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}

TEST_CASE("Stress_printf_VariableStr") {
#ifdef __linux__
  printf("Test: Stress_printf_VariableStr\n");
  bool TestPassed = true;
  // N provide the print limit
  unsigned int print_limit = 1;  // = 1 GB
  TestPassed = hipPrintfStressTest::test_variable_str(print_limit,
                   hipPrintfStressTest::kernel_printf_variablestr,
                   hipPrintfStressTest::EmpiricalValues1);
  REQUIRE(TestPassed);
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}

TEST_CASE("Stress_printf_DependentCalc") {
#ifdef __linux__
  printf("Test: Stress_printf_DependentCalc\n");
  bool TestPassed = true;
  // N provide the print limit
  unsigned int print_limit = 1;  // = 1 GB
  TestPassed = hipPrintfStressTest::test_variable_str(print_limit,
                       hipPrintfStressTest::kernel_dependent_calc,
                       hipPrintfStressTest::EmpiricalValues2);
  REQUIRE(TestPassed);
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}

TEST_CASE("Stress_printf_DecimalStr") {
#ifdef __linux__
  printf("Test: Stress_printf_DecimalStr\n");
  bool TestPassed = true;
  uint threads_per_block = BLOCK_SIZE;
  uint num_blocks = GRID_SIZE;
  // N provide the print limit
  unsigned int print_limit = 1;  // = 1 GB
  TestPassed = hipPrintfStressTest::test_decimal_str(num_blocks,
                  threads_per_block, print_limit);
  REQUIRE(TestPassed);
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}

TEST_CASE("Stress_printf_SharedMem") {
#ifdef __linux__
  printf("Test: Stress_printf_SharedMem\n");
  bool TestPassed = true;
  uint threads_per_block = BLOCK_SIZE;
  uint num_blocks = GRID_SIZE;
  // N provide the print limit
  unsigned int print_limit = 1;  // = 1 GB
  TestPassed = hipPrintfStressTest::test_shared_mem(num_blocks,
                  threads_per_block, print_limit);
  REQUIRE(TestPassed);
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}

TEST_CASE("Stress_printf_SynchronizedPrintf") {
#ifdef __linux__
  printf("Test: Stress_printf_SynchronizedPrintf\n");
  bool TestPassed = true;
  uint threads_per_block = BLOCK_SIZE;
  // N provide the print limit
  unsigned int print_limit = 1;  // = 1 GB
  TestPassed = hipPrintfStressTest::test_synchronized_printf(1,
                 threads_per_block, print_limit);
  REQUIRE(TestPassed);
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}

TEST_CASE("Stress_printf_AtomicCalc") {
#ifdef __linux__
  printf("Test: Stress_printf_AtomicCalc\n");
  bool TestPassed = true;
  // N provide the print limit
  unsigned int print_limit = 1;  // = 1 GB
  TestPassed = hipPrintfStressTest::test_variable_str(print_limit,
                 hipPrintfStressTest::kernel_dependent_calc_atomic,
                 hipPrintfStressTest::EmpiricalValues2);
  REQUIRE(TestPassed);
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}
