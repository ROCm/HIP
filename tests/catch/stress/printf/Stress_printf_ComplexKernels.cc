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

#define MAX_BLOCK_SIZE 523
#define MAX_GRID_SIZE 503
#define CHUNK_SIZE 1024
#define NUM_STREAM 4
#define CONST_WEIGHTING_FACT1 7
#define CONST_WEIGHTING_FACT2 5

namespace hipPrintfStressTest {
struct printInfo {
  uint32_t printSizeinBytes, lineCount;
};

__device__ __host__ struct printInfo startPrint(uint32_t tid,
                    uint32_t iterCount, uint32_t *a, uint32_t *b) {
  uint32_t printSize = 0;
  uint32_t lineCount = 0;
  // The 2nd modulus operand is arbitrarily chosen as 7 below to
  // diversify the printf output as much as possible while also being
  // a prime number. This number is fixed to 7 and should not be changed.
  uint32_t mod = tid % 7;
  // Perform some calculations and print the values.
  uint32_t uiresult;
  int32_t iresult;
  float fresult;
  for (uint32_t count = 0; count < iterCount; count++) {
    if (0 == mod) {
      // Perform Vector Multiplication a(i)*b(i)
      // Print both tid and result
      uiresult = a[tid]*b[tid];
      printSize +=
      printf("tid %u: Value of result=%u or %x\n",
              tid, uiresult, uiresult);
      lineCount++;
    } else if (1 == mod) {
      // Perform Array Addition a(i) + b(i)
      // Print both tid and result
      uiresult = a[tid] + b[tid];
      printSize +=
      printf("tid %u: Value of result=%u or %x \n",
              tid, uiresult, uiresult);
      lineCount++;
    } else if (2 == mod) {
      // Perform Array Subtraction a(i) - b(i)
      // Print both tid and result (as both int, uint)
      iresult = a[tid] - b[tid];
      printSize +=
      printf("tid %u: Value of result=%d or %x\n",
              tid, iresult, iresult);
      lineCount++;
    } else if (3 == mod) {
      // Perform Sum of Squares a(i)*a(i) + b(i)*b(i)
      // Print both tid and result
      uiresult = a[tid]*a[tid] + b[tid]*b[tid];
      printSize +=
      printf("tid %u: Value of result=%u or %x\n",
             tid, uiresult, uiresult);
      lineCount++;
    } else if (4 == mod) {
      // Perform (a(i)*a(i) + b(i)*b(i))/a(i)*b(i)
      // Print both tid and result (in float upto 2 decimal precision)
      fresult = (a[tid]*a[tid] + b[tid]*b[tid])/(a[tid]*b[tid]);
      printSize +=
      printf("tid %u: Value of result[%d] = %.2f or %.2e\n",
             tid, tid, fresult, fresult);
      lineCount++;
    } else if (5 == mod) {
      // Perform  (a(i)*a(i) - b(i)*b(i))/a(i)*b(i)
      // Print both tid and result (in float upto 4 decimal precision)
      fresult = (a[tid]*a[tid] - b[tid]*b[tid])/(a[tid]*b[tid]);
      printSize +=
      printf("tid %u: Value of result[%d] = %.4f or %.4e \n",
             tid, tid, fresult, fresult);
      lineCount++;
    } else if (6 == mod) {
      // Perform  (a(i)*a(i) + b(i)*b(i))/(a(i)*a(i) - b(i)*b(i))
      // Print both tid and result (in float upto 6 decimal precision)
      fresult = (a[tid]*a[tid] + b[tid]*b[tid])/
                (a[tid]*a[tid] - b[tid]*b[tid]);
      printSize +=
      printf("tid %u: Value of result[%d] = %.6f or %.6e \n",
             tid, tid, fresult, fresult);
      lineCount++;
    }
    // Print a random character string of variable size
    // and number.
    const char* msg;
    for (int i = 0; i < 12; i++) {
      int imod = (i % 6);
      if (0 == imod) {
        msg = "jhwehde2hl";
      } else if (1 == imod) {
        msg = "jhwehde2hlmc,prmlsl4";
      } else if (2 == imod) {
        msg = "xkdojdewnd34dMMnl2o4AAdeBEjbX0";
      } else if (3 == imod) {
        msg = "mcropkaA234dmelmfhja44ndalomkfokdMDFK328";
      } else if (4 == imod) {
        msg =
        "udnekc8939MDkdnjj3knsdlmnekdlgJNls328419i905409dfm";
      } else if (5 == imod) {
        msg =
        "lfjweknm4349u34sdlk09j3mAADDSDkeffe575675fdvfLKMWMORMFREKLkl";
      }
      printSize += printf("tid %u: %s imod = %d \n", tid, msg, imod);
      lineCount++;
    }
    // Print a long string with data
    msg =
    "jheku83290dnmnd##9u9BJKHFJLKsMMMMdkejwejjj232indnfdmsnndnsdn****bsXxZz";
    float pi = 3.141592;
    uint32_t unum = 123456789;
    int32_t inum = -123456789;
    printSize +=
    printf("%s,%d,%s,%u,%s,%x,%s,%f,%s,%e\n",
    msg, inum, msg, unum, msg, unum, msg, pi, msg, pi);
    lineCount++;
    // Print different data types using different specifiers
    float fmaxvalue = std::numeric_limits<float>::max();
    float fminvalue = std::numeric_limits<float>::min();
    double dmaxvalue = std::numeric_limits<double>::max();
    double dminvalue = std::numeric_limits<double>::min();
    printSize +=
    printf("%f, %f, %e, %e \n", fmaxvalue, fminvalue, fmaxvalue, fminvalue);
    printSize +=
    printf("%f, %f, %e, %e \n", dmaxvalue, dminvalue, dmaxvalue, dminvalue);
    printSize +=
    printf("%a, %a, %A, %A \n", fmaxvalue, fminvalue, fmaxvalue, fminvalue);
    printSize +=
    printf("%a, %a, %A, %A \n", dmaxvalue, dminvalue, dmaxvalue, dminvalue);
    lineCount+=4;
    size_t size_tmaxvalue = std::numeric_limits<size_t>::max();
    size_t size_tminvalue = std::numeric_limits<size_t>::min();
    long long llmaxvalue = std::numeric_limits<long long>::max();
    long long llminvalue = std::numeric_limits<long long>::min();
    unsigned long long ullmaxvalue =
                     std::numeric_limits<unsigned long long>::max();
    unsigned long long ullminvalue =
                     std::numeric_limits<unsigned long long>::min();
    long lmaxvalue = std::numeric_limits<long>::max();
    long lminvalue = std::numeric_limits<long>::min();
    unsigned long ulmaxvalue = std::numeric_limits<unsigned long>::max();
    unsigned long ulminvalue = std::numeric_limits<unsigned long>::min();
    short smaxvalue = std::numeric_limits<short>::max();
    short sminvalue = std::numeric_limits<short>::min();
    unsigned short usmaxvalue = std::numeric_limits<unsigned short>::max();
    unsigned short usminvalue = std::numeric_limits<unsigned short>::min();
    char cmaxvalue = std::numeric_limits<char>::max();
    char cminvalue = std::numeric_limits<char>::min();
    unsigned char ucmaxvalue = std::numeric_limits<unsigned char>::max();
    unsigned char ucminvalue = std::numeric_limits<unsigned char>::min();
    int32_t imaxvalue = std::numeric_limits<int32_t>::max();
    int32_t iminvalue = std::numeric_limits<int32_t>::min();
    uint32_t uimaxvalue = std::numeric_limits<uint32_t>::max();
    uint32_t uiminvalue = std::numeric_limits<uint32_t>::min();
    printSize +=
    printf("%zu, %zu, %lli, %lli, %llu, %llu, %li, %li, %lu, %lu\n",
          size_tmaxvalue, size_tminvalue, llmaxvalue, llminvalue,
          ullmaxvalue, ullminvalue, lmaxvalue, lminvalue,
          ulmaxvalue, ulminvalue);
    printSize +=
    printf("%zx, %zx, %llx, %llx, %llx, %llx, %lx, %lx, %lx, %lx\n",
          size_tmaxvalue, size_tminvalue, llmaxvalue, llminvalue,
          ullmaxvalue, ullminvalue, lmaxvalue, lminvalue,
          ulmaxvalue, ulminvalue);
    printSize +=
    printf("%zX, %zX, %llX, %llX, %llX, %llX, %lX, %lX, %lX, %lX\n",
          size_tmaxvalue, size_tminvalue, llmaxvalue, llminvalue,
          ullmaxvalue, ullminvalue, lmaxvalue, lminvalue,
          ulmaxvalue, ulminvalue);
    printSize +=
    printf("%zo, %zo, %llo, %llo, %llo, %llo, %lo, %lo, %lo, %lo\n",
          size_tmaxvalue, size_tminvalue, llmaxvalue, llminvalue,
          ullmaxvalue, ullminvalue, lmaxvalue, lminvalue,
          ulmaxvalue, ulminvalue);
    printSize +=
    printf("%hd, %hd, %hu, %hu, %hhd, %hhd, %hhu, %hhu, %d, %d, %u, %u\n",
         smaxvalue, sminvalue, usmaxvalue, usminvalue,
         cmaxvalue, cminvalue, ucmaxvalue, ucminvalue,
         imaxvalue, iminvalue, uimaxvalue, uiminvalue);
    printSize +=
    printf("%hx, %hx, %hx, %hx, %hhx, %hhx, %hhx, %hhx, %x, %x, %x, %x\n",
         smaxvalue, sminvalue, usmaxvalue, usminvalue,
         cmaxvalue, cminvalue, ucmaxvalue, ucminvalue,
         imaxvalue, iminvalue, uimaxvalue, uiminvalue);
    printSize +=
    printf("%hX, %hX, %hX, %hX, %hhX, %hhX, %hhX, %hhX, %X, %X, %X, %X\n",
         smaxvalue, sminvalue, usmaxvalue, usminvalue,
         cmaxvalue, cminvalue, ucmaxvalue, ucminvalue,
         imaxvalue, iminvalue, uimaxvalue, uiminvalue);
    printSize +=
    printf("%ho, %ho, %ho, %ho, %hho, %hho, %hho, %hho, %o, %o, %o, %o\n",
         smaxvalue, sminvalue, usmaxvalue, usminvalue,
         cmaxvalue, cminvalue, ucmaxvalue, ucminvalue,
         imaxvalue, iminvalue, uimaxvalue, uiminvalue);
    printSize +=
    printf("%c, %c, %c, %c\n", cmaxvalue, cminvalue, ucmaxvalue, ucminvalue);
    lineCount+=9;
  }
  struct printInfo pInfo = {printSize, lineCount};
  return pInfo;
}
// This kernel is launched only in X dimension
__global__ void kernel_complex_opX(uint32_t *a, uint32_t *b,
                                  uint32_t iterCount) {
  uint32_t tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  startPrint(tid, iterCount, a, b);
}
// This kernel is launched only in Y dimension
__global__ void kernel_complex_opY(uint32_t *a, uint32_t *b,
                                  uint32_t iterCount) {
  uint32_t tid = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;
  startPrint(tid, iterCount, a, b);
}
// This kernel is launched only in Z dimension
__global__ void kernel_complex_opZ(uint32_t *a, uint32_t *b,
                                  uint32_t iterCount) {
  uint32_t tid = hipThreadIdx_z + hipBlockIdx_z * hipBlockDim_z;
  startPrint(tid, iterCount, a, b);
}
#ifdef __linux__
// Performs printf stress test on a single GPU using multiple streams.
bool test_printf_multistream(uint32_t num_blocks,
                             uint32_t threads_per_block,
                             uint32_t iterCount) {
  uint32_t buffsize = num_blocks*threads_per_block;
  size_t actualFileSize = 0;
  uint32_t totalActualLinecount = 0;
  uint32_t *Ah, *Bh;
  uint32_t *Ad, *Bd;
  Ah = new uint32_t[buffsize];
  Bh = new uint32_t[buffsize];
  for (uint32_t i = 0; i < buffsize; i++) {
    Ah[i] = i + 1;
    Bh[i] = buffsize - i;
  }
  HIP_CHECK(hipMalloc(&Ad, buffsize*sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&Bd, buffsize*sizeof(uint32_t)));
  HIP_CHECK(hipMemcpy(Ad, Ah, buffsize*sizeof(uint32_t),
          hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, Bh, buffsize*sizeof(uint32_t),
           hipMemcpyHostToDevice));
  // DO NOT PUT ANY PRINTF WITHIN THIS BLOCK OF CODE
  {
    CaptureStream captured(stdout);
    hipStream_t stream[NUM_STREAM];
    for (int i = 0; i < NUM_STREAM; i++) {
      HIP_CHECK(hipStreamCreate(&stream[i]));
      hipLaunchKernelGGL(kernel_complex_opX, dim3(num_blocks, 1, 1),
                         dim3(threads_per_block, 1, 1),
                         0, stream[i], Ad, Bd, iterCount);
      hipLaunchKernelGGL(kernel_complex_opY, dim3(1, num_blocks, 1),
                         dim3(1, threads_per_block, 1),
                         0, stream[i], Ad, Bd, iterCount);
      hipLaunchKernelGGL(kernel_complex_opZ, dim3(1, 1, num_blocks),
                         dim3(1, 1, threads_per_block),
                         0, stream[i], Ad, Bd, iterCount);
    }
    HIP_CHECK(hipDeviceSynchronize());
    for (int i = 0; i < NUM_STREAM; i++) {
      HIP_CHECK(hipStreamDestroy(stream[i]));
    }
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
  struct printInfo pInfo;
  size_t estimatedPrintSize = 0;
  // DO NOT PUT ANY PRINTF WITHIN THIS BLOCK OF CODE
  uint32_t lop = 0;
  {
    CaptureStream captured(stdout);
    for (int j = 0; j < NUM_STREAM; j++) {
      for (uint32_t tid = 0; tid < (buffsize); tid++) {
        pInfo = startPrint(tid, iterCount, Ah, Bh);
        lop += pInfo.lineCount;
        estimatedPrintSize += pInfo.printSizeinBytes;
      }
      for (uint32_t tid = 0; tid < (buffsize); tid++) {
        pInfo = startPrint(tid, iterCount, Ah, Bh);
        lop += pInfo.lineCount;
        estimatedPrintSize += pInfo.printSizeinBytes;
      }
      for (uint32_t tid = 0; tid < (buffsize); tid++) {
        pInfo = startPrint(tid, iterCount, Ah, Bh);
        lop += pInfo.lineCount;
        estimatedPrintSize += pInfo.printSizeinBytes;
      }
    }
  }
  printf("estimatedPrintSize = %zu, actualFileSize = %zu\n",
         estimatedPrintSize, actualFileSize);
  printf("estimatedLinesPrinted = %u, actualLinesPrinted = %u\n",
         lop, totalActualLinecount-1);
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Ad));
  delete[] Bh;
  delete[] Ah;
  if ((estimatedPrintSize != actualFileSize)||
     (lop != (totalActualLinecount-1))) {
    return false;
  }
  return true;
}

bool test_printf_multigpu(int gpu,
                          uint32_t num_blocks,
                          uint32_t threads_per_block,
                          uint32_t iterCount,
                          size_t *actualFileSize,
                          uint32_t *totalActualLinecount) {
  uint32_t buffsize = num_blocks*threads_per_block;
  uint32_t *Ah, *Bh;
  uint32_t *Ad, *Bd;
  HIP_CHECK(hipSetDevice(gpu));
  Ah = new uint32_t[buffsize];
  Bh = new uint32_t[buffsize];
  for (uint32_t i = 0; i < buffsize; i++) {
    Ah[i] = i + 1;
    Bh[i] = buffsize - i;
  }
  HIP_CHECK(hipMalloc(&Ad, buffsize*sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&Bd, buffsize*sizeof(uint32_t)));
  HIP_CHECK(hipMemcpy(Ad, Ah, buffsize*sizeof(uint32_t),
           hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, Bh, buffsize*sizeof(uint32_t),
           hipMemcpyHostToDevice));
  // DO NOT PUT ANY PRINTF WITHIN THIS BLOCK OF CODE
  {
    CaptureStream captured(stdout);
    hipLaunchKernelGGL(kernel_complex_opX, dim3(num_blocks, 1, 1),
                       dim3(threads_per_block, 1, 1),
                       0, 0, Ad, Bd, iterCount);
    hipLaunchKernelGGL(kernel_complex_opY, dim3(1, num_blocks, 1),
                       dim3(1, threads_per_block, 1),
                       0, 0, Ad, Bd, iterCount);
    hipLaunchKernelGGL(kernel_complex_opZ, dim3(1, 1, num_blocks),
                       dim3(1, 1, threads_per_block),
                       0, 0, Ad, Bd, iterCount);
    HIP_CHECK(hipDeviceSynchronize());
    std::ifstream CapturedData = captured.getCapturedData();
    char *buffer = new char[CHUNK_SIZE];
    while (CapturedData.good()) {
      CapturedData.getline(buffer, CHUNK_SIZE);
      *totalActualLinecount += 1;
    }
    delete[] buffer;
    struct stat st;
    if (stat(captured.getTempFilename(), &st)) {
      printf("Temp File not found \n");
      return false;
    }
    *actualFileSize += st.st_size;
  }
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Ad));
  delete[] Bh;
  delete[] Ah;
  *totalActualLinecount -= 1;  // Removing Empty Line
  HIP_CHECK(hipSetDevice(0));
  return true;
}

// Performs printf stress test on all GPUs present in the system.
bool testPrintfMultGPU(int numOfGPUs,
                       uint32_t num_blocks,
                       uint32_t threads_per_block,
                       uint32_t iterCount) {
  uint32_t buffsize = num_blocks*threads_per_block;
  size_t actualFileSize = 0;
  uint32_t totalActualLinecount = 0;
  for (int gpu = 0; gpu < numOfGPUs; gpu++) {
    test_printf_multigpu(gpu, num_blocks, threads_per_block,
            iterCount, &actualFileSize, &totalActualLinecount);
  }
  struct printInfo pInfo;
  size_t estimatedPrintSize = 0;
  uint32_t *Ah, *Bh;
  Ah = new uint32_t[buffsize];
  Bh = new uint32_t[buffsize];
  for (uint32_t i = 0; i < buffsize; i++) {
    Ah[i] = i + 1;
    Bh[i] = buffsize - i;
  }
  // DO NOT PUT ANY PRINTF WITHIN THIS BLOCK OF CODE
  uint32_t lop = 0;
  {
    CaptureStream captured(stdout);
    for (int gpu = 0; gpu < numOfGPUs; gpu++) {
      for (uint32_t tid = 0; tid < (buffsize); tid++) {
        pInfo = startPrint(tid, iterCount, Ah, Bh);
        lop += pInfo.lineCount;
        estimatedPrintSize += pInfo.printSizeinBytes;
      }
      for (uint32_t tid = 0; tid < (buffsize); tid++) {
        pInfo = startPrint(tid, iterCount, Ah, Bh);
        lop += pInfo.lineCount;
        estimatedPrintSize += pInfo.printSizeinBytes;
      }
      for (uint32_t tid = 0; tid < (buffsize); tid++) {
        pInfo = startPrint(tid, iterCount, Ah, Bh);
        lop += pInfo.lineCount;
        estimatedPrintSize += pInfo.printSizeinBytes;
      }
    }
  }
  delete[] Bh;
  delete[] Ah;
  printf("estimatedPrintSize = %zu, actualFileSize = %zu\n",
         estimatedPrintSize, actualFileSize);
  printf("estimatedLinesPrinted = %u, actualLinesPrinted = %u\n",
         lop, totalActualLinecount);
  if ((estimatedPrintSize != actualFileSize)||
     (lop != totalActualLinecount)) {
    return false;
  }
  return true;
}
#endif
}  // namespace hipPrintfStressTest

TEST_CASE("Stress_printf_ComplexKernelMultStream") {
#ifdef __linux__
  printf("Test - Stress_printf_ComplexKernelMultStream start\n");
  bool TestPassed = true;
  uint threads_per_block = MAX_BLOCK_SIZE;
  // N provide the print limit
  unsigned int print_limit = 4;  // = 4 GB
  uint32_t iterCount = 1;
  // num_blocks is calculated using an approximate formula to arrive at
  // the required print data quantity. CONST_WEIGHTING_FACT1 and
  // CONST_WEIGHTING_FACT2 are empirically determined.
  uint32_t num_blocks = (MAX_GRID_SIZE*print_limit)/CONST_WEIGHTING_FACT1
                         - (CONST_WEIGHTING_FACT2*print_limit);
  TestPassed =
  hipPrintfStressTest::test_printf_multistream(num_blocks, threads_per_block,
                                              iterCount);
  REQUIRE(TestPassed);
  printf("Test - Stress_printf_ComplexKernelMultStream completed \n");
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}

TEST_CASE("Stress_printf_ComplexKernelMultStreamMultGpu") {
#ifdef __linux__
  printf("Test - Stress_printf_ComplexKernelMultStreamMultGpu start \n");
  bool TestPassed = true;
  uint threads_per_block = MAX_BLOCK_SIZE;
  // N provide the print limit
  unsigned int print_limit = 4;  // = 4 GB
  uint32_t iterCount = 1;
  int numOfGPUs = 0;
  hipGetDeviceCount(&numOfGPUs);
  if (numOfGPUs < 2) {
    printf("Skipping test because numOfGPUs < 2\n");
    return;
  }
  // num_blocks is calculated using an approximate formula to arrive at
  // the required print data quantity. CONST_WEIGHTING_FACT1 and
  // CONST_WEIGHTING_FACT2 are empirically determined.
  uint32_t num_blocks =
  (((MAX_GRID_SIZE*print_limit)/CONST_WEIGHTING_FACT1 -
  (CONST_WEIGHTING_FACT2*print_limit))*4)/numOfGPUs;
  TestPassed =
  hipPrintfStressTest::testPrintfMultGPU(numOfGPUs, num_blocks,
                                         threads_per_block,
                                         iterCount);
  REQUIRE(TestPassed);
  printf("Test - Stress_printf_ComplexKernelMultStreamMultGpu end \n");
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}
