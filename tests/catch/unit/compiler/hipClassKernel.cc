/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
 * TEST: %t
 * HIT_END
 */
#include "hipClassKernel.h"

__global__ void
ovrdClassKernel(bool* result_ecd){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    testOvrD tobj1;
     result_ecd[tid] = (tobj1.ovrdFunc1() == 30);
}

__global__ void
ovldClassKernel(bool* result_ecd){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    testFuncOvld tfo1;
     result_ecd[tid] = (tfo1.func1(10) == 20)
                       && (tfo1.func1(10,10) == 30);
}

TEST_CASE("Unit_hipClassKernel_Overload_Override") {
  bool *result_ecd, *result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

  hipLaunchKernelGGL(ovrdClassKernel,
                     dim3(BLOCKS),
                     dim3(THREADS_PER_BLOCK),
                     0,
                     0,
                     result_ecd);

  VerifyResult(result_ech,result_ecd);
  FreeMem(result_ech,result_ecd);

  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

  hipLaunchKernelGGL(ovldClassKernel,
                     dim3(BLOCKS),
                     dim3(THREADS_PER_BLOCK),
                     0,
                     0,
                     result_ecd);

  VerifyResult(result_ech,result_ecd);
  FreeMem(result_ech,result_ecd);
}

// check for friend
__global__ void
friendClassKernel(bool* result_ecd){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    testFrndB tfb1;
     result_ecd[tid] = (tfb1.showA() == 10);
}

TEST_CASE("Unit_hipClassKernel_Friend") {
  bool *result_ecd;
  result_ecd = AllocateDeviceMemory();
  hipLaunchKernelGGL(friendClassKernel,
                      dim3(BLOCKS),
                      dim3(THREADS_PER_BLOCK),
                      0,
                      0,
                      result_ecd);
}

// check sizeof empty class is 1
__global__ void
emptyClassKernel(bool* result_ecd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  testClassEmpty ob1,ob2;
  result_ecd[tid] = (sizeof(testClassEmpty) == 1)
                     && (&ob1 != &ob2);
}

TEST_CASE("Unit_hipClassKernel_Empty") {
  bool *result_ecd, *result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

  hipLaunchKernelGGL(emptyClassKernel,
                     dim3(BLOCKS),
                     dim3(THREADS_PER_BLOCK),
                     0,
                     0,
                     result_ecd);

  VerifyResult(result_ech,result_ecd);
  FreeMem(result_ech,result_ecd);
}

// tests for classes >8 bytes
__global__ void
 sizeClassBKernel(bool* result_ecd) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   result_ecd[tid] = (sizeof(testSizeB) == 12)
                      && (sizeof(testSizeC) == 16)
                      && (sizeof(testSizeP1) == 6)
                      && (sizeof(testSizeP2) == 13)
                      && (sizeof(testSizeP3) == 8);
}

TEST_CASE("Unit_hipClassKernel_BSize") {
  bool *result_ecd, *result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

   hipLaunchKernelGGL(sizeClassBKernel,
                      dim3(BLOCKS),
                      dim3(THREADS_PER_BLOCK),
                      0,
                      0,
                      result_ecd);

  VerifyResult(result_ech,result_ecd);
  FreeMem(result_ech,result_ecd);
}

__global__ void
sizeClassKernel(bool* result_ecd) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   result_ecd[tid] = (sizeof(testSizeA) == 16)
                     && (sizeof(testSizeDerived) == 24)
                     && (sizeof(testSizeDerived2) == 20);
}

TEST_CASE("Unit_hipClassKernel_Size") {
    bool *result_ecd, *result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

   hipLaunchKernelGGL(sizeClassKernel,
                      dim3(BLOCKS),
                      dim3(THREADS_PER_BLOCK),
                      0,
                      0,
                      result_ecd);

  VerifyResult(result_ech,result_ecd);
  FreeMem(result_ech,result_ecd);
}

__global__ void
 sizeVirtualClassKernel(bool* result_ecd) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   result_ecd[tid] = (sizeof(testSizeDV) == 16)
                      && (sizeof(testSizeDerivedDV) == 16)
                      && (sizeof(testSizeVirtDerPack) == 24)
                      && (sizeof(testSizeVirtDer) == 24)
                      && (sizeof(testSizeDerMulti) == 48) ;
 }

TEST_CASE("Unit_hipClassKernel_Virtual") {
  bool *result_ecd, *result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

  hipLaunchKernelGGL(sizeVirtualClassKernel,
                      dim3(BLOCKS),
                      dim3(THREADS_PER_BLOCK),
                      0,
                      0,
                      result_ecd);

  VerifyResult(result_ech,result_ecd);
  FreeMem(result_ech,result_ecd);
}

// check pass by value
__global__ void
passByValueKernel(testPassByValue obj, bool* result_ecd) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   result_ecd[tid] = (obj.exI == 10)
                      && (obj.exC == 'C');
}

TEST_CASE("Unit_hipClassKernel_Value") {
  bool *result_ecd,*result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

   testPassByValue exObj;
   exObj.exI = 10;
   exObj.exC = 'C';
   hipLaunchKernelGGL(passByValueKernel,
                      dim3(BLOCKS),
                      dim3(THREADS_PER_BLOCK),
                      0,
                      0,
                      exObj,
                      result_ecd);

  VerifyResult(result_ech,result_ecd);
  FreeMem(result_ech,result_ecd);
}