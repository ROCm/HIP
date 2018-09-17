/*
Copyright (c) 2015-Present Advanced Micro Devices, Inc. All rights reserved.

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
#include "hipClassKernel.h"

#ifdef ENABLE_OVERLOAD_OVERRIDE_TESTS
__global__ void
ovrdClassKernel(bool* result_ecd){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    testOvrD tobj1;
     result_ecd[tid] = (tobj1.ovrdFunc1() == 30);
}

void HipClassTests::TestForOverride(void){
  bool *result_ecd, *result_ech;
  result_ech = HipClassTests::AllocateHostMemory();
  result_ecd = HipClassTests::AllocateDeviceMemory();

  hipLaunchKernelGGL(ovrdClassKernel,
                     dim3(BLOCKS),
                     dim3(THREADS_PER_BLOCK),
                     0,
                     0,
                     result_ecd);
  
  HipClassTests::VerifyResult(result_ech,result_ecd);
  HipClassTests::FreeMem(result_ech,result_ecd);
}


__global__ void
ovldClassKernel(bool* result_ecd){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    testFuncOvld tfo1;
     result_ecd[tid] = (tfo1.func1(10) == 20)
                       && (tfo1.func1(10,10) == 30);
}

void HipClassTests::TestForOverload(void){
  bool *result_ecd, *result_ech;
  result_ech = HipClassTests::AllocateHostMemory();
  result_ecd = HipClassTests::AllocateDeviceMemory();

  hipLaunchKernelGGL(ovldClassKernel,
                     dim3(BLOCKS),
                     dim3(THREADS_PER_BLOCK),
                     0,
                     0,
                     result_ecd);
  
  HipClassTests::VerifyResult(result_ech,result_ecd);
  HipClassTests::FreeMem(result_ech,result_ecd);
}
#endif

#ifdef ENABLE_FRIEND_TEST 
// check for friend
__global__ void
friendClassKernel(bool* result_ecd){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    testFrndB tfb1;
     result_ecd[tid] = (tfb1.showA() == 10);
}
#endif

// check sizeof empty class is 1
__global__ void
emptyClassKernel(bool* result_ecd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  testClassEmpty ob1,ob2;
  result_ecd[tid] = (sizeof(testClassEmpty) == 1)
                     && (&ob1 != &ob2);
}

void HipClassTests::TestForEmptyClass(void){
  bool *result_ecd, *result_ech;
  result_ech = HipClassTests::AllocateHostMemory();
  result_ecd = HipClassTests::AllocateDeviceMemory();

  hipLaunchKernelGGL(emptyClassKernel,
                     dim3(BLOCKS),
                     dim3(THREADS_PER_BLOCK),
                     0,
                     0,
                     result_ecd);
  
  HipClassTests::VerifyResult(result_ech,result_ecd);
  HipClassTests::FreeMem(result_ech,result_ecd);
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

void HipClassTests::TestForClassBSize(void){
  bool *result_ecd, *result_ech;
  result_ech = HipClassTests::AllocateHostMemory();
  result_ecd = HipClassTests::AllocateDeviceMemory();

   hipLaunchKernelGGL(sizeClassBKernel,
                      dim3(BLOCKS),
                      dim3(THREADS_PER_BLOCK),
                      0,
                      0,
                      result_ecd);

  HipClassTests::VerifyResult(result_ech,result_ecd);
  HipClassTests::FreeMem(result_ech,result_ecd);
}

__global__ void
sizeClassKernel(bool* result_ecd) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   result_ecd[tid] = (sizeof(testSizeA) == 16)
                     && (sizeof(testSizeDerived) == 24)
                     && (sizeof(testSizeDerived2) == 20);
}

void HipClassTests::TestForClassSize(void){
    bool *result_ecd, *result_ech;
  result_ech = HipClassTests::AllocateHostMemory();
  result_ecd = HipClassTests::AllocateDeviceMemory();

   hipLaunchKernelGGL(sizeClassKernel,
                      dim3(BLOCKS),
                      dim3(THREADS_PER_BLOCK),
                      0,
                      0,
                      result_ecd);
  
  HipClassTests::VerifyResult(result_ech,result_ecd);
  HipClassTests::FreeMem(result_ech,result_ecd);
}

#ifdef ENABLE_VIRTUAL_TESTS
__global__ void
 sizeVirtualClassKernel(bool* result_ecd) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   result_ecd[tid] = (sizeof(testSizeDV) == 16)
                      && (sizeof(testSizeDerivedDV) == 16)
                      && (sizeof(testSizeVirtDerPack) == 24)
                      && (sizeof(testSizeVirtDer) == 24)
                      && (sizeof(testSizeDerMulti) == 48) ;
 }

void HipClassTests::TestForVirtualClassSize(void){
  bool *result_ecd, *result_ech;
  result_ech = HipClassTests::AllocateHostMemory();
  result_ecd = HipClassTests::AllocateDeviceMemory();

  hipLaunchKernelGGL(sizeVirtualClassKernel,
                      dim3(BLOCKS),
                      dim3(THREADS_PER_BLOCK),
                      0,
                      0,
                      result_ecd);

  HipClassTests::VerifyResult(result_ech,result_ecd);
  HipClassTests::FreeMem(result_ech,result_ecd);
}
#endif

// check pass by value
__global__ void
passByValueKernel(testPassByValue obj, bool* result_ecd) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   result_ecd[tid] = (obj.exI == 10)
                      && (obj.exC == 'C');
}

void HipClassTests::TestForPassByValue(void){
  bool *result_ecd,*result_ech;
  result_ech = HipClassTests::AllocateHostMemory();
  result_ecd = HipClassTests::AllocateDeviceMemory();

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

  HipClassTests::VerifyResult(result_ech,result_ecd);
  HipClassTests::FreeMem(result_ech,result_ecd);
}
 
 // check obj created with hipMalloc
__global__ void
mallocObjKernel(testPassByValue *obj, bool* result_ecd) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   result_ecd[tid] = (obj->exI == 100)
                     && (obj->exC == 'C');
}

void HipClassTests::TestForMallocPassByValue(void){
  bool *result_ecd,*result_ech;
  result_ech = HipClassTests::AllocateHostMemory();
  result_ecd = HipClassTests::AllocateDeviceMemory();


   testPassByValue *exObjM;
   HIPCHECK(hipMalloc(&exObjM, sizeof(testPassByValue)));
   exObjM->exI = 100;
   exObjM->exC = 'C';
   hipLaunchKernelGGL(mallocObjKernel,
                      dim3(BLOCKS),
                      dim3(THREADS_PER_BLOCK),
                      0,
                      0,
                      exObjM,
                      result_ecd);

  HipClassTests::VerifyResult(result_ech,result_ecd);
  HipClassTests::FreeMem(result_ech,result_ecd);

}

// check if constr and destr are accessible from kernel
#ifdef ENABLE_DESTRUCTOR_TEST
__global__ void
testDeviceClassKernel() {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   testDeviceClass ob1;
   testDeviceClass ob2;
   ob2.iVar = 10;
}

void HipClassTests::TestForConsrtDesrt(){
  testDeviceClass tDC;
  hipLaunchKernelGGL(testDeviceClassKernel,
                      dim3(BLOCKS),
                      dim3(THREADS_PER_BLOCK),
                      0,
                      0);
}
#endif

#ifdef ENABLE_FRIEND_TEST
void HipClassTests::TestForFriend(void){
    bool *result_ecd, *result_ech;
    result_ech = HipClassTests::AllocateHostMemory();
    result_ecd = HipClassTests::AllocateDeviceMemory();
    hipLaunchKernelGGL(friendClassKernel,
                       dim3(BLOCKS),
                       dim3(THREADS_PER_BLOCK),
                       0,
                       0,
                       result_ecd);
}
#endif

bool* HipClassTests::AllocateHostMemory(void){
  bool *result_ech;
  HIPCHECK(hipHostMalloc(&result_ech,
                         NBOOL,
                         hipHostMallocDefault));
  return result_ech;
}

bool* HipClassTests::AllocateDeviceMemory(void){
  bool* result_ecd; 
  HIPCHECK(hipMalloc(&result_ecd,
                     NBOOL));
  HIPCHECK(hipMemset(result_ecd,
                       false,
                       NBOOL));
  return result_ecd;
}

void HipClassTests::VerifyResult(bool* result_ech, bool* result_ecd){
  HIPCHECK(hipMemcpy(result_ech,
                       result_ecd,
                       BLOCKS*sizeof(bool),
                       hipMemcpyDeviceToHost));
    // validation on host side
  for (int i = 0; i < BLOCKS; i++) {
    HIPASSERT(result_ech[i] == true);
  }
}

void HipClassTests::FreeMem(bool* result_ech, bool* result_ecd){
  HIPCHECK(hipHostFree(result_ech));
  HIPCHECK(hipFree(result_ecd));
}

int main(){
  HipClassTests classTests;
  classTests.TestForEmptyClass();
  test_passed(TestForEmptyClass);
  classTests.TestForClassBSize();
  test_passed(TestForClassBSize);
  classTests.TestForClassSize();
  test_passed(TestForClassSize);
  classTests.TestForPassByValue();
  test_passed(TestForPassByValue);

#ifdef ENABLE_OVERLOAD_OVERRIDE_TESTS
  classTests.TestForOverload();
  test_passed(TestForOverload);
  classTests.TestForOverride();
  test_passed(TestForOverride);
#endif

#ifdef ENABLE_FRIEND_TEST
  classTests.TestForFriend();
  test_passed(TestForFriend);
#endif

//  classTests.TestForMallocPassByValue();
 // test_passed(TestForMallocPassByValue); #this test is crashing

#ifdef ENABLE_VIRTUAL_TESTS
  classTests.TestForVirtualClassSize();
  test_passed(TestForVirtualClassSize);
#endif

#ifdef ENABLE_DESTRUCTOR_TEST
  classTests.TestForConsrtDesrt();
  test_passed(TestForConsrtDesrt);
#endif 
}
