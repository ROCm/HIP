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
#ifndef _COMPILER_HIPCLASSKERNEL_H_
#define _COMPILER_HIPCLASSKERNEL_H_

#include <iostream>
#include <string>
#include "hip/hip_runtime.h"
#include "test_common.h"

static const int BLOCKS = 512;
static const int THREADS_PER_BLOCK = 1;
static const int ENABLE_DESTRUCTOR_TEST = 0;
static const int ENABLE_VIRTUAL_TESTS = 0;
static const int ENABLE_FRIEND_TEST = 0;
static const int ENABLE_OVERLAD_OVERRIDE_TESTS = 0;
size_t NBOOL = BLOCKS * sizeof(bool);

#define test_passed(test_name) printf("%s %s  PASSED!%s\n", KGRN, #test_name, KNRM);

#ifdef ENABLE_OVERLOAD_OVERRIDE_TESTS
class testFuncOvld{
  public:
  int __host__ __device__ func1(int a){
    return a + 10;  
  }

  int __host__ __device__ func1(int a , int b){
    return a + b + 10;  
  }

};


class testOvrB{
  public:
  int __host__ __device__ ovrdFunc1(){
    return  10;  
  }


};


class testOvrD: public testOvrB{
  public:
    int __host__ __device__ ovrdFunc1(){
    int x = testOvrB::ovrdFunc1();
     return x + 20;
    }

};
#endif

#ifdef ENABLE_FRIEND_TEST
class testFrndA{
   private:
       int fa = 10;
   public:
      friend class testFrndB;
};
 
class testFrndB{
    public:
      __host__ __device__ int showA(){
         testFrndA x;
          return x.fa;
     }
};
#endif

class testClassEmpty {};

class testPassByValue{
 public:
   int exI;
   char exC;
};

class testSizeA {
 public:
    float xa;
    int ia;
    double da;
    static char ca;
};

class testSizeDerived : testSizeA {
 public:
    float fd;
};

#pragma pack(push,4)
class testSizeDerived2 : testSizeA {
 public:
    float fd;
};
#pragma pack(pop)

class testSizeB {
 public:
    char ab;
    int ib;
    char cb;
};

#ifdef ENBABLE_VIRTUAL_TESTS
class testSizeVirtDer : public virtual testSizeB {
 public: 
    int ivd;
};

class testSizeVirtDer1 : public virtual testSizeB {
 public: 
    int ivd1;
};

class testSizeDerMulti : public testSizeVirtDer, public testSizeVirtDer1 {
 public: 
    int ivd2;
};

#pragma pack(push,4)
class testSizeVirtDerPack : public virtual testSizeB {
 public: 
    int ivd;
};
#pragma pack(pop)
#endif 

class testSizeC {
 public:
    char ac;
    int ic;
    int bc[2];
};

#ifdef ENABLE_VIRTUAL_TESTS
class testSizeDV {
 public:
    virtual void __host__ __device__ func1();
 private:
    int iDV;

};

class testSizeDerivedDV : testSizeDV {
 public:
    virtual void __host__ __device__ funcD1();
 private:
    int iDDV;
};
#endif

#pragma pack(push, 1)
class testSizeP1 {
 public:
    char ap;
    int ip;
    char cp;
};
#pragma pack(pop)

#pragma pack(push, 1) 
class testSizeP2 {
 public:
    char ap1;
    int ip1;
    int bp1[2];
};
#pragma pack(pop)

#pragma pack(push, 2)
class testSizeP3 {
 public:
    char ap2;
    int ip2;
    char cp2;
};
#pragma pack(pop)

#ifdef ENABLE_DESTRUCTOR_TEST
class testDeviceClass {
 public:
    int iVar;
    __host__ __device__ testDeviceClass();
    __host__ __device__ testDeviceClass(int a);
    __host__ __device__ ~testDeviceClass();
};

__host__ __device__
testDeviceClass::testDeviceClass() {
  iVar = 5;
}

__host__ __device__
testDeviceClass::testDeviceClass(int a) {
  iVar = a;
}
#endif

#endif  // _HIPCLASSKERNEL_H_

class HipClassTests{ 
  public:
    void TestForEmptyClass(void);
    void TestForClassBSize(void);
    void TestForClassSize(void);
    void TestForVirtualClassSize(void);
    void TestForPassByValue(void);
    void TestForMallocPassByValue(void);
    void TestForConsrtDesrt(void);
    void TestForOverload(void);
    void TestForOverride(void);
 
    bool* AllocateHostMemory(void);
    bool* AllocateDeviceMemory(void);
    void VerifyResult(bool* result_ech, bool* result_ecd);
    void FreeMem(bool* result_ech, bool* result_ecd);
};
