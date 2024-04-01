/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <cfloat>
#include <atomic>
#include "./defs.h"

static __device__ int* deviceAlloc(int test_type);
static __device__ void deviceWrite(int myId, int *devmem);
static __device__ void deviceFree(int *outputBuf, int *devmem,
                           int test_type, int myId);

/**
 * Allocation base and derived class to test dynamic allocation.
 */
class baseAlloc{
 public:
  virtual __device__ int* alloc(size_t size) = 0;
  virtual __device__ void free(int* ptr) = 0;
};

class derivedAlloc: public baseAlloc{
 public:
  virtual __device__ int* alloc(size_t size) {
    return new int[size];
  }
  virtual __device__ void free(int* ptr) {
    delete ptr;
  }
};

/**
 * Allocation Structure to test dynamic allocation.
 */
struct deviceAllocFunc{
  int* (*alloc)(int);
  void (*write)(int, int*);
  void (*free)(int*, int*, int, int);
};

/**
 * Simple Structure to test dynamic allocation.
 */
struct simpleStruct{
  int32_t i;
  double d;
  float f;
  int16_t s;
  char c;
  int32_t iarr[INTERNAL_BUFFER_SIZE];
  bool operator!=(const struct simpleStruct &inpStr) {
    if ((i != inpStr.i) || (d != inpStr.d) ||
    (f != inpStr.f) || (s != inpStr.s) || (c != inpStr.c)) {
      return true;
    }
    for (int32_t idx = 0; idx < INTERNAL_BUFFER_SIZE; idx++) {
      if (iarr[idx] != inpStr.iarr[idx]) {
        return true;
      }
    }
    return false;
  }
};

/**
 * Simple Structure containing thread information
 */
struct threadInfo{
  int threadid;
  int blockid;
  int32_t ival;
  double dval;
  float fval;
  int16_t sval;
  char cval;
};

/**
 * C/C++ Union
 */
union testInfoUnion{
  int32_t ival;
  double dval;
  float fval;
  int16_t sval;
  char cval;
};

/**
 * Complex (nested) Structure to test dynamic allocation using malloc.
 */
struct complexStructure{
  struct threadInfo *sthreadInfo;
  __device__ void alloc_internal_members(int test_type, size_t size) {
    sthreadInfo = nullptr;
    if (test_type == TEST_MALLOC_FREE) {
      sthreadInfo = reinterpret_cast<struct threadInfo*>(
                   malloc(size*sizeof(struct threadInfo)));
    } else {
      sthreadInfo = new struct threadInfo[size];
    }
  }

  __device__ void free_internal_members(int test_type) {
    if (test_type == TEST_MALLOC_FREE) {
      free(sthreadInfo);
    } else {
      delete[] sthreadInfo;
    }
    sthreadInfo = nullptr;
  }
};
