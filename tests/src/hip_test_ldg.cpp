/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip_runtime.h"
#include "test_common.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define WIDTH     8
#define HEIGHT    8

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  8
#define THREADS_PER_BLOCK_Y  8
#define THREADS_PER_BLOCK_Z  1

template<typename T>
__global__ void 
vectoradd_float(hipLaunchParm lp,
             T* a, const T*  bm, const T* cm, int width, int height) 

  {
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

      int i = y * width + x;
      if ( i < (width * height)) {
        a[i] = __ldg(&bm[i]) + __ldg(&cm[i]);
      }



  }

#if 0
__kernel__ void vectoradd_float(float* a, const float* b, const float* c, int width, int height) {

  
  int x = blockDimX * blockIdx.x + threadIdx.x;
  int y = blockDimY * blockIdy.y + threadIdx.y;

  int i = y * width + x;
  if ( i < (width * height)) {
    a[i] = b[i] + c[i];
  }
}
#endif

using namespace std;

template<typename T>
bool dataTypesRun(){
  T* hostA;
  T* hostB;
  T* hostC;

  T* deviceA;
  T* deviceB;
  T* deviceC;

  int i;
  int errors;

  hostA = (T*)malloc(NUM * sizeof(T));
  hostB = (T*)malloc(NUM * sizeof(T));
  hostC = (T*)malloc(NUM * sizeof(T));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (T)i;
    hostC[i] = (T)i;
  }
  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(T)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(T)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(T)));
  
  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(T), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM*sizeof(T), hipMemcpyHostToDevice));


  hipLaunchKernel(vectoradd_float, 
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  deviceA ,deviceB ,deviceC ,WIDTH ,HEIGHT);


  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(T), hipMemcpyDeviceToHost));

  bool ret = false;
  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
    ret = false;
  } else {
      ret = true;
  }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  return ret;

}


int main() {
  
  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  cout << " System minor " << devProp.minor << endl;
  cout << " System major " << devProp.major << endl;
  cout << " agent prop name " << devProp.name << endl;

    int errors;

    errors =    dataTypesRun<char>() &
                dataTypesRun<char1>() &
                dataTypesRun<char2>() &
                dataTypesRun<char3>() &
                dataTypesRun<char4>() &
                dataTypesRun<signed char>() &
                dataTypesRun<unsigned char>();

    if(errors == 1){
        errors = 0;
    }else{
        std::cout<<"Failed Char"<<std::endl;
        return -1;
    }

    errors =    dataTypesRun<short>() &
                dataTypesRun<short1>() &
                dataTypesRun<short2>() &
                dataTypesRun<short3>() &
                dataTypesRun<short4>() &
                dataTypesRun<unsigned short>();

    if(errors == 1){
        errors = 0;
    }else{
        std::cout<<"Failed Short"<<std::endl;
        return -1;
    }

    errors =    dataTypesRun<int>() &
                dataTypesRun<int1>() &
                dataTypesRun<int2>() &
                dataTypesRun<int3>() &
                dataTypesRun<int4>() &
                dataTypesRun<unsigned int>();

    if(errors == 1){
        errors = 0;
    }else{
        std::cout<<"Failed Int"<<std::endl;
        return -1;
    }

    errors =    dataTypesRun<long>() &
                dataTypesRun<long1>() &
                dataTypesRun<long2>() &
                dataTypesRun<long3>() &
                dataTypesRun<long4>() &
                dataTypesRun<unsigned long>();

    if(errors == 1){
        errors = 0;
    }else{
        std::cout<<"Failed Long"<<std::endl;
        return -1;
    }

    errors =    dataTypesRun<long long>() &
                dataTypesRun<longlong1>() &
                dataTypesRun<longlong2>() &
                dataTypesRun<longlong3>() &
                dataTypesRun<longlong4>() &
                dataTypesRun<unsigned long long>();

    if(errors == 1){
        errors = 0;
    }else{
        std::cout<<"Failed Long Long"<<std::endl;
        return -1;
    }

    errors =    dataTypesRun<uchar1>() &
                dataTypesRun<uchar2>() &
                dataTypesRun<uchar3>() &
                dataTypesRun<uchar4>();

    if(errors == 1){
        errors = 0;
    }else{
        std::cout<<"Failed Unsigned Char"<<std::endl;
        return -1;
    }

    errors =    dataTypesRun<ushort1>() &
                dataTypesRun<ushort2>() &
                dataTypesRun<ushort3>() &
                dataTypesRun<ushort4>();

    if(errors == 1){
        errors = 0;
    }else{
        std::cout<<"Failed Unsigned Short"<<std::endl;
        return -1;
    }

    errors =    dataTypesRun<uint1>() &
                dataTypesRun<uint2>() &
                dataTypesRun<uint3>() &
                dataTypesRun<uint4>();

    if(errors == 1){
        errors = 0;
    }else{
        std::cout<<"Failed Unsigned Int"<<std::endl;
        return -1;
    }

    errors =    dataTypesRun<ulonglong1>() &
                dataTypesRun<ulonglong2>() &
                dataTypesRun<ulonglong3>() &
                dataTypesRun<ulonglong4>();

    if(errors == 1){
        errors = 0;
    }else{
        std::cout<<"Failed Unsigned Long Long"<<std::endl;
        return -1;
    }

    errors =    dataTypesRun<float>() &
                dataTypesRun<float1>() &
                dataTypesRun<float2>() &
                dataTypesRun<float3>() &
                dataTypesRun<float4>();

    if(errors == 1){
        errors = 0;
    }else{
        std::cout<<"Failed Float"<<std::endl;
        return -1;
    }

    errors =    dataTypesRun<double>() &
                dataTypesRun<double1>() &
                dataTypesRun<double2>() &
                dataTypesRun<double3>() &
                dataTypesRun<double4>();

  //hipResetDefaultAccelerator();
    if(errors == 1){
        passed();
    }else{
        std::cout<<"Failed Float"<<std::endl;
        return -1;
    }
}
