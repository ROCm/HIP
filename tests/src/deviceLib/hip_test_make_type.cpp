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
#include "hip/hip_runtime.h"
#include "test_common.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define WIDTH     8
#define HEIGHT    8

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  8
#define THREADS_PER_BLOCK_Y  8
#define THREADS_PER_BLOCK_Z  1


__global__ void 
vectoradd_char1(hipLaunchParm lp,
             char1* a, const char1*  bm, const char1* cm, int width, int height) 

  {
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

      int i = y * width + x;
      if ( i < (width * height)) {
        a[i] = make_char1(bm[i].x) + make_char1(cm[i].x);
      }
  }

__global__ void 
vectoradd_char2(hipLaunchParm lp,
             char2* a, const char2*  bm, const char2* cm, int width, int height) 

  {
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

      int i = y * width + x;
      if ( i < (width * height)) {
        a[i] = make_char2(bm[i].x, bm[i].y) + make_char2(cm[i].x, cm[i].y);
      }
} 

__global__ void 
vectoradd_char3(hipLaunchParm lp,
             char3* a, const char3*  bm, const char3* cm, int width, int height) 

  {
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

      int i = y * width + x;
      if ( i < (width * height)) {
        a[i] = make_char3(bm[i].x, bm[i].y, bm[i].z) + make_char3(cm[i].x, cm[i].y, cm[i].z);
      }
}
__global__ void 
vectoradd_char4(hipLaunchParm lp,
             char4* a, const char4*  bm, const char4* cm, int width, int height) 

  {
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

      int i = y * width + x;
      if ( i < (width * height)) {
        a[i] = make_char4(bm[i].x, bm[i].y, bm[i].z, bm[i].w) + make_char4(cm[i].x, cm[i].y, cm[i].z, cm[i].w);
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

  hipLaunchKernel(HIP_KERNEL_NAME(vectoradd_char1), 
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

  hipLaunchKernel(HIP_KERNEL_NAME(vectoradd_char1), 
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

  hipLaunchKernel(HIP_KERNEL_NAME(vectoradd_char1), 
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

bool dataTypesRunChar4(){
  char4* hostA;
  char4* hostB;
  char4* hostC;

  char4* deviceA;
  char4* deviceB;
  char4* deviceC;

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

  hipLaunchKernel(HIP_KERNEL_NAME(vectoradd_char1), 
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

    errors =    dataTypesRun<char1>() &
                dataTypesRun<char2>() &
                dataTypesRun<char3>() &
                dataTypesRun<char4>();


  //hipResetDefaultAccelerator();
    if(errors == 1){
        passed();
    }else{
        std::cout<<"Failed Float"<<std::endl;
        return -1;
    }
}
