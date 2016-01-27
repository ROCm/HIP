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
#include <iostream>
#include "hip_runtime.h"


#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define WIDTH     32
#define HEIGHT    32

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  8
#define THREADS_PER_BLOCK_Y  8
#define THREADS_PER_BLOCK_Z  1

unsigned int  firstbit_u32(unsigned int a)
{  
   if (a == 0)
      return -1;  
   unsigned int pos = 0;  
   while ((int )a > 0) { 
      a <<= 1; pos++;
   }
   return pos;
}
unsigned int firstbit_s32(int a)
{  
   unsigned int u = a >= 0? a: ~a; // complement negative numbers  
   return firstbit_u32(u);
} 

unsigned int firstbit_u64(unsigned long long int a)
{  
   if (a == 0)
      return -1;  
   unsigned int pos = 0;  
   while ((long long int)a > 0) { 
      a <<= 1; pos++;
   }
   return pos;
}
unsigned int firstbit_s64(long long int a)
{  
	unsigned long long int u = a >= 0? a: ~a; // complement negative numbers  
   return firstbit_u64(u);
} 



__global__ void 
HIP_kernel(hipLaunchParm lp,
             unsigned int* a, unsigned int* b,unsigned int* c, unsigned long long int* d, 
			 unsigned int* e, int* f,unsigned  int* g,  long long int* h, int width, int height) 
  {
 
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

      int i = y * width + x;
      if ( i < (width * height)) {
        a[i] = __clz(b[i]);
	c[i] = __clzll(d[i]);
	e[i] = __clz(f[i]);
	g[i] = __clzll(h[i]);
      }

  }



#if 0
__kernel__ void HIP_kernel(unsigned int* a, unsigned int* b,unsigned int* c, unsigned long long int* d, 
			 unsigned int* e, int* f,unsigned  int* g,  long long int* h, int width, int height)  {

  
  int x = blockDimX * blockIdx.x + threadIdx.x;
  int y = blockDimY * blockIdy.y + threadIdx.y;

  int i = y * width + x;
  if ( i < (width * height)) {
     a[i] = __clz(b[i]);
		c[i] = __clzll(d[i]);
		e[i] = __clz(f[i]);
		g[i] = __clzll(h[i]);
  }
}
#endif

using namespace std;

int main() {
  
  unsigned int* hostA;
  unsigned int* hostB;
  unsigned int* hostC; 
  unsigned long long int* hostD;
  unsigned int* hostE;
  int* hostF;
  unsigned int* hostG; 
  long long int* hostH;

  unsigned int* deviceA;
  unsigned int* deviceB;
  unsigned int* deviceC;
  unsigned long long int* deviceD;
  unsigned int* deviceE;
  int* deviceF;
  unsigned int* deviceG;
  long long int* deviceH;

  hipDeviceProp_t devProp;
  hipDeviceGetProperties(&devProp, 0);
  cout << " System minor " << devProp.minor << endl;
  cout << " System major " << devProp.major << endl;
  cout << " agent prop name " << devProp.name << endl;

  cout << "hip Device prop succeeded " << endl ;


  int i;
  int errors;

  hostA = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostB = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostC = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostD = (unsigned long long int*)malloc(NUM * sizeof(unsigned long long int));
  hostE = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostF = (int*)malloc(NUM * sizeof(int));
  hostG = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostH = (long long int*)malloc(NUM * sizeof(long long int));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = i;
    hostD[i] = 1099511627776+i;
    hostF[i] = -2100+i;
    hostH[i] = 1099511627776+i;
  }
  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(unsigned int)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(unsigned int)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(unsigned int)));
  HIP_ASSERT(hipMalloc((void**)&deviceD, NUM * sizeof(unsigned long long int)));
  HIP_ASSERT(hipMalloc((void**)&deviceE, NUM * sizeof(unsigned int)));
  HIP_ASSERT(hipMalloc((void**)&deviceF, NUM * sizeof(int)));
  HIP_ASSERT(hipMalloc((void**)&deviceG, NUM * sizeof(unsigned int)));
  HIP_ASSERT(hipMalloc((void**)&deviceH, NUM * sizeof(long long int)));
  
  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(unsigned int), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceD, hostD, NUM*sizeof(unsigned long long int), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceF, hostF, NUM*sizeof(int), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceH, hostD, NUM*sizeof(long long int), hipMemcpyHostToDevice));

  hipLaunchKernel(HIP_kernel, 
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  deviceA ,deviceB, deviceC,deviceD ,deviceE ,deviceF, deviceG,deviceH, WIDTH ,HEIGHT);


  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(unsigned int), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(hostC, deviceC, NUM*sizeof(unsigned int), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(hostE, deviceE, NUM*sizeof(unsigned int), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(hostG, deviceG, NUM*sizeof(unsigned int), hipMemcpyDeviceToHost));
  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
	  printf("gpu_clz_u =%d, cpu_clz_u =%d \n",hostA[i],firstbit_u32(hostB[i]));
	  if (hostA[i] != firstbit_u32(hostB[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("__clz_u() for unsigned PASSED!\n");
  }
  errors = 0;
  for (i = 0; i < NUM; i++) {
	  printf("gpu_clzll_u =%d, cpu_clzll_u =%d \n",hostC[i],firstbit_u64(hostD[i]));
	  if (hostC[i] != firstbit_u64(hostD[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("__clzll_u() for unsigned PASSED!\n");
  }
  errors = 0;
  for (i = 0; i < NUM; i++) {
	  printf("gpu_clz_s =%d, cpu_clz_s =%d \n",hostE[i],firstbit_s32(hostF[i]));
	  if (hostE[i] != firstbit_s32(hostF[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("__clz_s() PASSED!\n");
  }
  errors = 0;
  for (i = 0; i < NUM; i++) {
	  printf("gpu_clzll_s =%d, cpu_clzll_s =%d \n",hostG[i],firstbit_s64(hostH[i]));
	  if (hostG[i] != firstbit_s64(hostH[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("__clzll_s() PASSED!\n");
  }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));
  HIP_ASSERT(hipFree(deviceD));
  HIP_ASSERT(hipFree(deviceE));
  HIP_ASSERT(hipFree(deviceF));
  HIP_ASSERT(hipFree(deviceG));
  HIP_ASSERT(hipFree(deviceH));

  free(hostA);
  free(hostB);
  free(hostC);
  free(hostD);
  free(hostE);
  free(hostF);
  free(hostG);
  free(hostH);


  //hipResetDefaultAccelerator();

  //return errors;
}











