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


#include <stdio.h>
#include <iostream>

#include <hip_runtime.h>
#define HIP_ASSERT(x) (assert((x)==hipSuccess))

__global__ void 
	warpvote(hipLaunchParm lp, int* device_any, int* device_all , int Num_Warps_per_Block)
{

   int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
   device_any[hipThreadIdx_x>>6] = __any(tid >77);
   device_all[hipThreadIdx_x>>6] = __all(tid >77);
}



int main(int argc, char *argv[])
{

  int Num_Threads_per_Block      = 1024;
  int Num_Blocks_per_Grid        = 1;
  int Num_Warps_per_Block        = Num_Threads_per_Block/64;
  int Num_Warps_per_Grid         = (Num_Threads_per_Block*Num_Blocks_per_Grid)/64;
  
  int * host_any  = ( int*)malloc(Num_Warps_per_Grid*sizeof(int));
  int * host_all  = ( int*)malloc(Num_Warps_per_Grid*sizeof(int));
  int *device_any; 
  int *device_all;
  HIP_ASSERT(hipMalloc((void**)&device_any,Num_Warps_per_Grid*sizeof( int)));
  HIP_ASSERT(hipMalloc((void**)&device_all,Num_Warps_per_Grid*sizeof(int)));
for (int i=0; i<Num_Warps_per_Grid; i++) 
{
	host_any[i] = 0;
	host_all[i] = 0;
}
  HIP_ASSERT(hipMemcpy(device_any, host_any,sizeof(int), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(device_all, host_all,sizeof(int), hipMemcpyHostToDevice));

  hipLaunchKernel(warpvote, dim3(Num_Blocks_per_Grid),dim3(Num_Threads_per_Block),0,0, device_any, device_all ,Num_Warps_per_Block);


  HIP_ASSERT(hipMemcpy(host_any, device_any, Num_Warps_per_Grid*sizeof(int), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(host_all, device_all, Num_Warps_per_Grid*sizeof(int), hipMemcpyDeviceToHost));
  for (int i=0; i<Num_Warps_per_Grid; i++) {

    printf("warp no. %d __any = %d \n",i,host_any[i]);
    printf("warp no. %d __all = %d \n",i,host_all[i]);


}


  return EXIT_SUCCESS;

}
