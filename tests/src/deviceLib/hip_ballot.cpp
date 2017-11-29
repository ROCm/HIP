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

/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/device_functions.h>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

__global__ void
	gpu_ballot(hipLaunchParm lp, unsigned int* device_ballot, int Num_Warps_per_Block,int pshift)
{

   int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
   const unsigned int warp_num = hipThreadIdx_x >> pshift;
#ifdef __HIP_PLATFORM_HCC__
   atomicAdd(&device_ballot[warp_num+hipBlockIdx_x*Num_Warps_per_Block],__popcll(__ballot(tid - 245)));
#else
	atomicAdd(&device_ballot[warp_num+hipBlockIdx_x*Num_Warps_per_Block],__popc(__ballot(tid - 245)));
#endif

}


int main(int argc, char *argv[])
{ int warpSize, pshift;
  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  warpSize = devProp.warpSize;

  int w = warpSize;
  pshift = 0;
  while (w >>= 1) ++pshift;

  unsigned int Num_Threads_per_Block      = 512;
  unsigned int Num_Blocks_per_Grid        = 1;
  unsigned int Num_Warps_per_Block        = Num_Threads_per_Block/warpSize;
  unsigned int Num_Warps_per_Grid         = (Num_Threads_per_Block*Num_Blocks_per_Grid)/warpSize;
  unsigned int* host_ballot = (unsigned int*)malloc(Num_Warps_per_Grid*sizeof(unsigned int));
  unsigned int* device_ballot;
  HIP_ASSERT(hipMalloc((void**)&device_ballot, Num_Warps_per_Grid*sizeof(unsigned int)));
  int divergent_count =0;
  for (int i=0; i<Num_Warps_per_Grid; i++) host_ballot[i] = 0;


  HIP_ASSERT(hipMemcpy(device_ballot, host_ballot, Num_Warps_per_Grid*sizeof(unsigned int), hipMemcpyHostToDevice));

  hipLaunchKernel(gpu_ballot, dim3(Num_Blocks_per_Grid),dim3(Num_Threads_per_Block),0,0, device_ballot,Num_Warps_per_Block,pshift);


  HIP_ASSERT(hipMemcpy(host_ballot, device_ballot, Num_Warps_per_Grid*sizeof(unsigned int), hipMemcpyDeviceToHost));
  for (int i=0; i<Num_Warps_per_Grid; i++) {

     if ((host_ballot[i] == 0)||(host_ballot[i]/warpSize == warpSize)) std::cout << "Warp " << i << " IS convergent- Predicate true for " << host_ballot[i]/warpSize << " threads\n";

     else {std::cout << " Warp " << i << " IS divergent - Predicate true for " << host_ballot[i]/warpSize<< " threads\n";
	  divergent_count++;}
}

if (divergent_count==1) printf("PASSED\n"); else printf("FAILED\n");
  return EXIT_SUCCESS;

}
