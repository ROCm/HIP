#include <iostream>

#include <hip_runtime.h>
#define HIP_ASSERT(x) (assert((x)==hipSuccess))

__global__ void 
	gpu_ballot(hipLaunchParm lp, unsigned int* device_ballot, int Num_Warps_per_Block,int pshift)
{

   int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
   const unsigned int warp_num = hipThreadIdx_x >> pshift;
   if (pshift ==6) {atomicAdd(&device_ballot[warp_num+hipBlockIdx_x*Num_Warps_per_Block],__popcll(__ballot(tid - 245)));}
	else {atomicAdd(&device_ballot[warp_num+hipBlockIdx_x*Num_Warps_per_Block],__popc(__ballot(tid - 245)));}
 
}


int main(int argc, char *argv[])
{ int warpSize, pshift;
  hipDeviceProp_t devProp;
  hipDeviceGetProperties(&devProp, 0);
  
  if(strncmp(devProp.name,"Fiji",1)==0)  
  {warpSize = 64; pshift =6;}
  else {warpSize =32; pshift =5;}
  
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
