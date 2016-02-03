#include <iostream>

#include <hip_runtime.h>
#define HIP_ASSERT(x) (assert((x)==hipSuccess))

__global__ void 
	gpu_ballot(hipLaunchParm lp, unsigned int* device_ballot, int Num_Warps_per_Block)
{

   int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
   const unsigned int warp_num = hipThreadIdx_x >> 6;
   atomicAdd(&device_ballot[warp_num+hipBlockIdx_x*Num_Warps_per_Block],__popcll(__ballot(tid - 245)));
 
}


int main(int argc, char *argv[])
{

  unsigned int Num_Threads_per_Block      = 512;
  unsigned int Num_Blocks_per_Grid        = 1;
  unsigned int Num_Warps_per_Block        = Num_Threads_per_Block/64;
  unsigned int Num_Warps_per_Grid         = (Num_Threads_per_Block*Num_Blocks_per_Grid)/64;
  unsigned int* host_ballot = (unsigned int*)malloc(Num_Warps_per_Grid*sizeof(unsigned int));
  unsigned int* device_ballot; 
  HIP_ASSERT(hipMalloc((void**)&device_ballot, Num_Warps_per_Grid*sizeof(unsigned int)));

  for (int i=0; i<Num_Warps_per_Grid; i++) host_ballot[i] = 0;

 
  HIP_ASSERT(hipMemcpy(device_ballot, host_ballot, Num_Warps_per_Grid*sizeof(unsigned int), hipMemcpyHostToDevice));

  hipLaunchKernel(gpu_ballot, dim3(Num_Blocks_per_Grid),dim3(Num_Threads_per_Block),0,0, device_ballot,Num_Warps_per_Block);


  HIP_ASSERT(hipMemcpy(host_ballot, device_ballot, Num_Warps_per_Grid*sizeof(unsigned int), hipMemcpyDeviceToHost));
  for (int i=0; i<Num_Warps_per_Grid; i++) {

     if ((host_ballot[i] == 0)||(host_ballot[i]/64 == 64)) std::cout << "Warp " << i << " IS convergent- Predicate true for " << host_ballot[i]/64 << " threads\n";

     else std::cout << "Warp " << i << " IS divergent - Predicate true for " << host_ballot[i]/64<< " threads\n";

}


  return EXIT_SUCCESS;

}
