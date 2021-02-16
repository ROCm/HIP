#include <hip/hip_runtime.h>
#include <math.h>
#include "test_common.h"

//Enable test when compiler support is available in mainline
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM amd
 * HIT_END
 */
#define N 1048576
__managed__ float A[N];   // Accessible by ALL CPU and GPU functions !!!
__managed__ float B[N];
__managed__  int  x = 0;

__global__ void add()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
        B[i] = A[i] + B[i];
}

__global__ void GPU_func() {
  x++;
}

bool managedSingleGPUTest() {
   bool testResult = true;

   for (int i = 0; i < N; i++) {
      A[i] = 1.0f;
      B[i] = 2.0f;
   }

   int blockSize = 256;
   int numBlocks = (N + blockSize - 1) / blockSize;
   dim3 dimGrid(numBlocks, 1, 1);
   dim3 dimBlock(blockSize, 1, 1);
   hipLaunchKernelGGL(add, dimGrid, dimBlock, 0, 0);

   hipDeviceSynchronize();

   float maxError = 0.0f;
   for (int i = 0; i < N; i++)
      maxError = fmax(maxError, fabs(B[i]-3.0f));

   if(maxError == 0.0f) {
      return true;
   }
   return false;
}

bool managedMultiGPUTest() {
   int numDevices = 0;
   hipGetDeviceCount(&numDevices);

   for (int i = 0; i < numDevices; i++) {
      hipSetDevice(i);
      GPU_func<<< 1, 1 >>>( );
      hipDeviceSynchronize();
   }
   if(x == numDevices) {
      return true;
   }
   return false;
}

int main(int argc, char *argv[]) {
   bool testStatus = true, OverAllStatus = true;
   testStatus = managedSingleGPUTest();
   if (!testStatus) {
      printf("managed keyword Single GPU Test failed\n");
      OverAllStatus = false;
   }
   testStatus = managedMultiGPUTest();
   if (!testStatus) {
      printf("managed keyword Multi GPU Test failed\n");
      OverAllStatus = false;
   }
   if (!OverAllStatus) {
      failed("");
   }
   passed();
}
