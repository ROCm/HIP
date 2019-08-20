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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD_CMD: vcpy_kernel.code %hc --genco %S/vcpy_kernel.cpp -o vcpy_kernel.code
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <iostream>

#define NUM 1000000
const unsigned threadsperblock = 32;
const unsigned blocks = (NUM/threadsperblock)+1;

uint32_t gridSize = 0;
uint32_t blockSize = 0;


#define HIP_CHECK(status)                                                                          \
    if (status != hipSuccess) {                                                                    \
        std::cout << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;            \
        exit(0);                                                                                   \
    }

// Device (Kernel) function
__global__ void multiply(float* C, float* A, float* B, int N){
      
    int tx = hipBlockDim_x*hipBlockIdx_x+hipThreadIdx_x;
    
    if (tx < N){
        C[tx] = A[tx] * B[tx];
    }
}
// CPU implementation
void multiplyCPU(float* C, float* A, float* B, int N){
    
    for(unsigned int i=0; i<N; i++){     
        C[i] = A[i] * B[i];      
    }

}

void launchKernel(float* C, float* A, float* B, int N, bool manual){
     
     hipDeviceProp_t devProp;
     HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

     hipEvent_t start, stop;
     HIP_CHECK(hipEventCreate(&start));
     HIP_CHECK(hipEventCreate(&stop));
     float eventMs = 1.0f;
     int activeWarps, maxWarps;
      
     if (manual){
      	blockSize = threadsperblock; 
     	gridSize  = blocks;
        std::cout << std::endl << "Manual Configuration with block size " << blockSize << std::endl;
     }
     else{
     	hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, multiply, 0, 0);
	std::cout << std::endl << "Automatic Configuation based on hipOccupancyMaxPotentialBlockSize " << std::endl;
        std::cout << "Suggested blocksize is " << blockSize << ", Minimum gridsize is " << gridSize << std::endl; 
     }

     // Record the start event
     HIP_CHECK(hipEventRecord(start, NULL));  

     // Launching the Kernel from Host
     hipLaunchKernelGGL(multiply, dim3(gridSize), dim3(blockSize), 0, 0, C, A, B, NUM);
     
     // Record the stop event
     HIP_CHECK(hipEventRecord(stop, NULL));
     HIP_CHECK(hipEventSynchronize(stop));
         
     HIP_CHECK(hipEventElapsedTime(&eventMs, start, stop));
     printf("kernel Execution time = %6.3fms\n", eventMs);

     //Calculate Occupancy
     uint32_t numBlock = 0;
     hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, multiply, blockSize, 0);
     
     //std::cout << "numBlock is " << numBlock << std::endl;
     activeWarps = numBlock* blockSize/devProp.warpSize;
     maxWarps = devProp.maxThreadsPerMultiProcessor/devProp.warpSize;

     //std::cout << activeWarps << "and " << maxWarps << "and blockSize is" << blockSize << std::endl;
     std::cout << "Theoretical Occupancy is " << (double)activeWarps/maxWarps * 100 << "%" << std::endl;

}

int main() {
     float *A, *B, *C, *cpuC;
   
     float *Ad, *Bd, *Cd;

     int errors;
     int i;
     int manual = 0;

     // initialize the input data
     A = (float *)malloc(NUM * sizeof(float));
     B = (float *)malloc(NUM * sizeof(float));
     C = (float *)malloc(NUM * sizeof(float));
     cpuC = (float *)malloc(NUM * sizeof(float));
     

     for(i=0; i< NUM; i++){
        A[i] = i;
	B[i] = i;
     }
    
     // allocate the memory on the device side   
     HIP_CHECK(hipMalloc((void**)&Ad, NUM * sizeof(float)));
     HIP_CHECK(hipMalloc((void**)&Bd, NUM * sizeof(float)));
     HIP_CHECK(hipMalloc((void**)&Cd, NUM * sizeof(float)));
 
     // Memory transfer from host to device
     HIP_CHECK(hipMemcpy(Ad,A,NUM * sizeof(float), hipMemcpyHostToDevice));
     HIP_CHECK(hipMemcpy(Bd,B,NUM * sizeof(float), hipMemcpyHostToDevice));

     //Kernel launch with manual/default block size
     launchKernel(Cd, Ad, Bd, NUM, 1);
     
     //Kernel launch with the block size suggested by hipOccupancyMaxPotentialBlockSize 
     launchKernel(Cd, Ad, Bd, NUM, 0);

     // Memory transfer from device to host
     HIP_CHECK(hipMemcpy(C,Cd, NUM * sizeof(float), hipMemcpyDeviceToHost));

     // CPU computation
     multiplyCPU(cpuC, A, B, NUM);

     //verify the results
     
     errors = 0;
     double eps = 1.0E-6;
     
     for (i = 0; i < NUM; i++) {
	  if (std::abs(C[i] - cpuC[i]) > eps) {
		  errors++;
	  }
     }
          
     if (errors != 0){
	     printf("\nFAILED: %d errors\n", errors);
     } else {
	     printf("\nPASSED!\n");
     }

     HIP_CHECK(hipFree(Ad));
     HIP_CHECK(hipFree(Bd));
     HIP_CHECK(hipFree(Cd));

     free(A);
     free(B);
     free(C);
     free(cpuC);
}
