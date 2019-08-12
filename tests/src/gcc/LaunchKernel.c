/*Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <stdio.h>
#include "LaunchKernel.h"

void LaunchKernelArg()
{
  hipError_t status = hipSuccess;
  dim3 blocks 	    = {1,1,1};
  dim3 threads      = {1,1,1};

  printf(" -----: Launch Zero Argument Kernel :------ \n");
  status = hipLaunchKernel(kernel, blocks, threads,NULL, 0, 0,0);
  printf("hipLaunchKernel status %s\n",hipGetErrorString(status));

  printf("Test Passed!!\n\n");
}

void LaunchKernelArg1()
{
  hipError_t status = hipSuccess;
  int A = 0;
  int *A_d = NULL;
  dim3 blocks       = {1,1,1};
  dim3 threads      = {1,1,1};

  printf(" -----: Launch One Argument Kernel :------ \n");

  // Allocate Device memory
  status = hipMalloc((void**)&A_d, sizeof(int));
  printf("hipMalloc status %s\n", hipGetErrorString(status));

  void* Args[]={A_d};
  hipLaunchKernel(kernel1, blocks, threads, Args,0,0,sizeof(Args));
  printf("hipLaunchKernel status %s\n",hipGetErrorString(status));

  // Get the result back to host memory
  status = hipMemcpy(&A, A_d, sizeof(int), hipMemcpyDeviceToHost);
  printf("hipMalloc status %s\n", hipGetErrorString(status));

  if(A == 333)
    printf("Test Passed!!\n\n");
  else
    printf("Test Failed :Value of A is %d\n\n",A);

  hipFree(A_d);
}

void LaunchKernelArg2()
{
  hipError_t status = hipSuccess;
  int A = 0;
  int B = 123;
  int *A_d = NULL;
  int *B_d = NULL;

  dim3 blocks       = {1,1,1};
  dim3 threads      = {1,1,1};

  printf(" -----: Launch Two Argument Kernel :------ \n");

  // Allocate Device memory
  status = hipMalloc((void**)&A_d, sizeof(int));
  printf("hipMalloc status %s\n", hipGetErrorString(status));

  status = hipMalloc((void**)&B_d, sizeof(int));
  printf("hipMalloc status %s\n", hipGetErrorString(status));

  // Copy data from host memory to device memory
  status = hipMemcpy(B_d,&B, sizeof(int), hipMemcpyHostToDevice);
  printf("hipMalloc status %s\n", hipGetErrorString(status));


  void* Args[]={A_d,B_d};
  hipLaunchKernel(kernel2, blocks, threads, Args,0,0,sizeof(Args));
  printf("hipLaunchKernel status %s\n", hipGetErrorString(status));

  // Get the result back to host memory
  status = hipMemcpy(&A, A_d, sizeof(int), hipMemcpyDeviceToHost);
  printf("hipMalloc status %s\n", hipGetErrorString(status));

  if(A == 123)
    printf("Test Passed!! \n\n");
  else
    printf("Test Failed : Value of A is %d\n\n",A);

  hipFree(A_d);
  hipFree(B_d);
}

void LaunchKernelArg3()
{
  hipError_t status = hipSuccess;
  int A = 321;
  int B = 123;
  int C = 0;
  int *A_d = NULL;
  int *B_d = NULL;
  int *C_d = NULL;

  dim3 blocks       = {1,1,1};
  dim3 threads      = {1,1,1};

  printf(" -----: Launch Three Argument Kernel :------ \n");

  // Allocate Device memory
  status = hipMalloc((void**)&A_d, sizeof(int));
  printf("hipMalloc status %s\n", hipGetErrorString(status));

  status = hipMalloc((void**)&B_d, sizeof(int));
  printf("hipMalloc status %s\n", hipGetErrorString(status));

  status = hipMalloc((void**)&C_d, sizeof(int));
  printf("hipMalloc status %s\n", hipGetErrorString(status));

  // Copy data from host memory to device memory
  status = hipMemcpy(A_d,&A, sizeof(int), hipMemcpyHostToDevice);
  printf("hipMalloc status %s\n", hipGetErrorString(status));

  status = hipMemcpy(B_d,&B, sizeof(int), hipMemcpyHostToDevice);
  printf("hipMalloc status %s\n", hipGetErrorString(status));


  void* Args[]={A_d,B_d,C_d};
  hipLaunchKernel(kernel3, blocks, threads, Args,0,0,sizeof(Args));
  printf("hipLaunchKernel status %s\n", hipGetErrorString(status));

  // Get the result back to host memory
  status = hipMemcpy(&C, C_d, sizeof(int), hipMemcpyDeviceToHost);
  printf("hipMalloc status %s\n", hipGetErrorString(status));

  if(C == 444)
    printf("Test Passed\n\n");
  else
    printf("Test Failed : Value of C is %d\n\n",C);

  hipFree(A_d);
  hipFree(B_d);
  hipFree(C_d);
}


int main()
{
  LaunchKernelArg();
  LaunchKernelArg1();
  LaunchKernelArg2();
  LaunchKernelArg3();

  printf("PASSED!\n");
}
