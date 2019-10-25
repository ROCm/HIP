/*
Copyright (c) 2019-Present Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD_CMD: matmul.code %hc --genco %S/matmul.cpp -o matmul.code
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include<chrono>
#include "hip/hip_runtime.h"
#include "hip/hip_hcc.h"
#include "test_common.h"
using namespace std::chrono;

#define fileName "matmul.code"
#define kernel_name1 "matmulK"
#define kernel_name2 "WaitKernel"

//hipDeviceReset() call is used to clear all the allocations in case of any api failure.
#define HIP_CHECK(status)     \
   if (status != hipSuccess) {   \
   std::cout << "Got Status: " << hipGetErrorString(status) << " at Line: " << __LINE__ << std::endl; \
	   hipDeviceReset(); \
   exit(0);  \
      }

int main()
{
	
int N=16384;
int SIZE=N*N;

int *A=new int[N*N*sizeof(int)];
int *B=new int[N*N*sizeof(int)];
int *C;

hipDeviceptr_t *Ad,*Bd;

for(int i=0;i<N;i++)
	for(int j=0;j<N;j++){
		A[i*N +j]=1;
		B[i*N +j]=1;
	}
hipStream_t stream1,stream2;
HIP_CHECK(hipStreamCreate(&stream1));

HIP_CHECK(hipMalloc((void**)&Ad, SIZE*sizeof(int)));
HIP_CHECK(hipMalloc((void**)&Bd, SIZE*sizeof(int)));
HIP_CHECK(hipHostMalloc((void**)&C, SIZE*sizeof(int)));

HIP_CHECK(hipMemcpy(Ad,A,SIZE*sizeof(int),hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(Bd,B,SIZE*sizeof(int),hipMemcpyHostToDevice));

hipModule_t Module;
hipFunction_t Function1,Function2;
HIP_CHECK(hipModuleLoad(&Module, fileName));
HIP_CHECK(hipModuleGetFunction(&Function1, Module, kernel_name1))
HIP_CHECK(hipModuleGetFunction(&Function2, Module, kernel_name2))

struct {
        void* _Ad;
        void* _Bd;
	void* _Cd;
	int _n;
       } args1,args2;

args1._Ad = Ad;
args1._Bd = Bd;
args1._Cd = C;
args1._n  = N;

args2._Ad = NULL;
args2._Bd = NULL;
args2._Cd = NULL;
args2._n  = 0; 

size_t size1=sizeof(args1);
size_t size2=sizeof(args2);
void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,HIP_LAUNCH_PARAM_END};
void* config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,HIP_LAUNCH_PARAM_END};

auto start=high_resolution_clock::now();
HIP_CHECK(hipExtModuleLaunchKernel(Function2, 1,1, 1, 1,1 ,1 , 0, stream1, NULL, (void**)&config2, NULL, NULL,0 ));
HIP_CHECK(hipExtModuleLaunchKernel(Function1, N,N, 1, 32,32 ,1 , 0, stream1, NULL, (void**)&config1, NULL, NULL,0 ));
HIP_CHECK(hipStreamSynchronize(stream1));

auto stop=high_resolution_clock::now();
auto duration1=duration_cast<microseconds>(stop-start);

start=high_resolution_clock::now();
HIP_CHECK(hipExtModuleLaunchKernel(Function2, 1,1, 1, 1,1 ,1 , 0, stream1, NULL, (void**)&config2, NULL, NULL,1 ));
HIP_CHECK(hipExtModuleLaunchKernel(Function1, N,N, 1, 32,32 ,1 , 0, stream1, NULL, (void**)&config1, NULL, NULL,1 ));
HIP_CHECK(hipStreamSynchronize(stream1));

stop=high_resolution_clock::now();
auto duration2=duration_cast<microseconds>(stop-start);

bool TEST_STATUS = true;

if(! (duration2.count() < duration1.count())){
	std::cout<<"Test failed as there was no time gain observed when two kernels were launched using hipExtModuleLaunchKernel() with flag 1."<<std::endl;
	TEST_STATUS=false;}

unsigned long int mismatch=0;
for(int i=0;i<N;i++){
	for(int j=0;j<N;j++){
		if(C[i*N + j] != N)
			mismatch++;
	}}
if(! (mismatch == 0)){
	std::cout<<"Test failed as the result of matrix multiplication was found incorrect."<<std::endl;
	TEST_STATUS=false;}

free(A);
free(B);
HIP_CHECK(hipFree(Ad));
HIP_CHECK(hipFree(Bd));
HIP_CHECK(hipHostFree(C));
if(TEST_STATUS == true)
passed();
}
