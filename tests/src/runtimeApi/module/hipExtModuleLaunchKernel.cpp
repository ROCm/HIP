/*
Copyright (c) 2015-2019 Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD_CMD: matadd.code %hc --genco %S/matadd.cpp -o matadd.code
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */


#include<iostream>
#include"hip/hip_runtime.h"
#include<time.h>
#include "test_common.h"

using namespace std;

#define fileName "matadd.code"
#define kernel_name "mataddK"

#define HIP_CHECK(status)     \
   if (status != hipSuccess) {   \
   std::cout << "Got Status: " << hipGetErrorString(status) << " at Line: " << __LINE__ << std::endl; \
   exit(0);  \
      }



int main()
{
	
int N=20480,i,j;
int SIZE=N*N;

int *A=new int[N*N*sizeof(int)];
int *B=new int[N*N*sizeof(int)];
int *C=new int[N*N*sizeof(int)];
int *C1=new int[N*N*sizeof(int)];

time_t start,end;
hipDeviceptr_t *Ad,*Bd,*Cd,*Cd1,*Ad1,*Bd1;

for(i=0;i<N;i++)
	for(j=0;j<N;j++){
		A[i*N +j]=1;
		B[i*N +j]=1;
	}
hipStream_t stream1;
HIP_CHECK(hipStreamCreate(&stream1));

HIP_CHECK(hipMalloc((void**)&Ad, SIZE*sizeof(int)));
HIP_CHECK(hipMalloc((void**)&Bd, SIZE*sizeof(int)));
HIP_CHECK(hipMalloc((void**)&Cd, SIZE*sizeof(int)));
HIP_CHECK(hipMalloc((void**)&Cd1, SIZE*sizeof(int)));

HIP_CHECK(hipMemcpy(Ad,A,SIZE*sizeof(int),hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(Bd,B,SIZE*sizeof(int),hipMemcpyHostToDevice));

hipModule_t Module;
hipFunction_t Function;
HIP_CHECK(hipModuleLoad(&Module, fileName));
HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name))

struct {
        void* _Ad;
        void* _Bd;
	void* _Cd;
	int _n;
       } args,args1;

args._Ad = Ad;
args._Bd = Bd;
args._Cd = Cd;
args._n  = N;

args1._Ad = Ad;
args1._Bd = Bd;
args1._Cd = Cd1;
args1._n  = N;

size_t size=sizeof(args);
size_t size1=sizeof(args1);
void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,HIP_LAUNCH_PARAM_END};
void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,HIP_LAUNCH_PARAM_END};

time(&start);
for(i=0;i<200;i++)
HIP_CHECK(hipExtModuleLaunchKernel(Function, N,N, 1, 32,32 ,1 , 0, stream1, NULL, (void**)&config, NULL, NULL,1 ));
HIP_CHECK(hipDeviceSynchronize());
time(&end);
double dif1 = difftime (end,start);
cout<<"Time taken to complete 200 kernel launches: "<<dif1<<" seconds."<<endl;

time(&start);
for(i=0;i<200;i++)
HIP_CHECK(hipExtModuleLaunchKernel(Function, N,N, 1, 32,32 ,1 , 0, stream1, NULL, (void**)&config, NULL, NULL,1 ));
HIP_CHECK(hipExtModuleLaunchKernel(Function, N,N, 1, 32,32 ,1 , 0, stream1, NULL, (void**)&config1, NULL, NULL,1 ));
HIP_CHECK(hipDeviceSynchronize());
time(&end);
double dif2 = difftime (end,start);
cout<<"Time taken to complete 400 kernel launches: "<<dif2<<" seconds."<<endl;

if(dif2 < (2*dif1)){
	cout<<"Flag functionality of hipExtModuleLaunchKernel() worked fine."<<endl;
	cout<<"Result: Passed!"<<endl;}
else{
	cout<<"Flag functionality of hipExtModuleLaunchKernel() did not work as expected."<<endl;
	cout<<"Result: Failed!"<<endl;}

HIP_CHECK(hipMemcpyDtoHAsync(C, Cd, SIZE*sizeof(int),stream1));
HIP_CHECK(hipMemcpyDtoHAsync(C1, Cd1, SIZE*sizeof(int),stream1));

cout<<"Verifying the result of first kernel operation."<<endl;
int mismatch=0;
for(i=0;i<N;i++){
	for(j=0;j<N;j++){
		if(C[i*N + j] != 2)
			mismatch++;
	}
}
if(mismatch == 0){
	cout<<"No mismatch found."<<endl;
	cout<<"Result: Passed!"<<endl;}
else{
	cout<<"Mismatch found."<<endl;
	cout<<"Result: Failed!"<<endl;}

cout<<"Verifying the result of second kernel operation."<<endl;
 mismatch=0;
for(i=0;i<N;i++){
	for(j=0;j<N;j++){
		if(C1[i*N + j] != 2)
			mismatch++;
	}
}
if(mismatch == 0){
	cout<<"No mismatch found."<<endl;
	cout<<"Result: Passed!"<<endl;}
else{
	cout<<"Mismatch found."<<endl;
	cout<<"Result: Failed!"<<endl;}

free(A);
free(B);
free(C);
HIP_CHECK(hipFree(Ad));
HIP_CHECK(hipFree(Bd));
HIP_CHECK(hipFree(Cd));
passed();
}
