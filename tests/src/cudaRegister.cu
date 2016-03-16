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

#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
#include<unistd.h>
#include<stdio.h>
#include<malloc.h>

#define LEN 1024
#define SIZE LEN * sizeof(float)
#define ITER 1024*1024

#define check(msg, status){ \
if(status != cudaSuccess) { \
	printf("%s failed. \n", #msg); \
} \
}

__global__ void Inc1(float *Ad, float *Bd){
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	if(tx < 1 ){
		for(int i=0;i<ITER;i++){
			Ad[tx] = Ad[tx] + 1.0f;
			for(int j=0;j<256;j++){
				Bd[tx] = Ad[tx];
			}
		}
	}
}

__global__ void Inc2(float *Ad, float *Bd){
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	if(tx < 1024){
		for(int i=0;i<ITER;i++){
			Ad[tx] = Ad[tx] + 1.0f;
			for(int j=0;j<256;j++){
				Bd[tx] = Ad[tx];
			}
		}
	}
}

int main(){
	float *A, *Ad, *Bd;
	A = new float[LEN];
	for(int i=0;i<LEN;i++){
		A[i] = 0.0f;
	}
	cudaError_t status;
	status = cudaHostRegister(A, SIZE, cudaHostRegisterMapped);
	check("Registering A",status);
	cudaHostGetDevicePointer(&Ad, A, 0);
	cudaMalloc((void**)&Bd, SIZE);
	dim3 dimGrid(LEN/512,1,1);
	dim3 dimBlock(512,1,1);
	Inc1<<<dimGrid, dimBlock>>>(Ad, Bd);
	sleep(3);
	A[0] = -(ITER*1.0f);
	std::cout<<"Same cache line before completion: \t"<< A[0]<<std::endl;
	cudaDeviceSynchronize();
	std::cout<<"Same cache line after completion: \t"<< A[0]<<std::endl;

	for(int i=0;i<LEN;i++){
		A[i] = 0.0f;
	}
	Inc2<<<dimGrid, dimBlock>>>(Ad, Bd);
	sleep(3);
	A[0] = -(ITER*1.0f);
	std::cout<<"Diff cache line before completion: \t"<<A[0]<<std::endl;
	cudaDeviceSynchronize();
	std::cout<<"Diff cache line after completion: \t"<<A[0]<<std::endl;
}
