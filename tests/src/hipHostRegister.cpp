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

#include"test_common.h"
#include<malloc.h>

__global__ void Inc(hipLaunchParm lp, float *Ad){
int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
Ad[tx] = Ad[tx] + float(1);
}

int main(){
	float *A, *Ad;
	const size_t size = N * sizeof(float);
#ifdef __HIP_PLATFORM_NVCC__
	A = (float*)malloc(size*2);
#else
	A = (float*)memalign(64, size);
#endif
	HIPCHECK(hipHostRegister(A, size, 0));

	for(int i=0;i<N;i++){
		A[i] = float(1);
	}

	HIPCHECK(hipMalloc(&Ad, size));

	HIPCHECK(hipMemcpy(Ad, A, size, hipMemcpyHostToDevice));

	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, 0, Ad);
	HIPCHECK(hipDeviceSynchronize());

	HIPCHECK(hipMemcpy(A, Ad, size, hipMemcpyDeviceToHost));

	HIPASSERT(A[10] == 2.0f);
	HIPCHECK(hipHostUnregister(A));
	passed();
}
