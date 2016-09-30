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
 * BUILD: %t %s test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include"test_common.h"
#include<malloc.h>

__global__ void Inc(hipLaunchParm lp, float *Ad){
int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
Ad[tx] = Ad[tx] + float(1);
}

int main(){
	float *A, **Ad;
	int num_devices;
	HIPCHECK(hipGetDeviceCount(&num_devices));
	Ad = new float*[num_devices];
	const size_t size = N * sizeof(float);
	A = (float*)malloc(size);
	HIPCHECK(hipHostRegister(A, size, 0));
	for(int i=0;i<N;i++){
		A[i] = float(1);
	}
	for(int i=0;i<num_devices;i++){
	HIPCHECK(hipSetDevice(i));
	HIPCHECK(hipHostGetDevicePointer((void**)&Ad[i], A, 0));
	}

	for(int i=0;i<num_devices;i++){
	HIPCHECK(hipSetDevice(i));
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, 0, Ad[i]);

	HIPCHECK(hipDeviceSynchronize());

	}
	HIPASSERT(A[10] == 1.0f + float(num_devices));
	HIPCHECK(hipHostUnregister(A));
	passed();
}
