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
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */

// Test under-development. Call hipStreamAddCallback function and see if it works as expected.

#include "hip/hip_runtime.h"
#include "test_common.h"
#define HIPRT_CB
const int NN = 1 << 21;

__global__ void kernel(hipLaunchParm lp, float *x, float *y, int n){
	int tid = hipThreadIdx_x;
	if(tid < 1){
		for(int i=0;i<n;i++){
			x[i] = sqrt(pow(3.14159,i));
		}
		y[tid] = y[tid] + 1.0f;
	}
}
__global__ void nKernel(hipLaunchParm lp, float *y){
	int tid = hipThreadIdx_x;
	y[tid] = y[tid] + 1.0f;
}

class CallbackClass
{
public:
    static void HIPRT_CB Callback(hipStream_t stream, hipError_t status, void *userData);

private:
    void callbackFunc(hipError_t status);
};

void HIPRT_CB CallbackClass::Callback(hipStream_t stream, hipError_t status, void *userData)
{
    CallbackClass* obj = (CallbackClass*) userData;
    obj->callbackFunc(status);
}

void CallbackClass::callbackFunc(hipError_t status)
{
     HIPASSERT(status==hipSuccess);
}

int main(){
	const int num_streams = 8;
	hipStream_t streams[num_streams];
   /* float *data[num_streams], *yd, *xd;*/
	//float y = 1.0f, x = 1.0f;
	//HIPCHECK(hipMalloc((void**)&yd, sizeof(float)));
	//HIPCHECK(hipMalloc((void**)&xd, sizeof(float)));
	//HIPCHECK(hipMemcpy(yd, &y, sizeof(float), hipMemcpyHostToDevice));
	//HIPCHECK(hipMemcpy(xd, &x, sizeof(float), hipMemcpyHostToDevice));
	//for(int i=0;i<num_streams;i++){
		//HIPCHECK(hipStreamCreate(&streams[i]));
		//HIPCHECK(hipMalloc(&data[i], NN * sizeof(float)));
		//hipLaunchKernel(HIP_KERNEL_NAME(kernel), dim3(1), dim3(1), 0, streams[i], data[i], xd, N);
		//hipLaunchKernel(HIP_KERNEL_NAME(nKernel), dim3(1), dim3(1), 0, 0, yd);
	//}

	//HIPCHECK(hipMemcpy(&x, xd, sizeof(float), hipMemcpyDeviceToHost));
	//HIPCHECK(hipMemcpy(&y, yd, sizeof(float), hipMemcpyDeviceToHost));
	//std::cout<<x<<" "<<y<<std::endl;
	//HIPASSERT(x<y);

    //hipStream_t mystream = streams[0];
    hipStream_t mystream = NULL;
     CallbackClass* obj = new CallbackClass;
     hipStreamAddCallback(mystream, CallbackClass::Callback, obj, 0);


	passed();
}
