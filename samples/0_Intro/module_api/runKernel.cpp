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

#include<hip_runtime.h>
#include<hip_runtime_api.h>
#include<iostream>
#include<fstream>
#include<vector>

#define LEN 64
#define SIZE LEN<<2

#ifdef __HIP_PLATFORM_HCC__ 
#define fileName "vcpy_kernel.co"
#define kernel_name "ZN12_GLOBAL__N_146_Z11hello_world16grid_launch_parmPfS0__functor19__cxxamp_trampolineEiiiiiiPKfPf"
#endif

#ifdef __HIP_PLATFORM_NVCC__
#define fileName "vcpy_kernel.ptx"
#define kernel_name "hello_world"
#endif

int main(){
    float *A, *B;
    hipDeviceptr_t Ad, Bd;
    A = new float[LEN];
    B = new float[LEN];

    for(uint32_t i=0;i<LEN;i++){
        A[i] = i*1.0f;
        B[i] = 0.0f;
    }


#ifdef __HIP_PLATFORM_NVCC__
  	hipInit(0);
	  hipDevice_t device;
	  hipCtx_t context;
	  hipDeviceGet(&device, 0);
	  hipCtxCreate(&context, 0, device);
#endif

    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);

    hipMemcpyHtoD(Ad, A, SIZE);
    hipMemcpyHtoD(Bd, B, SIZE);
    hipModule_t Module;
    hipFunction_t Function;
    hipModuleLoad(&Module, fileName);
    hipModuleGetFunction(&Function, Module, kernel_name);

#ifdef __HIP_PLATFORM_HCC__
		uint32_t len = LEN;
		uint32_t one = 1;

    std::vector<void*>argBuffer(5);
    uint32_t *ptr32_t = (uint32_t*)&argBuffer[0];
    memcpy(ptr32_t + 0, &one, sizeof(uint32_t));
    memcpy(ptr32_t + 1, &one, sizeof(uint32_t));
    memcpy(ptr32_t + 2, &one, sizeof(uint32_t));
    memcpy(ptr32_t + 3, &len, sizeof(uint32_t));
    memcpy(ptr32_t + 4, &one, sizeof(uint32_t));
    memcpy(ptr32_t + 5, &one, sizeof(uint32_t));
    memcpy(&argBuffer[3], &Ad, sizeof(void*));
    memcpy(&argBuffer[4], &Bd, sizeof(void*));
#endif

#ifdef __HIP_PLATFORM_NVCC__
    std::vector<void*>argBuffer(2);
    memcpy(&argBuffer[0], &Ad, sizeof(void*));
    memcpy(&argBuffer[1], &Bd, sizeof(void*));
#endif


    size_t size = argBuffer.size()*sizeof(void*);

    void *config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, &argBuffer[0],
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
      HIP_LAUNCH_PARAM_END
    };

    hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL, (void**)&config); 

    hipMemcpyDtoH(B, Bd, SIZE);
    for(uint32_t i=LEN-4;i<LEN;i++){
        std::cout<<A[i]<<" - "<<B[i]<<std::endl;
    }

#ifdef __HIP_PLATFORM_NVCC__
	  hipCtxDetach(context);
#endif 

    return 0;
}
