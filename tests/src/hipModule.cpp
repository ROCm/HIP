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

#define fileName "vcpy_isa.co"
#define kernel_name "hello_world"

__global__ void Cpy(hipLaunchParm lp, float *Ad, float* Bd){
  int tx = hipThreadIdx_x;
  Bd[tx] = Ad[tx];
}

int main(){
  float *A, *B, *Ad, *Bd;
  A = new float[LEN];
  B = new float[LEN];

  for(uint32_t i=0;i<LEN;i++){
    A[i] = i*1.0f;
    B[i] = 0.0f;
    std::cout<<A[i] << " "<<B[i]<<std::endl;
  }

  hipMalloc((void**)&Ad, SIZE);
  hipMalloc((void**)&Bd, SIZE);

  hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
  hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice);
  hipModule_t Module;
  hipFunction_t Function;
  hipModuleLoad(&Module, fileName);
  hipModuleGetFunction(&Function, Module, kernel_name);
  hipStream_t stream;
  hipStreamCreate(&stream);
  void *args[2] = {&Ad, &Bd};


    std::vector<void*>argBuffer(2);
    memcpy(&argBuffer[0], &Ad, sizeof(void*));
    memcpy(&argBuffer[1], &Bd, sizeof(void*));

    size_t size = argBuffer.size()*sizeof(void*);

    void *config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, &argBuffer[0],
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
      HIP_LAUNCH_PARAM_END
    };

    hipLaunchModuleKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream, NULL, (void**)&config); 

    hipStreamDestroy(stream);

 hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);

  for(uint32_t i=0;i<LEN;i++){
  std::cout<<A[i]<<" - "<<B[i]<<std::endl;
  }

  return 0;
}
