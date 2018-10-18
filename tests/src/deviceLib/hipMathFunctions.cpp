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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

#if __HIP_ARCH_GFX803__ || __HIP_ARCH_GFX900__ || __HIP_ARCH_GFX906__

__global__ void kernel_abs_int64(long long *input, long long *output) {
    int tx = threadIdx.x;
    output[tx] = abs(input[tx]);
}

__global__ void kernel_lgamma_double(double *input, double *output) {
    int tx = threadIdx.x;
    output[tx] = lgamma(input[tx]);
}

#endif

#define CHECK_LGAMMA_DOUBLE(IN, OUT, EXP)		 \
  {						 \
    if (OUT != EXP)  {				 \
      failed("check_abs_int64 failed on %f (output = %f, expected = %fd)\n", IN, OUT, EXP); \
    }						 \
  }

#define CHECK_ABS_INT64(IN, OUT, EXP)		 \
  {						 \
    if (OUT != EXP)  {				 \
      failed("check_abs_int64 failed on %lld (output = %lld, expected = %lld)\n", IN, OUT, EXP); \
    }						 \
  }

void check_lgamma_double() {

  using datatype_t = double;
  
  const int NUM_INPUTS = 8;
  auto memsize = NUM_INPUTS * sizeof(datatype_t);
  
  // allocate memories
  datatype_t *inputCPU = (datatype_t *) malloc(memsize);
  datatype_t *outputCPU = (datatype_t *) malloc(memsize);
  datatype_t *inputGPU = nullptr; hipMalloc((void**)&inputGPU, memsize);
  datatype_t *outputGPU = nullptr; hipMalloc((void**)&outputGPU, memsize);
  
  // populate input
  for (int i=0; i<NUM_INPUTS; i++) {
    inputCPU[i] = -3.5 + i;
  }
  
  // copy inputs to device
  hipMemcpy(inputGPU, inputCPU, memsize, hipMemcpyHostToDevice);

  // launch kernel
  hipLaunchKernelGGL(kernel_lgamma_double, dim3(1), dim3(NUM_INPUTS), 0, 0, inputGPU, outputGPU);

  // copy outputs from device
  hipMemcpy(outputCPU, outputGPU, memsize, hipMemcpyDeviceToHost);

  // check outputs
  for (int i=0; i<NUM_INPUTS; i++) {
    CHECK_LGAMMA_DOUBLE(inputCPU[i], outputCPU[i], lgamma(inputCPU[i]));
  }
  
  // free memories
  hipFree(inputGPU);
  hipFree(outputGPU);
  free(inputCPU);
  free(outputCPU);
  
  // done
  return;
}


void check_abs_int64() {

  using datatype_t = long long;

  const int NUM_INPUTS = 8;
  auto memsize = NUM_INPUTS * sizeof(datatype_t);
  
  // allocate memories
  datatype_t *inputCPU = (datatype_t *) malloc(memsize);
  datatype_t *outputCPU = (datatype_t *) malloc(memsize);
  datatype_t *inputGPU = nullptr; hipMalloc((void**)&inputGPU, memsize);
  datatype_t *outputGPU = nullptr; hipMalloc((void**)&outputGPU, memsize);
  
  // populate input
  inputCPU[0] = -81985529216486895ll; 
  inputCPU[1] =  81985529216486895ll;
  inputCPU[2] = -1250999896491ll;
  inputCPU[3] =  1250999896491ll;
  inputCPU[4] = -19088743ll;
  inputCPU[5] =  19088743ll;
  inputCPU[6] = -291ll;
  inputCPU[7] =  291ll;
  
  // copy inputs to device
  hipMemcpy(inputGPU, inputCPU, memsize, hipMemcpyHostToDevice);

  // launch kernel
  hipLaunchKernelGGL(kernel_abs_int64, dim3(1), dim3(NUM_INPUTS), 0, 0, inputGPU, outputGPU);

  // copy outputs from device
  hipMemcpy(outputCPU, outputGPU, memsize, hipMemcpyDeviceToHost);

  // check outputs
  CHECK_ABS_INT64(inputCPU[0], outputCPU[0], outputCPU[1]);
  CHECK_ABS_INT64(inputCPU[1], outputCPU[1], outputCPU[1]);
  CHECK_ABS_INT64(inputCPU[2], outputCPU[2], outputCPU[3]);
  CHECK_ABS_INT64(inputCPU[3], outputCPU[3], outputCPU[3]);
  CHECK_ABS_INT64(inputCPU[4], outputCPU[4], outputCPU[5]);
  CHECK_ABS_INT64(inputCPU[5], outputCPU[5], outputCPU[5]);
  CHECK_ABS_INT64(inputCPU[6], outputCPU[6], outputCPU[7]);
  CHECK_ABS_INT64(inputCPU[7], outputCPU[7], outputCPU[7]);
  
  // free memories
  hipFree(inputGPU);
  hipFree(outputGPU);
  free(inputCPU);
  free(outputCPU);
  
  // done
  return;
}
  

int main(int argc, char* argv[]) {
    HipTest::parseStandardArguments(argc, argv, true);

    check_abs_int64();

    // check_lgamma_double();
    
    passed();
}
