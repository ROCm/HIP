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

#include<iostream>

// hip header file
#include "hip/hip_runtime.h"
#include "hip/hip_profile.h"

#define WIDTH     1024

#define NUM       (WIDTH*WIDTH)

#define THREADS_PER_BLOCK_X  4
#define THREADS_PER_BLOCK_Y  4
#define THREADS_PER_BLOCK_Z  1

#define ITERATIONS 10

// Cmdline parms to control start and stop triggers
int startTriggerIteration=-1;
int stopTriggerIteration=-1;

// Device (Kernel) function, it must be void
// hipLaunchParm provides the execution configuration
__global__ void matrixTranspose(hipLaunchParm lp,
                                float *out,
                                float *in,
                                const int width)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    out[y * width + x] = in[x * width + y];
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(
    float * output,
    float * input,
    const unsigned int width)
{
    for(unsigned int j=0; j < width; j++)
    {
        for(unsigned int i=0; i < width; i++)
        {
            output[i*width + j] = input[j*width + i];
        }
    }
}


// Use a separate function to demonstrate how to use function name as part of scoped marker:
void runGPU(float *Matrix, float *TransposeMatrix, 
            float* gpuMatrix, float* gpuTransposeMatrix)  {

  // __func__ is a standard C++ macro which expands to the name of the function, in this case "runGPU"
  HIP_SCOPED_MARKER(__func__, "MyGroup");

  for (int i=0; i<ITERATIONS; i++) {

    if (i==startTriggerIteration) {
      hipProfilerStart();
    }
    if (i==stopTriggerIteration) {
      hipProfilerStop();
    }

    float eventMs = 0.0f;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);


    // Record the start event
    hipEventRecord(start, NULL);

    // Memory transfer from host to device
    hipMemcpy(gpuMatrix, Matrix, NUM*sizeof(float), hipMemcpyHostToDevice);

    // Record the stop event
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf ("hipMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);

    // Record the start event
    hipEventRecord(start, NULL);

    // Lauching kernel from host
    hipLaunchKernel(matrixTranspose,
                    dim3(WIDTH/THREADS_PER_BLOCK_X, WIDTH/THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                    0, 0,
                    gpuTransposeMatrix , gpuMatrix, WIDTH);

    // Record the stop event
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf ("kernel Execution time             = %6.3fms\n", eventMs);

    // Record the start event
    hipEventRecord(start, NULL);

    // Memory transfer from device to host
    hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM*sizeof(float), hipMemcpyDeviceToHost);

    // Record the stop event
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf ("hipMemcpyDeviceToHost time taken  = %6.3fms\n", eventMs);
  }
};


int main(int argc, char *argv[]) {

  if (argc >= 2) {
    startTriggerIteration = atoi(argv[1]);
    printf ("info : will start tracing at iteration:%d\n", startTriggerIteration);
  } 
  if (argc >= 3) {
    stopTriggerIteration = atoi(argv[2]);
    printf ("info : will stop tracing at iteration:%d\n", stopTriggerIteration);
  }

  float* Matrix;
  float* TransposeMatrix;
  float* cpuTransposeMatrix;

  float* gpuMatrix;
  float* gpuTransposeMatrix;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  std::cout << "Device name " << devProp.name << std::endl;

  {
      // Show example of how to create a "scoped marker".  
      // The scoped marker records the time spent inside the { scope } of the marker - the begin timestamp is at the
      // beginning of the code scope, and the end is recorded when the SCOPE exits.  This can be viewed in CodeXL
      // timeline relative to other GPU and CPU events.
      // This marker captures the time spent in setup including host allocation, initialization, and device memory allocation.
      HIP_SCOPED_MARKER("Setup", "MyGroup");



      Matrix = (float*)malloc(NUM * sizeof(float));
      TransposeMatrix = (float*)malloc(NUM * sizeof(float));
      cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

      // initialize the input data
      for (int i = 0; i < NUM; i++) {
        Matrix[i] = (float)i*10.0f;
      }


      // allocate the memory on the device side
      hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
      hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

      // FYI, the scoped-marker will be destroyed here when the scope exits, and will record its "end" timestamp.
  }

  runGPU(Matrix, TransposeMatrix, gpuMatrix, gpuTransposeMatrix);


  // show how to use explicit begin/end markers:
  // We begin the timed region with HIP_BEGIN_MARKER, passing in the markerName and group:
  // The region will stop when HIP_END_MARKER is called
  // This is another way to mark begin/end - as an alternative to scoped markers.
  HIP_BEGIN_MARKER("Check&TearDown", "MyGroup");

  int errors = 0;

  // CPU MatrixTranspose computation
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

  // verify the results
  double eps = 1.0E-6;
  for (int i = 0; i < NUM; i++) {
    if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps ) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
    printf ("PASSED!\n");
  }

  //free the resources on device side
  hipFree(gpuMatrix);
  hipFree(gpuTransposeMatrix);

  //free the resources on host side
  free(Matrix);
  free(TransposeMatrix);
  free(cpuTransposeMatrix);

  // This ends the last marker started in this thread, in this case "Check&TearDown"
  HIP_END_MARKER();  
  
  return errors;
}
