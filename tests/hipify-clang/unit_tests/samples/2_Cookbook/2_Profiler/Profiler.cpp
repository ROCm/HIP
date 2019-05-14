// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

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

// NOTE: Profiler API is under development.
// NOTE: This is NOT WORKING example.
// TODO: Get rid of HIP_SCOPED_MARKER, HIP_BEGIN_MARKER, HIP_END_MARKER, declared in hip/hip_profile.h or
// TODO: find out a way to hipify it in particular place (signatures are to obtain).

#include <iostream>

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
// CHECK: #include <hip/hip_profile.h>
#include <cuda_profiler_api.h>

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

#define ITERATIONS 10

// Cmdline parms to control start and stop triggers
int startTriggerIteration = -1;
int stopTriggerIteration = -1;

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    out[y * width + x] = in[x * width + y];
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

// Use a separate function to demonstrate how to use function name as part of scoped marker:
void runGPU(float* Matrix, float* TransposeMatrix, float* gpuMatrix, float* gpuTransposeMatrix) {
  // __func__ is a standard C++ macro which expands to the name of the function, in this case
  // "runGPU"
// TODO: Find out signatures to generate the following:
// HIP_SCOPED_MARKER(__func__, "MyGroup");

  for (int i = 0; i < ITERATIONS; i++) {
    if (i == startTriggerIteration) {
      // CHECK: hipProfilerStart();
      cudaProfilerStart();
    }
    if (i == stopTriggerIteration) {
      // CHECK: hipProfilerStop();
      cudaProfilerStop();
    }

    float eventMs = 0.0f;

    // CHECK: hipEvent_t start, stop;
    cudaEvent_t start, stop;
    // CHECK: hipEventCreate(&start);
    cudaEventCreate(&start);
    // CHECK: hipEventCreate(&stop);
    cudaEventCreate(&stop);

    // Record the start event
    // CHECK: hipEventRecord(start, NULL);
    cudaEventRecord(start, NULL);

    // Memory transfer from host to device
    // CHECK: hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);
    cudaMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), cudaMemcpyHostToDevice);

    // Record the stop event
    // CHECK: hipEventRecord(stop, NULL);
    cudaEventRecord(stop, NULL);
    // CHECK: hipEventSynchronize(stop);
    cudaEventSynchronize(stop);

    // CHECK: hipEventElapsedTime(&eventMs, start, stop);
    cudaEventElapsedTime(&eventMs, start, stop);

    // CHECK: printf("hipMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);
    printf("cudaMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);

    // Record the start event
    // CHECK: hipEventRecord(start, NULL);
    cudaEventRecord(start, NULL);

    // Lauching kernel from host
    dim3 dimGrid(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y);
    dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    // CHECK: hipLaunchKernelGGL(matrixTranspose, dim3(dimGrid), dim3(dimBlock), 0, 0, gpuTransposeMatrix, gpuMatrix, WIDTH);
    matrixTranspose <<<dimGrid, dimBlock >>> (gpuTransposeMatrix, gpuMatrix, WIDTH);

    // Record the stop event
    // CHECK: hipEventRecord(stop, NULL);
    cudaEventRecord(stop, NULL);
    // CHECK: hipEventSynchronize(stop);
    cudaEventSynchronize(stop);
    // CHECK: hipEventElapsedTime(&eventMs, start, stop);
    cudaEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs);

    // Record the start event
    // CHECK: hipEventRecord(start, NULL);
    cudaEventRecord(start, NULL);

    // Memory transfer from device to host
    // CHECK: hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);
    cudaMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), cudaMemcpyDeviceToHost);

    // Record the stop event
    // CHECK: hipEventRecord(stop, NULL);
    cudaEventRecord(stop, NULL);
    // CHECK: hipEventSynchronize(stop);
    cudaEventSynchronize(stop);

    // CHECK: hipEventElapsedTime(&eventMs, start, stop);
    cudaEventElapsedTime(&eventMs, start, stop);

    // CHECK: printf("hipMemcpyDeviceToHost time taken  = %6.3fms\n", eventMs);
    printf("cudaMemcpyDeviceToHost time taken  = %6.3fms\n", eventMs);
  }
};

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        startTriggerIteration = atoi(argv[1]);
        printf("info : will start tracing at iteration:%d\n", startTriggerIteration);
    }
    if (argc >= 3) {
        stopTriggerIteration = atoi(argv[2]);
        printf("info : will stop tracing at iteration:%d\n", stopTriggerIteration);
    }

    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;

    // CHECK: hipDeviceProp_t devProp;
    cudaDeviceProp devProp;
    // CHECK: hipGetDeviceProperties(&devProp, 0);
    cudaGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    {
      // Show example of how to create a "scoped marker".
      // The scoped marker records the time spent inside the { scope } of the marker - the begin
      // timestamp is at the beginning of the code scope, and the end is recorded when the SCOPE
      // exits. This can be viewed in CodeXL timeline relative to other GPU and CPU events. This
      // marker captures the time spent in setup including host allocation, initialization, and
      // device memory allocation.
// TODO: Find out signatures to generate the following:
// HIP_SCOPED_MARKER("Setup", "MyGroup");

      Matrix = (float*)malloc(NUM * sizeof(float));
      TransposeMatrix = (float*)malloc(NUM * sizeof(float));
      cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

      // initialize the input data
      for (int i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
      }

      // allocate the memory on the device side
      // CHECK: hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
      cudaMalloc((void**)&gpuMatrix, NUM * sizeof(float));
      // CHECK: hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));
      cudaMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

      // FYI, the scoped-marker will be destroyed here when the scope exits, and will record its
      // "end" timestamp.
    }

    runGPU(Matrix, TransposeMatrix, gpuMatrix, gpuTransposeMatrix);

    // show how to use explicit begin/end markers:
    // We begin the timed region with HIP_BEGIN_MARKER, passing in the markerName and group:
    // The region will stop when HIP_END_MARKER is called
    // This is another way to mark begin/end - as an alternative to scoped markers.
// TODO: Find out signatures to generate the following:
// HIP_BEGIN_MARKER("Check&TearDown", "MyGroup");

    int errors = 0;

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    double eps = 1.0E-6;
    for (int i = 0; i < NUM; i++) {
        if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
            errors++;
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }

    // free the resources on device side
    // CHECK: hipFree(gpuMatrix);
    cudaFree(gpuMatrix);
    // CHECK: hipFree(gpuTransposeMatrix);
    cudaFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    // This ends the last marker started in this thread, in this case "Check&TearDown"
// TODO: Find out signatures to generate the following:
// HIP_END_MARKER();

    return errors;
}
