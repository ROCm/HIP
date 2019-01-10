// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
// CHECK: #include <hipfft.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

#define DATASIZE 8
#define BATCH 2

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// CHECK: inline void gpuAssert(hipError_t code, const char *file, int line, bool abort = true)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  // CHECK: if (code != hipSuccess)
  if (code != cudaSuccess)
  {
    // CHECK: fprintf(stderr, "GPUassert: %s %s %dn", hipGetErrorString(code), file, line);
    fprintf(stderr, "GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

int main()
{
  // --- Host side input data allocation and initialization
  // CHECK: hipfftReal *hostInputData = (hipfftReal*)malloc(DATASIZE*BATCH * sizeof(hipfftReal));
  cufftReal *hostInputData = (cufftReal*)malloc(DATASIZE*BATCH * sizeof(cufftReal));
  for (int i = 0; i<BATCH; i++)
    for (int j = 0; j<DATASIZE; j++) hostInputData[i*DATASIZE + j] = (cufftReal)(i + 1);

  // --- Device side input data allocation and initialization
  cufftReal *deviceInputData; gpuErrchk(cudaMalloc((void**)&deviceInputData, DATASIZE * BATCH * sizeof(cufftReal)));
  // CHECK: hipMemcpy(deviceInputData, hostInputData, DATASIZE * BATCH * sizeof(hipfftReal), hipMemcpyHostToDevice);
  cudaMemcpy(deviceInputData, hostInputData, DATASIZE * BATCH * sizeof(cufftReal), cudaMemcpyHostToDevice);

  // --- Host side output data allocation
  cufftComplex *hostOutputData = (cufftComplex*)malloc((DATASIZE / 2 + 1) * BATCH * sizeof(cufftComplex));

  // --- Device side output data allocation
  cufftComplex *deviceOutputData; gpuErrchk(cudaMalloc((void**)&deviceOutputData, (DATASIZE / 2 + 1) * BATCH * sizeof(cufftComplex)));

  // --- Batched 1D FFTs
  // CHECK: hipfftHandle handle;
  cufftHandle handle;
  int rank = 1;                           // --- 1D FFTs
  int n[] = { DATASIZE };                 // --- Size of the Fourier transform
  int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
  int idist = DATASIZE, odist = (DATASIZE / 2 + 1); // --- Distance between batches
  int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
  int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
  int batch = BATCH;                      // --- Number of batched executions
  // CHECK: hipfftPlanMany(&handle, rank, n,
  cufftPlanMany(&handle, rank, n,
    inembed, istride, idist,
    // CHECK: onembed, ostride, odist, HIPFFT_R2C, batch);
    onembed, ostride, odist, CUFFT_R2C, batch);

  // CHECK: hipfftExecR2C(handle, deviceInputData, deviceOutputData);
  cufftExecR2C(handle, deviceInputData, deviceOutputData);

  // --- Device->Host copy of the results
  // CHECK: gpuErrchk(hipMemcpy(hostOutputData, deviceOutputData, (DATASIZE / 2 + 1) * BATCH * sizeof(hipfftComplex), hipMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(hostOutputData, deviceOutputData, (DATASIZE / 2 + 1) * BATCH * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

  for (int i = 0; i<BATCH; i++)
    for (int j = 0; j<(DATASIZE / 2 + 1); j++)
      printf("%i %i %f %fn", i, j, hostOutputData[i*(DATASIZE / 2 + 1) + j].x, hostOutputData[i*(DATASIZE / 2 + 1) + j].y);

  // CHECK: hipfftDestroy(handle);
  cufftDestroy(handle);
  // CHECK: gpuErrchk(hipFree(deviceOutputData));
  // CHECK: gpuErrchk(hipFree(deviceInputData));
  gpuErrchk(cudaFree(deviceOutputData));
  gpuErrchk(cudaFree(deviceInputData));
}
