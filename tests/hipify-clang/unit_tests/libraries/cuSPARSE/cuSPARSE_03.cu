// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "hipsparse.h"
#include "cusparse.h"

int main(int argc, char*argv[])
{
  // CHECK: hipsparseHandle_t handle = NULL;
  cusparseHandle_t handle = NULL;
  // CHECK: hipStream_t stream = NULL;
  cudaStream_t stream = NULL;
  // CHECK: hipsparseStatus_t status = HIPSPARSE_STATUS_SUCCESS;
  cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
  // CHECK: hipError_t cudaStat1 = hipSuccess;
  // CHECK: hipError_t cudaStat2 = hipSuccess;
  // CHECK: hipError_t cudaStat3 = hipSuccess;
  // CHECK: hipError_t cudaStat4 = hipSuccess;
  // CHECK: hipError_t cudaStat5 = hipSuccess;
  // CHECK: hipError_t cudaStat6 = hipSuccess;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  cudaError_t cudaStat4 = cudaSuccess;
  cudaError_t cudaStat5 = cudaSuccess;
  cudaError_t cudaStat6 = cudaSuccess;

  /*
   * A is a 3x3 sparse matrix
   *     | 1 2 0 |
   * A = | 0 5 0 |
   *     | 0 8 0 |
   */
  const int m = 3;
  const int n = 3;
  const int nnz = 4;

#if 0
  /* index starts at 0 */
  int h_cooRows[nnz] = { 2, 1, 0, 0 };
  int h_cooCols[nnz] = { 1, 1, 0, 1 };
#else
  /* index starts at -2 */
  int h_cooRows[nnz] = { 0, -1, -2, -2 };
  int h_cooCols[nnz] = { -1, -1, -2, -1 };
#endif
  double h_cooVals[nnz] = { 8.0, 5.0, 1.0, 2.0 };
  int h_P[nnz];

  int *d_cooRows = NULL;
  int *d_cooCols = NULL;
  int *d_P = NULL;
  double *d_cooVals = NULL;
  double *d_cooVals_sorted = NULL;
  size_t pBufferSizeInBytes = 0;
  void *pBuffer = NULL;

  printf("m = %d, n = %d, nnz=%d \n", m, n, nnz);

  /* step 1: create cusparse handle, bind a stream */
  // CHECK: cudaStat1 = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: status = hipsparseCreate(&handle);
  status = cusparseCreate(&handle);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  // CHECK: status = hipsparseSetStream(handle, stream);
  status = cusparseSetStream(handle, stream);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  /* step 2: allocate buffer */
  // TODO: status = hipsparseXcoosort_bufferSizeExt(
  status = cusparseXcoosort_bufferSizeExt(
    handle,
    m,
    n,
    nnz,
    d_cooRows,
    d_cooCols,
    &pBufferSizeInBytes
  );
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  printf("pBufferSizeInBytes = %lld bytes \n", (long long)pBufferSizeInBytes);

  // CHECK: cudaStat1 = hipMalloc(&d_cooRows, sizeof(int)*nnz);
  cudaStat1 = cudaMalloc(&d_cooRows, sizeof(int)*nnz);
  // CHECK: cudaStat2 = hipMalloc(&d_cooCols, sizeof(int)*nnz);
  cudaStat2 = cudaMalloc(&d_cooCols, sizeof(int)*nnz);
  // CHECK: cudaStat3 = hipMalloc(&d_P, sizeof(int)*nnz);
  cudaStat3 = cudaMalloc(&d_P, sizeof(int)*nnz);
  // CHECK: cudaStat4 = hipMalloc(&d_cooVals, sizeof(double)*nnz);
  cudaStat4 = cudaMalloc(&d_cooVals, sizeof(double)*nnz);
  // CHECK: cudaStat5 = hipMalloc(&d_cooVals_sorted, sizeof(double)*nnz);
  cudaStat5 = cudaMalloc(&d_cooVals_sorted, sizeof(double)*nnz);
  // CHECK: cudaStat6 = hipMalloc(&pBuffer, sizeof(char)* pBufferSizeInBytes);
  cudaStat6 = cudaMalloc(&pBuffer, sizeof(char)* pBufferSizeInBytes);

  // CHECK: assert(hipSuccess == cudaStat1);
  // CHECK: assert(hipSuccess == cudaStat2);
  // CHECK: assert(hipSuccess == cudaStat3);
  // CHECK: assert(hipSuccess == cudaStat4);
  // CHECK: assert(hipSuccess == cudaStat5);
  // CHECK: assert(hipSuccess == cudaStat6);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);
  assert(cudaSuccess == cudaStat4);
  assert(cudaSuccess == cudaStat5);
  assert(cudaSuccess == cudaStat6);

  // CHECK: cudaStat1 = hipMemcpy(d_cooRows, h_cooRows, sizeof(int)*nnz, hipMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_cooRows, h_cooRows, sizeof(int)*nnz, cudaMemcpyHostToDevice);
  // CHECK: cudaStat2 = hipMemcpy(d_cooCols, h_cooCols, sizeof(int)*nnz, hipMemcpyHostToDevice);
  cudaStat2 = cudaMemcpy(d_cooCols, h_cooCols, sizeof(int)*nnz, cudaMemcpyHostToDevice);
  // CHECK: cudaStat3 = hipMemcpy(d_cooVals, h_cooVals, sizeof(double)*nnz, hipMemcpyHostToDevice);
  cudaStat3 = cudaMemcpy(d_cooVals, h_cooVals, sizeof(double)*nnz, cudaMemcpyHostToDevice);
  // CHECK: cudaStat4 = hipDeviceSynchronize();
  cudaStat4 = cudaDeviceSynchronize();

  // CHECK: assert(hipSuccess == cudaStat1);
  // CHECK: assert(hipSuccess == cudaStat2);
  // CHECK: assert(hipSuccess == cudaStat3);
  // CHECK: assert(hipSuccess == cudaStat4);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);
  assert(cudaSuccess == cudaStat4);

  /* step 3: setup permutation vector P to identity */
  // TODO: status = hipsparseCreateIdentityPermutation(
  status = cusparseCreateIdentityPermutation(
    handle,
    nnz,
    d_P);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  /* step 4: sort COO format by Row */
  // TODO: status = hipsparseXcoosortByRow(
  status = cusparseXcoosortByRow(
    handle,
    m,
    n,
    nnz,
    d_cooRows,
    d_cooCols,
    d_P,
    pBuffer
  );
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  /* step 5: gather sorted cooVals */
  // CHECK: status = hipsparseDgthr(
  // CHECK: HIPSPARSE_INDEX_BASE_ZERO
  status = cusparseDgthr(
    handle,
    nnz,
    d_cooVals,
    d_cooVals_sorted,
    d_P,
    CUSPARSE_INDEX_BASE_ZERO
  );
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  /* wait until the computation is done */
  // CHECK: cudaStat1 = hipDeviceSynchronize();
  cudaStat1 = cudaDeviceSynchronize();
  // CHECK: cudaStat2 = hipMemcpy(h_cooRows, d_cooRows, sizeof(int)*nnz, hipMemcpyDeviceToHost);
  cudaStat2 = cudaMemcpy(h_cooRows, d_cooRows, sizeof(int)*nnz, cudaMemcpyDeviceToHost);
  // CHECK: cudaStat3 = hipMemcpy(h_cooCols, d_cooCols, sizeof(int)*nnz, hipMemcpyDeviceToHost);
  cudaStat3 = cudaMemcpy(h_cooCols, d_cooCols, sizeof(int)*nnz, cudaMemcpyDeviceToHost);
  // CHECK: cudaStat4 = hipMemcpy(h_P, d_P, sizeof(int)*nnz, hipMemcpyDeviceToHost);
  cudaStat4 = cudaMemcpy(h_P, d_P, sizeof(int)*nnz, cudaMemcpyDeviceToHost);
  // CHECK: cudaStat5 = hipMemcpy(h_cooVals, d_cooVals_sorted, sizeof(double)*nnz, hipMemcpyDeviceToHost);
  cudaStat5 = cudaMemcpy(h_cooVals, d_cooVals_sorted, sizeof(double)*nnz, cudaMemcpyDeviceToHost);
  // CHECK: cudaStat6 = hipDeviceSynchronize();
  cudaStat6 = cudaDeviceSynchronize();
  // CHECK: assert(hipSuccess == cudaStat1);
  // CHECK: assert(hipSuccess == cudaStat2);
  // CHECK: assert(hipSuccess == cudaStat3);
  // CHECK: assert(hipSuccess == cudaStat4);
  // CHECK: assert(hipSuccess == cudaStat5);
  // CHECK: assert(hipSuccess == cudaStat6);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);
  assert(cudaSuccess == cudaStat4);
  assert(cudaSuccess == cudaStat5);
  assert(cudaSuccess == cudaStat6);

  printf("sorted coo: \n");
  for (int j = 0; j < nnz; j++) {
    printf("(%d, %d, %f) \n", h_cooRows[j], h_cooCols[j], h_cooVals[j]);
  }

  for (int j = 0; j < nnz; j++) {
    printf("P[%d] = %d \n", j, h_P[j]);
  }

  /* free resources */
  // CHECK: if (d_cooRows) hipFree(d_cooRows);
  if (d_cooRows) cudaFree(d_cooRows);
  // CHECK: if (d_cooCols) hipFree(d_cooCols);
  if (d_cooCols) cudaFree(d_cooCols);
  // CHECK: if (d_P) hipFree(d_P);
  if (d_P) cudaFree(d_P);
  // CHECK: if (d_cooVals) hipFree(d_cooVals);
  if (d_cooVals) cudaFree(d_cooVals);
  // CHECK: if (d_cooVals_sorted) hipFree(d_cooVals_sorted);
  if (d_cooVals_sorted) cudaFree(d_cooVals_sorted);
  // CHECK: if (pBuffer) hipFree(pBuffer);
  if (pBuffer) cudaFree(pBuffer);
  // if (handle) hipsparseDestroy(handle);
  if (handle) cusparseDestroy(handle);
  // CHECK: if (stream) hipStreamDestroy(stream);
  if (stream) cudaStreamDestroy(stream);
  // CHECK: hipDeviceReset();
  cudaDeviceReset();
  return 0;
}
