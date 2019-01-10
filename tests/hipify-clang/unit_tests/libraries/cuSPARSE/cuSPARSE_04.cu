// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include <hipsparse.h>
#include <cusparse.h>

void printMatrix(int m, int n, const float*A, int lda, const char* name)
{
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float Areg = A[row + col * lda];
      printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
    }
  }
}

void printCsr(
  int m,
  int n,
  int nnz,
  // CHECK: const hipsparseMatDescr_t descrA,
  const cusparseMatDescr_t descrA,
  const float *csrValA,
  const int *csrRowPtrA,
  const int *csrColIndA,
  const char* name)
{
  // CHECK: const int base = (hipsparseGetMatIndexBase(descrA) != HIPSPARSE_INDEX_BASE_ONE) ? 0 : 1;
  const int base = (cusparseGetMatIndexBase(descrA) != CUSPARSE_INDEX_BASE_ONE) ? 0 : 1;

  printf("matrix %s is %d-by-%d, nnz=%d, base=%d\n", name, m, n, nnz, base);
  for (int row = 0; row < m; row++) {
    const int start = csrRowPtrA[row] - base;
    const int end = csrRowPtrA[row + 1] - base;
    for (int colidx = start; colidx < end; colidx++) {
      const int col = csrColIndA[colidx] - base;
      const float Areg = csrValA[colidx];
      printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
    }
  }
}

int main(int argc, char*argv[])
{
  // CHECK: hipsparseHandle_t handle = NULL;
  cusparseHandle_t handle = NULL;
  // CHECK: hipStream_t stream = NULL;
  cudaStream_t stream = NULL;
  // CHECK: hipsparseMatDescr_t descrC = NULL;
  cusparseMatDescr_t descrC = NULL;
  // CHECK: hipsparseStatus_t status = HIPSPARSE_STATUS_SUCCESS;
  cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
  // CHECK: hipError_t cudaStat1 = hipSuccess;
  // CHECK: hipError_t cudaStat2 = hipSuccess;
  // CHECK: hipError_t cudaStat3 = hipSuccess;
  // CHECK: hipError_t cudaStat4 = hipSuccess;
  // CHECK: hipError_t cudaStat5 = hipSuccess;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  cudaError_t cudaStat4 = cudaSuccess;
  cudaError_t cudaStat5 = cudaSuccess;
  const int m = 4;
  const int n = 4;
  const int lda = m;
  /*
   *      |    1     0     2     -3  |
   *      |    0     4     0     0   |
   *  A = |    5     0     6     7   |
   *      |    0     8     0     9   |
   *
   */
  const float A[lda*n] = { 1, 0, 5, 0, 0, 4, 0, 8, 2, 0, 6, 0, -3, 0, 7, 9 };
  int* csrRowPtrC = NULL;
  int* csrColIndC = NULL;
  float* csrValC = NULL;

  float *d_A = NULL;
  int *d_csrRowPtrC = NULL;
  int *d_csrColIndC = NULL;
  float *d_csrValC = NULL;

  size_t lworkInBytes = 0;
  char *d_work = NULL;

  int nnzC = 0;

  float threshold = 4.1; /* remove Aij <= 4.1 */
//    float threshold = 0; /* remove zeros */

  printf("example of pruneDense2csr \n");

  printf("prune |A(i,j)| <= threshold \n");
  printf("threshold = %E \n", threshold);

  printMatrix(m, n, A, lda, "A");

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

  /* step 2: configuration of matrix C */
  // CHECK: status = hipsparseCreateMatDescr(&descrC);
  status = cusparseCreateMatDescr(&descrC);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  // CHECK: hipsparseSetMatIndexBase(descrC, HIPSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
  // CHECK: hipsparseSetMatType(descrC, HIPSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_A, sizeof(float)*lda*n);
  cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float)*lda*n);
  // CHECK: cudaStat2 = hipMalloc((void**)&d_csrRowPtrC, sizeof(int)*(m + 1));
  cudaStat2 = cudaMalloc((void**)&d_csrRowPtrC, sizeof(int)*(m + 1));
  // CHECK: assert(hipSuccess == cudaStat1);
  // CHECK: assert(hipSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);

  /* step 3: query workspace */
  // CHECK: cudaStat1 = hipMemcpy(d_A, A, sizeof(float)*lda*n, hipMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_A, A, sizeof(float)*lda*n, cudaMemcpyHostToDevice);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // TODO: status = hipsparseSpruneDense2csr_bufferSizeExt(
  status = cusparseSpruneDense2csr_bufferSizeExt(
    handle,
    m,
    n,
    d_A,
    lda,
    &threshold,
    descrC,
    d_csrValC,
    d_csrRowPtrC,
    d_csrColIndC,
    &lworkInBytes);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);
  // CHECK: if (NULL != d_work) { hipFree(d_work); }
  if (NULL != d_work) { cudaFree(d_work); }
  // CHECK: cudaStat1 = hipMalloc((void**)&d_work, lworkInBytes);
  cudaStat1 = cudaMalloc((void**)&d_work, lworkInBytes);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  /* step 4: compute csrRowPtrC and nnzC */
  // TODO: status = hipsparseSpruneDense2csrNnz(
  status = cusparseSpruneDense2csrNnz(
    handle,
    m,
    n,
    d_A,
    lda,
    &threshold,
    descrC,
    d_csrRowPtrC,
    &nnzC, /* host */
    d_work);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  // CHECK: cudaStat1 = hipDeviceSynchronize();
  cudaStat1 = cudaDeviceSynchronize();
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  printf("nnzC = %d\n", nnzC);
  if (0 == nnzC) {
    printf("C is empty \n");
    return 0;
  }

  /* step 5: compute csrColIndC and csrValC */
  // CHECK: cudaStat1 = hipMalloc((void**)&d_csrColIndC, sizeof(int) * nnzC);
  cudaStat1 = cudaMalloc((void**)&d_csrColIndC, sizeof(int) * nnzC);
  // CHECK: cudaStat2 = hipMalloc((void**)&d_csrValC, sizeof(float) * nnzC);
  cudaStat2 = cudaMalloc((void**)&d_csrValC, sizeof(float) * nnzC);
  // CHECK: assert(hipSuccess == cudaStat1);
  // CHECK: assert(hipSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  // TODO: status = hipsparseSpruneDense2csr(
  status = cusparseSpruneDense2csr(
    handle,
    m,
    n,
    d_A,
    lda,
    &threshold,
    descrC,
    d_csrValC,
    d_csrRowPtrC,
    d_csrColIndC,
    d_work);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  // CHECK: cudaStat1 = hipDeviceSynchronize();
  cudaStat1 = cudaDeviceSynchronize();
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  /* step 6: output C */
  csrRowPtrC = (int*)malloc(sizeof(int)*(m + 1));
  csrColIndC = (int*)malloc(sizeof(int)*nnzC);
  csrValC = (float*)malloc(sizeof(float)*nnzC);
  assert(NULL != csrRowPtrC);
  assert(NULL != csrColIndC);
  assert(NULL != csrValC);
  // CHECK: cudaStat1 = hipMemcpy(csrRowPtrC, d_csrRowPtrC, sizeof(int)*(m + 1), hipMemcpyDeviceToHost);
  cudaStat1 = cudaMemcpy(csrRowPtrC, d_csrRowPtrC, sizeof(int)*(m + 1), cudaMemcpyDeviceToHost);
  // CHECK: cudaStat2 = hipMemcpy(csrColIndC, d_csrColIndC, sizeof(int)*nnzC, hipMemcpyDeviceToHost);
  cudaStat2 = cudaMemcpy(csrColIndC, d_csrColIndC, sizeof(int)*nnzC, cudaMemcpyDeviceToHost);
  // CHECK: cudaStat3 = hipMemcpy(csrValC, d_csrValC, sizeof(float)*nnzC, hipMemcpyDeviceToHost);
  cudaStat3 = cudaMemcpy(csrValC, d_csrValC, sizeof(float)*nnzC, cudaMemcpyDeviceToHost);
  // CHECK: assert(hipSuccess == cudaStat1);
  // CHECK: assert(hipSuccess == cudaStat2);
  // CHECK: assert(hipSuccess == cudaStat3);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);

  printCsr(m, n, nnzC, descrC, csrValC, csrRowPtrC, csrColIndC, "C");

  /* free resources */
  // CHECK: if (d_A) hipFree(d_A);
  if (d_A) cudaFree(d_A);
  // CHECK: if (d_csrRowPtrC) hipFree(d_csrRowPtrC);
  if (d_csrRowPtrC) cudaFree(d_csrRowPtrC);
  // CHECK: if (d_csrColIndC) hipFree(d_csrColIndC);
  if (d_csrColIndC) cudaFree(d_csrColIndC);
  // CHECK: if (d_csrValC) hipFree(d_csrValC);
  if (d_csrValC) cudaFree(d_csrValC);

  if (csrRowPtrC) free(csrRowPtrC);
  if (csrColIndC) free(csrColIndC);
  if (csrValC) free(csrValC);
  // CHECK: if (handle) hipsparseDestroy(handle);
  if (handle) cusparseDestroy(handle);
  // CHECK: if (stream) hipStreamDestroy(stream);
  if (stream) cudaStreamDestroy(stream);
  // CHECK: if (descrC) hipsparseDestroyMatDescr(descrC);
  if (descrC) cusparseDestroyMatDescr(descrC);
  // CHECK: hipDeviceReset();
  cudaDeviceReset();
  return 0;
}
