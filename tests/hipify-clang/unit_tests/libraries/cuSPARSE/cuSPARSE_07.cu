// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include <hipsparse.h>
#include <cusparse.h>

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

  printf("matrix %s is %d-by-%d, nnz=%d, base=%d, output base-1\n", name, m, n, nnz, base);
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
  // CHECK: hipsparseMatDescr_t descrA = NULL;
  cusparseMatDescr_t descrA = NULL;
  // CHECK: hipsparseMatDescr_t descrC = NULL;
  cusparseMatDescr_t descrC = NULL;
  pruneInfo_t info = NULL;
  // CHECK: hipsparseStatus_t status = HIPSPARSE_STATUS_SUCCESS;
  cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
  // CHECK: hipError_t cudaStat1 = hipSuccess;
  cudaError_t cudaStat1 = cudaSuccess;
  const int m = 4;
  const int n = 4;
  const int nnzA = 9;
  /*
   *      |    1     0     2     -3  |
   *      |    0     4     0     0   |
   *  A = |    5     0     6     7   |
   *      |    0     8     0     9   |
   *
   */

  const int csrRowPtrA[m + 1] = { 1, 4, 5, 8, 10 };
  const int csrColIndA[nnzA] = { 1, 3, 4, 2, 1, 3, 4, 2, 4 };
  const float csrValA[nnzA] = { 1, 2, -3, 4, 5, 6, 7, 8, 9 };

  int* csrRowPtrC = NULL;
  int* csrColIndC = NULL;
  float* csrValC = NULL;

  int *d_csrRowPtrA = NULL;
  int *d_csrColIndA = NULL;
  float *d_csrValA = NULL;

  int *d_csrRowPtrC = NULL;
  int *d_csrColIndC = NULL;
  float *d_csrValC = NULL;

  size_t lworkInBytes = 0;
  char *d_work = NULL;

  int nnzC = 0;

  float percentage = 20; /* remove 20% of nonzeros */

  printf("example of pruneCsr2csrByPercentage \n");

  printf("prune %.1f percent of nonzeros \n", percentage);

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
  // TODO: status = hipsparseCreatePruneInfo(&info);
  status = cusparseCreatePruneInfo(&info);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  /* step 2: configuration of matrix C */
  // CHECK: status = hipsparseCreateMatDescr(&descrA);
  status = cusparseCreateMatDescr(&descrA);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  /* A is base-1*/
  // CHECK: hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ONE);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
  // CHECK: hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  // CHECK: status = hipsparseCreateMatDescr(&descrC);
  status = cusparseCreateMatDescr(&descrC);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  /* C is base-0 */
  // CHECK: hipsparseSetMatIndexBase(descrC, HIPSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
  // CHECK: hipsparseSetMatType(descrC, HIPSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);

  printCsr(m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, "A");
  // CHECK: cudaStat1 = hipMalloc((void**)&d_csrRowPtrA, sizeof(int)*(m + 1));
  cudaStat1 = cudaMalloc((void**)&d_csrRowPtrA, sizeof(int)*(m + 1));
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_csrColIndA, sizeof(int)*nnzA);
  cudaStat1 = cudaMalloc((void**)&d_csrColIndA, sizeof(int)*nnzA);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_csrValA, sizeof(float)*nnzA);
  cudaStat1 = cudaMalloc((void**)&d_csrValA, sizeof(float)*nnzA);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_csrRowPtrC, sizeof(int)*(m + 1));
  cudaStat1 = cudaMalloc((void**)&d_csrRowPtrC, sizeof(int)*(m + 1));
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int)*(m + 1), hipMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int)*(m + 1), cudaMemcpyHostToDevice);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMemcpy(d_csrColIndA, csrColIndA, sizeof(int)*nnzA, hipMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int)*nnzA, cudaMemcpyHostToDevice);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMemcpy(d_csrValA, csrValA, sizeof(float)*nnzA, hipMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_csrValA, csrValA, sizeof(float)*nnzA, cudaMemcpyHostToDevice);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  /* step 3: query workspace */
  // TODO: status = hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(
  status = cusparseSpruneCsr2csrByPercentage_bufferSizeExt(
    handle,
    m,
    n,
    nnzA,
    descrA,
    d_csrValA,
    d_csrRowPtrA,
    d_csrColIndA,
    percentage,
    descrC,
    d_csrValC,
    d_csrRowPtrC,
    d_csrColIndC,
    info,
    &lworkInBytes);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  printf("lworkInBytes = %lld \n", (long long)lworkInBytes);
  // CHECK: if (NULL != d_work) { hipFree(d_work); }
  if (NULL != d_work) { cudaFree(d_work); }
  // CHECK: cudaStat1 = hipMalloc((void**)&d_work, lworkInBytes);
  cudaStat1 = cudaMalloc((void**)&d_work, lworkInBytes);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  /* step 4: compute csrRowPtrC and nnzC */
  // TODO: status = hipsparseSpruneCsr2csrNnzByPercentage(
  status = cusparseSpruneCsr2csrNnzByPercentage(
    handle,
    m,
    n,
    nnzA,
    descrA,
    d_csrValA,
    d_csrRowPtrA,
    d_csrColIndA,
    percentage,
    descrC,
    d_csrRowPtrC,
    &nnzC, /* host */
    info,
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
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_csrValC, sizeof(float) * nnzC);
  cudaStat1 = cudaMalloc((void**)&d_csrValC, sizeof(float) * nnzC);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // TODO: status = hipsparseSpruneCsr2csrByPercentage(
  status = cusparseSpruneCsr2csrByPercentage(
    handle,
    m,
    n,
    nnzA,
    descrA,
    d_csrValA,
    d_csrRowPtrA,
    d_csrColIndA,
    percentage,
    descrC,
    d_csrValC,
    d_csrRowPtrC,
    d_csrColIndC,
    info,
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
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMemcpy(csrColIndC, d_csrColIndC, sizeof(int)*nnzC, hipMemcpyDeviceToHost);
  cudaStat1 = cudaMemcpy(csrColIndC, d_csrColIndC, sizeof(int)*nnzC, cudaMemcpyDeviceToHost);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMemcpy(csrValC, d_csrValC, sizeof(float)*nnzC, hipMemcpyDeviceToHost);
  cudaStat1 = cudaMemcpy(csrValC, d_csrValC, sizeof(float)*nnzC, cudaMemcpyDeviceToHost);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  printCsr(m, n, nnzC, descrC, csrValC, csrRowPtrC, csrColIndC, "C");

  /* free resources */
  // CHECK: if (d_csrRowPtrA) hipFree(d_csrRowPtrA);
  if (d_csrRowPtrA) cudaFree(d_csrRowPtrA);
  // CHECK: if (d_csrColIndA) hipFree(d_csrColIndA);
  if (d_csrColIndA) cudaFree(d_csrColIndA);
  // CHECK: if (d_csrValA) hipFree(d_csrValA);
  if (d_csrValA) cudaFree(d_csrValA);
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
  // CHECK: if (descrA) hipsparseDestroyMatDescr(descrA);
  if (descrA) cusparseDestroyMatDescr(descrA);
  // CHECK: if (descrC) hipsparseDestroyMatDescr(descrC);
  if (descrC) cusparseDestroyMatDescr(descrC);
  // TODO: if (info) hipsparseDestroyPruneInfo(info);
  if (info) cusparseDestroyPruneInfo(info);
  // CHECK: hipDeviceReset();
  cudaDeviceReset();

  return 0;
}
