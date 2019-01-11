// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include <hipsparse.h>
#include <cusparse.h>

// NOTE: CUDA 10.0

/* compute | b - A*x|_inf */
void residaul_eval(
  int n,
  // CHECK: const hipsparseMatDescr_t descrA,
  const cusparseMatDescr_t descrA,
  const float *csrVal,
  const int *csrRowPtr,
  const int *csrColInd,
  const float *b,
  const float *x,
  float *r_nrminf_ptr)
{
  // CHECK: const int base = (hipsparseGetMatIndexBase(descrA) != HIPSPARSE_INDEX_BASE_ONE) ? 0 : 1;
  const int base = (cusparseGetMatIndexBase(descrA) != CUSPARSE_INDEX_BASE_ONE) ? 0 : 1;
  // CHECK: const int lower = (HIPSPARSE_FILL_MODE_LOWER == hipsparseGetMatFillMode(descrA)) ? 1 : 0;
  const int lower = (CUSPARSE_FILL_MODE_LOWER == cusparseGetMatFillMode(descrA)) ? 1 : 0;
  // CHECK: const int unit = (HIPSPARSE_DIAG_TYPE_UNIT == hipsparseGetMatDiagType(descrA)) ? 1 : 0;
  const int unit = (CUSPARSE_DIAG_TYPE_UNIT == cusparseGetMatDiagType(descrA)) ? 1 : 0;

  float r_nrminf = 0;
  for (int row = 0; row < n; row++) {
    const int start = csrRowPtr[row] - base;
    const int end = csrRowPtr[row + 1] - base;
    float dot = 0;
    for (int colidx = start; colidx < end; colidx++) {
      const int col = csrColInd[colidx] - base;
      float Aij = csrVal[colidx];
      float xj = x[col];
      if ((row == col) && unit) {
        Aij = 1.0;
      }
      int valid = (row >= col) && lower ||
        (row <= col) && !lower;
      if (valid) {
        dot += Aij * xj;
      }
    }
    float ri = b[row] - dot;
    r_nrminf = (r_nrminf > fabs(ri)) ? r_nrminf : fabs(ri);
  }
  *r_nrminf_ptr = r_nrminf;
}

int main(int argc, char*argv[])
{
  // CHECK: hipsparseHandle_t handle = NULL;
  cusparseHandle_t handle = NULL;
  // CHECK: hipStream_t stream = NULL;
  cudaStream_t stream = NULL;
  // CHECK: hipsparseMatDescr_t descrA = NULL;
  cusparseMatDescr_t descrA = NULL;
  // NOTE: CUDA 10.0
  // TODO: csrsm2Info_t info = NULL;
  csrsm2Info_t info = NULL;
  // CHECK: hipsparseStatus_t status = HIPSPARSE_STATUS_SUCCESS;
  cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
  // CHECK: hipError_t cudaStat1 = hipSuccess;
  cudaError_t cudaStat1 = cudaSuccess;
  const int nrhs = 2;
  const int n = 4;
  const int nnzA = 9;
  // CHECK: const hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const float h_one = 1.0;
  /*
   *      |    1     0     2     -3  |
   *      |    0     4     0     0   |
   *  A = |    5     0     6     7   |
   *      |    0     8     0     9   |
   *
   *  Regard A as a lower triangle matrix L with non-unit diagonal.
   *             | 1  5 |              | 1          5      |
   *  Given  B = | 2  6 |, X = L \ B = | 0.5       1.5     |
   *             | 3  7 |              | -0.3333   -3      |
   *             | 4  8 |              |  0        -0.4444 |
   */
  const int csrRowPtrA[n + 1] = { 1, 4, 5, 8, 10 };
  const int csrColIndA[nnzA] = { 1, 3, 4, 2, 1, 3, 4, 2, 4 };
  const float csrValA[nnzA] = { 1, 2, -3, 4, 5, 6, 7, 8, 9 };
  const float B[n*nrhs] = { 1,2,3,4,5,6,7,8 };
  float X[n*nrhs];

  int *d_csrRowPtrA = NULL;
  int *d_csrColIndA = NULL;
  float *d_csrValA = NULL;
  float *d_B = NULL;

  size_t lworkInBytes = 0;
  char *d_work = NULL;

  const int algo = 0; /* non-block version */

  printf("example of csrsm2 \n");

  /* step 1: create cusparse handle, bind a stream */
  // CHECK: cudaStat1 = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: status = hipsparseCreate(&handle);
  status = cusparseCreate(&handle);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  status = cusparseSetStream(handle, stream);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  // NOTE: CUDA 10.0
  // TODO: status = hipsparseCreateCsrsm2Info(&info);
  status = cusparseCreateCsrsm2Info(&info);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  /* step 2: configuration of matrix A */
  status = cusparseCreateMatDescr(&descrA);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  /* A is base-1*/
  // CHECK: hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ONE);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
  // CHECK: hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  /* A is lower triangle */
  // CHECK: hipsparseSetMatFillMode(descrA, HIPSPARSE_FILL_MODE_LOWER);
  cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER);
  /* A has non unit diagonal */
  // CHECK: hipsparseSetMatDiagType(descrA, HIPSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_csrRowPtrA, sizeof(int)*(n + 1));
  cudaStat1 = cudaMalloc((void**)&d_csrRowPtrA, sizeof(int)*(n + 1));
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
  // CHECK: cudaStat1 = hipMalloc((void**)&d_B, sizeof(float)*n*nrhs);
  cudaStat1 = cudaMalloc((void**)&d_B, sizeof(float)*n*nrhs);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int)*(n + 1), hipMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int)*(n + 1), cudaMemcpyHostToDevice);
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
  // CHECK: cudaStat1 = hipMemcpy(d_B, B, sizeof(float)*n*nrhs, hipMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_B, B, sizeof(float)*n*nrhs, cudaMemcpyHostToDevice);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  /* step 3: query workspace */
  // NOTE: CUDA 10.0
  // TODO: status = hipsparseScsrsm2_bufferSizeExt(
  // CHECK: HIPSPARSE_OPERATION_NON_TRANSPOSE,
  // CHECK: HIPSPARSE_OPERATION_NON_TRANSPOSE,
  status = cusparseScsrsm2_bufferSizeExt(
    handle,
    algo,
    CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
    CUSPARSE_OPERATION_NON_TRANSPOSE, /* transB */
    n,
    nrhs,
    nnzA,
    &h_one,
    descrA,
    d_csrValA,
    d_csrRowPtrA,
    d_csrColIndA,
    d_B,
    n,   /* ldb */
    info,
    policy,
    &lworkInBytes);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  printf("lworkInBytes  = %lld \n", (long long)lworkInBytes);
  // CHECK: if (NULL != d_work) { hipFree(d_work); }
  if (NULL != d_work) { cudaFree(d_work); }
  // CHECK: cudaStat1 = hipMalloc((void**)&d_work, lworkInBytes);
  cudaStat1 = cudaMalloc((void**)&d_work, lworkInBytes);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  /* step 4: analysis */
  // NOTE: CUDA 10.0
  // TODO: status = hipsparseScsrsm2_analysis(
  // CHECK: HIPSPARSE_OPERATION_NON_TRANSPOSE,
  // CHECK: HIPSPARSE_OPERATION_NON_TRANSPOSE,
  status = cusparseScsrsm2_analysis(
    handle,
    algo,
    CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
    CUSPARSE_OPERATION_NON_TRANSPOSE, /* transB */
    n,
    nrhs,
    nnzA,
    &h_one,
    descrA,
    d_csrValA,
    d_csrRowPtrA,
    d_csrColIndA,
    d_B,
    n,   /* ldb */
    info,
    policy,
    d_work);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  /* step 5: solve L * X = B */
  // NOTE: CUDA 10.0
  // TODO: status = hipsparseScsrsm2_solve(
  // CHECK: HIPSPARSE_OPERATION_NON_TRANSPOSE,
  // CHECK: HIPSPARSE_OPERATION_NON_TRANSPOSE,
  status = cusparseScsrsm2_solve(
    handle,
    algo,
    CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
    CUSPARSE_OPERATION_NON_TRANSPOSE, /* transB */
    n,
    nrhs,
    nnzA,
    &h_one,
    descrA,
    d_csrValA,
    d_csrRowPtrA,
    d_csrColIndA,
    d_B,
    n,   /* ldb */
    info,
    policy,
    d_work);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  // CHECK: cudaStat1 = hipDeviceSynchronize();
  cudaStat1 = cudaDeviceSynchronize();
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  /* step 6:measure residual B - A*X */
  // CHECK: cudaStat1 = hipMemcpy(X, d_B, sizeof(float)*n*nrhs, hipMemcpyDeviceToHost);
  cudaStat1 = cudaMemcpy(X, d_B, sizeof(float)*n*nrhs, cudaMemcpyDeviceToHost);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: hipDeviceSynchronize();
  cudaDeviceSynchronize();

  printf("==== x1 = inv(A)*b1 \n");
  for (int j = 0; j < n; j++) {
    printf("x1[%d] = %f\n", j, X[j]);
  }
  float r1_nrminf;
  residaul_eval(
    n,
    descrA,
    csrValA,
    csrRowPtrA,
    csrColIndA,
    B,
    X,
    &r1_nrminf
  );
  printf("|b1 - A*x1| = %E\n", r1_nrminf);

  printf("==== x2 = inv(A)*b2 \n");
  for (int j = 0; j < n; j++) {
    printf("x2[%d] = %f\n", j, X[n + j]);
  }
  float r2_nrminf;
  residaul_eval(
    n,
    descrA,
    csrValA,
    csrRowPtrA,
    csrColIndA,
    B + n,
    X + n,
    &r2_nrminf
  );
  printf("|b2 - A*x2| = %E\n", r2_nrminf);

  /* free resources */
  // CHECK: if (d_csrRowPtrA) hipFree(d_csrRowPtrA);
  if (d_csrRowPtrA) cudaFree(d_csrRowPtrA);
  // CHECK: if (d_csrColIndA) hipFree(d_csrColIndA);
  if (d_csrColIndA) cudaFree(d_csrColIndA);
  // CHECK: if (d_csrValA) hipFree(d_csrValA);
  if (d_csrValA) cudaFree(d_csrValA);
  // CHECK: if (d_B) hipFree(d_B);
  if (d_B) cudaFree(d_B);
  // CHECK: if (handle) hipsparseDestroy(handle);
  if (handle) cusparseDestroy(handle);
  // CHECK: if (stream) hipStreamDestroy(stream);
  if (stream) cudaStreamDestroy(stream);
  // CHECK: if (descrA) hipsparseDestroyMatDescr(descrA);
  if (descrA) cusparseDestroyMatDescr(descrA);
  // NOTE: CUDA 10.0
  // TODO: if (info) hipsparseDestroyCsrsm2Info(info);
  if (info) cusparseDestroyCsrsm2Info(info);
  // CHECK: hipDeviceReset();
  cudaDeviceReset();

  return 0;
}
