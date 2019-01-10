// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include <hipsparse.h>
#include <cusparse.h>
// CHECK: #include <hipblas.h>
#include <cublas_v2.h>

// NOTE: CUDA 10.0

/*
 * compute | b - A*x|_inf
 */
void residaul_eval(
  int n,
  const float *dl,
  const float *d,
  const float *du,
  const float *b,
  const float *x,
  float *r_nrminf_ptr)
{
  float r_nrminf = 0;
  for (int i = 0; i < n; i++) {
    float dot = 0;
    if (i > 0) {
      dot += dl[i] * x[i - 1];
    }
    dot += d[i] * x[i];
    if (i < (n - 1)) {
      dot += du[i] * x[i + 1];
    }
    float ri = b[i] - dot;
    r_nrminf = (r_nrminf > fabs(ri)) ? r_nrminf : fabs(ri);
  }

  *r_nrminf_ptr = r_nrminf;
}

int main(int argc, char*argv[])
{
  // CHECK: hipsparseHandle_t cusparseH = NULL;
  cusparseHandle_t cusparseH = NULL;
  // CHECK: hipblasHandle_t cublasH = NULL;
  cublasHandle_t cublasH = NULL;
  // CHECK: hipStream_t stream = NULL;
  cudaStream_t stream = NULL;
  // CHECK: hipsparseStatus_t status = HIPSPARSE_STATUS_SUCCESS;
  cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
  // CHECK: hipblasStatus_t cublasStat = HIPBLAS_STATUS_SUCCESS;
  cublasStatus_t cublasStat = CUBLAS_STATUS_SUCCESS;
  // CHECK: hipError_t cudaStat1 = hipSuccess;
  cudaError_t cudaStat1 = cudaSuccess;

  const int n = 3;
  const int batchSize = 2;
  /*
   *      |    1     6     0  |       | 1 |       | -0.603960 |
   *  A1 =|    4     2     7  |, b1 = | 2 |, x1 = |  0.267327 |
   *      |    0     5     3  |       | 3 |       |  0.554455 |
   *
   *      |    8    13     0  |       | 4 |       | -0.063291 |
   *  A2 =|   11     9    14  |, b2 = | 5 |, x2 = |  0.346641 |
   *      |    0    12    10  |       | 6 |       |  0.184031 |
   */

   /*
    * A = (dl, d, du), B and X are in aggregate format
    */
  const float dl[n * batchSize] = { 0, 4, 5,  0, 11, 12 };
  const float  d[n * batchSize] = { 1, 2, 3,  8,  9, 10 };
  const float du[n * batchSize] = { 6, 7, 0, 13, 14,  0 };
  const float  B[n * batchSize] = { 1, 2, 3,  4,  5,  6 };
  float X[n * batchSize]; /* Xj = Aj \ Bj */

/* device memory
 * (d_dl0, d_d0, d_du0) is aggregate format
 * (d_dl, d_d, d_du) is interleaved format
 */
  float *d_dl0 = NULL;
  float *d_d0 = NULL;
  float *d_du0 = NULL;
  float *d_dl = NULL;
  float *d_d = NULL;
  float *d_du = NULL;
  float *d_B = NULL;
  float *d_X = NULL;

  size_t lworkInBytes = 0;
  char *d_work = NULL;

  /*
   * algo = 0: cuThomas (unstable)
   * algo = 1: LU with pivoting (stable)
   * algo = 2: QR (stable)
   */
  const int algo = 2;

  const float h_one = 1;
  const float h_zero = 0;

  printf("example of gtsv (interleaved format) \n");
  printf("choose algo = 0,1,2 to select different algorithms \n");
  printf("n = %d, batchSize = %d, algo = %d \n", n, batchSize, algo);

  /* step 1: create cusparse/cublas handle, bind a stream */
  // CHECK: cudaStat1 = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: status = hipsparseCreate(&cusparseH);
  status = cusparseCreate(&cusparseH);
  //CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  // CHECK: status = hipsparseSetStream(cusparseH, stream);
  status = cusparseSetStream(cusparseH, stream);
  //CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  // CHECK: cublasStat = hipblasCreate(&cublasH);
  cublasStat = cublasCreate(&cublasH);
  // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
  assert(CUBLAS_STATUS_SUCCESS == cublasStat);
  // CHECK: cublasStat = hipblasSetStream(cublasH, stream);
  cublasStat = cublasSetStream(cublasH, stream);
  // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
  assert(CUBLAS_STATUS_SUCCESS == cublasStat);

  /* step 2: allocate device memory */
  // CHECK: cudaStat1 = hipMalloc((void**)&d_dl0, sizeof(float)*n*batchSize);
  cudaStat1 = cudaMalloc((void**)&d_dl0, sizeof(float)*n*batchSize);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_d0, sizeof(float)*n*batchSize);
  cudaStat1 = cudaMalloc((void**)&d_d0, sizeof(float)*n*batchSize);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_du0, sizeof(float)*n*batchSize);
  cudaStat1 = cudaMalloc((void**)&d_du0, sizeof(float)*n*batchSize);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_dl, sizeof(float)*n*batchSize);
  cudaStat1 = cudaMalloc((void**)&d_dl, sizeof(float)*n*batchSize);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_d, sizeof(float)*n*batchSize);
  cudaStat1 = cudaMalloc((void**)&d_d, sizeof(float)*n*batchSize);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_du, sizeof(float)*n*batchSize);
  cudaStat1 = cudaMalloc((void**)&d_du, sizeof(float)*n*batchSize);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_B, sizeof(float)*n*batchSize);
  cudaStat1 = cudaMalloc((void**)&d_B, sizeof(float)*n*batchSize);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_X, sizeof(float)*n*batchSize);
  cudaStat1 = cudaMalloc((void**)&d_X, sizeof(float)*n*batchSize);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  /* step 3: prepare data in device, interleaved format */
  // CHECK: cudaStat1 = hipMemcpy(d_dl0, dl, sizeof(float)*n*batchSize, hipMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_dl0, dl, sizeof(float)*n*batchSize, cudaMemcpyHostToDevice);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMemcpy(d_d0, d, sizeof(float)*n*batchSize, hipMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_d0, d, sizeof(float)*n*batchSize, cudaMemcpyHostToDevice);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMemcpy(d_du0, du, sizeof(float)*n*batchSize, hipMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_du0, du, sizeof(float)*n*batchSize, cudaMemcpyHostToDevice);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: cudaStat1 = hipMemcpy(d_B, B, sizeof(float)*n*batchSize, hipMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_B, B, sizeof(float)*n*batchSize, cudaMemcpyHostToDevice);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: hipDeviceSynchronize();
  cudaDeviceSynchronize();
  /* convert dl to interleaved format
   *  dl = transpose(dl0)
   */
  // CHECK: cublasStat = hipblasSgeam(
  // CHECK: HIPBLAS_OP_T,
  // CHECK: HIPBLAS_OP_T,
  cublasStat = cublasSgeam(
    cublasH,
    CUBLAS_OP_T, /* transa */
    CUBLAS_OP_T, /* transb, don't care */
    batchSize, /* number of rows of dl */
    n,         /* number of columns of dl */
    &h_one,
    d_dl0,  /* dl0 is n-by-batchSize */
    n, /* leading dimension of dl0 */
    &h_zero,
    NULL,
    n,         /* don't cae */
    d_dl,      /* dl is batchSize-by-n */
    batchSize  /* leading dimension of dl */
  );
  // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
  assert(CUBLAS_STATUS_SUCCESS == cublasStat);
  /* convert d to interleaved format
   *  d = transpose(d0)
   */
  // CHECK: cublasStat = hipblasSgeam(
  // CHECK: HIPBLAS_OP_T,
  // CHECK: HIPBLAS_OP_T,
  cublasStat = cublasSgeam(
    cublasH,
    CUBLAS_OP_T, /* transa */
    CUBLAS_OP_T, /* transb, don't care */
    batchSize, /* number of rows of d */
    n,         /* number of columns of d */
    &h_one,
    d_d0, /* d0 is n-by-batchSize */
    n, /* leading dimension of d0 */
    &h_zero,
    NULL,
    n,         /* don't cae */
    d_d,       /* d is batchSize-by-n */
    batchSize  /* leading dimension of d */
  );
  // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
  assert(CUBLAS_STATUS_SUCCESS == cublasStat);

  /* convert du to interleaved format
   *  du = transpose(du0)
   */
  // CHECK: cublasStat = hipblasSgeam(
  // CHECK: HIPBLAS_OP_T,
  // CHECK: HIPBLAS_OP_T,
  cublasStat = cublasSgeam(
    cublasH,
    CUBLAS_OP_T, /* transa */
    CUBLAS_OP_T, /* transb, don't care */
    batchSize, /* number of rows of du */
    n,         /* number of columns of du */
    &h_one,
    d_du0, /* du0 is n-by-batchSize */
    n, /* leading dimension of du0 */
    &h_zero,
    NULL,
    n,         /* don't cae */
    d_du,      /* du is batchSize-by-n */
    batchSize  /* leading dimension of du */
  );
  // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
  assert(CUBLAS_STATUS_SUCCESS == cublasStat);

  /* convert B to interleaved format
   *  X = transpose(B)
   */
  // CHECK: cublasStat = hipblasSgeam(
  // CHECK: HIPBLAS_OP_T,
  // CHECK: HIPBLAS_OP_T,
  cublasStat = cublasSgeam(
    cublasH,
    CUBLAS_OP_T, /* transa */
    CUBLAS_OP_T, /* transb, don't care */
    batchSize, /* number of rows of X */
    n,         /* number of columns of X */
    &h_one,
    d_B, /* B is n-by-batchSize */
    n, /* leading dimension of B */
    &h_zero,
    NULL,
    n,         /* don't cae */
    d_X,       /* X is batchSize-by-n */
    batchSize  /* leading dimension of X */
  );
  // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
  assert(CUBLAS_STATUS_SUCCESS == cublasStat);
  /* step 4: prepare workspace */
  // NOTE: CUDA 10.0
  // TODO: status = hipsparseSgtsvInterleavedBatch_bufferSizeExt(
  status = cusparseSgtsvInterleavedBatch_bufferSizeExt(
    cusparseH,
    algo,
    n,
    d_dl,
    d_d,
    d_du,
    d_X,
    batchSize,
    &lworkInBytes);
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);

  printf("lworkInBytes = %lld \n", (long long)lworkInBytes);
  // CHECK: cudaStat1 = hipMalloc((void**)&d_work, lworkInBytes);
  cudaStat1 = cudaMalloc((void**)&d_work, lworkInBytes);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  /* step 5: solve Aj*xj = bj */
  // NOTE: CUDA 10.0
  // TODO: status = hipsparseSgtsvInterleavedBatch(
   status = cusparseSgtsvInterleavedBatch(
    cusparseH,
    algo,
    n,
    d_dl,
    d_d,
    d_du,
    d_X,
    batchSize,
    d_work);
  // CHECK: cudaStat1 = hipDeviceSynchronize();
  cudaStat1 = cudaDeviceSynchronize();
  // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == status);
  assert(CUSPARSE_STATUS_SUCCESS == status);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);

  /* step 6: convert X back to aggregate format  */
      /* B = transpose(X) */
  // CHECK: cublasStat = hipblasSgeam(
  // CHECK: HIPBLAS_OP_T,
  // CHECK: HIPBLAS_OP_T,
  cublasStat = cublasSgeam(
    cublasH,
    CUBLAS_OP_T, /* transa */
    CUBLAS_OP_T, /* transb, don't care */
    n,         /* number of rows of B */
    batchSize, /* number of columns of B */
    &h_one,
    d_X,       /* X is batchSize-by-n */
    batchSize, /* leading dimension of X */
    &h_zero,
    NULL,
    n, /* don't cae */
    d_B, /* B is n-by-batchSize */
    n  /* leading dimension of B */
  );
  // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
  assert(CUBLAS_STATUS_SUCCESS == cublasStat);
  // CHECK: hipDeviceSynchronize();
  cudaDeviceSynchronize();

  /* step 7: residual evaluation */
  // CHECK: cudaStat1 = hipMemcpy(X, d_B, sizeof(float)*n*batchSize, hipMemcpyDeviceToHost);
  cudaStat1 = cudaMemcpy(X, d_B, sizeof(float)*n*batchSize, cudaMemcpyDeviceToHost);
  // CHECK: assert(hipSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat1);
  // CHECK: hipDeviceSynchronize();
  cudaDeviceSynchronize();
  printf("==== x1 = inv(A1)*b1 \n");
  for (int j = 0; j < n; j++) {
    printf("x1[%d] = %f\n", j, X[j]);
  }

  float r1_nrminf;
  residaul_eval(
    n,
    dl,
    d,
    du,
    B,
    X,
    &r1_nrminf
  );
  printf("|b1 - A1*x1| = %E\n", r1_nrminf);

  printf("\n==== x2 = inv(A2)*b2 \n");
  for (int j = 0; j < n; j++) {
    printf("x2[%d] = %f\n", j, X[n + j]);
  }

  float r2_nrminf;
  residaul_eval(
    n,
    dl + n,
    d + n,
    du + n,
    B + n,
    X + n,
    &r2_nrminf
  );
  printf("|b2 - A2*x2| = %E\n", r2_nrminf);

  /* free resources */
  // CHECK: if (d_dl0) hipFree(d_dl0);
  if (d_dl0) cudaFree(d_dl0);
  // CHECK: if (d_d0) hipFree(d_d0);
  if (d_d0) cudaFree(d_d0);
  // CHECK: if (d_du0) hipFree(d_du0);
  if (d_du0) cudaFree(d_du0);
  // CHECK: if (d_dl) hipFree(d_dl);
  if (d_dl) cudaFree(d_dl);
  // CHECK: if (d_d) hipFree(d_d);
  if (d_d) cudaFree(d_d);
  // CHECK: if (d_du) hipFree(d_du);
  if (d_du) cudaFree(d_du);
  // CHECK: if (d_B) hipFree(d_B);
  if (d_B) cudaFree(d_B);
  // CHECK: if (d_X) hipFree(d_X);
  if (d_X) cudaFree(d_X);
  // CHECK: if (cusparseH) hipsparseDestroy(cusparseH);
  if (cusparseH) cusparseDestroy(cusparseH);
  // CHECK: if (cublasH) hipblasDestroy(cublasH);
  if (cublasH) cublasDestroy(cublasH);
  // CHECK: if (stream) hipStreamDestroy(stream);
  if (stream) cudaStreamDestroy(stream);
  // CHECK: hipDeviceReset();
  cudaDeviceReset();

  return 0;
}
