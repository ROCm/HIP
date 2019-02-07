// RUN: %run_test hipify "%s" "%t" %hipify_args "-roc" %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// CHECK: #include "rocblas.h"
#include "cublas.h"
#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
static __inline__ void modify(float *m, int ldm, int n, int p, int q, float
  alpha, float beta) {
  // CHECK: rocblas_sscal(n - p, alpha, &m[IDX2C(p, q, ldm)], ldm);
  // CHECK: rocblas_sscal(ldm - p, beta, &m[IDX2C(p, q, ldm)], 1);
  cublasSscal(n - p, alpha, &m[IDX2C(p, q, ldm)], ldm);
  cublasSscal(ldm - p, beta, &m[IDX2C(p, q, ldm)], 1);
}
int main(void) {
  int i, j;
  // CHECK: rocblas_status stat;
  cublasStatus stat;
  float* devPtrA;
  float* a = 0;
  a = (float *)malloc(M * N * sizeof(*a));
  if (!a) {
    printf("host memory allocation failed");
    return EXIT_FAILURE;
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      a[IDX2C(i, j, M)] = (float)(i * M + j + 1);
    }
  }
  // cublasInit is not supported yet
  cublasInit();
  // cublasAlloc is not supported yet
  stat = cublasAlloc(M*N, sizeof(*a), (void**)&devPtrA);
  // CHECK: if (stat != rocblas_status_success) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("device memory allocation failed");
    // cublasShutdown is not supported yet
    cublasShutdown();
    return EXIT_FAILURE;
  }
  // CHECK: stat = rocblas_set_matrix(M, N, sizeof(*a), a, M, devPtrA, M);
  stat = cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
  // CHECK: if (stat != rocblas_status_success) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    // cublasFree is not supported yet
    cublasFree(devPtrA);
    // cublasShutdown is not supported yet
    cublasShutdown();
    return EXIT_FAILURE;
  }
  modify(devPtrA, M, N, 1, 2, 16.0f, 12.0f);
  // CHECK: stat = rocblas_get_matrix(M, N, sizeof(*a), devPtrA, M, a, M);
  stat = cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
  // CHECK: if (stat != rocblas_status_success) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data upload failed");
    // cublasFree is not supported yet
    cublasFree(devPtrA);
    // cublasShutdown is not supported yet
    cublasShutdown();
    return EXIT_FAILURE;
  }
  // cublasFree is not supported yet
  cublasFree(devPtrA);
  // cublasShutdown is not supported yet
  cublasShutdown();
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      printf("%7.0f", a[IDX2C(i, j, M)]);
    }
    printf("\n");
  }
  free(a);
  return EXIT_SUCCESS;
}
