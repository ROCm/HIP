// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// CHECK: #include "hipblas.h"
#include "cublas.h"
#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
static __inline__ void modify(float *m, int ldm, int n, int p, int q, float
  alpha, float beta) {
  // CHECK: hipblasSscal(n - p, alpha, &m[IDX2C(p, q, ldm)], ldm);
  // CHECK: hipblasSscal(ldm - p, beta, &m[IDX2C(p, q, ldm)], 1);
  cublasSscal(n - p, alpha, &m[IDX2C(p, q, ldm)], ldm);
  cublasSscal(ldm - p, beta, &m[IDX2C(p, q, ldm)], 1);
}
int main(void) {
  int i, j;
  // CHECK: hipblasStatus_t stat;
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
  stat = cublasAlloc(M*N, sizeof(*a), (void**)&devPtrA);
  // CHECK: if (stat != HIPBLAS_STATUS_SUCCESS) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("device memory allocation failed");
    // cublasShutdown is not supported yet
    cublasShutdown();
    return EXIT_FAILURE;
  }
  // CHECK: stat = hipblasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
  stat = cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
  // CHECK: if (stat != HIPBLAS_STATUS_SUCCESS) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    // cublasFree is not supported yet
    cublasFree(devPtrA);
    // cublasShutdown is not supported yet
    cublasShutdown();
    return EXIT_FAILURE;
  }
  modify(devPtrA, M, N, 1, 2, 16.0f, 12.0f);
  // CHECK: stat = hipblasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
  stat = cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
  // CHECK: if (stat != HIPBLAS_STATUS_SUCCESS) {
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
