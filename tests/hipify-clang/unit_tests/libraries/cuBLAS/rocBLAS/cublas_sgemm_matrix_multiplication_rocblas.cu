// RUN: %run_test hipify "%s" "%t" %hipify_args "-roc" %clang_args

#include <stdio.h>
#include <stdlib.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "rocblas.h"
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define m 6
#define n 4
#define k 5
int main(void) {
  // CHECK: hipError_t cudaStat;
  // CHECK: rocblas_status stat;
  // CHECK: rocblas_handle handle;
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int i, j;
  float * a;
  float * b;
  float * c;
  a = (float *)malloc(m*k * sizeof(float));
  b = (float *)malloc(k*n * sizeof(float));
  c = (float *)malloc(m*n * sizeof(float));
  int ind = 11;
  for (j = 0; j<k; j++) {
    for (i = 0; i<m; i++) {
      a[IDX2C(i, j, m)] = (float)ind++;
    }
  }
  printf("a:\n");
  for (i = 0; i<m; i++) {
    for (j = 0; j<k; j++) {
      printf(" %5.0f", a[IDX2C(i, j, m)]);
    }
    printf("\n");
  }
  ind = 11;
  for (j = 0; j<n; j++) {
    for (i = 0; i<k; i++) {
      b[IDX2C(i, j, k)] = (float)ind++;
    }
  }
  printf("b:\n");
  for (i = 0; i<k; i++) {
    for (j = 0; j<n; j++) {
      printf(" %5.0f", b[IDX2C(i, j, k)]);
    }
    printf("\n");
  }
  ind = 11;
  for (j = 0; j<n; j++) {
    for (i = 0; i<m; i++) {
      c[IDX2C(i, j, m)] = (float)ind++;
    }
  }
  printf("c:\n");
  for (i = 0; i<m; i++) {
    for (j = 0; j<n; j++) {
      printf(" %5.0f", c[IDX2C(i, j, m)]);
    }
    printf("\n");
  }
  float * d_a;
  float * d_b;
  float * d_c;
  // CHECK: cudaStat = hipMalloc((void **)& d_a, m*k * sizeof(*a));
  // CHECK: cudaStat = hipMalloc((void **)& d_b, k*n * sizeof(*b));
  // CHECK: cudaStat = hipMalloc((void **)& d_c, m*n * sizeof(*c));
  cudaStat = cudaMalloc((void **)& d_a, m*k * sizeof(*a));
  cudaStat = cudaMalloc((void **)& d_b, k*n * sizeof(*b));
  cudaStat = cudaMalloc((void **)& d_c, m*n * sizeof(*c));
  // CHECK: stat = rocblas_create_handle(&handle);
  stat = cublasCreate(&handle);
  // CHECK: stat = rocblas_set_matrix(m, k, sizeof(*a), a, m, d_a, m);
  // CHECK: stat = rocblas_set_matrix(k, n, sizeof(*b), b, k, d_b, k);
  // CHECK: stat = rocblas_set_matrix(m, n, sizeof(*c), c, m, d_c, m);
  stat = cublasSetMatrix(m, k, sizeof(*a), a, m, d_a, m);
  stat = cublasSetMatrix(k, n, sizeof(*b), b, k, d_b, k);
  stat = cublasSetMatrix(m, n, sizeof(*c), c, m, d_c, m);
  float al = 1.0f;
  float bet = 1.0f;
  // CHECK: stat = rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, &al, d_a, m, d_b, k, &bet, d_c, m);
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, d_a, m, d_b, k, &bet, d_c, m);
  // CHECK: stat = rocblas_get_matrix(m, n, sizeof(*c), d_c, m, c, m);
  stat = cublasGetMatrix(m, n, sizeof(*c), d_c, m, c, m);
  printf("c after Sgemm :\n");
  for (i = 0; i<m; i++) {
    for (j = 0; j<n; j++) {
      printf(" %7.0f", c[IDX2C(i, j, m)]);
    }
    printf("\n");
  }
  // CHECK: hipFree(d_a);
  // CHECK: hipFree(d_b);
  // CHECK: hipFree(d_c);
  // CHECK: rocblas_destroy_handle(handle);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);
  free(a);
  free(b);
  free(c);
  return EXIT_SUCCESS;
}
