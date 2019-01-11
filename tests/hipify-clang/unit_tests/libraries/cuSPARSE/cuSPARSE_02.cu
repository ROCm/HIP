// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include <hipblas.h>
#include <cublas_v2.h>
// CHECK: #include "hipsparse.h"
#include "cusparse.h"

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

int main(int argc, char*argv[])
{
    // CHECK: hipblasHandle_t cublasH = NULL;
    cublasHandle_t cublasH = NULL;
    // CHECK: hipsparseHandle_t cusparseH = NULL;
    cusparseHandle_t cusparseH = NULL;
    // CHECK: hipStream_t stream = NULL;
    cudaStream_t stream = NULL;
    // CHECK: hipsparseMatDescr_t descrA = NULL;
    cusparseMatDescr_t descrA = NULL;
    // CHECK: hipblasStatus_t cublasStat = HIPBLAS_STATUS_SUCCESS;
    cublasStatus_t cublasStat = CUBLAS_STATUS_SUCCESS;
    // CHECK: hipsparseStatus_t cusparseStat = HIPSPARSE_STATUS_SUCCESS;
    cusparseStatus_t cusparseStat = CUSPARSE_STATUS_SUCCESS;
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
    const int n = 4;
    const int nnzA = 9;
/* 
 *      |    1     0     2     3   |
 *      |    0     4     0     0   |
 *  A = |    5     0     6     7   |
 *      |    0     8     0     9   |
 *
 * eigevales are { -0.5311, 7.5311, 9.0000, 4.0000 }
 *
 * The largest eigenvaluse is 9 and corresponding eigenvector is
 *
 *      | 0.3029  |
 * v =  |     0   |
 *      | 0.9350  |
 *      | 0.1844  |
 */
    const int csrRowPtrA[n+1] = { 0, 3, 4, 7, 9 };
    const int csrColIndA[nnzA] = {0, 2, 3, 1, 0, 2, 3, 1, 3 };
    const double csrValA[nnzA] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    const double lambda_exact[n] = { 9.0000, 7.5311, 4.0000, -0.5311 };
    const double x0[n] = {1.0, 2.0, 3.0, 4.0 }; /* initial guess */
    double x[n]; /* numerical eigenvector */

    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    double *d_csrValA = NULL;

    double *d_x = NULL; /* eigenvector */
    double *d_y = NULL; /* workspace */

    const double tol = 1.e-6;
    const int max_ites = 30;

    const double h_one  = 1.0;
    const double h_zero = 0.0;

    printf("example of csrmv_mp \n");
    printf("tol = %E \n", tol);
    printf("max. iterations = %d \n", max_ites);

    printf("1st eigenvaluse is %f\n", lambda_exact[0] );
    printf("2nd eigenvaluse is %f\n", lambda_exact[1] );

    double alpha = lambda_exact[1]/lambda_exact[0] ;
    printf("convergence rate is %f\n", alpha );

    double est_iterations = log(tol)/log(alpha);
    printf("# of iterations required is %d\n", (int)ceil(est_iterations));

    // step 1: create cublas/cusparse handle, bind a stream
    // CHECK: cudaStat1 = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // CHECK: assert(hipSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat1);
    // CHECK: cublasStat = hipblasCreate(&cublasH);
    cublasStat = cublasCreate(&cublasH);
    // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
    assert(CUBLAS_STATUS_SUCCESS == cublasStat);
    // CHECK: cublasStat = hipblasSetStream(cublasH, stream);
    cublasStat = cublasSetStream(cublasH, stream);
    // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
    assert(CUBLAS_STATUS_SUCCESS == cublasStat);
    // CHECK: cusparseStat = hipsparseCreate(&cusparseH);
    cusparseStat = cusparseCreate(&cusparseH);
    // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == cusparseStat);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
    // CHECK: cusparseStat = hipsparseSetStream(cusparseH, stream);
    cusparseStat = cusparseSetStream(cusparseH, stream);
    // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == cusparseStat);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

    // step 2: configuration of matrix A
    // CHECK: cusparseStat = hipsparseCreateMatDescr(&descrA);
    cusparseStat = cusparseCreateMatDescr(&descrA);
    // assert(HIPSPARSE_STATUS_SUCCESS == cusparseStat);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
    // CHECK: hipsparseSetMatIndexBase(descrA,HIPSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    // CHECK: hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL );
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    // step 3: copy A and x0 to device
    // CHECK: cudaStat1 = hipMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (n+1) );
    cudaStat1 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (n+1) );
    // CHECK: cudaStat2 = hipMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA );
    cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA );
    // CHECK: cudaStat3 = hipMalloc ((void**)&d_csrValA   , sizeof(double) * nnzA );
    cudaStat3 = cudaMalloc ((void**)&d_csrValA   , sizeof(double) * nnzA );
    // CHECK: cudaStat4 = hipMalloc ((void**)&d_x         , sizeof(double) * n );
    cudaStat4 = cudaMalloc ((void**)&d_x         , sizeof(double) * n );
    // CHECK: cudaStat5 = hipMalloc ((void**)&d_y         , sizeof(double) * n );
    cudaStat5 = cudaMalloc ((void**)&d_y         , sizeof(double) * n );
    // CHECK: assert(hipSuccess == cudaStat1);
    // CHECK: assert(hipSuccess == cudaStat2);
    // CHECK: assert(hipSuccess == cudaStat3);
    // CHECK: assert(hipSuccess == cudaStat4);
    // CHECK: assert(hipSuccess == cudaStat5);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);

    // CHECK: cudaStat1 = hipMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (n+1)   , hipMemcpyHostToDevice);
    cudaStat1 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (n+1)   , cudaMemcpyHostToDevice);
    // CHECK: cudaStat2 = hipMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA    , hipMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA    , cudaMemcpyHostToDevice);
    // CHECK: cudaStat3 = hipMemcpy(d_csrValA   , csrValA   , sizeof(double) * nnzA , hipMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_csrValA   , csrValA   , sizeof(double) * nnzA , cudaMemcpyHostToDevice);
    // CHECK: assert(hipSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat1);
    // CHECK: assert(hipSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat2);
    // CHECK: assert(hipSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat3);

    // step 4: power method
    double lambda = 0.0;
    double lambda_next = 0.0;

    // 4.1: initial guess x0
    cudaStat1 = cudaMemcpy(d_x, x0, sizeof(double) * n, cudaMemcpyHostToDevice);
    // CHECK: assert(hipSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat1);

    for(int ite = 0 ; ite < max_ites ; ite++ ){
        // 4.2: normalize vector x
        // x = x / |x|
        double nrm2_x;
        // TODO: cublasStat = hipblasDnrm2_v2(cublasH,
        cublasStat = cublasDnrm2_v2(cublasH,
                                    n,
                                    d_x,
                                    1, // incx,
                                    &nrm2_x  /* host pointer */
                                   );
        // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
        assert(CUBLAS_STATUS_SUCCESS == cublasStat);
        double one_over_nrm2_x = 1.0 / nrm2_x;
        // TODO: cublasStat = hipblasDscal_v2( cublasH,
        cublasStat = cublasDscal_v2( cublasH,
                                     n,
                                     &one_over_nrm2_x,  /* host pointer */
                                     d_x,
                                     1 // incx
                                    );
        // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
        assert(CUBLAS_STATUS_SUCCESS == cublasStat);

        // 4.3: y = A*x
        // TODO: hipsparseStat = cusparseDcsrmv_mp(cusparseH,
        // CHECK: HIPSPARSE_OPERATION_NON_TRANSPOSE
        cusparseStat = cusparseDcsrmv_mp(cusparseH,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         n,
                                         n,
                                         nnzA,
                                         &h_one,
                                         descrA,
                                         d_csrValA,
                                         d_csrRowPtrA,
                                         d_csrColIndA,
                                         d_x,
                                         &h_zero,
                                         d_y);
        // CHECK: assert(HIPSPARSE_STATUS_SUCCESS == cusparseStat);
        assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

        // 4.4: lambda = y**T*x
        // TODO: cublasStat = hipblasDdot_v2 ( cublasH,
        cublasStat = cublasDdot_v2 ( cublasH,
                                     n,
                                     d_x,
                                     1, // incx,
                                     d_y,
                                     1, // incy,
                                     &lambda_next  /* host pointer */
                                   );
        // CHECK: assert(HIPBLAS_STATUS_SUCCESS == cublasStat);
        assert(CUBLAS_STATUS_SUCCESS == cublasStat);

        double lambda_err = fabs( lambda_next - lambda_exact[0] );
        printf("ite %d: lambda = %f, error = %E\n", ite, lambda_next, lambda_err );

        // 4.5: check if converges
        if ( (ite > 0) &&
             fabs( lambda - lambda_next ) < tol
        ){
            break; // converges
        }

        /*
          *  4.6: x := y
          *       lambda = lambda_next
          *
          *  so new approximation is (lambda, x), x is not normalized.
          */
        // CHECK: cudaStat1 = hipMemcpy(d_x, d_y, sizeof(double) * n , hipMemcpyDeviceToDevice);
        cudaStat1 = cudaMemcpy(d_x, d_y, sizeof(double) * n , cudaMemcpyDeviceToDevice);
        // CHECK: assert(hipSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat1);

        lambda = lambda_next;
    }
    // step 5: report eigen-pair
    // CHECK: cudaStat1 = hipMemcpy(x, d_x, sizeof(double) * n, hipMemcpyDeviceToHost);
    cudaStat1 = cudaMemcpy(x, d_x, sizeof(double) * n, cudaMemcpyDeviceToHost);
    // CHECK: assert(hipSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat1);

    printf("largest eigenvalue is %E\n", lambda );
    printf("eigenvector = (matlab base-1)\n");
    printMatrix(n, 1, x, n, "V0");
    printf("=====\n");

    // free resources
    // CHECK: if (d_csrRowPtrA  ) hipFree(d_csrRowPtrA);
    if (d_csrRowPtrA  ) cudaFree(d_csrRowPtrA);
    // CHECK: if (d_csrColIndA  ) hipFree(d_csrColIndA);
    if (d_csrColIndA  ) cudaFree(d_csrColIndA);
    // CHECK: if (d_csrValA     ) hipFree(d_csrValA);
    if (d_csrValA     ) cudaFree(d_csrValA);
    // CHECK: if (d_x           ) hipFree(d_x);
    if (d_x           ) cudaFree(d_x);
    // CHeCK: if (d_y           ) hipFree(d_y);
    if (d_y           ) cudaFree(d_y);
    // CHECK: if (cublasH       ) hipblasDestroy(cublasH);
    if (cublasH       ) cublasDestroy(cublasH);
    // CHECK: if (cusparseH     ) hipsparseDestroy(cusparseH);
    if (cusparseH     ) cusparseDestroy(cusparseH);
    // CHECK: if (stream        ) hipStreamDestroy(stream);
    if (stream        ) cudaStreamDestroy(stream);
    // CHECK: if (descrA        ) hipsparseDestroyMatDescr(descrA);
    if (descrA        ) cusparseDestroyMatDescr(descrA);
    // CHECK: hipDeviceReset();
    cudaDeviceReset();
    return 0;
}
