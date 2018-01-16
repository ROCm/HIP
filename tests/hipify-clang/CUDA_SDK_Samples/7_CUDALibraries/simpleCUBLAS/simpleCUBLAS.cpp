// RUN: %run_test hipify "%s" "%t" %cuda_args

/*
* This software contains source code provided by NVIDIA Corporation.
*/

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This example demonstrates how to use the CUBLAS library
 * by scaling an array of floating-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
// CHECK: #include <hip/hip_runtime.h>
// CHECK: #include <hipblas.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

/* Matrix size */
#define N  (275)

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C)
{
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

/* Main */
int main(int argc, char **argv)
{
    // CHECK: hipblasStatus_t status;
    cublasStatus_t status;
    float *h_A;
    float *h_B;
    float *h_C;
    float *h_C_ref;
    float *d_A = 0;
    float *d_B = 0;
    float *d_C = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * N;
    int i;
    float error_norm;
    float ref_norm;
    float diff;
    // CHECK: hipblasHandle_t handle;
    cublasHandle_t handle;

    int dev = findCudaDevice(argc, (const char **) argv);

    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

    /* Initialize CUBLAS */
    printf("simpleCUBLAS test running..\n");
    // CHECK: status = hipblasCreate(&handle);
    status = cublasCreate(&handle);
    // CHECK: if (status != HIPBLAS_STATUS_SUCCESS)
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    /* Allocate host memory for the matrices */
    h_A = (float *)malloc(n2 * sizeof(h_A[0]));

    if (h_A == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }

    h_B = (float *)malloc(n2 * sizeof(h_B[0]));

    if (h_B == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }

    h_C = (float *)malloc(n2 * sizeof(h_C[0]));

    if (h_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Fill the matrices with test data */
    for (i = 0; i < n2; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = rand() / (float)RAND_MAX;
    }

    /* Allocate device memory for the matrices */
    // CHECK: if (hipMalloc((void **)&d_A, n2 * sizeof(d_A[0])) != hipSuccess)
    if (cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }
    // CHECK: if (hipMalloc((void **)&d_B, n2 * sizeof(d_B[0])) != hipSuccess)
    if (cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }
    // CHECK: if (hipMalloc((void **)&d_C, n2 * sizeof(d_C[0])) != hipSuccess)
    if (cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    // CHECK: status = hipblasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
    status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
    // CHECK: if (status != HIPBLAS_STATUS_SUCCESS)
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }
    // CHECK: status = hipblasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
    status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
    // CHECK: if (status != HIPBLAS_STATUS_SUCCESS)
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }
    // CHECK: status = hipblasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);
    status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);
    // CHECK: if (status != HIPBLAS_STATUS_SUCCESS)
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

    /* Performs operation using plain C code */
    simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
    h_C_ref = h_C;

    /* Performs operation using cublas */
    // CHECK: status = hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    // CHECK: if (status != HIPBLAS_STATUS_SUCCESS)
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    /* Allocate host memory for reading back the result from device memory */
    h_C = (float *)malloc(n2 * sizeof(h_C[0]));

    if (h_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Read the result back */
    // CHECK: status = hipblasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);
    status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);
    // CHECK: if (status != HIPBLAS_STATUS_SUCCESS)
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }

    /* Check result against reference */
    error_norm = 0;
    ref_norm = 0;

    for (i = 0; i < n2; ++i)
    {
        diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_C_ref[i] * h_C_ref[i];
    }

    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);

    if (fabs(ref_norm) < 1e-7)
    {
        fprintf(stderr, "!!!! reference norm is 0\n");
        return EXIT_FAILURE;
    }

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    // CHECK: if (hipFree(d_A) != hipSuccess)
    if (cudaFree(d_A) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }
    // CHECK: if (hipFree(d_B) != hipSuccess)
    if (cudaFree(d_B) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }
    // CHECK: if (hipFree(d_C) != hipSuccess)
    if (cudaFree(d_C) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    // CHECK: status = hipblasDestroy(handle);
    status = cublasDestroy(handle);
    // CHECK: if (status != HIPBLAS_STATUS_SUCCESS)
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }

    if (error_norm / ref_norm < 1e-6f)
    {
        printf("simpleCUBLAS test passed.\n");
        exit(EXIT_SUCCESS);
    }
    else
    {
        printf("simpleCUBLAS test failed.\n");
        exit(EXIT_FAILURE);
    }
}
