/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cublas_v2.h>

//HGSOS for Kalmar leave it as C++, only cublas needs C linkage.

#ifdef __cplusplus
extern "C" {
#endif


typedef cublasHandle_t hipblasHandle_t ;
typedef cuComplex hipComplex;

/* Unsupported types
		"cublasFillMode_t",
		"cublasDiagType_t",
		"cublasSideMode_t",
		"cublasPointerMode_t",
		"cublasAtomicsMode_t",
		"cublasDataType_t"
*/		


inline static cublasOperation_t hipOperationToCudaOperation( hipblasOperation_t op)
{
	switch (op)
	{
		case HIPBLAS_OP_N:
			return CUBLAS_OP_N;
		
		case HIPBLAS_OP_T:
			return CUBLAS_OP_T;
			
		case HIPBLAS_OP_C:
			return CUBLAS_OP_C;
			
		default:
			throw "Non existent OP";
	}
}

inline static hipblasOperation_t CudaOperationToHIPOperation( cublasOperation_t op)
{
	switch (op)
	{
		case CUBLAS_OP_N :
			return HIPBLAS_OP_N;
		
		case CUBLAS_OP_T :
			return HIPBLAS_OP_T;
			
		case CUBLAS_OP_C :
			return HIPBLAS_OP_C;
			
		default:
			throw "Non existent OP";
	}
}


inline static hipblasStatus_t hipCUBLASStatusToHIPStatus(cublasStatus_t cuStatus) 
{
	switch(cuStatus) 
	{
		case CUBLAS_STATUS_SUCCESS:
			return HIPBLAS_STATUS_SUCCESS;
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return HIPBLAS_STATUS_NOT_INITIALIZED;
		case CUBLAS_STATUS_ALLOC_FAILED:
			return HIPBLAS_STATUS_ALLOC_FAILED;
		case CUBLAS_STATUS_INVALID_VALUE:
			return HIPBLAS_STATUS_INVALID_VALUE;
		case CUBLAS_STATUS_MAPPING_ERROR:
			return HIPBLAS_STATUS_MAPPING_ERROR;
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return HIPBLAS_STATUS_EXECUTION_FAILED;
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return HIPBLAS_STATUS_INTERNAL_ERROR;
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return HIPBLAS_STATUS_NOT_SUPPORTED;
		default:
			throw "Unimplemented status";
	}
}


inline static hipblasStatus_t hipblasCreate(hipblasHandle_t* handle) {
    return hipCUBLASStatusToHIPStatus(cublasCreate(&*handle)); 
}

//TODO broke common API semantics, think about this again.
inline static hipblasStatus_t hipblasDestroy(hipblasHandle_t handle) {
    return hipCUBLASStatusToHIPStatus(cublasDestroy(handle)); 
}

//note: no handle
inline static hipblasStatus_t hipblasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy){
	return hipCUBLASStatusToHIPStatus(cublasSetVector(n, elemSize, x, incx, y, incy)); //HGSOS no need for handle
}

//note: no handle
inline static hipblasStatus_t hipblasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy){
	return hipCUBLASStatusToHIPStatus(cublasGetVector(n, elemSize, x, incx, y, incy)); //HGSOS no need for handle
}

//note: no handle
inline static hipblasStatus_t hipblasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb){
	return hipCUBLASStatusToHIPStatus(cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb));
}

//note: no handle
inline static hipblasStatus_t hipblasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb){
	return hipCUBLASStatusToHIPStatus(cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb));
}

inline static hipblasStatus_t  hipblasSasum(hipblasHandle_t handle, int n, float *x, int incx, float  *result){
	return hipCUBLASStatusToHIPStatus(cublasSasum(handle, n, x, incx, result));
}

inline static hipblasStatus_t  hipblasDasum(hipblasHandle_t handle, int n, double *x, int incx, double *result){
	return hipCUBLASStatusToHIPStatus(cublasDasum( handle, n, x, incx, result));
}

inline static hipblasStatus_t  hipblasSasumBatched(hipblasHandle_t handle, int n, float *x, int incx, float  *result, int batchCount){
	//TODO warn user that function was demoted to ignore batch
	return hipCUBLASStatusToHIPStatus(cublasSasum( handle, n, x, incx, result));
}

inline static hipblasStatus_t  hipblasDasumBatched(hipblasHandle_t handle, int n, double *x, int incx, double *result, int batchCount){
	//TODO warn user that function was demoted to ignore batch
	return hipCUBLASStatusToHIPStatus(cublasDasum(handle, n, x, incx, result));
}

inline static hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float *alpha,   const float *x, int incx, float *y, int incy) {
	return hipCUBLASStatusToHIPStatus(cublasSaxpy(handle, n, alpha, x, incx, y, incy));
}  
 
inline static hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t handle, int n, const float *alpha, const float *x, int incx,  float *y, int incy, int batchCount){
	//TODO warn user that function was demoted to ignore batch
	return hipCUBLASStatusToHIPStatus(cublasSaxpy(handle, n, alpha, x, incx, y, incy));
}

inline static hipblasStatus_t hipblasScopy(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy){
	return hipCUBLASStatusToHIPStatus(cublasScopy( handle, n, x, incx, y, incy));
}

inline static hipblasStatus_t hipblasDcopy(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy){
	return hipCUBLASStatusToHIPStatus(cublasDcopy( handle, n, x, incx, y, incy));
}

inline static hipblasStatus_t hipblasScopyBatched(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy, int batchCount){
	//TODO warn user that function was demoted to ignore batch
	return hipCUBLASStatusToHIPStatus(cublasScopy( handle, n, x, incx, y, incy));
}
 
inline static hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy, int batchCount){
	//TODO warn user that function was demoted to ignore batch
	return hipCUBLASStatusToHIPStatus(cublasDcopy( handle, n, x, incx, y, incy));
}

inline static hipblasStatus_t hipblasSdot (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result){
	return 	hipCUBLASStatusToHIPStatus(cublasSdot ( handle, n, x, incx, y, incy, result));			   
}

inline static hipblasStatus_t hipblasDdot (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result){
	return 	hipCUBLASStatusToHIPStatus(cublasDdot ( handle, n, x, incx, y, incy, result));			   
}

inline static hipblasStatus_t hipblasSdotBatched (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result, int batchCount){
	//TODO warn user that function was demoted to ignore batch
	return 	hipCUBLASStatusToHIPStatus(cublasSdot ( handle, n, x, incx, y, incy, result));			   
}

inline static hipblasStatus_t hipblasDdotBatched (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result, int batchCount){
	//TODO warn user that function was demoted to ignore batch
	return 	hipCUBLASStatusToHIPStatus(cublasDdot ( handle, n, x, incx, y, incy, result));			   
}

inline static hipblasStatus_t  hipblasSscal(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx){
	return hipCUBLASStatusToHIPStatus(cublasSscal(handle, n, alpha,  x, incx));
}
inline static hipblasStatus_t  hipblasDscal(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx){
	return hipCUBLASStatusToHIPStatus(cublasDscal(handle, n, alpha,  x, incx));
}
inline static hipblasStatus_t  hipblasSscalBatched(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx, int batchCount){
	//TODO warn user that function was demoted to ignore batch
	return hipCUBLASStatusToHIPStatus(cublasSscal(handle, n, alpha,  x, incx));
}
inline static hipblasStatus_t  hipblasDscalBatched(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx, int batchCount){
	//TODO warn user that function was demoted to ignore batch
	return hipCUBLASStatusToHIPStatus(cublasDscal(handle, n, alpha,  x, incx));
}

inline static hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, float *A, int lda,
                           float *x, int incx,  const float *beta,  float *y, int incy){
	return hipCUBLASStatusToHIPStatus(cublasSgemv(handle, hipOperationToCudaOperation(trans),  m,  n, alpha, A, lda, x, incx, beta,  y, incy));						   
}

inline static hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, float *A, int lda,
                           float *x, int incx,  const float *beta,  float *y, int incy, int batchCount){
	//TODO warn user that function was demoted to ignore batch
	return hipCUBLASStatusToHIPStatus(cublasSgemv(handle, hipOperationToCudaOperation(trans),  m,  n, alpha, A, lda, x, incx, beta,  y, incy));						   
}

inline static hipblasStatus_t  hipblasSger(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda){
	return hipCUBLASStatusToHIPStatus(cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

inline static hipblasStatus_t  hipblasSgerBatched(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda, int batchCount){
	//TODO warn user that function was demoted to ignore batch
	return hipCUBLASStatusToHIPStatus(cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

inline static hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const float *alpha, float *A, int lda, float *B, int ldb, const float *beta, float *C, int ldc){
	return hipCUBLASStatusToHIPStatus(cublasSgemm( handle, hipOperationToCudaOperation(transa),  hipOperationToCudaOperation(transb), m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc));
}

inline static hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, hipComplex *A, int lda, hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc){
	return hipCUBLASStatusToHIPStatus(cublasCgemm( handle, hipOperationToCudaOperation(transa),  hipOperationToCudaOperation(transb), m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc));
}
		
inline static hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const float *alpha, float *A, int lda, float *B, int ldb, const float *beta, float *C, int ldc, int batchCount){
	//TODO incompatible API
	return HIPBLAS_STATUS_NOT_SUPPORTED;
	//return hipCUBLASStatusToHIPStatus(cublasSgemmBatched( handle, hipOperationToCudaOperation(transa),  hipOperationToCudaOperation(transb), m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc, batchCount));
}


inline static hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, hipComplex *A, int lda, hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc, int batchCount){

	//TODO incompatible API
	return HIPBLAS_STATUS_NOT_SUPPORTED;
	//return hipCUBLASStatusToHIPStatus(cublasCgemmBatched( handle, hipOperationToCudaOperation(transa),  hipOperationToCudaOperation(transb), m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc, batchCount));
}

#ifdef __cplusplus
}
#endif


