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

#include <hip_runtime_api.h>
#include <hcblas.h>

//HGSOS for Kalmar leave it as C++, only cublas needs C linkage.

#ifdef __cplusplus
extern "C" {
#endif

typedef hcblasHandle_t hipblasHandle_t;
typedef hcComplex hipComplex ;

static hipblasHandle_t dummyGlobal;

/* Unsupported types
		"cublasFillMode_t",
		"cublasDiagType_t",
		"cublasSideMode_t",
		"cublasPointerMode_t",
		"cublasAtomicsMode_t",
		"cublasDataType_t"
*/	

inline static hcblasOperation_t hipOperationToHCCOperation( hipblasOperation_t op)
{
	switch (op)
	{
		case HIPBLAS_OP_N:
			return HCBLAS_OP_N;
		
		case HIPBLAS_OP_T:
			return HCBLAS_OP_T;
			
		case HIPBLAS_OP_C:
			return HCBLAS_OP_C;
			
		default:
			throw "Non existent OP";
	}
}

inline static hipblasOperation_t HCCOperationToHIPOperation( hcblasOperation_t op)
{
	switch (op)
	{
		case HCBLAS_OP_N :
			return HIPBLAS_OP_N;
		
		case HCBLAS_OP_T :
			return HIPBLAS_OP_T;
			
		case HCBLAS_OP_C :
			return HIPBLAS_OP_C;
			
		default:
			throw "Non existent OP";
	}
}


inline static hipblasStatus_t hipHCBLASStatusToHIPStatus(hcblasStatus_t hcStatus) 
{
	switch(hcStatus)
	{
		case HCBLAS_STATUS_SUCCESS:
			return HIPBLAS_STATUS_SUCCESS;
		case HCBLAS_STATUS_NOT_INITIALIZED:
			return HIPBLAS_STATUS_NOT_INITIALIZED;
		case HCBLAS_STATUS_ALLOC_FAILED:
			return HIPBLAS_STATUS_ALLOC_FAILED;
		case HCBLAS_STATUS_INVALID_VALUE:
			return HIPBLAS_STATUS_INVALID_VALUE;
		case HCBLAS_STATUS_MAPPING_ERROR:
			return HIPBLAS_STATUS_MAPPING_ERROR;
		case HCBLAS_STATUS_EXECUTION_FAILED:
			return HIPBLAS_STATUS_EXECUTION_FAILED;
		case HCBLAS_STATUS_INTERNAL_ERROR:
			return HIPBLAS_STATUS_INTERNAL_ERROR;
		default:
			throw "Unimplemented status";
	}
}



inline static hipblasStatus_t hipblasCreate(hipblasHandle_t* handle) {
    hipblasStatus_t retVal = hipHCBLASStatusToHIPStatus(hcblasCreate(&*handle));
    return retVal;
}

inline static hipblasStatus_t hipblasDestroy(hipblasHandle_t& handle) {
    return hipHCBLASStatusToHIPStatus(hcblasDestroy(handle)); 
}

//note: no handle
inline static hipblasStatus_t hipblasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy){
	return hipHCBLASStatusToHIPStatus(hcblasSetVector(dummyGlobal, n, elemSize, x, incx, y, incy)); //HGSOS no need for handle moving forward
}

//note: no handle
inline static hipblasStatus_t hipblasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy){
	return hipHCBLASStatusToHIPStatus(hcblasGetVector(dummyGlobal, n, elemSize, x, incx, y, incy)); //HGSOS no need for handle
}

//note: no handle
inline static hipblasStatus_t hipblasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb){
	return hipHCBLASStatusToHIPStatus(hcblasSetMatrix(dummyGlobal, rows, cols, elemSize, A, lda, B, ldb));
}

//note: no handle
inline static hipblasStatus_t hipblasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb){
	return hipHCBLASStatusToHIPStatus(hcblasGetMatrix(dummyGlobal, rows, cols, elemSize, A, lda, B, ldb));
}

inline static hipblasStatus_t  hipblasSasum(hipblasHandle_t handle, int n, const float *x, int incx, float  *result){
	return hipHCBLASStatusToHIPStatus(hcblasSasum(handle, n, const_cast<float*>(x), incx, result));
}

inline static hipblasStatus_t  hipblasDasum(hipblasHandle_t handle, int n, const double *x, int incx, double *result){
	return hipHCBLASStatusToHIPStatus(hcblasDasum(handle, n, const_cast<double*>(x), incx, result));
}

inline static hipblasStatus_t  hipblasSasumBatched(hipblasHandle_t handle, int n, float *x, int incx, float  *result, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSasumBatched( handle, n, x, incx, result, batchCount));
}

inline static hipblasStatus_t  hipblasDasumBatched(hipblasHandle_t handle, int n, double *x, int incx, double *result, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasDasumBatched(handle, n, x, incx, result, batchCount));
}

inline static hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float *alpha,   const float *x, int incx, float *y, int incy) {
	return hipHCBLASStatusToHIPStatus(hcblasSaxpy(handle, n, alpha, x, incx, y, incy));
}

inline static hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n, const double *alpha,   const double *x, int incx, double *y, int incy) {
	return hipHCBLASStatusToHIPStatus(hcblasDaxpy(handle, n, alpha, x, incx, y, incy));
}

inline static hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t handle, int n, const float *alpha, const float *x, int incx,  float *y, int incy, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSaxpyBatched(handle, n, alpha, x, incx, y, incy, batchCount));
}

inline static hipblasStatus_t hipblasScopy(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy){
	return hipHCBLASStatusToHIPStatus(hcblasScopy( handle, n, x, incx, y, incy));
}

inline static hipblasStatus_t hipblasDcopy(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy){
	return hipHCBLASStatusToHIPStatus(hcblasDcopy( handle, n, x, incx, y, incy));
}

inline static hipblasStatus_t hipblasScopyBatched(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasScopyBatched( handle, n, x, incx, y, incy, batchCount));
}

inline static hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasDcopyBatched( handle, n, x, incx, y, incy, batchCount));
}

inline static hipblasStatus_t hipblasSdot (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result){
	return 	hipHCBLASStatusToHIPStatus(hcblasSdot(handle, n, x, incx, y, incy, result));			   
}

inline static hipblasStatus_t hipblasDdot (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result){
	return 	hipHCBLASStatusToHIPStatus(hcblasDdot(handle, n, x, incx, y, incy, result));			   
}

inline static hipblasStatus_t hipblasSdotBatched (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result, int batchCount){
	return 	hipHCBLASStatusToHIPStatus(hcblasSdotBatched(handle, n, x, incx, y, incy, result, batchCount));			   
}

inline static hipblasStatus_t hipblasDdotBatched (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result, int batchCount){
	return 	hipHCBLASStatusToHIPStatus(hcblasDdotBatched ( handle, n, x, incx, y, incy, result, batchCount));			   
}

inline static hipblasStatus_t  hipblasSscal(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx){
	return hipHCBLASStatusToHIPStatus(hcblasSscal(handle, n, alpha,  x, incx));
}

inline static hipblasStatus_t  hipblasDscal(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx){
	return hipHCBLASStatusToHIPStatus(hcblasDscal(handle, n, alpha,  x, incx));
}

inline static hipblasStatus_t  hipblasSscalBatched(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSscalBatched(handle, n, alpha,  x, incx, batchCount));
}

inline static hipblasStatus_t  hipblasDscalBatched(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasDscalBatched(handle, n, alpha,  x, incx, batchCount));
}

inline static hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda,
                           const float *x, int incx,  const float *beta,  float *y, int incy){
        // TODO: Remove const_cast
	return hipHCBLASStatusToHIPStatus(hcblasSgemv(handle, hipOperationToHCCOperation(trans),  m,  n, alpha, const_cast<float*>(A), lda, const_cast<float*>(x), incx, beta,  y, incy));						   
}

inline static hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda,
                           const double *x, int incx,  const double *beta,  double *y, int incy){
        // TODO: Remove const_cast
	return hipHCBLASStatusToHIPStatus(hcblasDgemv(handle, hipOperationToHCCOperation(trans),  m,  n, alpha, const_cast<double*>(A), lda, const_cast<double*>(x), incx, beta,  y, incy));						   
}
inline static hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, float *A, int lda,
                           float *x, int incx,  const float *beta,  float *y, int incy, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSgemvBatched(handle, hipOperationToHCCOperation(trans),  m,  n, alpha, A, lda, x, incx, beta,  y, incy, batchCount));						   
}

inline static hipblasStatus_t  hipblasSger(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda){
	return hipHCBLASStatusToHIPStatus(hcblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

inline static hipblasStatus_t  hipblasSgerBatched(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSgerBatched(handle, m, n, alpha, x, incx, y, incy, A, lda, batchCount));
}

inline static hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc){
   // TODO: Remove const_cast
	return hipHCBLASStatusToHIPStatus(hcblasSgemm( handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, const_cast<float*>(A),  lda, const_cast<float*>(B),  ldb, beta, C,  ldc));
}

inline static hipblasStatus_t hipblasDgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc){
	return hipHCBLASStatusToHIPStatus(hcblasDgemm( handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, const_cast<double*>(A),  lda, const_cast<double*>(B),  ldb, beta, C,  ldc));
}

inline static hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, hipComplex *A, int lda, hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc){
	return hipHCBLASStatusToHIPStatus(hcblasCgemm( handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc));
}

inline static hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const float *alpha, float *A, int lda, float *B, int ldb, const float *beta, float *C, int ldc, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSgemmBatched( handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc, batchCount));
}

inline static hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const double *alpha, double *A, int lda, double *B, int ldb, const double *beta, double *C, int ldc, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasDgemmBatched( handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc, batchCount));
}

inline static hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, hipComplex *A, int lda, hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc, int batchCount){
	return HIPBLAS_STATUS_NOT_SUPPORTED;
	//return hipHCBLASStatusToHIPStatus(hcblasCgemmBatched( handle, transa,  transb, m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc, batchCount));
}




#ifdef __cplusplus
}
#endif


