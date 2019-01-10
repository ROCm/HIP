// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
#include <stdio.h>
#include <stdlib.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "hipsparse.h"
#include "cusparse.h"

// CHECK: if (y)                  hipFree(y);
// CHECK: if (z)                  hipFree(z);
// CHECK: if (xInd)               hipFree(xInd);
// CHECK: if (xVal)               hipFree(xVal);
// CHECK: if (csrRowPtr)          hipFree(csrRowPtr);
// CHECK: if (cooRowIndex)        hipFree(cooRowIndex);
// CHECK: if (cooColIndex)        hipFree(cooColIndex);
// CHECK: if (cooVal)             hipFree(cooVal);
// CHECK: if (descr)              hipsparseDestroyMatDescr(descr);
// CHECK: if (handle)             hipsparseDestroy(handle);
// CHECK: hipDeviceReset();
#define CLEANUP(s)                                   \
do {                                                 \
    printf ("%s\n", s);                              \
    if (yHostPtr)           free(yHostPtr);          \
    if (zHostPtr)           free(zHostPtr);          \
    if (xIndHostPtr)        free(xIndHostPtr);       \
    if (xValHostPtr)        free(xValHostPtr);       \
    if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);\
    if (cooColIndexHostPtr) free(cooColIndexHostPtr);\
    if (cooValHostPtr)      free(cooValHostPtr);     \
    if (y)                  cudaFree(y);             \
    if (z)                  cudaFree(z);             \
    if (xInd)               cudaFree(xInd);          \
    if (xVal)               cudaFree(xVal);          \
    if (csrRowPtr)          cudaFree(csrRowPtr);     \
    if (cooRowIndex)        cudaFree(cooRowIndex);   \
    if (cooColIndex)        cudaFree(cooColIndex);   \
    if (cooVal)             cudaFree(cooVal);        \
    if (descr)              cusparseDestroyMatDescr(descr);\
    if (handle)             cusparseDestroy(handle); \
    cudaDeviceReset();          \
    fflush (stdout);                                 \
} while (0)

int main(){
    // CHECK: hipError_t cudaStat1,cudaStat2,cudaStat3,cudaStat4,cudaStat5,cudaStat6;
    cudaError_t cudaStat1,cudaStat2,cudaStat3,cudaStat4,cudaStat5,cudaStat6;
    // CHECK: hipsparseStatus_t status;
    cusparseStatus_t status;
    // CHECK: hipsparseHandle_t handle=0;
    cusparseHandle_t handle=0;
    // CHECK: hipsparseMatDescr_t descr=0;
    cusparseMatDescr_t descr=0;
    int *    cooRowIndexHostPtr=0;
    int *    cooColIndexHostPtr=0;
    double * cooValHostPtr=0;
    int *    cooRowIndex=0;
    int *    cooColIndex=0;
    double * cooVal=0;
    int *    xIndHostPtr=0;
    double * xValHostPtr=0;
    double * yHostPtr=0;
    int *    xInd=0;
    double * xVal=0;
    double * y=0;
    int *    csrRowPtr=0;
    double * zHostPtr=0;
    double * z=0;
    int      n, nnz, nnz_vector;
    double dzero =0.0;
    double dtwo  =2.0;
    double dthree=3.0;
    double dfive =5.0;
    printf("testing example\n");
    /* create the following sparse test matrix in COO format */
    /* |1.0     2.0 3.0|
       |    4.0        |
       |5.0     6.0 7.0|
       |    8.0     9.0| */
    n=4; nnz=9;
    cooRowIndexHostPtr = (int *)   malloc(nnz*sizeof(cooRowIndexHostPtr[0]));
    cooColIndexHostPtr = (int *)   malloc(nnz*sizeof(cooColIndexHostPtr[0]));
    cooValHostPtr      = (double *)malloc(nnz*sizeof(cooValHostPtr[0]));
    if ((!cooRowIndexHostPtr) || (!cooColIndexHostPtr) || (!cooValHostPtr)){
        CLEANUP("Host malloc failed (matrix)");
        return 1;
    }
    cooRowIndexHostPtr[0]=0; cooColIndexHostPtr[0]=0; cooValHostPtr[0]=1.0;
    cooRowIndexHostPtr[1]=0; cooColIndexHostPtr[1]=2; cooValHostPtr[1]=2.0;
    cooRowIndexHostPtr[2]=0; cooColIndexHostPtr[2]=3; cooValHostPtr[2]=3.0;
    cooRowIndexHostPtr[3]=1; cooColIndexHostPtr[3]=1; cooValHostPtr[3]=4.0;
    cooRowIndexHostPtr[4]=2; cooColIndexHostPtr[4]=0; cooValHostPtr[4]=5.0;
    cooRowIndexHostPtr[5]=2; cooColIndexHostPtr[5]=2; cooValHostPtr[5]=6.0;
    cooRowIndexHostPtr[6]=2; cooColIndexHostPtr[6]=3; cooValHostPtr[6]=7.0;
    cooRowIndexHostPtr[7]=3; cooColIndexHostPtr[7]=1; cooValHostPtr[7]=8.0;
    cooRowIndexHostPtr[8]=3; cooColIndexHostPtr[8]=3; cooValHostPtr[8]=9.0;
    nnz_vector = 3;
    xIndHostPtr = (int *)   malloc(nnz_vector*sizeof(xIndHostPtr[0]));
    xValHostPtr = (double *)malloc(nnz_vector*sizeof(xValHostPtr[0]));
    yHostPtr    = (double *)malloc(2*n       *sizeof(yHostPtr[0]));
    zHostPtr    = (double *)malloc(2*(n+1)   *sizeof(zHostPtr[0]));
    if((!xIndHostPtr) || (!xValHostPtr) || (!yHostPtr) || (!zHostPtr)) {
        CLEANUP("Host malloc failed (vectors)");
        return 1;
    }
    yHostPtr[0] = 10.0;
    xIndHostPtr[0]=0;
    xValHostPtr[0]=100.0;
    yHostPtr[1] = 20.0;
    xIndHostPtr[1]=1;
    xValHostPtr[1]=200.0;
    yHostPtr[2] = 30.0;
    yHostPtr[3] = 40.0;
    xIndHostPtr[2]=3;
    xValHostPtr[2]=400.0;
    yHostPtr[4] = 50.0;
    yHostPtr[5] = 60.0;
    yHostPtr[6] = 70.0;
    yHostPtr[7] = 80.0;
    /* allocate GPU memory and copy the matrix and vectors into it */
    // CHECK: cudaStat1 = hipMalloc((void**)&cooRowIndex,nnz*sizeof(cooRowIndex[0]));
    cudaStat1 = cudaMalloc((void**)&cooRowIndex,nnz*sizeof(cooRowIndex[0]));
    // CHECK: cudaStat2 = hipMalloc((void**)&cooColIndex,nnz*sizeof(cooColIndex[0]));
    cudaStat2 = cudaMalloc((void**)&cooColIndex,nnz*sizeof(cooColIndex[0]));
    // CHECK: cudaStat3 = hipMalloc((void**)&cooVal,     nnz*sizeof(cooVal[0]));
    cudaStat3 = cudaMalloc((void**)&cooVal,     nnz*sizeof(cooVal[0]));
    // CHECK: cudaStat4 = hipMalloc((void**)&y,          2*n*sizeof(y[0]));
    cudaStat4 = cudaMalloc((void**)&y,          2*n*sizeof(y[0]));
    // CHECK: cudaStat5 = hipMalloc((void**)&xInd,nnz_vector*sizeof(xInd[0]));
    cudaStat5 = cudaMalloc((void**)&xInd,nnz_vector*sizeof(xInd[0]));
    // CHECK: cudaStat6 = hipMalloc((void**)&xVal,nnz_vector*sizeof(xVal[0]));
    cudaStat6 = cudaMalloc((void**)&xVal,nnz_vector*sizeof(xVal[0]));
    // CHECK: if ((cudaStat1 != hipSuccess) ||
    // CHECK: (cudaStat2 != hipSuccess) ||
    // CHECK: (cudaStat3 != hipSuccess) ||
    // CHECK: (cudaStat4 != hipSuccess) ||
    // CHECK: (cudaStat5 != hipSuccess) ||
    // CHECK: (cudaStat6 != hipSuccess)) {
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
        (cudaStat4 != cudaSuccess) ||
        (cudaStat5 != cudaSuccess) ||
        (cudaStat6 != cudaSuccess)) {
        CLEANUP("Device malloc failed");
        return 1;
    }
    // CHECK: cudaStat1 = hipMemcpy(cooRowIndex, cooRowIndexHostPtr,
    // CHECK: hipMemcpyHostToDevice);
    cudaStat1 = cudaMemcpy(cooRowIndex, cooRowIndexHostPtr,
                           (size_t)(nnz*sizeof(cooRowIndex[0])),
                           cudaMemcpyHostToDevice);
    // CHECK: cudaStat2 = hipMemcpy(cooColIndex, cooColIndexHostPtr,
    // CHECK: hipMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(cooColIndex, cooColIndexHostPtr,
                           (size_t)(nnz*sizeof(cooColIndex[0])),
                           cudaMemcpyHostToDevice);
    // CHECK: cudaStat3 = hipMemcpy(cooVal,      cooValHostPtr,
    // CHECK: hipMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(cooVal,      cooValHostPtr,
                           (size_t)(nnz*sizeof(cooVal[0])),
                           cudaMemcpyHostToDevice);
    // CHECK: cudaStat4 = hipMemcpy(y,           yHostPtr,
    // CHECK: hipMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(y,           yHostPtr,
                           (size_t)(2*n*sizeof(y[0])),
                           cudaMemcpyHostToDevice);
    // CHECK: cudaStat5 = hipMemcpy(xInd,        xIndHostPtr,
    // CHECK: hipMemcpyHostToDevice);
    cudaStat5 = cudaMemcpy(xInd,        xIndHostPtr,
                           (size_t)(nnz_vector*sizeof(xInd[0])),
                           cudaMemcpyHostToDevice);
    // CHECK: cudaStat6 = hipMemcpy(xVal,        xValHostPtr,
    // CHECK: hipMemcpyHostToDevice);
    cudaStat6 = cudaMemcpy(xVal,        xValHostPtr,
                           (size_t)(nnz_vector*sizeof(xVal[0])),
                           cudaMemcpyHostToDevice);
    // CHECK: if ((cudaStat1 != hipSuccess) ||
    // CHECK: (cudaStat2 != hipSuccess) ||
    // CHECK: (cudaStat3 != hipSuccess) ||
    // CHECK: (cudaStat4 != hipSuccess) ||
    // CHECK: (cudaStat5 != hipSuccess) ||
    // CHECK: (cudaStat6 != hipSuccess)) {
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
        (cudaStat4 != cudaSuccess) ||
        (cudaStat5 != cudaSuccess) ||
        (cudaStat6 != cudaSuccess)) {
        CLEANUP("Memcpy from Host to Device failed");
        return 1;
    }
    /* initialize cusparse library */
    // CHECK: status= hipsparseCreate(&handle);
    status= cusparseCreate(&handle);
    // CHECK: if (status != HIPSPARSE_STATUS_SUCCESS) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library initialization failed");
        return 1;
    }
    /* create and setup matrix descriptor */
    // CHECK: status= hipsparseCreateMatDescr(&descr);
    status= cusparseCreateMatDescr(&descr);
    // CHECK: if (status != HIPSPARSE_STATUS_SUCCESS) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    // CHECK: hipsparseSetMatType(descr,HIPSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    // CHECK: hipsparseSetMatIndexBase(descr,HIPSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    /* exercise conversion routines (convert matrix from COO 2 CSR format) */
    // CHECK: cudaStat1 = hipMalloc((void**)&csrRowPtr,(n+1)*sizeof(csrRowPtr[0]));
    cudaStat1 = cudaMalloc((void**)&csrRowPtr,(n+1)*sizeof(csrRowPtr[0]));
    // CHECK: if (cudaStat1 != hipSuccess) {
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed (csrRowPtr)");
        return 1;
    }
    status= cusparseXcoo2csr(handle,cooRowIndex,nnz,n,
    // CHECK: csrRowPtr,HIPSPARSE_INDEX_BASE_ZERO);
                             csrRowPtr,CUSPARSE_INDEX_BASE_ZERO);
    // CHECK: if (status != HIPSPARSE_STATUS_SUCCESS) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Conversion from COO to CSR format failed");
        return 1;
    }
    //csrRowPtr = [0 3 4 7 9]
    // The following test only works for compute capability 1.3 and above 
    // because it needs double precision.
    int devId;
    // CHECK: hipDeviceProp_t prop;
    cudaDeviceProp prop;
    // CHECK: hipError_t cudaStat;
    cudaError_t cudaStat;
    // CHECK: cudaStat = hipGetDevice(&devId);
    cudaStat = cudaGetDevice(&devId);
    // CHECK: if (hipSuccess != cudaStat){
    if (cudaSuccess != cudaStat){
        // CLEANUP("hipGetDevice failed");
        CLEANUP("cudaGetDevice failed");
        // printf("Error: cudaStat %d, %s\n", cudaStat, hipGetErrorString(cudaStat));
        printf("Error: cudaStat %d, %s\n", cudaStat, cudaGetErrorString(cudaStat));
        return 1;
    }
    // CHECK: cudaStat = hipGetDeviceProperties( &prop, devId);
    cudaStat = cudaGetDeviceProperties( &prop, devId);
    // CHECK: if (hipSuccess != cudaStat) {
    if (cudaSuccess != cudaStat) {
        // CHECK: CLEANUP("hipGetDeviceProperties failed");
        CLEANUP("cudaGetDeviceProperties failed");
        // CHECK: printf("Error: cudaStat %d, %s\n", cudaStat, hipGetErrorString(cudaStat));
        printf("Error: cudaStat %d, %s\n", cudaStat, cudaGetErrorString(cudaStat));
        return 1;
    }
    int cc = 100*prop.major + 10*prop.minor;
    if (cc < 130){
        CLEANUP("waive the test because only sm13 and above are supported\n");
        printf("the device has compute capability %d\n", cc);
        printf("example test WAIVED");
        return 2;
    }
    /* exercise Level 1 routines (scatter vector elements) */
    // CHECK: status= hipsparseDsctr(handle, nnz_vector, xVal, xInd,
    // CHECK: &y[n], HIPSPARSE_INDEX_BASE_ZERO);
    status= cusparseDsctr(handle, nnz_vector, xVal, xInd,
                          &y[n], CUSPARSE_INDEX_BASE_ZERO);
    // CHECK: if (status != HIPSPARSE_STATUS_SUCCESS) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Scatter from sparse to dense vector failed");
        return 1;
    }  
    //y = [10 20 30 40 | 100 200 70 400]
    /* exercise Level 2 routines (csrmv) */
    // CHECK: status= hipsparseDcsrmv(handle,HIPSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
    status= cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
                           &dtwo, descr, cooVal, csrRowPtr, cooColIndex,
                           &y[0], &dthree, &y[n]);
    // CHECK: if (status != HIPSPARSE_STATUS_SUCCESS) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix-vector multiplication failed");
        return 1;
    }    
    //y = [10 20 30 40 | 680 760 1230 2240]
    // CHECK: hipMemcpy(yHostPtr, y, (size_t)(2*n*sizeof(y[0])), hipMemcpyDeviceToHost);
    cudaMemcpy(yHostPtr, y, (size_t)(2*n*sizeof(y[0])), cudaMemcpyDeviceToHost);
    /* exercise Level 3 routines (csrmm) */
    // cudaStat1 = hipMalloc((void**)&z, 2*(n+1)*sizeof(z[0]));
    cudaStat1 = cudaMalloc((void**)&z, 2*(n+1)*sizeof(z[0]));
    // CHECK: if (cudaStat1 != hipSuccess) {
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed (z)");
        return 1;
    }
    // CHECK: cudaStat1 = hipMemset((void *)z,0, 2*(n+1)*sizeof(z[0]));
    cudaStat1 = cudaMemset((void *)z,0, 2*(n+1)*sizeof(z[0]));
    // CHECK: if (cudaStat1 != hipSuccess) {
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memset on Device failed");
        return 1;
    }
    // CHECK: status= hipsparseDcsrmm(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, n, 2, n,
    status= cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, 2, n,
                           nnz, &dfive, descr, cooVal, csrRowPtr, cooColIndex,
                           y, n, &dzero, z, n+1);
    // CHECK: if (status != HIPSPARSE_STATUS_SUCCESS) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix-matrix multiplication failed");
        return 1;
    }
    /* print final results (z) */
    // CHECK: cudaStat1 = hipMemcpy(zHostPtr, z,
    // CHECK: hipMemcpyDeviceToHost);
    cudaStat1 = cudaMemcpy(zHostPtr, z,
                           (size_t)(2*(n+1)*sizeof(z[0])),
                           cudaMemcpyDeviceToHost);
    // CHECK: if (cudaStat1 != hipSuccess) {
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memcpy from Device to Host failed");
        return 1;
    }
    //z = [950 400 2550 2600 0 | 49300 15200 132300 131200 0]
    /* destroy matrix descriptor */
    // status = hipsparseDestroyMatDescr(descr);
    status = cusparseDestroyMatDescr(descr);
    descr = 0;
    // CHECK: if (status != HIPSPARSE_STATUS_SUCCESS) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor destruction failed");
        return 1;
    }
    /* destroy handle */
    // CHECK: status = hipsparseDestroy(handle);
    status = cusparseDestroy(handle);
    handle = 0;
    // CHECK: if (status != HIPSPARSE_STATUS_SUCCESS) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library release of resources failed");
        return 1;
    }
    /* check the results */
    // Notice that CLEANUP() contains a call to cusparseDestroy(handle)
    if ((zHostPtr[0] != 950.0)    ||
        (zHostPtr[1] != 400.0)    ||
        (zHostPtr[2] != 2550.0)   ||
        (zHostPtr[3] != 2600.0)   ||
        (zHostPtr[4] != 0.0)      ||
        (zHostPtr[5] != 49300.0)  ||
        (zHostPtr[6] != 15200.0)  ||
        (zHostPtr[7] != 132300.0) ||
        (zHostPtr[8] != 131200.0) ||
        (zHostPtr[9] != 0.0)      ||
        (yHostPtr[0] != 10.0)     ||
        (yHostPtr[1] != 20.0)     ||
        (yHostPtr[2] != 30.0)     ||
        (yHostPtr[3] != 40.0)     ||
        (yHostPtr[4] != 680.0)    ||
        (yHostPtr[5] != 760.0)    ||
        (yHostPtr[6] != 1230.0)   ||
        (yHostPtr[7] != 2240.0)) {
        CLEANUP("example test FAILED");
        return 1;
    } else {
        CLEANUP("example test PASSED");
        return 0;
    }
}