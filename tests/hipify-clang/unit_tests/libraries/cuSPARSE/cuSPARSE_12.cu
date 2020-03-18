// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
// CHECK: #include <hip/hip_runtime.h>
// CHECK: #include <hipsparse.h>
// CHECK: #include <hipblas.h>
#include <cuda.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cstring>
#include <cassert>
#include <cstdio>

#define Min(x,y) ((x)<(y)?(x):(y))
#define Max(x,y) ((x)>(y)?(x):(y))
#define Abs(x) ((x)>(0)?(x):-(x))
// CHECK: static void CudaCheckCore(hipError_t code, const char *file, int line) {
static void CudaCheckCore(cudaError_t code, const char *file, int line) {
  // CHECK: if (code != hipSuccess) {
  if (code != cudaSuccess) {
    // CHECK: fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, hipGetErrorString(code), file, line);
    fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define CudaCheck( test ) { CudaCheckCore((test), __FILE__, __LINE__); }
// CHECK: #define CudaCheckAfterCall() { CudaCheckCore((hipGetLastError()), __FILE__, __LINE__); }
#define CudaCheckAfterCall() { CudaCheckCore((cudaGetLastError()), __FILE__, __LINE__); }

// CHECK: static const char * GetErrorString(hipsparseStatus_t error) {
static const char * GetErrorString(cusparseStatus_t error) {
  switch (error) {
    // CHECK: case HIPSPARSE_STATUS_SUCCESS:
    case CUSPARSE_STATUS_SUCCESS:
      return "The operation completed successfully.";
    // CHECK: case HIPSPARSE_STATUS_NOT_INITIALIZED:
    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "The cuSPARSE library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSPARSE routine, or an error in the hardware setup.\n" \
             "To correct: call cusparseCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";
    // CHECK: case HIPSPARSE_STATUS_ALLOC_FAILED:
    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "Resource allocation failed inside the cuSPARSE library. This is usually caused by a cudaMalloc() failure.\n"\
             "To correct: prior to the function call, deallocate previously allocated memory as much as possible.";
    // CHECK: case HIPSPARSE_STATUS_INVALID_VALUE:
    case CUSPARSE_STATUS_INVALID_VALUE:
      return "An unsupported value or parameter was passed to the function (a negative vector size, for example).\n"\
             "To correct: ensure that all the parameters being passed have valid values.";
    // CHECK: case HIPSPARSE_STATUS_ARCH_MISMATCH:
    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision.\n"\
              "To correct: compile and run the application on a device with appropriate compute capability, which is 1.1 for 32-bit atomic operations and 1.3 for double precision.";
    // CHECK: case HIPSPARSE_STATUS_MAPPING_ERROR:
    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.\n"\
             "To correct: prior to the function call, unbind any previously bound textures.";
    // CHECK: case HIPSPARSE_STATUS_EXECUTION_FAILED:
    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\n"\
             "To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";
    // CHECK: case HIPSPARSE_STATUS_INTERNAL_ERROR:
    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "An internal cuSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\n"\
             "To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.";
    // CHECK: case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    // CHECK: "To correct: check that the fields in hipsparseMatDescr_t descrA were set correctly.";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.\n"\
             "To correct: check that the fields in cusparseMatDescr_t descrA were set correctly.";
  }
  return "<unknown>";
}

// CHECK: static void CudaSparseCheckCore(hipsparseStatus_t code, const char *file, int line) {
static void CudaSparseCheckCore(cusparseStatus_t code, const char *file, int line) {
  // CHECK: if (code != HIPSPARSE_STATUS_SUCCESS) {
  if (code != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, GetErrorString(code), file, line);
    exit(code);
  }
}

#define CudaSparseCheck( test ) { CudaSparseCheckCore((test), __FILE__, __LINE__); }

// Alloc and copy
template <class ObjectType>
ObjectType* allocAndCopy(const ObjectType src[], const int size) {
  ObjectType* dest = NULL;
  // CHECK: CudaCheck( hipMalloc(&dest,size*sizeof(ObjectType)) );
  CudaCheck( cudaMalloc(&dest,size*sizeof(ObjectType)) );
  // CHECK: CudaCheck( hipMemcpy(dest, src, size*sizeof(ObjectType), hipMemcpyHostToDevice ) );
  CudaCheck( cudaMemcpy(dest, src, size*sizeof(ObjectType), cudaMemcpyHostToDevice ) );
  return dest;
}

template <class ObjectType>
ObjectType* alloc(const int size) {
  ObjectType* dest = NULL;
  // CHECK: CudaCheck( hipMalloc(&dest,size*sizeof(ObjectType)) );
  CudaCheck( cudaMalloc(&dest,size*sizeof(ObjectType)) );
  return dest;
}

template <class ObjectType>
ObjectType* allocAndCopyPart(const ObjectType src[], const int size, const int allocSize) {
  ObjectType* dest = NULL;
  assert(size <= allocSize);
  // CHECK: CudaCheck( hipMalloc(&dest,allocSize*sizeof(ObjectType)) );
  // CHECK: CudaCheck( hipMemcpy(dest, src, size*sizeof(ObjectType), hipMemcpyHostToDevice ) );
  // CHECK: CudaCheck( hipMemset(&dest[size],0,(allocSize-size)*sizeof(ObjectType)) );
  CudaCheck( cudaMalloc(&dest,allocSize*sizeof(ObjectType)) );
  CudaCheck( cudaMemcpy(dest, src, size*sizeof(ObjectType), cudaMemcpyHostToDevice ) );
  CudaCheck( cudaMemset(&dest[size],0,(allocSize-size)*sizeof(ObjectType)) );
  return dest;
}

// COO part
#include <algorithm>

struct Ijv {
  int i, j;
  double v;
};

bool IjvComp(const Ijv& v1, const Ijv& v2) {
  return v1.i < v2.i || (v1.i == v2.i && v1.j < v2.j);
}

struct COOArrays {
  int m;
  int nnz;
  double *val;/*values(NNZ)*/
  int *rowind;/*i(NNZ)*/
  int *colind;/*j(NNZ)*/

  COOArrays() {
    val = NULL;
    rowind = NULL;
    colind = NULL;
  }

  ~COOArrays() {
    delete[] val;
    delete[] rowind;
    delete[] colind;
  }

  void sortToRowMajor() {
    Ijv* ijvs = new Ijv[nnz];
    for(int idxCopy = 0 ; idxCopy < nnz ; ++idxCopy){
      ijvs[idxCopy].i = rowind[idxCopy];
      ijvs[idxCopy].j = colind[idxCopy];
      ijvs[idxCopy].v = val[idxCopy];
    }
    std::sort(ijvs, ijvs+nnz, IjvComp);
    for(int idxCopy = 0 ; idxCopy < nnz ; ++idxCopy){
      rowind[idxCopy] = ijvs[idxCopy].i;
      colind[idxCopy] = ijvs[idxCopy].j;
      val[idxCopy] = ijvs[idxCopy].v;
    }
    delete[] ijvs;
  }
};

void compute_COO(COOArrays& coo, double *x , double *y ) {
  for(int idxVal = 0 ; idxVal < coo.nnz ; ++idxVal){
    y[coo.rowind[idxVal]] += x[coo.colind[idxVal]] * coo.val[idxVal];
  }
}

// COO part
struct CRSArrays {
  int m;  //< the dim of the matrix
  int nnz;//< the number of nnz (== ia[m])
  double *cu_csrValA;  //< the values (of size NNZ)
  int *cu_csrRowPtrA;//< the usual rowptr (of size m+1)
  int *cu_csrColIndA;//< the colidx of each NNZ (of size nnz)
  // CHECK: hipStream_t streamId;
  // CHECK: hipsparseHandle_t cusparseHandle;
  cudaStream_t streamId;
  cusparseHandle_t cusparseHandle;

  CRSArrays() {
    cu_csrValA = NULL;
    cu_csrRowPtrA = NULL;
    cu_csrColIndA = NULL;
    // Create sparse handle (needed to call sparse functions
    streamId = 0;
    // CHECK-NOT: hipsparseHandle = 0;
    cusparseHandle = 0;
    // CHECK: CudaSparseCheck(hipsparseCreate(&cusparseHandle));
    // CHECK: CudaSparseCheck(hipsparseSetStream(cusparseHandle, streamId));
    CudaSparseCheck(cusparseCreate(&cusparseHandle));
    CudaSparseCheck(cusparseSetStream(cusparseHandle, streamId));
  }

  ~CRSArrays() {
    // CHECK: CudaCheck(hipFree(cu_csrValA));
    // CHECK: CudaCheck(hipFree(cu_csrRowPtrA));
    // CHECK: CudaCheck(hipFree(cu_csrColIndA));
    CudaCheck(cudaFree(cu_csrValA));
    CudaCheck(cudaFree(cu_csrRowPtrA));
    CudaCheck(cudaFree(cu_csrColIndA));
    // Destroy sparse handle
    // CHECK: CudaSparseCheck(hipsparseDestroy(cusparseHandle));
    CudaSparseCheck(cusparseDestroy(cusparseHandle));
  }
};

void COO_to_CRS(COOArrays& coo, CRSArrays* crs) {
  // We need COO to be sorted by row (and column)
  coo.sortToRowMajor();
  crs->m = coo.m;
  crs->nnz = coo.nnz;
  // Convert COO to CSR (it is just for the rows idx)
  crs->cu_csrRowPtrA = alloc<int>(coo.m+1);
  {
    int* cu_cooRowIndA = allocAndCopy(coo.rowind, coo.nnz);
    // CHECK: CudaSparseCheck(hipsparseXcoo2csr(crs->cusparseHandle, cu_cooRowIndA,
    // CHECK: coo.nnz, coo.m, crs->cu_csrRowPtrA, HIPSPARSE_INDEX_BASE_ZERO));
    CudaSparseCheck(cusparseXcoo2csr(crs->cusparseHandle, cu_cooRowIndA,
                coo.nnz, coo.m, crs->cu_csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO));
    // CHECK: CudaCheck(hipFree(cu_cooRowIndA));
    CudaCheck(cudaFree(cu_cooRowIndA));
  }
  // Copy cols idx and values that are unchanged
  crs->cu_csrValA = allocAndCopy(coo.val, coo.nnz);
  crs->cu_csrColIndA = allocAndCopy(coo.colind, coo.nnz);
}

double compute_CRS( CRSArrays& crs, double *x , double *y) {
  // For blas 2 gemv y = alpha.x.A + Beta.y
  const double alpha = 1.0;
  const double beta = 0.0;
  // Copy input
  double* cu_x = allocAndCopy(x, crs.m);
  double* cu_y = allocAndCopy(y, crs.m);
  // Init matrix properties
  // CHECK: hipsparseMatDescr_t descr = 0;
  cusparseMatDescr_t descr = 0;
  // CHECK: CudaSparseCheck(hipsparseCreateMatDescr(&descr));
  CudaSparseCheck(cusparseCreateMatDescr(&descr));
  // CHECK: hipsparseSetMatType(descr,HIPSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  // CHECK: hipsparseSetMatIndexBase(descr,HIPSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
  // Compute gemv
  float gemvComputeTume = 0;
  {
    // CHECK: hipEvent_t startTime, stopTime;
    // CHECK: hipEventCreate(&startTime);
    // CHECK: hipEventCreate(&stopTime);
    // CHECK: hipEventRecord(startTime, crs.streamId);
    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaEventRecord(startTime, crs.streamId);
    // CHECK: CudaSparseCheck(hipsparseDcsrmv(crs.cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
    CudaSparseCheck(cusparseDcsrmv(crs.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   crs.m, crs.m, crs.nnz, &alpha,
                                   descr, crs.cu_csrValA, crs.cu_csrRowPtrA,
                                   crs.cu_csrColIndA, cu_x, &beta, cu_y));
    // CHECK: hipEventRecord(stopTime, crs.streamId);
    // CHECK: hipEventSynchronize(stopTime);
    // CHECK: hipEventElapsedTime(&gemvComputeTume, startTime, stopTime);
    cudaEventRecord(stopTime, crs.streamId);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&gemvComputeTume, startTime, stopTime);
    gemvComputeTume /=1000.0;
  }
  // Get back result
  // CHECK: CudaCheck( hipMemcpy(y, cu_y, crs.m*sizeof(double), hipMemcpyDeviceToHost ) );
  CudaCheck( cudaMemcpy(y, cu_y, crs.m*sizeof(double), cudaMemcpyDeviceToHost ) );
  // Dealloc vectors
  // CHECK: CudaCheck(hipFree(cu_x));
  // CHECK: CudaCheck(hipFree(cu_y));
  CudaCheck(cudaFree(cu_x));
  CudaCheck(cudaFree(cu_y));
  return gemvComputeTume;
}

// BCSR part
struct BCRSArrays {
  int m;
  int nnz;
  int nbBlocks;
  int nbBlockRow;
  int blockSize;
  int* cu_bsrRowPtrC;
  int* cu_bsrColIndC;
  double* cu_bsrValC;
  // CHECK: hipStream_t streamId;
  cudaStream_t streamId;
  // CHECK: hipsparseHandle_t cusparseHandle;
  cusparseHandle_t cusparseHandle;

  BCRSArrays() {
    cu_bsrRowPtrC = NULL;
    cu_bsrColIndC = NULL;
    cu_bsrValC = NULL;
    // Create sparse handle (needed to call sparse functions
    streamId = 0;
    // CHECK: CudaSparseCheck(hipsparseCreate(&cusparseHandle));
    // CHECK: CudaSparseCheck(hipsparseSetStream(cusparseHandle, streamId));
    CudaSparseCheck(cusparseCreate(&cusparseHandle));
    CudaSparseCheck(cusparseSetStream(cusparseHandle, streamId));
  }

  ~BCRSArrays() {
    // CHECK: CudaCheck(hipFree(cu_bsrRowPtrC));
    // CHECK: CudaCheck(hipFree(cu_bsrColIndC));
    // CHECK: CudaCheck(hipFree(cu_bsrValC));
    CudaCheck(cudaFree(cu_bsrRowPtrC));
    CudaCheck(cudaFree(cu_bsrColIndC));
    CudaCheck(cudaFree(cu_bsrValC));
    // Destroy sparse handle
    // CHECK: CudaSparseCheck(hipsparseDestroy(cusparseHandle));
    CudaSparseCheck(cusparseDestroy(cusparseHandle));
  }
};

void CRS_to_BCRS(CRSArrays& csr, BCRSArrays* bcrs, const int blockSize) {
  bcrs->m = csr.m;
  bcrs->nnz = csr.nnz;
  bcrs->blockSize = blockSize;
  bcrs->nbBlockRow = (csr.m + blockSize-1)/blockSize;
  // CHECK: hipMalloc((void**)&bcrs->cu_bsrRowPtrC, sizeof(int) *(bcrs->nbBlockRow+1));
  cudaMalloc((void**)&bcrs->cu_bsrRowPtrC, sizeof(int) *(bcrs->nbBlockRow+1));
  // CHECK: hipsparseMatDescr_t descr = 0;
  cusparseMatDescr_t descr = 0;
  // CHECK: CudaSparseCheck(hipsparseCreateMatDescr(&descr));
  // CHECK: hipsparseSetMatType(descr,HIPSPARSE_MATRIX_TYPE_GENERAL);
  // CHECK: hipsparseSetMatIndexBase(descr,HIPSPARSE_INDEX_BASE_ZERO);
  CudaSparseCheck(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
  int nbNnzBlocks;
  // NOTE: cusparseXcsr2bsrNnz and CUSPARSE_DIRECTION_COLUMN (of type cusparseDirection_t) are yet unsupported by HIP
  // CHECK-NOT: hipsparseXcsr2bsrNnz(bcrs->cusparseHandle, HIPSPARSE_DIRECTION_COLUMN, csr.m, csr.m, descr, csr.cu_csrRowPtrA, csr.cu_csrColIndA,
  cusparseXcsr2bsrNnz(bcrs->cusparseHandle, CUSPARSE_DIRECTION_COLUMN, csr.m, csr.m, descr, csr.cu_csrRowPtrA, csr.cu_csrColIndA,
                      blockSize, descr, bcrs->cu_bsrRowPtrC, &nbNnzBlocks);
  {
    int firstBlockIdx, lastBlockIdx;
    // CHECK: hipMemcpy(&lastBlockIdx, bcrs->cu_bsrRowPtrC+bcrs->nbBlockRow, sizeof(int), hipMemcpyDeviceToHost);
    // CHECK: hipMemcpy(&firstBlockIdx, bcrs->cu_bsrRowPtrC, sizeof(int), hipMemcpyDeviceToHost);
    cudaMemcpy(&lastBlockIdx, bcrs->cu_bsrRowPtrC+bcrs->nbBlockRow, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&firstBlockIdx, bcrs->cu_bsrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
    assert(firstBlockIdx == 0); // we are in base 0
    assert(nbNnzBlocks == lastBlockIdx - firstBlockIdx);
  }
  bcrs->nbBlocks = nbNnzBlocks;
  // CHECK: CudaCheck(hipMalloc((void**)&bcrs->cu_bsrColIndC, sizeof(int)*nbNnzBlocks));
  // CHECK: CudaCheck(hipMalloc((void**)&bcrs->cu_bsrValC, sizeof(double)*(blockSize*blockSize)*nbNnzBlocks));
  CudaCheck(cudaMalloc((void**)&bcrs->cu_bsrColIndC, sizeof(int)*nbNnzBlocks));
  CudaCheck(cudaMalloc((void**)&bcrs->cu_bsrValC, sizeof(double)*(blockSize*blockSize)*nbNnzBlocks));
  // NOTE: cusparseDcsr2bsr and CUSPARSE_DIRECTION_COLUMN (of type cusparseDirection_t) are yet unsupported by HIP
  // CHECK-NOT: hipsparseDcsr2bsr(bcrs->cusparseHandle, HIPSPARSE_DIRECTION_COLUMN,
  cusparseDcsr2bsr(bcrs->cusparseHandle, CUSPARSE_DIRECTION_COLUMN,
                   csr.m, csr.m, descr, csr.cu_csrValA, csr.cu_csrRowPtrA, csr.cu_csrColIndA, blockSize, descr, bcrs->cu_bsrValC, bcrs->cu_bsrRowPtrC, bcrs->cu_bsrColIndC);
}

double compute_BSR(BCRSArrays& bcsr, double *x , double *y){
  // For blas 2 gemv y = alpha.x.A + Beta.y
  const double alpha = 1.0;
  const double beta = 0.0;
  // Copy input
  const int sizeMultipleBlockSize = ((bcsr.m+bcsr.blockSize-1)/bcsr.blockSize)*bcsr.blockSize;
  double* cu_x = allocAndCopyPart(x, bcsr.m, sizeMultipleBlockSize);
  double* cu_y = allocAndCopyPart(y, bcsr.m, sizeMultipleBlockSize);
  // Init matrix properties
  // CHECK: hipsparseMatDescr_t descr = 0;
  // CHECK: CudaSparseCheck(hipsparseCreateMatDescr(&descr));
  // CHECK: hipsparseSetMatType(descr,HIPSPARSE_MATRIX_TYPE_GENERAL);
  // CHECK: hipsparseSetMatIndexBase(descr,HIPSPARSE_INDEX_BASE_ZERO);
  cusparseMatDescr_t descr = 0;
  CudaSparseCheck(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
  // Compute gemv
  float gemvComputeTume = 0;
  {
    // CHECK: hipEvent_t startTime, stopTime;
    // CHECK: hipEventCreate(&startTime);
    // CHECK: hipEventCreate(&stopTime);
    // CHECK: hipEventRecord(startTime, bcsr.streamId);
    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaEventRecord(startTime, bcsr.streamId);
    // CHECK: cusparseDbsrmv(bcsr.cusparseHandle, HIPSPARSE_DIRECTION_COLUMN, HIPSPARSE_OPERATION_NON_TRANSPOSE,
    cusparseDbsrmv(bcsr.cusparseHandle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   bcsr.nbBlockRow, bcsr.m, bcsr.nbBlocks, &alpha, descr,
                   bcsr.cu_bsrValC, bcsr.cu_bsrRowPtrC, bcsr.cu_bsrColIndC, bcsr.blockSize,
                   cu_x, &beta, cu_y);
    // CHECK: hipEventRecord(stopTime, bcsr.streamId);
    // CHECK: hipEventSynchronize(stopTime);
    // CHECK: hipEventElapsedTime(&gemvComputeTume, startTime, stopTime);
    cudaEventRecord(stopTime, bcsr.streamId);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&gemvComputeTume, startTime, stopTime);
    gemvComputeTume /=1000.0;
  }
  // Get back result
  // CHECK: CudaCheck( hipMemcpy(y, cu_y, bcsr.m*sizeof(double), hipMemcpyDeviceToHost ) );
  CudaCheck( cudaMemcpy(y, cu_y, bcsr.m*sizeof(double), cudaMemcpyDeviceToHost ) );
  // Dealloc vectors
  // CHECK: CudaCheck(hipFree(cu_x));
  // CHECK: CudaCheck(hipFree(cu_y));
  CudaCheck(cudaFree(cu_x));
  CudaCheck(cudaFree(cu_y));
  return gemvComputeTume;
}
