# CUBLAS API supported by HIP

## **2. CUBLAS Data types**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`CUBLAS_ATOMICS_ALLOWED`|  |  |  | `HIPBLAS_ATOMICS_ALLOWED` | 3.10.0 |  |  |
|`CUBLAS_ATOMICS_NOT_ALLOWED`|  |  |  | `HIPBLAS_ATOMICS_NOT_ALLOWED` | 3.10.0 |  |  |
|`CUBLAS_COMPUTE_16F`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_COMPUTE_16F_PEDANTIC`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_COMPUTE_32F`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_COMPUTE_32F_FAST_16BF`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_COMPUTE_32F_FAST_16F`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_COMPUTE_32F_FAST_TF32`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_COMPUTE_32F_PEDANTIC`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_COMPUTE_32I`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_COMPUTE_32I_PEDANTIC`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_COMPUTE_64F`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_COMPUTE_64F_PEDANTIC`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_DEFAULT_MATH`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_DIAG_NON_UNIT`|  |  |  | `HIPBLAS_DIAG_NON_UNIT` | 1.8.2 |  |  |
|`CUBLAS_DIAG_UNIT`|  |  |  | `HIPBLAS_DIAG_UNIT` | 1.8.2 |  |  |
|`CUBLAS_FILL_MODE_FULL`| 10.1 |  |  | `HIPBLAS_FILL_MODE_FULL` | 1.8.2 |  |  |
|`CUBLAS_FILL_MODE_LOWER`|  |  |  | `HIPBLAS_FILL_MODE_LOWER` | 1.8.2 |  |  |
|`CUBLAS_FILL_MODE_UPPER`|  |  |  | `HIPBLAS_FILL_MODE_UPPER` | 1.8.2 |  |  | 
|`CUBLAS_GEMM_ALGO0`| 8.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO0_TENSOR_OP`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO1`| 8.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO10`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO10_TENSOR_OP`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO11`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO11_TENSOR_OP`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO12`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO12_TENSOR_OP`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO13`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO13_TENSOR_OP`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO14`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO14_TENSOR_OP`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO15`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO15_TENSOR_OP`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO16`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO17`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO18`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO19`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO1_TENSOR_OP`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO2`| 8.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO20`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO21`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO22`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO23`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO2_TENSOR_OP`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO3`| 8.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO3_TENSOR_OP`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO4`| 8.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO4_TENSOR_OP`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO5`| 8.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO5_TENSOR_OP`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO6`| 8.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO6_TENSOR_OP`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO7`| 8.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO7_TENSOR_OP`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO8`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO8_TENSOR_OP`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO9`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_ALGO9_TENSOR_OP`| 9.2 |  |  |  |  |  |  |
|`CUBLAS_GEMM_DEFAULT`| 8.0 |  |  | `HIPBLAS_GEMM_DEFAULT` | 1.8.2 |  |  |
|`CUBLAS_GEMM_DEFAULT_TENSOR_OP`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_GEMM_DFALT`| 8.0 |  |  | `HIPBLAS_GEMM_DEFAULT` | 1.8.2 |  |  |
|`CUBLAS_GEMM_DFALT_TENSOR_OP`| 9.0 |  |  |  |  |  |  |
|`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_OP_C`|  |  |  | `HIPBLAS_OP_C` | 1.8.2 |  |  |
|`CUBLAS_OP_CONJG`| 10.1 |  |  |  |  |  |  |
|`CUBLAS_OP_HERMITAN`| 10.1 |  |  | `HIPBLAS_OP_C` | 1.8.2 |  |  |
|`CUBLAS_OP_N`|  |  |  | `HIPBLAS_OP_N` | 1.8.2 |  |  |
|`CUBLAS_OP_T`|  |  |  | `HIPBLAS_OP_T` | 1.8.2 |  |  |
|`CUBLAS_PEDANTIC_MATH`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_POINTER_MODE_DEVICE`|  |  |  | `HIPBLAS_POINTER_MODE_DEVICE` | 1.8.2 |  |  |
|`CUBLAS_POINTER_MODE_HOST`|  |  |  | `HIPBLAS_POINTER_MODE_HOST` | 1.8.2 |  |  |
|`CUBLAS_SIDE_LEFT`|  |  |  | `HIPBLAS_SIDE_LEFT` | 1.8.2 |  |  |
|`CUBLAS_SIDE_RIGHT`|  |  |  | `HIPBLAS_SIDE_RIGHT` | 1.8.2 |  |  |
|`CUBLAS_STATUS_ALLOC_FAILED`|  |  |  | `HIPBLAS_STATUS_ALLOC_FAILED` | 1.8.2 |  |  |
|`CUBLAS_STATUS_ARCH_MISMATCH`|  |  |  | `HIPBLAS_STATUS_ARCH_MISMATCH` | 1.8.2 |  |  |
|`CUBLAS_STATUS_EXECUTION_FAILED`|  |  |  | `HIPBLAS_STATUS_EXECUTION_FAILED` | 1.8.2 |  |  |
|`CUBLAS_STATUS_INTERNAL_ERROR`|  |  |  | `HIPBLAS_STATUS_INTERNAL_ERROR` | 1.8.2 |  |  |
|`CUBLAS_STATUS_INVALID_VALUE`|  |  |  | `HIPBLAS_STATUS_INVALID_VALUE` | 1.8.2 |  |  |
|`CUBLAS_STATUS_LICENSE_ERROR`|  |  |  |  |  |  |  |
|`CUBLAS_STATUS_MAPPING_ERROR`|  |  |  | `HIPBLAS_STATUS_MAPPING_ERROR` | 1.8.2 |  |  |
|`CUBLAS_STATUS_NOT_INITIALIZED`|  |  |  | `HIPBLAS_STATUS_NOT_INITIALIZED` | 1.8.2 |  |  |
|`CUBLAS_STATUS_NOT_SUPPORTED`|  |  |  | `HIPBLAS_STATUS_NOT_SUPPORTED` | 1.8.2 |  |  |
|`CUBLAS_STATUS_SUCCESS`|  |  |  | `HIPBLAS_STATUS_SUCCESS` | 1.8.2 |  |  |
|`CUBLAS_TENSOR_OP_MATH`| 9.0 | 11.0 |  |  |  |  |  |
|`CUBLAS_TF32_TENSOR_OP_MATH`| 11.0 |  |  |  |  |  |  |
|`CUBLAS_VERSION`| 10.1 |  |  |  |  |  |  |
|`CUBLAS_VER_BUILD`| 10.2 |  |  |  |  |  |  |
|`CUBLAS_VER_MAJOR`| 10.1 |  |  |  |  |  |  |
|`CUBLAS_VER_MINOR`| 10.1 |  |  |  |  |  |  |
|`CUBLAS_VER_PATCH`| 10.1 |  |  |  |  |  |  |
|`cublasAtomicsMode_t`|  |  |  | `hipblasAtomicsMode_t` | 3.10.0 |  |  |
|`cublasComputeType_t`| 11.0 |  |  |  |  |  |  |
|`cublasContext`|  |  |  |  |  |  |  |
|`cublasDataType_t`| 7.5 |  |  | `hipblasDatatype_t` | 1.8.2 |  |  |
|`cublasDiagType_t`|  |  |  | `hipblasDiagType_t` | 1.8.2 |  |  |
|`cublasFillMode_t`|  |  |  | `hipblasFillMode_t` | 1.8.2 |  |  |
|`cublasGemmAlgo_t`| 8.0 |  |  | `hipblasGemmAlgo_t` | 1.8.2 |  |  |
|`cublasHandle_t`|  |  |  | `hipblasHandle_t` | 3.0.0 |  |  |
|`cublasMath_t`| 9.0 |  |  |  |  |  |  |
|`cublasOperation_t`|  |  |  | `hipblasOperation_t` | 1.8.2 |  |  |
|`cublasPointerMode_t`|  |  |  | `hipblasPointerMode_t` | 1.8.2 |  |  |
|`cublasSideMode_t`|  |  |  | `hipblasSideMode_t` | 1.8.2 |  |  |
|`cublasStatus`|  |  |  | `hipblasStatus_t` | 1.8.2 |  |  |
|`cublasStatus_t`|  |  |  | `hipblasStatus_t` | 1.8.2 |  |  |

## **3. CUDA Datatypes Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`CUDA_C_16F`| 8.0 |  |  | `HIPBLAS_C_16F` | 1.8.2 |  |  |
|`CUDA_C_32F`| 8.0 |  |  | `HIPBLAS_C_32F` | 1.8.2 |  |  |
|`CUDA_C_32I`| 8.0 |  |  | `HIPBLAS_C_32I` | 3.0.0 |  |  |
|`CUDA_C_32U`| 8.0 |  |  | `HIPBLAS_C_32U` | 3.0.0 |  |  |
|`CUDA_C_64F`| 8.0 |  |  | `HIPBLAS_C_64F` | 1.8.2 |  |  |
|`CUDA_C_8I`| 8.0 |  |  | `HIPBLAS_C_8I` | 3.0.0 |  |  |
|`CUDA_C_8U`| 8.0 |  |  | `HIPBLAS_C_8U` | 3.0.0 |  |  |
|`CUDA_R_16F`| 8.0 |  |  | `HIPBLAS_R_16F` | 1.8.2 |  |  |
|`CUDA_R_32F`| 8.0 |  |  | `HIPBLAS_R_32F` | 1.8.2 |  |  |
|`CUDA_R_32I`| 8.0 |  |  | `HIPBLAS_R_32I` | 3.0.0 |  |  |
|`CUDA_R_32U`| 8.0 |  |  | `HIPBLAS_R_32U` | 3.0.0 |  |  |
|`CUDA_R_64F`| 8.0 |  |  | `HIPBLAS_R_64F` | 1.8.2 |  |  |
|`CUDA_R_8I`| 8.0 |  |  | `HIPBLAS_R_8I` | 3.0.0 |  |  |
|`CUDA_R_8U`| 8.0 |  |  | `HIPBLAS_R_8U` | 3.0.0 |  |  |
|`cudaDataType`| 8.0 |  |  | `hipblasDatatype_t` | 1.8.2 |  |  |
|`cudaDataType_t`| 8.0 |  |  | `hipblasDatatype_t` | 1.8.2 |  |  |

## **4. CUBLAS Helper Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cublasAlloc`|  |  |  |  |  |  |  |
|`cublasCreate`|  |  |  | `hipblasCreate` | 1.8.2 |  |  |
|`cublasCreate_v2`|  |  |  | `hipblasCreate` | 1.8.2 |  |  |
|`cublasDestroy`|  |  |  | `hipblasDestroy` | 1.8.2 |  |  |
|`cublasDestroy_v2`|  |  |  | `hipblasDestroy` | 1.8.2 |  |  |
|`cublasFree`|  |  |  |  |  |  |  |
|`cublasGetAtomicsMode`|  |  |  | `hipblasGetAtomicsMode` | 3.10.0 |  |  |
|`cublasGetCudartVersion`| 10.1 |  |  |  |  |  |  |
|`cublasGetError`|  |  |  |  |  |  |  |
|`cublasGetLoggerCallback`| 9.2 |  |  |  |  |  |  |
|`cublasGetMathMode`| 9.0 |  |  |  |  |  |  |
|`cublasGetMatrix`|  |  |  | `hipblasGetMatrix` | 1.8.2 |  |  |
|`cublasGetMatrixAsync`|  |  |  | `hipblasGetMatrixAsync` | 3.7.0 |  |  |
|`cublasGetPointerMode`|  |  |  | `hipblasGetPointerMode` | 1.8.2 |  |  |
|`cublasGetPointerMode_v2`|  |  |  | `hipblasGetPointerMode` | 1.8.2 |  |  |
|`cublasGetProperty`|  |  |  |  |  |  |  |
|`cublasGetStream`|  |  |  | `hipblasGetStream` | 1.8.2 |  |  |
|`cublasGetStream_v2`|  |  |  | `hipblasGetStream` | 1.8.2 |  |  |
|`cublasGetVector`|  |  |  | `hipblasGetVector` | 1.8.2 |  |  |
|`cublasGetVectorAsync`|  |  |  | `hipblasGetVectorAsync` | 3.7.0 |  |  |
|`cublasGetVersion`|  |  |  |  |  |  |  |
|`cublasGetVersion_v2`|  |  |  |  |  |  |  |
|`cublasInit`|  |  |  |  |  |  |  |
|`cublasLogCallback`| 9.2 |  |  |  |  |  |  |
|`cublasLoggerConfigure`| 9.2 |  |  |  |  |  |  |
|`cublasMigrateComputeType`| 11.0 |  |  |  |  |  |  |
|`cublasSetAtomicsMode`|  |  |  | `hipblasSetAtomicsMode` | 3.10.0 |  |  |
|`cublasSetKernelStream`|  |  |  |  |  |  |  |
|`cublasSetLoggerCallback`| 9.2 |  |  |  |  |  |  |
|`cublasSetMathMode`|  |  |  |  |  |  |  |
|`cublasSetMatrix`|  |  |  | `hipblasSetMatrix` | 1.8.2 |  |  |
|`cublasSetMatrixAsync`|  |  |  | `hipblasSetMatrixAsync` | 3.7.0 |  |  |
|`cublasSetPointerMode`|  |  |  | `hipblasSetPointerMode` | 1.8.2 |  |  |
|`cublasSetPointerMode_v2`|  |  |  | `hipblasSetPointerMode` | 1.8.2 |  |  |
|`cublasSetStream`|  |  |  | `hipblasSetStream` | 1.8.2 |  |  |
|`cublasSetStream_v2`|  |  |  | `hipblasSetStream` | 1.8.2 |  |  |
|`cublasSetVector`|  |  |  | `hipblasSetVector` | 1.8.2 |  |  |
|`cublasSetVectorAsync`|  |  |  | `hipblasSetVectorAsync` | 3.7.0 |  |  |
|`cublasShutdown`|  |  |  |  |  |  |  |
|`cublasXerbla`|  |  |  |  |  |  |  |

## **5. CUBLAS Level-1 Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cublasCaxpy`|  |  |  | `hipblasCaxpy` | 3.0.0 |  |  |
|`cublasCaxpy_v2`|  |  |  | `hipblasCaxpy` | 3.0.0 |  |  |
|`cublasCcopy`|  |  |  | `hipblasCcopy` | 3.0.0 |  |  |
|`cublasCcopy_v2`|  |  |  | `hipblasCcopy` | 3.0.0 |  |  |
|`cublasCdotc`|  |  |  | `hipblasCdotc` | 3.0.0 |  |  |
|`cublasCdotc_v2`|  |  |  | `hipblasCdotc` | 3.0.0 |  |  |
|`cublasCdotu`|  |  |  | `hipblasCdotu` | 3.0.0 |  |  |
|`cublasCdotu_v2`|  |  |  | `hipblasCdotu` | 3.0.0 |  |  |
|`cublasCrot`|  |  |  | `hipblasCrot` | 3.0.0 |  |  |
|`cublasCrot_v2`|  |  |  | `hipblasCrot` | 3.0.0 |  |  |
|`cublasCrotg`|  |  |  | `hipblasCrotg` | 3.0.0 |  |  |
|`cublasCrotg_v2`|  |  |  | `hipblasCrotg` | 3.0.0 |  |  |
|`cublasCscal`|  |  |  | `hipblasCscal` | 1.8.2 |  |  |
|`cublasCscal_v2`|  |  |  | `hipblasCscal` | 1.8.2 |  |  |
|`cublasCsrot`|  |  |  | `hipblasCsrot` | 3.0.0 |  |  |
|`cublasCsrot_v2`|  |  |  | `hipblasCsrot` | 3.0.0 |  |  |
|`cublasCsscal`|  |  |  | `hipblasCsscal` | 3.0.0 |  |  |
|`cublasCsscal_v2`|  |  |  | `hipblasCsscal` | 3.0.0 |  |  |
|`cublasCswap`|  |  |  | `hipblasCswap` | 3.0.0 |  |  |
|`cublasCswap_v2`|  |  |  | `hipblasCswap` | 3.0.0 |  |  |
|`cublasDasum`|  |  |  | `hipblasDasum` | 1.8.2 |  |  |
|`cublasDasum_v2`|  |  |  | `hipblasDasum` | 1.8.2 |  |  |
|`cublasDaxpy`|  |  |  | `hipblasDaxpy` | 1.8.2 |  |  |
|`cublasDaxpy_v2`|  |  |  | `hipblasDaxpy` | 1.8.2 |  |  |
|`cublasDcopy`|  |  |  | `hipblasDcopy` | 1.8.2 |  |  |
|`cublasDcopy_v2`|  |  |  | `hipblasDcopy` | 1.8.2 |  |  |
|`cublasDdot`|  |  |  | `hipblasDdot` | 3.0.0 |  |  |
|`cublasDdot_v2`|  |  |  | `hipblasDdot` | 3.0.0 |  |  |
|`cublasDnrm2`|  |  |  | `hipblasDnrm2` | 1.8.2 |  |  |
|`cublasDnrm2_v2`|  |  |  | `hipblasDnrm2` | 1.8.2 |  |  |
|`cublasDrot`|  |  |  | `hipblasDrot` | 3.0.0 |  |  |
|`cublasDrot_v2`|  |  |  | `hipblasDrot` | 3.0.0 |  |  |
|`cublasDrotg`|  |  |  | `hipblasDrotg` | 3.0.0 |  |  |
|`cublasDrotg_v2`|  |  |  | `hipblasDrotg` | 3.0.0 |  |  |
|`cublasDrotm`|  |  |  | `hipblasDrotm` | 3.0.0 |  |  |
|`cublasDrotm_v2`|  |  |  | `hipblasDrotm` | 3.0.0 |  |  |
|`cublasDrotmg`|  |  |  | `hipblasDrotmg` | 3.0.0 |  |  |
|`cublasDrotmg_v2`|  |  |  | `hipblasDrotmg` | 3.0.0 |  |  |
|`cublasDscal`|  |  |  | `hipblasDscal` | 1.8.2 |  |  |
|`cublasDscal_v2`|  |  |  | `hipblasDscal` | 1.8.2 |  |  |
|`cublasDswap`|  |  |  | `hipblasDswap` | 3.0.0 |  |  |
|`cublasDswap_v2`|  |  |  | `hipblasDswap` | 3.0.0 |  |  |
|`cublasDzasum`|  |  |  | `hipblasDzasum` | 3.0.0 |  |  |
|`cublasDzasum_v2`|  |  |  | `hipblasDzasum` | 3.0.0 |  |  |
|`cublasDznrm2`|  |  |  | `hipblasDznrm2` | 3.0.0 |  |  |
|`cublasDznrm2_v2`|  |  |  | `hipblasDznrm2` | 3.0.0 |  |  |
|`cublasIcamax`|  |  |  | `hipblasIcamax` | 3.0.0 |  |  |
|`cublasIcamax_v2`|  |  |  | `hipblasIcamax` | 3.0.0 |  |  |
|`cublasIcamin`|  |  |  | `hipblasIcamin` | 3.0.0 |  |  |
|`cublasIcamin_v2`|  |  |  | `hipblasIcamin` | 3.0.0 |  |  |
|`cublasIdamax`|  |  |  | `hipblasIdamax` | 1.8.2 |  |  |
|`cublasIdamax_v2`|  |  |  | `hipblasIdamax` | 1.8.2 |  |  |
|`cublasIdamin`|  |  |  | `hipblasIdamin` | 3.0.0 |  |  |
|`cublasIdamin_v2`|  |  |  | `hipblasIdamin` | 3.0.0 |  |  |
|`cublasIsamax`|  |  |  | `hipblasIsamax` | 1.8.2 |  |  |
|`cublasIsamax_v2`|  |  |  | `hipblasIsamax` | 1.8.2 |  |  |
|`cublasIsamin`|  |  |  | `hipblasIsamin` | 3.0.0 |  |  |
|`cublasIsamin_v2`|  |  |  | `hipblasIsamin` | 3.0.0 |  |  |
|`cublasIzamax`|  |  |  | `hipblasIzamax` | 3.0.0 |  |  |
|`cublasIzamax_v2`|  |  |  | `hipblasIzamax` | 3.0.0 |  |  |
|`cublasIzamin`|  |  |  | `hipblasIzamin` | 3.0.0 |  |  |
|`cublasIzamin_v2`|  |  |  | `hipblasIzamin` | 3.0.0 |  |  |
|`cublasNrm2Ex`| 8.0 |  |  |  |  |  |  |
|`cublasSasum`|  |  |  | `hipblasSasum` | 1.8.2 |  |  |
|`cublasSasum_v2`|  |  |  | `hipblasSasum` | 1.8.2 |  |  |
|`cublasSaxpy`|  |  |  | `hipblasSaxpy` | 1.8.2 |  |  |
|`cublasSaxpy_v2`|  |  |  | `hipblasSaxpy` | 1.8.2 |  |  |
|`cublasScasum`|  |  |  | `hipblasScasum` | 3.0.0 |  |  |
|`cublasScasum_v2`|  |  |  | `hipblasScasum` | 3.0.0 |  |  |
|`cublasScnrm2`|  |  |  | `hipblasScnrm2` | 3.0.0 |  |  |
|`cublasScnrm2_v2`|  |  |  | `hipblasScnrm2` | 3.0.0 |  |  |
|`cublasScopy`|  |  |  | `hipblasScopy` | 1.8.2 |  |  |
|`cublasScopy_v2`|  |  |  | `hipblasScopy` | 1.8.2 |  |  |
|`cublasSdot`|  |  |  | `hipblasSdot` | 3.0.0 |  |  |
|`cublasSdot_v2`|  |  |  | `hipblasSdot` | 3.0.0 |  |  |
|`cublasSnrm2`|  |  |  | `hipblasSnrm2` | 1.8.2 |  |  |
|`cublasSnrm2_v2`|  |  |  | `hipblasSnrm2` | 1.8.2 |  |  |
|`cublasSrot`|  |  |  | `hipblasSrot` | 3.0.0 |  |  |
|`cublasSrot_v2`|  |  |  | `hipblasSrot` | 3.0.0 |  |  |
|`cublasSrotg`|  |  |  | `hipblasSrotg` | 3.0.0 |  |  |
|`cublasSrotg_v2`|  |  |  | `hipblasSrotg` | 3.0.0 |  |  |
|`cublasSrotm`|  |  |  | `hipblasSrotm` | 3.0.0 |  |  |
|`cublasSrotm_v2`|  |  |  | `hipblasSrotm` | 3.0.0 |  |  |
|`cublasSrotmg`|  |  |  | `hipblasSrotmg` | 3.0.0 |  |  |
|`cublasSrotmg_v2`|  |  |  | `hipblasSrotmg` | 3.0.0 |  |  |
|`cublasSscal`|  |  |  | `hipblasSscal` | 1.8.2 |  |  |
|`cublasSscal_v2`|  |  |  | `hipblasSscal` | 1.8.2 |  |  |
|`cublasSswap`|  |  |  | `hipblasSswap` | 3.0.0 |  |  |
|`cublasSswap_v2`|  |  |  | `hipblasSswap` | 3.0.0 |  |  |
|`cublasZaxpy`|  |  |  | `hipblasZaxpy` | 3.0.0 |  |  |
|`cublasZaxpy_v2`|  |  |  | `hipblasZaxpy` | 3.0.0 |  |  |
|`cublasZcopy`|  |  |  | `hipblasZcopy` | 3.0.0 |  |  |
|`cublasZcopy_v2`|  |  |  | `hipblasZcopy` | 3.0.0 |  |  |
|`cublasZdotc`|  |  |  | `hipblasZdotc` | 3.0.0 |  |  |
|`cublasZdotc_v2`|  |  |  | `hipblasZdotc` | 3.0.0 |  |  |
|`cublasZdotu`|  |  |  | `hipblasZdotu` | 3.0.0 |  |  |
|`cublasZdotu_v2`|  |  |  | `hipblasZdotu` | 3.0.0 |  |  |
|`cublasZdrot`|  |  |  | `hipblasZdrot` | 3.0.0 |  |  |
|`cublasZdrot_v2`|  |  |  | `hipblasZdrot` | 3.0.0 |  |  |
|`cublasZdscal`|  |  |  | `hipblasZdscal` | 3.0.0 |  |  |
|`cublasZdscal_v2`|  |  |  | `hipblasZdscal` | 3.0.0 |  |  |
|`cublasZrot`|  |  |  | `hipblasZrot` | 3.0.0 |  |  |
|`cublasZrot_v2`|  |  |  | `hipblasZrot` | 3.0.0 |  |  |
|`cublasZrotg`|  |  |  | `hipblasZrotg` | 3.0.0 |  |  |
|`cublasZrotg_v2`|  |  |  | `hipblasZrotg` | 3.0.0 |  |  |
|`cublasZscal`|  |  |  | `hipblasZscal` | 1.8.2 |  |  |
|`cublasZscal_v2`|  |  |  | `hipblasZscal` | 1.8.2 |  |  |
|`cublasZswap`|  |  |  | `hipblasZswap` | 3.0.0 |  |  |
|`cublasZswap_v2`|  |  |  | `hipblasZswap` | 3.0.0 |  |  |

## **6. CUBLAS Level-2 Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cublasCgbmv`|  |  |  | `hipblasCgbmv` | 3.5.0 |  |  |
|`cublasCgbmv_v2`|  |  |  | `hipblasCgbmv` | 3.5.0 |  |  |
|`cublasCgemv`|  |  |  | `hipblasCgemv` | 3.0.0 |  |  |
|`cublasCgemv_v2`|  |  |  | `hipblasCgemv` | 3.0.0 |  |  |
|`cublasCgerc`|  |  |  | `hipblasCgerc` | 3.5.0 |  |  |
|`cublasCgerc_v2`|  |  |  | `hipblasCgerc` | 3.5.0 |  |  |
|`cublasCgeru`|  |  |  | `hipblasCgeru` | 3.5.0 |  |  |
|`cublasCgeru_v2`|  |  |  | `hipblasCgeru` | 3.5.0 |  |  |
|`cublasChbmv`|  |  |  | `hipblasChbmv` | 3.5.0 |  |  |
|`cublasChbmv_v2`|  |  |  | `hipblasChbmv` | 3.5.0 |  |  |
|`cublasChemv`|  |  |  | `hipblasChemv` | 3.5.0 |  |  |
|`cublasChemv_v2`|  |  |  | `hipblasChemv` | 3.5.0 |  |  |
|`cublasCher`|  |  |  | `hipblasCher` | 3.5.0 |  |  |
|`cublasCher2`|  |  |  | `hipblasCher2` | 3.5.0 |  |  |
|`cublasCher2_v2`|  |  |  | `hipblasCher2` | 3.5.0 |  |  |
|`cublasCher_v2`|  |  |  | `hipblasCher` | 3.5.0 |  |  |
|`cublasChpmv`|  |  |  | `hipblasChpmv` | 3.5.0 |  |  |
|`cublasChpmv_v2`|  |  |  | `hipblasChpmv` | 3.5.0 |  |  |
|`cublasChpr`|  |  |  | `hipblasChpr` | 3.5.0 |  |  |
|`cublasChpr2`|  |  |  | `hipblasChpr2` | 3.5.0 |  |  |
|`cublasChpr2_v2`|  |  |  | `hipblasChpr2` | 3.5.0 |  |  |
|`cublasChpr_v2`|  |  |  | `hipblasChpr` | 3.5.0 |  |  |
|`cublasCsymv`|  |  |  | `hipblasCsymv` | 3.5.0 |  |  |
|`cublasCsymv_v2`|  |  |  | `hipblasCsymv` | 3.5.0 |  |  |
|`cublasCsyr`|  |  |  | `hipblasCsyr` | 3.5.0 |  |  |
|`cublasCsyr2`|  |  |  | `hipblasCsyr2` | 3.5.0 |  |  |
|`cublasCsyr2_v2`|  |  |  | `hipblasCsyr2` | 3.5.0 |  |  |
|`cublasCsyr_v2`|  |  |  | `hipblasCsyr` | 3.5.0 |  |  |
|`cublasCtbmv`|  |  |  | `hipblasCtbmv` | 3.5.0 |  |  | 
|`cublasCtbmv_v2`|  |  |  | `hipblasCtbmv` | 3.5.0 |  |  | 
|`cublasCtbsv`|  |  |  | `hipblasCtbsv` | 3.6.0 |  |  | 
|`cublasCtbsv_v2`|  |  |  | `hipblasCtbsv` | 3.6.0 |  |  | 
|`cublasCtpmv`|  |  |  | `hipblasCtpmv` | 3.5.0 |  |  | 
|`cublasCtpmv_v2`|  |  |  | `hipblasCtpmv` | 3.5.0 |  |  | 
|`cublasCtpsv`|  |  |  | `hipblasCtpsv` | 3.5.0 |  |  | 
|`cublasCtpsv_v2`|  |  |  | `hipblasCtpsv` | 3.5.0 |  |  | 
|`cublasCtrmv`|  |  |  | `hipblasCtrmv` | 3.5.0 |  |  | 
|`cublasCtrmv_v2`|  |  |  | `hipblasCtrmv` | 3.5.0 |  |  | 
|`cublasCtrsv`|  |  |  | `hipblasCtrsv` | 3.5.0 |  |  | 
|`cublasCtrsv_v2`|  |  |  | `hipblasCtrsv` | 3.5.0 |  |  | 
|`cublasDgbmv`|  |  |  | `hipblasDgbmv` | 3.5.0 |  |  | 
|`cublasDgbmv_v2`|  |  |  | `hipblasDgbmv` | 3.5.0 |  |  | 
|`cublasDgemv`|  |  |  | `hipblasDgemv` | 1.8.2 |  |  | 
|`cublasDgemv_v2`|  |  |  | `hipblasDgemv` | 1.8.2 |  |  | 
|`cublasDger`|  |  |  | `hipblasDger` | 1.8.2 |  |  | 
|`cublasDger_v2`|  |  |  | `hipblasDger` | 1.8.2 |  |  | 
|`cublasDsbmv`|  |  |  | `hipblasDsbmv` | 3.5.0 |  |  | 
|`cublasDsbmv_v2`|  |  |  | `hipblasDsbmv` | 3.5.0 |  |  | 
|`cublasDspmv`|  |  |  | `hipblasDspmv` | 3.5.0 |  |  | 
|`cublasDspmv_v2`|  |  |  | `hipblasDspmv` | 3.5.0 |  |  | 
|`cublasDspr`|  |  |  | `hipblasDspr` | 3.5.0 |  |  | 
|`cublasDspr2`|  |  |  | `hipblasDspr2` | 3.5.0 |  |  | 
|`cublasDspr2_v2`|  |  |  | `hipblasDspr2` | 3.5.0 |  |  | 
|`cublasDspr_v2`|  |  |  | `hipblasDspr` | 3.5.0 |  |  | 
|`cublasDsymv`|  |  |  | `hipblasDsymv` | 3.5.0 |  |  | 
|`cublasDsymv_v2`|  |  |  | `hipblasDsymv` | 3.5.0 |  |  | 
|`cublasDsyr`|  |  |  | `hipblasDsyr` | 3.0.0 |  |  | 
|`cublasDsyr2`|  |  |  | `hipblasDsyr2` | 3.5.0 |  |  | 
|`cublasDsyr2_v2`|  |  |  | `hipblasDsyr2` | 3.5.0 |  |  | 
|`cublasDsyr_v2`|  |  |  | `hipblasDsyr` | 3.0.0 |  |  | 
|`cublasDtbmv`|  |  |  | `hipblasDtbmv` | 3.5.0 |  |  | 
|`cublasDtbmv_v2`|  |  |  | `hipblasDtbmv` | 3.5.0 |  |  | 
|`cublasDtbsv`|  |  |  | `hipblasDtbsv` | 3.6.0 |  |  | 
|`cublasDtbsv_v2`|  |  |  | `hipblasDtbsv` | 3.6.0 |  |  | 
|`cublasDtpmv`|  |  |  | `hipblasDtpmv` | 3.5.0 |  |  | 
|`cublasDtpmv_v2`|  |  |  | `hipblasDtpmv` | 3.5.0 |  |  | 
|`cublasDtpsv`|  |  |  | `hipblasDtpsv` | 3.5.0 |  |  | 
|`cublasDtpsv_v2`|  |  |  | `hipblasDtpsv` | 3.5.0 |  |  | 
|`cublasDtrmv`|  |  |  | `hipblasDtrmv` | 3.5.0 |  |  | 
|`cublasDtrmv_v2`|  |  |  | `hipblasDtrmv` | 3.5.0 |  |  | 
|`cublasDtrsv`|  |  |  | `hipblasDtrsv` | 3.0.0 |  |  | 
|`cublasDtrsv_v2`|  |  |  | `hipblasDtrsv` | 3.0.0 |  |  | 
|`cublasSgbmv`|  |  |  | `hipblasSgbmv` | 3.5.0 |  |  | 
|`cublasSgbmv_v2`|  |  |  | `hipblasSgbmv` | 3.5.0 |  |  | 
|`cublasSgemv`|  |  |  | `hipblasSgemv` | 1.8.2 |  |  | 
|`cublasSgemv_v2`|  |  |  | `hipblasSgemv` | 1.8.2 |  |  | 
|`cublasSger`|  |  |  | `hipblasSger` | 1.8.2 |  |  | 
|`cublasSger_v2`|  |  |  | `hipblasSger` | 1.8.2 |  |  | 
|`cublasSsbmv`|  |  |  | `hipblasSsbmv` | 3.5.0 |  |  | 
|`cublasSsbmv_v2`|  |  |  | `hipblasSsbmv` | 3.5.0 |  |  | 
|`cublasSspmv`|  |  |  | `hipblasSspmv` | 3.5.0 |  |  | 
|`cublasSspmv_v2`|  |  |  | `hipblasSspmv` | 3.5.0 |  |  | 
|`cublasSspr`|  |  |  | `hipblasSspr` | 3.5.0 |  |  | 
|`cublasSspr2`|  |  |  | `hipblasSspr2` | 3.5.0 |  |  | 
|`cublasSspr2_v2`|  |  |  | `hipblasSspr2` | 3.5.0 |  |  | 
|`cublasSspr_v2`|  |  |  | `hipblasSspr` | 3.5.0 |  |  | 
|`cublasSsymv`|  |  |  | `hipblasSsymv` | 3.5.0 |  |  | 
|`cublasSsymv_v2`|  |  |  | `hipblasSsymv` | 3.5.0 |  |  | 
|`cublasSsyr`|  |  |  | `hipblasSsyr` | 3.0.0 |  |  | 
|`cublasSsyr2`|  |  |  | `hipblasSsyr2` | 3.5.0 |  |  | 
|`cublasSsyr2_v2`|  |  |  | `hipblasSsyr2` | 3.5.0 |  |  | 
|`cublasSsyr_v2`|  |  |  | `hipblasSsyr` | 3.0.0 |  |  | 
|`cublasStbmv`|  |  |  | `hipblasStbmv` | 3.5.0 |  |  | 
|`cublasStbmv_v2`|  |  |  | `hipblasStbmv` | 3.5.0 |  |  | 
|`cublasStbsv`|  |  |  | `hipblasStbsv` | 3.6.0 |  |  | 
|`cublasStbsv_v2`|  |  |  | `hipblasStbsv` | 3.6.0 |  |  | 
|`cublasStpmv`|  |  |  | `hipblasStpmv` | 3.5.0 |  |  | 
|`cublasStpmv_v2`|  |  |  | `hipblasStpmv` | 3.5.0 |  |  | 
|`cublasStpsv`|  |  |  | `hipblasStpsv` | 3.5.0 |  |  | 
|`cublasStpsv_v2`|  |  |  | `hipblasStpsv` | 3.5.0 |  |  | 
|`cublasStrmv`|  |  |  | `hipblasStrmv` | 3.5.0 |  |  | 
|`cublasStrmv_v2`|  |  |  | `hipblasStrmv` | 3.5.0 |  |  | 
|`cublasStrsv`|  |  |  | `hipblasStrsv` | 3.0.0 |  |  | 
|`cublasStrsv_v2`|  |  |  | `hipblasStrsv` | 3.0.0 |  |  | 
|`cublasZgbmv`|  |  |  | `hipblasZgbmv` | 3.5.0 |  |  | 
|`cublasZgbmv_v2`|  |  |  | `hipblasZgbmv` | 3.5.0 |  |  | 
|`cublasZgemv`|  |  |  | `hipblasZgemv` | 3.0.0 |  |  | 
|`cublasZgemv_v2`|  |  |  | `hipblasZgemv` | 3.0.0 |  |  | 
|`cublasZgerc`|  |  |  | `hipblasZgerc` | 3.5.0 |  |  | 
|`cublasZgerc_v2`|  |  |  | `hipblasZgerc` | 3.5.0 |  |  | 
|`cublasZgeru`|  |  |  | `hipblasZgeru` | 3.5.0 |  |  | 
|`cublasZgeru_v2`|  |  |  | `hipblasZgeru` | 3.5.0 |  |  | 
|`cublasZhbmv`|  |  |  | `hipblasZhbmv` | 3.5.0 |  |  | 
|`cublasZhbmv_v2`|  |  |  | `hipblasZhbmv` | 3.5.0 |  |  | 
|`cublasZhemv`|  |  |  | `hipblasZhemv` | 3.5.0 |  |  | 
|`cublasZhemv_v2`|  |  |  | `hipblasZhemv` | 3.5.0 |  |  | 
|`cublasZher`|  |  |  | `hipblasZher` | 3.5.0 |  |  | 
|`cublasZher2`|  |  |  | `hipblasZher2` | 3.5.0 |  |  | 
|`cublasZher2_v2`|  |  |  | `hipblasZher2` | 3.5.0 |  |  | 
|`cublasZher_v2`|  |  |  | `hipblasZher` | 3.5.0 |  |  | 
|`cublasZhpmv`|  |  |  | `hipblasZhpmv` | 3.5.0 |  |  | 
|`cublasZhpmv_v2`|  |  |  | `hipblasZhpmv` | 3.5.0 |  |  | 
|`cublasZhpr`|  |  |  | `hipblasZhpr` | 3.5.0 |  |  | 
|`cublasZhpr2`|  |  |  | `hipblasZhpr2` | 3.5.0 |  |  | 
|`cublasZhpr2_v2`|  |  |  | `hipblasZhpr2` | 3.5.0 |  |  | 
|`cublasZhpr_v2`|  |  |  | `hipblasZhpr` | 3.5.0 |  |  | 
|`cublasZsymv`|  |  |  | `hipblasZsymv` | 3.5.0 |  |  | 
|`cublasZsymv_v2`|  |  |  | `hipblasZsymv` | 3.5.0 |  |  | 
|`cublasZsyr`|  |  |  | `hipblasZsyr` | 3.5.0 |  |  | 
|`cublasZsyr2`|  |  |  | `hipblasZsyr2` | 3.5.0 |  |  | 
|`cublasZsyr2_v2`|  |  |  | `hipblasZsyr2` | 3.5.0 |  |  | 
|`cublasZsyr_v2`|  |  |  | `hipblasZsyr` | 3.5.0 |  |  | 
|`cublasZtbmv`|  |  |  | `hipblasZtbmv` | 3.5.0 |  |  | 
|`cublasZtbmv_v2`|  |  |  | `hipblasZtbmv` | 3.5.0 |  |  | 
|`cublasZtbsv`|  |  |  | `hipblasZtbsv` | 3.6.0 |  |  | 
|`cublasZtbsv_v2`|  |  |  | `hipblasZtbsv` | 3.6.0 |  |  | 
|`cublasZtpmv`|  |  |  | `hipblasZtpmv` | 3.5.0 |  |  | 
|`cublasZtpmv_v2`|  |  |  | `hipblasZtpmv` | 3.5.0 |  |  | 
|`cublasZtpsv`|  |  |  | `hipblasZtpsv` | 3.5.0 |  |  | 
|`cublasZtpsv_v2`|  |  |  | `hipblasZtpsv` | 3.5.0 |  |  | 
|`cublasZtrmv`|  |  |  | `hipblasZtrmv` | 3.5.0 |  |  | 
|`cublasZtrmv_v2`|  |  |  | `hipblasZtrmv` | 3.5.0 |  |  | 
|`cublasZtrsv`|  |  |  | `hipblasZtrsv` | 3.5.0 |  |  | 
|`cublasZtrsv_v2`|  |  |  | `hipblasZtrsv` | 3.5.0 |  |  | 

## **7. CUBLAS Level-3 Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cublasCgemm`|  |  |  | `hipblasCgemm` | 1.8.2 |  |  | 
|`cublasCgemm3m`| 8.0 |  |  |  |  |  |  | 
|`cublasCgemm3mBatched`| 8.0 |  |  |  |  |  |  | 
|`cublasCgemm3mEx`| 8.0 |  |  |  |  |  |  | 
|`cublasCgemm3mStridedBatched`| 8.0 |  |  |  |  |  |  | 
|`cublasCgemmBatched`|  |  |  | `hipblasCgemmBatched` | 3.0.0 |  |  | 
|`cublasCgemmStridedBatched`| 8.0 |  |  | `hipblasCgemmStridedBatched` | 3.0.0 |  |  | 
|`cublasCgemm_v2`|  |  |  | `hipblasCgemm` | 1.8.2 |  |  | 
|`cublasChemm`|  |  |  | `hipblasChemm` | 3.6.0 |  |  | 
|`cublasChemm_v2`|  |  |  | `hipblasChemm` | 3.6.0 |  |  | 
|`cublasCher2k`|  |  |  | `hipblasCher2k` | 3.5.0 |  |  | 
|`cublasCher2k_v2`|  |  |  | `hipblasCher2k` | 3.5.0 |  |  | 
|`cublasCherk`|  |  |  | `hipblasCherk` | 3.5.0 |  |  | 
|`cublasCherk_v2`|  |  |  | `hipblasCherk` | 3.5.0 |  |  | 
|`cublasCherkx`|  |  |  | `hipblasCherkx` | 3.5.0 |  |  | 
|`cublasCsymm`|  |  |  | `hipblasCsymm` | 3.6.0 |  |  | 
|`cublasCsymm_v2`|  |  |  | `hipblasCsymm` | 3.6.0 |  |  | 
|`cublasCsyr2k`|  |  |  | `hipblasCsyr2k` | 3.5.0 |  |  | 
|`cublasCsyr2k_v2`|  |  |  | `hipblasCsyr2k` | 3.5.0 |  |  | 
|`cublasCsyrk`|  |  |  | `hipblasCsyrk` | 3.5.0 |  |  | 
|`cublasCsyrk_v2`|  |  |  | `hipblasCsyrk` | 3.5.0 |  |  | 
|`cublasCsyrkx`|  |  |  | `hipblasCsyrkx` | 3.5.0 |  |  | 
|`cublasCtrmm`|  |  |  | `hipblasCtrmm` | 3.5.0 |  |  | 
|`cublasCtrmm_v2`|  |  |  | `hipblasCtrmm` | 3.5.0 |  |  | 
|`cublasCtrsm`|  |  |  | `hipblasCtrsm` | 3.5.0 |  |  | 
|`cublasCtrsm_v2`|  |  |  | `hipblasCtrsm` | 3.5.0 |  |  | 
|`cublasDgemm`|  |  |  | `hipblasDgemm` | 1.8.2 |  |  | 
|`cublasDgemmBatched`|  |  |  | `hipblasDgemmBatched` | 1.8.2 |  |  | 
|`cublasDgemmStridedBatched`| 8.0 |  |  | `hipblasDgemmStridedBatched` | 1.8.2 |  |  | 
|`cublasDgemm_v2`|  |  |  | `hipblasDgemm` | 1.8.2 |  |  | 
|`cublasDsymm`|  |  |  | `hipblasDsymm` | 3.6.0 |  |  | 
|`cublasDsymm_v2`|  |  |  | `hipblasDsymm` | 3.6.0 |  |  | 
|`cublasDsyr2k`|  |  |  | `hipblasDsyr2k` | 3.5.0 |  |  | 
|`cublasDsyr2k_v2`|  |  |  | `hipblasDsyr2k` | 3.5.0 |  |  | 
|`cublasDsyrk`|  |  |  | `hipblasDsyrk` | 3.5.0 |  |  | 
|`cublasDsyrk_v2`|  |  |  | `hipblasDsyrk` | 3.5.0 |  |  | 
|`cublasDsyrkx`|  |  |  | `hipblasDsyrkx` | 3.5.0 |  |  | 
|`cublasDtrmm`|  |  |  | `hipblasDtrmm` | 3.2.0 |  |  | 
|`cublasDtrmm_v2`|  |  |  | `hipblasDtrmm` | 3.2.0 |  |  | 
|`cublasDtrsm`|  |  |  | `hipblasDtrsm` | 1.8.2 |  |  | 
|`cublasDtrsm_v2`|  |  |  | `hipblasDtrsm` | 1.8.2 |  |  | 
|`cublasHgemm`| 7.5 |  |  | `hipblasHgemm` | 1.8.2 |  |  | 
|`cublasHgemmBatched`| 9.0 |  |  | `hipblasHgemmBatched` | 3.0.0 |  |  | 
|`cublasHgemmStridedBatched`| 8.0 |  |  | `hipblasHgemmStridedBatched` | 3.0.0 |  |  | 
|`cublasSgemm`|  |  |  | `hipblasSgemm` | 1.8.2 |  |  | 
|`cublasSgemmBatched`|  |  |  | `hipblasSgemmBatched` | 1.8.2 |  |  | 
|`cublasSgemmStridedBatched`| 8.0 |  |  | `hipblasSgemmStridedBatched` | 1.8.2 |  |  | 
|`cublasSgemm_v2`|  |  |  | `hipblasSgemm` | 1.8.2 |  |  | 
|`cublasSsymm`|  |  |  | `hipblasSsymm` | 3.6.0 |  |  | 
|`cublasSsymm_v2`|  |  |  | `hipblasSsymm` | 3.6.0 |  |  | 
|`cublasSsyr2k`|  |  |  | `hipblasSsyr2k` | 3.5.0 |  |  | 
|`cublasSsyr2k_v2`|  |  |  | `hipblasSsyr2k` | 3.5.0 |  |  | 
|`cublasSsyrk`|  |  |  | `hipblasSsyrk` | 3.5.0 |  |  | 
|`cublasSsyrk_v2`|  |  |  | `hipblasSsyrk` | 3.5.0 |  |  | 
|`cublasSsyrkx`|  |  |  | `hipblasSsyrkx` | 3.5.0 |  |  | 
|`cublasStrmm`|  |  |  | `hipblasStrmm` | 3.2.0 |  |  | 
|`cublasStrmm_v2`|  |  |  | `hipblasStrmm` | 3.2.0 |  |  | 
|`cublasStrsm`|  |  |  | `hipblasStrsm` | 1.8.2 |  |  | 
|`cublasStrsm_v2`|  |  |  | `hipblasStrsm` | 1.8.2 |  |  | 
|`cublasZgemm`|  |  |  | `hipblasZgemm` | 1.8.2 |  |  | 
|`cublasZgemm3m`| 8.0 |  |  |  |  |  |  | 
|`cublasZgemmBatched`|  |  |  | `hipblasZgemmBatched` | 3.0.0 |  |  | 
|`cublasZgemmStridedBatched`| 8.0 |  |  | `hipblasZgemmStridedBatched` | 3.0.0 |  |  | 
|`cublasZgemm_v2`|  |  |  | `hipblasZgemm` | 1.8.2 |  |  | 
|`cublasZhemm`|  |  |  | `hipblasZhemm` | 3.6.0 |  |  | 
|`cublasZhemm_v2`|  |  |  | `hipblasZhemm` | 3.6.0 |  |  | 
|`cublasZher2k`|  |  |  | `hipblasZher2k` | 3.5.0 |  |  | 
|`cublasZher2k_v2`|  |  |  | `hipblasZher2k` | 3.5.0 |  |  | 
|`cublasZherk`|  |  |  | `hipblasZherk` | 3.5.0 |  |  | 
|`cublasZherk_v2`|  |  |  | `hipblasZherk` | 3.5.0 |  |  | 
|`cublasZherkx`|  |  |  | `hipblasZherkx` | 3.5.0 |  |  | 
|`cublasZsymm`|  |  |  | `hipblasZsymm` | 3.6.0 |  |  | 
|`cublasZsymm_v2`|  |  |  | `hipblasZsymm` | 3.6.0 |  |  | 
|`cublasZsyr2k`|  |  |  | `hipblasZsyr2k` | 3.5.0 |  |  | 
|`cublasZsyr2k_v2`|  |  |  | `hipblasZsyr2k` | 3.5.0 |  |  | 
|`cublasZsyrk`|  |  |  | `hipblasZsyrk` | 3.5.0 |  |  | 
|`cublasZsyrk_v2`|  |  |  | `hipblasZsyrk` | 3.5.0 |  |  | 
|`cublasZsyrkx`|  |  |  | `hipblasZsyrkx` | 3.5.0 |  |  | 
|`cublasZtrmm`|  |  |  | `hipblasZtrmm` | 3.5.0 |  |  | 
|`cublasZtrmm_v2`|  |  |  | `hipblasZtrmm` | 3.5.0 |  |  | 
|`cublasZtrsm`|  |  |  | `hipblasZtrsm` | 3.5.0 |  |  | 
|`cublasZtrsm_v2`|  |  |  | `hipblasZtrsm` | 3.5.0 |  |  | 

## **8. BLAS-like Extension**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cublasAsumEx`| 10.1 |  |  |  |  |  |  | 
|`cublasAxpyEx`| 8.0 |  |  |  |  |  |  | 
|`cublasCdgmm`|  |  |  | `hipblasCdgmm` | 3.6.0 |  |  | 
|`cublasCgeam`|  |  |  | `hipblasCgeam` | 3.6.0 |  |  | 
|`cublasCgelsBatched`|  |  |  |  |  |  |  | 
|`cublasCgemmEx`| 8.0 |  |  |  |  |  |  | 
|`cublasCgeqrfBatched`|  |  |  | `hipblasCgeqrfBatched` | 3.5.0 |  |  | 
|`cublasCgetrfBatched`|  |  |  | `hipblasCgetrfBatched` | 3.5.0 |  |  | 
|`cublasCgetriBatched`|  |  |  | `hipblasCgetriBatched` | 3.7.0 |  |  | 
|`cublasCgetrsBatched`|  |  |  | `hipblasCgetrsBatched` | 3.5.0 |  |  | 
|`cublasCherk3mEx`| 8.0 |  |  |  |  |  |  | 
|`cublasCherkEx`| 8.0 |  |  |  |  |  |  | 
|`cublasCmatinvBatched`|  |  |  |  |  |  |  | 
|`cublasCopyEx`| 10.1 |  |  |  |  |  |  | 
|`cublasCsyrk3mEx`| 8.0 |  |  |  |  |  |  | 
|`cublasCsyrkEx`| 8.0 |  |  |  |  |  |  | 
|`cublasCtpttr`|  |  |  |  |  |  |  | 
|`cublasCtrsmBatched`|  |  |  | `hipblasCtrsmBatched` | 3.5.0 |  |  | 
|`cublasCtrttp`|  |  |  |  |  |  |  | 
|`cublasDdgmm`|  |  |  | `hipblasDdgmm` | 3.6.0 |  |  | 
|`cublasDgeam`|  |  |  | `hipblasDgeam` | 1.8.2 |  |  | 
|`cublasDgelsBatched`|  |  |  |  |  |  |  | 
|`cublasDgeqrfBatched`|  |  |  | `hipblasDgeqrfBatched` | 3.5.0 |  |  | 
|`cublasDgetrfBatched`|  |  |  | `hipblasDgetrfBatched` | 3.5.0 |  |  | 
|`cublasDgetriBatched`|  |  |  | `hipblasDgetriBatched` | 3.7.0 |  |  | 
|`cublasDgetrsBatched`|  |  |  | `hipblasDgetrsBatched` | 3.5.0 |  |  | 
|`cublasDmatinvBatched`|  |  |  |  |  |  |  | 
|`cublasDotEx`| 8.0 |  |  |  |  |  |  | 
|`cublasDotcEx`| 8.0 |  |  |  |  |  |  | 
|`cublasDtpttr`|  |  |  |  |  |  |  | 
|`cublasDtrsmBatched`|  |  |  | `hipblasDtrsmBatched` | 3.2.0 |  |  | 
|`cublasDtrttp`|  |  |  |  |  |  |  | 
|`cublasGemmBatchedEx`| 9.1 |  |  | `hipblasGemmBatchedEx` | 3.6.0 |  |  | 
|`cublasGemmEx`| 8.0 |  |  | `hipblasGemmEx` | 1.8.2 |  |  | 
|`cublasGemmStridedBatchedEx`| 9.1 |  |  | `hipblasGemmStridedBatchedEx` | 3.6.0 |  |  | 
|`cublasIamaxEx`| 10.1 |  |  |  |  |  |  | 
|`cublasIaminEx`| 10.1 |  |  |  |  |  |  | 
|`cublasRotEx`| 10.1 |  |  |  |  |  |  | 
|`cublasRotgEx`| 10.1 |  |  |  |  |  |  | 
|`cublasRotmEx`| 10.1 |  |  |  |  |  |  | 
|`cublasRotmgEx`| 10.1 |  |  |  |  |  |  | 
|`cublasScalEx`| 8.0 |  |  |  |  |  |  | 
|`cublasSdgmm`|  |  |  | `hipblasSdgmm` | 3.6.0 |  |  | 
|`cublasSgeam`|  |  |  | `hipblasSgeam` | 1.8.2 |  |  | 
|`cublasSgelsBatched`|  |  |  |  |  |  |  | 
|`cublasSgemmEx`| 7.5 |  |  |  |  |  |  | 
|`cublasSgeqrfBatched`|  |  |  | `hipblasSgeqrfBatched` | 3.5.0 |  |  | 
|`cublasSgetrfBatched`|  |  |  | `hipblasSgetrfBatched` | 3.5.0 |  |  | 
|`cublasSgetriBatched`|  |  |  | `hipblasSgetriBatched` | 3.7.0 |  |  | 
|`cublasSgetrsBatched`|  |  |  | `hipblasSgetrsBatched` | 3.5.0 |  |  | 
|`cublasSmatinvBatched`|  |  |  |  |  |  |  | 
|`cublasStpttr`|  |  |  |  |  |  |  | 
|`cublasStrsmBatched`|  |  |  | `hipblasStrsmBatched` | 3.2.0 |  |  | 
|`cublasStrttp`|  |  |  |  |  |  |  | 
|`cublasSwapEx`| 10.1 |  |  |  |  |  |  | 
|`cublasUint8gemmBias`| 8.0 |  |  |  |  |  |  | 
|`cublasZdgmm`|  |  |  | `hipblasZdgmm` | 3.6.0 |  |  | 
|`cublasZgeam`|  |  |  | `hipblasZgeam` | 3.6.0 |  |  | 
|`cublasZgelsBatched`|  |  |  |  |  |  |  | 
|`cublasZgeqrfBatched`|  |  |  | `hipblasZgeqrfBatched` | 3.5.0 |  |  | 
|`cublasZgetrfBatched`|  |  |  | `hipblasZgetrfBatched` | 3.5.0 |  |  | 
|`cublasZgetriBatched`|  |  |  | `hipblasZgetriBatched` | 3.7.0 |  |  | 
|`cublasZgetrsBatched`|  |  |  | `hipblasZgetrsBatched` | 3.5.0 |  |  | 
|`cublasZmatinvBatched`|  |  |  |  |  |  |  | 
|`cublasZtpttr`|  |  |  |  |  |  |  | 
|`cublasZtrsmBatched`|  |  |  | `hipblasZtrsmBatched` | 3.5.0 |  |  | 
|`cublasZtrttp`|  |  |  |  |  |  |  | 


\*A - Added; D - Deprecated; R - Removed