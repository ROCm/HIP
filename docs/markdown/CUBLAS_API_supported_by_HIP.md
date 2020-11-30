# CUBLAS API supported by HIP

## **2. CUBLAS Data types**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`CUBLAS_ATOMICS_ALLOWED`|  |  |  |`HIPBLAS_ATOMICS_ALLOWED`|
|`CUBLAS_ATOMICS_NOT_ALLOWED`|  |  |  |`HIPBLAS_ATOMICS_NOT_ALLOWED`|
|`CUBLAS_COMPUTE_16F`| 11.0 |  |  |`HIPBLAS_COMPUTE_16F`|
|`CUBLAS_COMPUTE_16F_PEDANTIC`| 11.0 |  |  |`HIPBLAS_COMPUTE_16F_PEDANTIC`|
|`CUBLAS_COMPUTE_32F`| 11.0 |  |  |`HIPBLAS_COMPUTE_32F`|
|`CUBLAS_COMPUTE_32F_FAST_16BF`| 11.0 |  |  |`HIPBLAS_COMPUTE_32F_FAST_16BF`|
|`CUBLAS_COMPUTE_32F_FAST_16F`| 11.0 |  |  |`HIPBLAS_COMPUTE_32F_FAST_16F`|
|`CUBLAS_COMPUTE_32F_FAST_TF32`| 11.0 |  |  |`HIPBLAS_COMPUTE_32F_FAST_TF32`|
|`CUBLAS_COMPUTE_32F_PEDANTIC`| 11.0 |  |  |`HIPBLAS_COMPUTE_32F_PEDANTIC`|
|`CUBLAS_COMPUTE_32I`| 11.0 |  |  |`HIPBLAS_COMPUTE_32I`|
|`CUBLAS_COMPUTE_32I_PEDANTIC`| 11.0 |  |  |`HIPBLAS_COMPUTE_32I_PEDANTIC`|
|`CUBLAS_COMPUTE_64F`| 11.0 |  |  |`HIPBLAS_COMPUTE_64F`|
|`CUBLAS_COMPUTE_64F_PEDANTIC`| 11.0 |  |  |`HIPBLAS_COMPUTE_64F_PEDANTIC`|
|`CUBLAS_DEFAULT_MATH`| 9.0 |  |  |`HIPBLAS_DEFAULT_MATH`|
|`CUBLAS_DIAG_NON_UNIT`|  |  |  |`HIPBLAS_DIAG_NON_UNIT`|
|`CUBLAS_DIAG_UNIT`|  |  |  |`HIPBLAS_DIAG_UNIT`|
|`CUBLAS_FILL_MODE_FULL`| 10.1 |  |  |`HIPBLAS_FILL_MODE_FULL`|
|`CUBLAS_FILL_MODE_LOWER`|  |  |  |`HIPBLAS_FILL_MODE_LOWER`|
|`CUBLAS_FILL_MODE_UPPER`|  |  |  |`HIPBLAS_FILL_MODE_UPPER`|
|`CUBLAS_GEMM_ALGO0`| 8.0 |  |  |`HIPBLAS_GEMM_ALGO0`|
|`CUBLAS_GEMM_ALGO0_TENSOR_OP`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO0_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO1`| 8.0 |  |  |`HIPBLAS_GEMM_ALGO1`|
|`CUBLAS_GEMM_ALGO10`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO10`|
|`CUBLAS_GEMM_ALGO10_TENSOR_OP`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO10_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO11`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO11`|
|`CUBLAS_GEMM_ALGO11_TENSOR_OP`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO11_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO12`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO12`|
|`CUBLAS_GEMM_ALGO12_TENSOR_OP`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO12_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO13`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO13`|
|`CUBLAS_GEMM_ALGO13_TENSOR_OP`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO13_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO14`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO14`|
|`CUBLAS_GEMM_ALGO14_TENSOR_OP`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO14_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO15`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO15`|
|`CUBLAS_GEMM_ALGO15_TENSOR_OP`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO15_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO16`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO16`|
|`CUBLAS_GEMM_ALGO17`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO17`|
|`CUBLAS_GEMM_ALGO18`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO18`|
|`CUBLAS_GEMM_ALGO19`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO19`|
|`CUBLAS_GEMM_ALGO1_TENSOR_OP`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO1_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO2`| 8.0 |  |  |`HIPBLAS_GEMM_ALGO2`|
|`CUBLAS_GEMM_ALGO20`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO20`|
|`CUBLAS_GEMM_ALGO21`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO21`|
|`CUBLAS_GEMM_ALGO22`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO22`|
|`CUBLAS_GEMM_ALGO23`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO23`|
|`CUBLAS_GEMM_ALGO2_TENSOR_OP`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO2_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO3`| 8.0 |  |  |`HIPBLAS_GEMM_ALGO3`|
|`CUBLAS_GEMM_ALGO3_TENSOR_OP`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO3_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO4`| 8.0 |  |  |`HIPBLAS_GEMM_ALGO4`|
|`CUBLAS_GEMM_ALGO4_TENSOR_OP`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO4_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO5`| 8.0 |  |  |`HIPBLAS_GEMM_ALGO5`|
|`CUBLAS_GEMM_ALGO5_TENSOR_OP`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO5_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO6`| 8.0 |  |  |`HIPBLAS_GEMM_ALGO6`|
|`CUBLAS_GEMM_ALGO6_TENSOR_OP`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO6_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO7`| 8.0 |  |  |`HIPBLAS_GEMM_ALGO7`|
|`CUBLAS_GEMM_ALGO7_TENSOR_OP`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO7_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO8`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO8`|
|`CUBLAS_GEMM_ALGO8_TENSOR_OP`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO8_TENSOR_OP`|
|`CUBLAS_GEMM_ALGO9`| 9.0 |  |  |`HIPBLAS_GEMM_ALGO9`|
|`CUBLAS_GEMM_ALGO9_TENSOR_OP`| 9.2 |  |  |`HIPBLAS_GEMM_ALGO9_TENSOR_OP`|
|`CUBLAS_GEMM_DEFAULT`| 8.0 |  |  |`HIPBLAS_GEMM_DEFAULT`|
|`CUBLAS_GEMM_DEFAULT_TENSOR_OP`| 9.0 |  |  |`HIPBLAS_GEMM_DEFAULT_TENSOR_OP`|
|`CUBLAS_GEMM_DFALT`| 8.0 |  |  |`HIPBLAS_GEMM_DEFAULT`|
|`CUBLAS_GEMM_DFALT_TENSOR_OP`| 9.0 |  |  |`HIPBLAS_GEMM_DFALT_TENSOR_OP`|
|`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`| 11.0 |  |  |`HIPBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`|
|`CUBLAS_OP_C`|  |  |  |`HIPBLAS_OP_C`|
|`CUBLAS_OP_CONJG`| 10.1 |  |  |`HIPBLAS_OP_CONJG`|
|`CUBLAS_OP_HERMITAN`| 10.1 |  |  |`HIPBLAS_OP_C`|
|`CUBLAS_OP_N`|  |  |  |`HIPBLAS_OP_N`|
|`CUBLAS_OP_T`|  |  |  |`HIPBLAS_OP_T`|
|`CUBLAS_PEDANTIC_MATH`| 11.0 |  |  |`HIPBLAS_PEDANTIC_MATH`|
|`CUBLAS_POINTER_MODE_DEVICE`|  |  |  |`HIPBLAS_POINTER_MODE_DEVICE`|
|`CUBLAS_POINTER_MODE_HOST`|  |  |  |`HIPBLAS_POINTER_MODE_HOST`|
|`CUBLAS_SIDE_LEFT`|  |  |  |`HIPBLAS_SIDE_LEFT`|
|`CUBLAS_SIDE_RIGHT`|  |  |  |`HIPBLAS_SIDE_RIGHT`|
|`CUBLAS_STATUS_ALLOC_FAILED`|  |  |  |`HIPBLAS_STATUS_ALLOC_FAILED`|
|`CUBLAS_STATUS_ARCH_MISMATCH`|  |  |  |`HIPBLAS_STATUS_ARCH_MISMATCH`|
|`CUBLAS_STATUS_EXECUTION_FAILED`|  |  |  |`HIPBLAS_STATUS_EXECUTION_FAILED`|
|`CUBLAS_STATUS_INTERNAL_ERROR`|  |  |  |`HIPBLAS_STATUS_INTERNAL_ERROR`|
|`CUBLAS_STATUS_INVALID_VALUE`|  |  |  |`HIPBLAS_STATUS_INVALID_VALUE`|
|`CUBLAS_STATUS_LICENSE_ERROR`|  |  |  |`HIPBLAS_STATUS_LICENSE_ERROR`|
|`CUBLAS_STATUS_MAPPING_ERROR`|  |  |  |`HIPBLAS_STATUS_MAPPING_ERROR`|
|`CUBLAS_STATUS_NOT_INITIALIZED`|  |  |  |`HIPBLAS_STATUS_NOT_INITIALIZED`|
|`CUBLAS_STATUS_NOT_SUPPORTED`|  |  |  |`HIPBLAS_STATUS_NOT_SUPPORTED`|
|`CUBLAS_STATUS_SUCCESS`|  |  |  |`HIPBLAS_STATUS_SUCCESS`|
|`CUBLAS_TENSOR_OP_MATH`| 9.0 |  |  |`HIPBLAS_TENSOR_OP_MATH`|
|`CUBLAS_TF32_TENSOR_OP_MATH`| 11.0 |  |  |`HIPBLAS_TF32_TENSOR_OP_MATH`|
|`CUBLAS_VERSION`| 10.1 |  |  ||
|`CUBLAS_VER_BUILD`| 10.2 |  |  ||
|`CUBLAS_VER_MAJOR`| 10.1 |  |  ||
|`CUBLAS_VER_MINOR`| 10.1 |  |  ||
|`CUBLAS_VER_PATCH`| 10.1 |  |  ||
|`cublasAtomicsMode_t`|  |  |  |`hipblasAtomicsMode_t`|
|`cublasComputeType_t`| 11.0 |  |  |`hipblasComputeType_t`|
|`cublasContext`|  |  |  ||
|`cublasDataType_t`| 7.5 |  |  |`hipblasDatatype_t`|
|`cublasDiagType_t`|  |  |  |`hipblasDiagType_t`|
|`cublasFillMode_t`|  |  |  |`hipblasFillMode_t`|
|`cublasGemmAlgo_t`| 8.0 |  |  |`hipblasGemmAlgo_t`|
|`cublasHandle_t`|  |  |  |`hipblasHandle_t`|
|`cublasMath_t`| 9.0 |  |  |`hipblasMath_t`|
|`cublasOperation_t`|  |  |  |`hipblasOperation_t`|
|`cublasPointerMode_t`|  |  |  |`hipblasPointerMode_t`|
|`cublasSideMode_t`|  |  |  |`hipblasSideMode_t`|
|`cublasStatus`|  |  |  |`hipblasStatus_t`|
|`cublasStatus_t`|  |  |  |`hipblasStatus_t`|

## **3. CUDA Datatypes Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`CUDA_C_16F`| 8.0 |  |  |`HIPBLAS_C_16F`|
|`CUDA_C_32F`| 8.0 |  |  |`HIPBLAS_C_32F`|
|`CUDA_C_32I`| 8.0 |  |  |`HIPBLAS_C_32I`|
|`CUDA_C_32U`| 8.0 |  |  |`HIPBLAS_C_32U`|
|`CUDA_C_64F`| 8.0 |  |  |`HIPBLAS_C_64F`|
|`CUDA_C_8I`| 8.0 |  |  |`HIPBLAS_C_8I`|
|`CUDA_C_8U`| 8.0 |  |  |`HIPBLAS_C_8U`|
|`CUDA_R_16F`| 8.0 |  |  |`HIPBLAS_R_16F`|
|`CUDA_R_32F`| 8.0 |  |  |`HIPBLAS_R_32F`|
|`CUDA_R_32I`| 8.0 |  |  |`HIPBLAS_R_32I`|
|`CUDA_R_32U`| 8.0 |  |  |`HIPBLAS_R_32U`|
|`CUDA_R_64F`| 8.0 |  |  |`HIPBLAS_R_64F`|
|`CUDA_R_8I`| 8.0 |  |  |`HIPBLAS_R_8I`|
|`CUDA_R_8U`| 8.0 |  |  |`HIPBLAS_R_8U`|
|`cudaDataType`| 8.0 |  |  |`hipblasDatatype_t`|
|`cudaDataType_t`| 8.0 |  |  |`hipblasDatatype_t`|

## **4. CUBLAS Helper Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cublasAlloc`|  |  |  |`hipblasAlloc`|
|`cublasCreate`|  |  |  |`hipblasCreate`|
|`cublasCreate_v2`|  |  |  |`hipblasCreate`|
|`cublasDestroy`|  |  |  |`hipblasDestroy`|
|`cublasDestroy_v2`|  |  |  |`hipblasDestroy`|
|`cublasFree`|  |  |  |`hipblasFree`|
|`cublasGetAtomicsMode`|  |  |  |`hipblasGetAtomicsMode`|
|`cublasGetCudartVersion`| 10.1 |  |  |`hipblasGetCudartVersion`|
|`cublasGetError`|  |  |  |`hipblasGetError`|
|`cublasGetLoggerCallback`| 9.2 |  |  |`hipblasGetLoggerCallback`|
|`cublasGetMathMode`| 9.0 |  |  |`hipblasGetMathMode`|
|`cublasGetMatrix`|  |  |  |`hipblasGetMatrix`|
|`cublasGetMatrixAsync`|  |  |  |`hipblasGetMatrixAsync`|
|`cublasGetPointerMode`|  |  |  |`hipblasGetPointerMode`|
|`cublasGetPointerMode_v2`|  |  |  |`hipblasGetPointerMode`|
|`cublasGetProperty`|  |  |  |`hipblasGetProperty`|
|`cublasGetStream`|  |  |  |`hipblasGetStream`|
|`cublasGetStream_v2`|  |  |  |`hipblasGetStream`|
|`cublasGetVector`|  |  |  |`hipblasGetVector`|
|`cublasGetVectorAsync`|  |  |  |`hipblasGetVectorAsync`|
|`cublasGetVersion`|  |  |  |`hipblasGetVersion`|
|`cublasGetVersion_v2`|  |  |  |`hipblasGetVersion`|
|`cublasInit`|  |  |  |`hipblasInit`|
|`cublasLogCallback`| 9.2 |  |  |`hipblasLogCallback`|
|`cublasLoggerConfigure`| 9.2 |  |  |`hipblasLoggerConfigure`|
|`cublasMigrateComputeType`| 11.0 |  |  |`hipblasMigrateComputeType`|
|`cublasSetAtomicsMode`|  |  |  |`hipblasSetAtomicsMode`|
|`cublasSetKernelStream`|  |  |  |`hipblasSetKernelStream`|
|`cublasSetLoggerCallback`| 9.2 |  |  |`hipblasSetLoggerCallback`|
|`cublasSetMathMode`|  |  |  |`hipblasSetMathMode`|
|`cublasSetMatrix`|  |  |  |`hipblasSetMatrix`|
|`cublasSetMatrixAsync`|  |  |  |`hipblasSetMatrixAsync`|
|`cublasSetPointerMode`|  |  |  |`hipblasSetPointerMode`|
|`cublasSetPointerMode_v2`|  |  |  |`hipblasSetPointerMode`|
|`cublasSetStream`|  |  |  |`hipblasSetStream`|
|`cublasSetStream_v2`|  |  |  |`hipblasSetStream`|
|`cublasSetVector`|  |  |  |`hipblasSetVector`|
|`cublasSetVectorAsync`|  |  |  |`hipblasSetVectorAsync`|
|`cublasShutdown`|  |  |  |`hipblasShutdown`|
|`cublasXerbla`|  |  |  |`hipblasXerbla`|

## **5. CUBLAS Level-1 Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cublasCaxpy`|  |  |  |`hipblasCaxpy`|
|`cublasCaxpy_v2`|  |  |  |`hipblasCaxpy`|
|`cublasCcopy`|  |  |  |`hipblasCcopy`|
|`cublasCcopy_v2`|  |  |  |`hipblasCcopy`|
|`cublasCdotc`|  |  |  |`hipblasCdotc`|
|`cublasCdotc_v2`|  |  |  |`hipblasCdotc`|
|`cublasCdotu`|  |  |  |`hipblasCdotu`|
|`cublasCdotu_v2`|  |  |  |`hipblasCdotu`|
|`cublasCrot`|  |  |  |`hipblasCrot`|
|`cublasCrot_v2`|  |  |  |`hipblasCrot`|
|`cublasCrotg`|  |  |  |`hipblasCrotg`|
|`cublasCrotg_v2`|  |  |  |`hipblasCrotg`|
|`cublasCscal`|  |  |  |`hipblasCscal`|
|`cublasCscal_v2`|  |  |  |`hipblasCscal`|
|`cublasCsrot`|  |  |  |`hipblasCsrot`|
|`cublasCsrot_v2`|  |  |  |`hipblasCsrot`|
|`cublasCsscal`|  |  |  |`hipblasCsscal`|
|`cublasCsscal_v2`|  |  |  |`hipblasCsscal`|
|`cublasCswap`|  |  |  |`hipblasCswap`|
|`cublasCswap_v2`|  |  |  |`hipblasCswap`|
|`cublasDasum`|  |  |  |`hipblasDasum`|
|`cublasDasum_v2`|  |  |  |`hipblasDasum`|
|`cublasDaxpy`|  |  |  |`hipblasDaxpy`|
|`cublasDaxpy_v2`|  |  |  |`hipblasDaxpy`|
|`cublasDcopy`|  |  |  |`hipblasDcopy`|
|`cublasDcopy_v2`|  |  |  |`hipblasDcopy`|
|`cublasDdot`|  |  |  |`hipblasDdot`|
|`cublasDdot_v2`|  |  |  |`hipblasDdot`|
|`cublasDnrm2`|  |  |  |`hipblasDnrm2`|
|`cublasDnrm2_v2`|  |  |  |`hipblasDnrm2`|
|`cublasDrot`|  |  |  |`hipblasDrot`|
|`cublasDrot_v2`|  |  |  |`hipblasDrot`|
|`cublasDrotg`|  |  |  |`hipblasDrotg`|
|`cublasDrotg_v2`|  |  |  |`hipblasDrotg`|
|`cublasDrotm`|  |  |  |`hipblasDrotm`|
|`cublasDrotm_v2`|  |  |  |`hipblasDrotm`|
|`cublasDrotmg`|  |  |  |`hipblasDrotmg`|
|`cublasDrotmg_v2`|  |  |  |`hipblasDrotmg`|
|`cublasDscal`|  |  |  |`hipblasDscal`|
|`cublasDscal_v2`|  |  |  |`hipblasDscal`|
|`cublasDswap`|  |  |  |`hipblasDswap`|
|`cublasDswap_v2`|  |  |  |`hipblasDswap`|
|`cublasDzasum`|  |  |  |`hipblasDzasum`|
|`cublasDzasum_v2`|  |  |  |`hipblasDzasum`|
|`cublasDznrm2`|  |  |  |`hipblasDznrm2`|
|`cublasDznrm2_v2`|  |  |  |`hipblasDznrm2`|
|`cublasIcamax`|  |  |  |`hipblasIcamax`|
|`cublasIcamax_v2`|  |  |  |`hipblasIcamax`|
|`cublasIcamin`|  |  |  |`hipblasIcamin`|
|`cublasIcamin_v2`|  |  |  |`hipblasIcamin`|
|`cublasIdamax`|  |  |  |`hipblasIdamax`|
|`cublasIdamax_v2`|  |  |  |`hipblasIdamax`|
|`cublasIdamin`|  |  |  |`hipblasIdamin`|
|`cublasIdamin_v2`|  |  |  |`hipblasIdamin`|
|`cublasIsamax`|  |  |  |`hipblasIsamax`|
|`cublasIsamax_v2`|  |  |  |`hipblasIsamax`|
|`cublasIsamin`|  |  |  |`hipblasIsamin`|
|`cublasIsamin_v2`|  |  |  |`hipblasIsamin`|
|`cublasIzamax`|  |  |  |`hipblasIzamax`|
|`cublasIzamax_v2`|  |  |  |`hipblasIzamax`|
|`cublasIzamin`|  |  |  |`hipblasIzamin`|
|`cublasIzamin_v2`|  |  |  |`hipblasIzamin`|
|`cublasNrm2Ex`| 8.0 |  |  |`hipblasNrm2Ex`|
|`cublasSasum`|  |  |  |`hipblasSasum`|
|`cublasSasum_v2`|  |  |  |`hipblasSasum`|
|`cublasSaxpy`|  |  |  |`hipblasSaxpy`|
|`cublasSaxpy_v2`|  |  |  |`hipblasSaxpy`|
|`cublasScasum`|  |  |  |`hipblasScasum`|
|`cublasScasum_v2`|  |  |  |`hipblasScasum`|
|`cublasScnrm2`|  |  |  |`hipblasScnrm2`|
|`cublasScnrm2_v2`|  |  |  |`hipblasScnrm2`|
|`cublasScopy`|  |  |  |`hipblasScopy`|
|`cublasScopy_v2`|  |  |  |`hipblasScopy`|
|`cublasSdot`|  |  |  |`hipblasSdot`|
|`cublasSdot_v2`|  |  |  |`hipblasSdot`|
|`cublasSnrm2`|  |  |  |`hipblasSnrm2`|
|`cublasSnrm2_v2`|  |  |  |`hipblasSnrm2`|
|`cublasSrot`|  |  |  |`hipblasSrot`|
|`cublasSrot_v2`|  |  |  |`hipblasSrot`|
|`cublasSrotg`|  |  |  |`hipblasSrotg`|
|`cublasSrotg_v2`|  |  |  |`hipblasSrotg`|
|`cublasSrotm`|  |  |  |`hipblasSrotm`|
|`cublasSrotm_v2`|  |  |  |`hipblasSrotm`|
|`cublasSrotmg`|  |  |  |`hipblasSrotmg`|
|`cublasSrotmg_v2`|  |  |  |`hipblasSrotmg`|
|`cublasSscal`|  |  |  |`hipblasSscal`|
|`cublasSscal_v2`|  |  |  |`hipblasSscal`|
|`cublasSswap`|  |  |  |`hipblasSswap`|
|`cublasSswap_v2`|  |  |  |`hipblasSswap`|
|`cublasZaxpy`|  |  |  |`hipblasZaxpy`|
|`cublasZaxpy_v2`|  |  |  |`hipblasZaxpy`|
|`cublasZcopy`|  |  |  |`hipblasZcopy`|
|`cublasZcopy_v2`|  |  |  |`hipblasZcopy`|
|`cublasZdotc`|  |  |  |`hipblasZdotc`|
|`cublasZdotc_v2`|  |  |  |`hipblasZdotc`|
|`cublasZdotu`|  |  |  |`hipblasZdotu`|
|`cublasZdotu_v2`|  |  |  |`hipblasZdotu`|
|`cublasZdrot`|  |  |  |`hipblasZdrot`|
|`cublasZdrot_v2`|  |  |  |`hipblasZdrot`|
|`cublasZdscal`|  |  |  |`hipblasZdscal`|
|`cublasZdscal_v2`|  |  |  |`hipblasZdscal`|
|`cublasZrot`|  |  |  |`hipblasZrot`|
|`cublasZrot_v2`|  |  |  |`hipblasZrot`|
|`cublasZrotg`|  |  |  |`hipblasZrotg`|
|`cublasZrotg_v2`|  |  |  |`hipblasZrotg`|
|`cublasZscal`|  |  |  |`hipblasZscal`|
|`cublasZscal_v2`|  |  |  |`hipblasZscal`|
|`cublasZswap`|  |  |  |`hipblasZswap`|
|`cublasZswap_v2`|  |  |  |`hipblasZswap`|

## **6. CUBLAS Level-2 Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cublasCgbmv`|  |  |  |`hipblasCgbmv`|
|`cublasCgbmv_v2`|  |  |  |`hipblasCgbmv`|
|`cublasCgemv`|  |  |  |`hipblasCgemv`|
|`cublasCgemv_v2`|  |  |  |`hipblasCgemv`|
|`cublasCgerc`|  |  |  |`hipblasCgerc`|
|`cublasCgerc_v2`|  |  |  |`hipblasCgerc`|
|`cublasCgeru`|  |  |  |`hipblasCgeru`|
|`cublasCgeru_v2`|  |  |  |`hipblasCgeru`|
|`cublasChbmv`|  |  |  |`hipblasChbmv`|
|`cublasChbmv_v2`|  |  |  |`hipblasChbmv`|
|`cublasChemv`|  |  |  |`hipblasChemv`|
|`cublasChemv_v2`|  |  |  |`hipblasChemv`|
|`cublasCher`|  |  |  |`hipblasCher`|
|`cublasCher2`|  |  |  |`hipblasCher2`|
|`cublasCher2_v2`|  |  |  |`hipblasCher2`|
|`cublasCher_v2`|  |  |  |`hipblasCher`|
|`cublasChpmv`|  |  |  |`hipblasChpmv`|
|`cublasChpmv_v2`|  |  |  |`hipblasChpmv`|
|`cublasChpr`|  |  |  |`hipblasChpr`|
|`cublasChpr2`|  |  |  |`hipblasChpr2`|
|`cublasChpr2_v2`|  |  |  |`hipblasChpr2`|
|`cublasChpr_v2`|  |  |  |`hipblasChpr`|
|`cublasCsymv`|  |  |  |`hipblasCsymv`|
|`cublasCsymv_v2`|  |  |  |`hipblasCsymv`|
|`cublasCsyr`|  |  |  |`hipblasCsyr`|
|`cublasCsyr2`|  |  |  |`hipblasCsyr2`|
|`cublasCsyr2_v2`|  |  |  |`hipblasCsyr2`|
|`cublasCsyr_v2`|  |  |  |`hipblasCsyr`|
|`cublasCtbmv`|  |  |  |`hipblasCtbmv`|
|`cublasCtbmv_v2`|  |  |  |`hipblasCtbmv`|
|`cublasCtbsv`|  |  |  |`hipblasCtbsv`|
|`cublasCtbsv_v2`|  |  |  |`hipblasCtbsv`|
|`cublasCtpmv`|  |  |  |`hipblasCtpmv`|
|`cublasCtpmv_v2`|  |  |  |`hipblasCtpmv`|
|`cublasCtpsv`|  |  |  |`hipblasCtpsv`|
|`cublasCtpsv_v2`|  |  |  |`hipblasCtpsv`|
|`cublasCtrmv`|  |  |  |`hipblasCtrmv`|
|`cublasCtrmv_v2`|  |  |  |`hipblasCtrmv`|
|`cublasCtrsv`|  |  |  |`hipblasCtrsv`|
|`cublasCtrsv_v2`|  |  |  |`hipblasCtrsv`|
|`cublasDgbmv`|  |  |  |`hipblasDgbmv`|
|`cublasDgbmv_v2`|  |  |  |`hipblasDgbmv`|
|`cublasDgemv`|  |  |  |`hipblasDgemv`|
|`cublasDgemv_v2`|  |  |  |`hipblasDgemv`|
|`cublasDger`|  |  |  |`hipblasDger`|
|`cublasDger_v2`|  |  |  |`hipblasDger`|
|`cublasDsbmv`|  |  |  |`hipblasDsbmv`|
|`cublasDsbmv_v2`|  |  |  |`hipblasDsbmv`|
|`cublasDspmv`|  |  |  |`hipblasDspmv`|
|`cublasDspmv_v2`|  |  |  |`hipblasDspmv`|
|`cublasDspr`|  |  |  |`hipblasDspr`|
|`cublasDspr2`|  |  |  |`hipblasDspr2`|
|`cublasDspr2_v2`|  |  |  |`hipblasDspr2`|
|`cublasDspr_v2`|  |  |  |`hipblasDspr`|
|`cublasDsymv`|  |  |  |`hipblasDsymv`|
|`cublasDsymv_v2`|  |  |  |`hipblasDsymv`|
|`cublasDsyr`|  |  |  |`hipblasDsyr`|
|`cublasDsyr2`|  |  |  |`hipblasDsyr2`|
|`cublasDsyr2_v2`|  |  |  |`hipblasDsyr2`|
|`cublasDsyr_v2`|  |  |  |`hipblasDsyr`|
|`cublasDtbmv`|  |  |  |`hipblasDtbmv`|
|`cublasDtbmv_v2`|  |  |  |`hipblasDtbmv`|
|`cublasDtbsv`|  |  |  |`hipblasDtbsv`|
|`cublasDtbsv_v2`|  |  |  |`hipblasDtbsv`|
|`cublasDtpmv`|  |  |  |`hipblasDtpmv`|
|`cublasDtpmv_v2`|  |  |  |`hipblasDtpmv`|
|`cublasDtpsv`|  |  |  |`hipblasDtpsv`|
|`cublasDtpsv_v2`|  |  |  |`hipblasDtpsv`|
|`cublasDtrmv`|  |  |  |`hipblasDtrmv`|
|`cublasDtrmv_v2`|  |  |  |`hipblasDtrmv`|
|`cublasDtrsv`|  |  |  |`hipblasDtrsv`|
|`cublasDtrsv_v2`|  |  |  |`hipblasDtrsv`|
|`cublasSgbmv`|  |  |  |`hipblasSgbmv`|
|`cublasSgbmv_v2`|  |  |  |`hipblasSgbmv`|
|`cublasSgemv`|  |  |  |`hipblasSgemv`|
|`cublasSgemv_v2`|  |  |  |`hipblasSgemv`|
|`cublasSger`|  |  |  |`hipblasSger`|
|`cublasSger_v2`|  |  |  |`hipblasSger`|
|`cublasSsbmv`|  |  |  |`hipblasSsbmv`|
|`cublasSsbmv_v2`|  |  |  |`hipblasSsbmv`|
|`cublasSspmv`|  |  |  |`hipblasSspmv`|
|`cublasSspmv_v2`|  |  |  |`hipblasSspmv`|
|`cublasSspr`|  |  |  |`hipblasSspr`|
|`cublasSspr2`|  |  |  |`hipblasSspr2`|
|`cublasSspr2_v2`|  |  |  |`hipblasSspr2`|
|`cublasSspr_v2`|  |  |  |`hipblasSspr`|
|`cublasSsymv`|  |  |  |`hipblasSsymv`|
|`cublasSsymv_v2`|  |  |  |`hipblasSsymv`|
|`cublasSsyr`|  |  |  |`hipblasSsyr`|
|`cublasSsyr2`|  |  |  |`hipblasSsyr2`|
|`cublasSsyr2_v2`|  |  |  |`hipblasSsyr2`|
|`cublasSsyr_v2`|  |  |  |`hipblasSsyr`|
|`cublasStbmv`|  |  |  |`hipblasStbmv`|
|`cublasStbmv_v2`|  |  |  |`hipblasStbmv`|
|`cublasStbsv`|  |  |  |`hipblasStbsv`|
|`cublasStbsv_v2`|  |  |  |`hipblasStbsv`|
|`cublasStpmv`|  |  |  |`hipblasStpmv`|
|`cublasStpmv_v2`|  |  |  |`hipblasStpmv`|
|`cublasStpsv`|  |  |  |`hipblasStpsv`|
|`cublasStpsv_v2`|  |  |  |`hipblasStpsv`|
|`cublasStrmv`|  |  |  |`hipblasStrmv`|
|`cublasStrmv_v2`|  |  |  |`hipblasStrmv`|
|`cublasStrsv`|  |  |  |`hipblasStrsv`|
|`cublasStrsv_v2`|  |  |  |`hipblasStrsv`|
|`cublasZgbmv`|  |  |  |`hipblasZgbmv`|
|`cublasZgbmv_v2`|  |  |  |`hipblasZgbmv`|
|`cublasZgemv`|  |  |  |`hipblasZgemv`|
|`cublasZgemv_v2`|  |  |  |`hipblasZgemv`|
|`cublasZgerc`|  |  |  |`hipblasZgerc`|
|`cublasZgerc_v2`|  |  |  |`hipblasZgerc`|
|`cublasZgeru`|  |  |  |`hipblasZgeru`|
|`cublasZgeru_v2`|  |  |  |`hipblasZgeru`|
|`cublasZhbmv`|  |  |  |`hipblasZhbmv`|
|`cublasZhbmv_v2`|  |  |  |`hipblasZhbmv`|
|`cublasZhemv`|  |  |  |`hipblasZhemv`|
|`cublasZhemv_v2`|  |  |  |`hipblasZhemv`|
|`cublasZher`|  |  |  |`hipblasZher`|
|`cublasZher2`|  |  |  |`hipblasZher2`|
|`cublasZher2_v2`|  |  |  |`hipblasZher2`|
|`cublasZher_v2`|  |  |  |`hipblasZher`|
|`cublasZhpmv`|  |  |  |`hipblasZhpmv`|
|`cublasZhpmv_v2`|  |  |  |`hipblasZhpmv`|
|`cublasZhpr`|  |  |  |`hipblasZhpr`|
|`cublasZhpr2`|  |  |  |`hipblasZhpr2`|
|`cublasZhpr2_v2`|  |  |  |`hipblasZhpr2`|
|`cublasZhpr_v2`|  |  |  |`hipblasZhpr`|
|`cublasZsymv`|  |  |  |`hipblasZsymv`|
|`cublasZsymv_v2`|  |  |  |`hipblasZsymv`|
|`cublasZsyr`|  |  |  |`hipblasZsyr`|
|`cublasZsyr2`|  |  |  |`hipblasZsyr2`|
|`cublasZsyr2_v2`|  |  |  |`hipblasZsyr2`|
|`cublasZsyr_v2`|  |  |  |`hipblasZsyr`|
|`cublasZtbmv`|  |  |  |`hipblasZtbmv`|
|`cublasZtbmv_v2`|  |  |  |`hipblasZtbmv`|
|`cublasZtbsv`|  |  |  |`hipblasZtbsv`|
|`cublasZtbsv_v2`|  |  |  |`hipblasZtbsv`|
|`cublasZtpmv`|  |  |  |`hipblasZtpmv`|
|`cublasZtpmv_v2`|  |  |  |`hipblasZtpmv`|
|`cublasZtpsv`|  |  |  |`hipblasZtpsv`|
|`cublasZtpsv_v2`|  |  |  |`hipblasZtpsv`|
|`cublasZtrmv`|  |  |  |`hipblasZtrmv`|
|`cublasZtrmv_v2`|  |  |  |`hipblasZtrmv`|
|`cublasZtrsv`|  |  |  |`hipblasZtrsv`|
|`cublasZtrsv_v2`|  |  |  |`hipblasZtrsv`|

## **7. CUBLAS Level-3 Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cublasCgemm`|  |  |  |`hipblasCgemm`|
|`cublasCgemm3m`| 8.0 |  |  |`hipblasCgemm3m`|
|`cublasCgemm3mBatched`| 8.0 |  |  |`hipblasCgemm3mBatched`|
|`cublasCgemm3mEx`| 8.0 |  |  |`hipblasCgemm3mEx`|
|`cublasCgemm3mStridedBatched`| 8.0 |  |  |`hipblasCgemm3mStridedBatched`|
|`cublasCgemmBatched`|  |  |  |`hipblasCgemmBatched`|
|`cublasCgemmStridedBatched`| 8.0 |  |  |`hipblasCgemmStridedBatched`|
|`cublasCgemm_v2`|  |  |  |`hipblasCgemm`|
|`cublasChemm`|  |  |  |`hipblasChemm`|
|`cublasChemm_v2`|  |  |  |`hipblasChemm`|
|`cublasCher2k`|  |  |  |`hipblasCher2k`|
|`cublasCher2k_v2`|  |  |  |`hipblasCher2k`|
|`cublasCherk`|  |  |  |`hipblasCherk`|
|`cublasCherk_v2`|  |  |  |`hipblasCherk`|
|`cublasCherkx`|  |  |  |`hipblasCherkx`|
|`cublasCsymm`|  |  |  |`hipblasCsymm`|
|`cublasCsymm_v2`|  |  |  |`hipblasCsymm`|
|`cublasCsyr2k`|  |  |  |`hipblasCsyr2k`|
|`cublasCsyr2k_v2`|  |  |  |`hipblasCsyr2k`|
|`cublasCsyrk`|  |  |  |`hipblasCsyrk`|
|`cublasCsyrk_v2`|  |  |  |`hipblasCsyrk`|
|`cublasCsyrkx`|  |  |  |`hipblasCsyrkx`|
|`cublasCtrmm`|  |  |  |`hipblasCtrmm`|
|`cublasCtrmm_v2`|  |  |  |`hipblasCtrmm`|
|`cublasCtrsm`|  |  |  |`hipblasCtrsm`|
|`cublasCtrsm_v2`|  |  |  |`hipblasCtrsm`|
|`cublasDgemm`|  |  |  |`hipblasDgemm`|
|`cublasDgemmBatched`|  |  |  |`hipblasDgemmBatched`|
|`cublasDgemmStridedBatched`| 8.0 |  |  |`hipblasDgemmStridedBatched`|
|`cublasDgemm_v2`|  |  |  |`hipblasDgemm`|
|`cublasDsymm`|  |  |  |`hipblasDsymm`|
|`cublasDsymm_v2`|  |  |  |`hipblasDsymm`|
|`cublasDsyr2k`|  |  |  |`hipblasDsyr2k`|
|`cublasDsyr2k_v2`|  |  |  |`hipblasDsyr2k`|
|`cublasDsyrk`|  |  |  |`hipblasDsyrk`|
|`cublasDsyrk_v2`|  |  |  |`hipblasDsyrk`|
|`cublasDsyrkx`|  |  |  |`hipblasDsyrkx`|
|`cublasDtrmm`|  |  |  |`hipblasDtrmm`|
|`cublasDtrmm_v2`|  |  |  |`hipblasDtrmm`|
|`cublasDtrsm`|  |  |  |`hipblasDtrsm`|
|`cublasDtrsm_v2`|  |  |  |`hipblasDtrsm`|
|`cublasHgemm`| 7.5 |  |  |`hipblasHgemm`|
|`cublasHgemmBatched`| 9.0 |  |  |`hipblasHgemmBatched`|
|`cublasHgemmStridedBatched`| 8.0 |  |  |`hipblasHgemmStridedBatched`|
|`cublasSgemm`|  |  |  |`hipblasSgemm`|
|`cublasSgemmBatched`|  |  |  |`hipblasSgemmBatched`|
|`cublasSgemmStridedBatched`| 8.0 |  |  |`hipblasSgemmStridedBatched`|
|`cublasSgemm_v2`|  |  |  |`hipblasSgemm`|
|`cublasSsymm`|  |  |  |`hipblasSsymm`|
|`cublasSsymm_v2`|  |  |  |`hipblasSsymm`|
|`cublasSsyr2k`|  |  |  |`hipblasSsyr2k`|
|`cublasSsyr2k_v2`|  |  |  |`hipblasSsyr2k`|
|`cublasSsyrk`|  |  |  |`hipblasSsyrk`|
|`cublasSsyrk_v2`|  |  |  |`hipblasSsyrk`|
|`cublasSsyrkx`|  |  |  |`hipblasSsyrkx`|
|`cublasStrmm`|  |  |  |`hipblasStrmm`|
|`cublasStrmm_v2`|  |  |  |`hipblasStrmm`|
|`cublasStrsm`|  |  |  |`hipblasStrsm`|
|`cublasStrsm_v2`|  |  |  |`hipblasStrsm`|
|`cublasZgemm`|  |  |  |`hipblasZgemm`|
|`cublasZgemm3m`| 8.0 |  |  |`hipblasZgemm3m`|
|`cublasZgemmBatched`|  |  |  |`hipblasZgemmBatched`|
|`cublasZgemmStridedBatched`| 8.0 |  |  |`hipblasZgemmStridedBatched`|
|`cublasZgemm_v2`|  |  |  |`hipblasZgemm`|
|`cublasZhemm`|  |  |  |`hipblasZhemm`|
|`cublasZhemm_v2`|  |  |  |`hipblasZhemm`|
|`cublasZher2k`|  |  |  |`hipblasZher2k`|
|`cublasZher2k_v2`|  |  |  |`hipblasZher2k`|
|`cublasZherk`|  |  |  |`hipblasZherk`|
|`cublasZherk_v2`|  |  |  |`hipblasZherk`|
|`cublasZherkx`|  |  |  |`hipblasZherkx`|
|`cublasZsymm`|  |  |  |`hipblasZsymm`|
|`cublasZsymm_v2`|  |  |  |`hipblasZsymm`|
|`cublasZsyr2k`|  |  |  |`hipblasZsyr2k`|
|`cublasZsyr2k_v2`|  |  |  |`hipblasZsyr2k`|
|`cublasZsyrk`|  |  |  |`hipblasZsyrk`|
|`cublasZsyrk_v2`|  |  |  |`hipblasZsyrk`|
|`cublasZsyrkx`|  |  |  |`hipblasZsyrkx`|
|`cublasZtrmm`|  |  |  |`hipblasZtrmm`|
|`cublasZtrmm_v2`|  |  |  |`hipblasZtrmm`|
|`cublasZtrsm`|  |  |  |`hipblasZtrsm`|
|`cublasZtrsm_v2`|  |  |  |`hipblasZtrsm`|

## **8. BLAS-like Extension**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cublasAsumEx`| 10.1 |  |  |`hipblasAsumEx`|
|`cublasAxpyEx`| 8.0 |  |  |`hipblasAxpyEx`|
|`cublasCdgmm`|  |  |  |`hipblasCdgmm`|
|`cublasCgeam`|  |  |  |`hipblasCgeam`|
|`cublasCgelsBatched`|  |  |  |`hipblasCgelsBatched`|
|`cublasCgemmEx`| 8.0 |  |  |`hipblasCgemmEx`|
|`cublasCgeqrfBatched`|  |  |  |`hipblasCgeqrfBatched`|
|`cublasCgetrfBatched`|  |  |  |`hipblasCgetrfBatched`|
|`cublasCgetriBatched`|  |  |  |`hipblasCgetriBatched`|
|`cublasCgetrsBatched`|  |  |  |`hipblasCgetrsBatched`|
|`cublasCherk3mEx`| 8.0 |  |  |`hipblasCherk3mEx`|
|`cublasCherkEx`| 8.0 |  |  |`hipblasCherkEx`|
|`cublasCmatinvBatched`|  |  |  |`hipblasCmatinvBatched`|
|`cublasCopyEx`| 10.1 |  |  |`hipblasCopyEx`|
|`cublasCsyrk3mEx`| 8.0 |  |  |`hipblasCsyrk3mEx`|
|`cublasCsyrkEx`| 8.0 |  |  |`hipblasCsyrkEx`|
|`cublasCtpttr`|  |  |  |`hipblasCtpttr`|
|`cublasCtrsmBatched`|  |  |  |`hipblasCtrsmBatched`|
|`cublasCtrttp`|  |  |  |`hipblasCtrttp`|
|`cublasDdgmm`|  |  |  |`hipblasDdgmm`|
|`cublasDgeam`|  |  |  |`hipblasDgeam`|
|`cublasDgelsBatched`|  |  |  |`hipblasDgelsBatched`|
|`cublasDgeqrfBatched`|  |  |  |`hipblasDgeqrfBatched`|
|`cublasDgetrfBatched`|  |  |  |`hipblasDgetrfBatched`|
|`cublasDgetriBatched`|  |  |  |`hipblasDgetriBatched`|
|`cublasDgetrsBatched`|  |  |  |`hipblasDgetrsBatched`|
|`cublasDmatinvBatched`|  |  |  |`hipblasDmatinvBatched`|
|`cublasDotEx`| 8.0 |  |  |`hipblasDotEx`|
|`cublasDotcEx`| 8.0 |  |  |`hipblasDotcEx`|
|`cublasDtpttr`|  |  |  |`hipblasDtpttr`|
|`cublasDtrsmBatched`|  |  |  |`hipblasDtrsmBatched`|
|`cublasDtrttp`|  |  |  |`hipblasDtrttp`|
|`cublasGemmBatchedEx`| 9.1 |  |  |`hipblasGemmBatchedEx`|
|`cublasGemmEx`| 8.0 |  |  |`hipblasGemmEx`|
|`cublasGemmStridedBatchedEx`| 9.1 |  |  |`hipblasGemmStridedBatchedEx`|
|`cublasIamaxEx`| 10.1 |  |  |`hipblasIamaxEx`|
|`cublasIaminEx`| 10.1 |  |  |`hipblasIaminEx`|
|`cublasRotEx`| 10.1 |  |  |`hipblasRotEx`|
|`cublasRotgEx`| 10.1 |  |  |`hipblasRotgEx`|
|`cublasRotmEx`| 10.1 |  |  |`hipblasRotmEx`|
|`cublasRotmgEx`| 10.1 |  |  |`hipblasRotmgEx`|
|`cublasScalEx`| 8.0 |  |  |`hipblasScalEx`|
|`cublasSdgmm`|  |  |  |`hipblasSdgmm`|
|`cublasSgeam`|  |  |  |`hipblasSgeam`|
|`cublasSgelsBatched`|  |  |  |`hipblasSgelsBatched`|
|`cublasSgemmEx`| 7.5 |  |  |`hipblasSgemmEx`|
|`cublasSgeqrfBatched`|  |  |  |`hipblasSgeqrfBatched`|
|`cublasSgetrfBatched`|  |  |  |`hipblasSgetrfBatched`|
|`cublasSgetriBatched`|  |  |  |`hipblasSgetriBatched`|
|`cublasSgetrsBatched`|  |  |  |`hipblasSgetrsBatched`|
|`cublasSmatinvBatched`|  |  |  |`hipblasSmatinvBatched`|
|`cublasStpttr`|  |  |  |`hipblasStpttr`|
|`cublasStrsmBatched`|  |  |  |`hipblasStrsmBatched`|
|`cublasStrttp`|  |  |  |`hipblasStrttp`|
|`cublasSwapEx`| 10.1 |  |  |`hipblasSwapEx`|
|`cublasUint8gemmBias`| 8.0 |  |  |`hipblasUint8gemmBias`|
|`cublasZdgmm`|  |  |  |`hipblasZdgmm`|
|`cublasZgeam`|  |  |  |`hipblasZgeam`|
|`cublasZgelsBatched`|  |  |  |`hipblasZgelsBatched`|
|`cublasZgeqrfBatched`|  |  |  |`hipblasZgeqrfBatched`|
|`cublasZgetrfBatched`|  |  |  |`hipblasZgetrfBatched`|
|`cublasZgetriBatched`|  |  |  |`hipblasZgetriBatched`|
|`cublasZgetrsBatched`|  |  |  |`hipblasZgetrsBatched`|
|`cublasZmatinvBatched`|  |  |  |`hipblasZmatinvBatched`|
|`cublasZtpttr`|  |  |  |`hipblasZtpttr`|
|`cublasZtrsmBatched`|  |  |  |`hipblasZtrsmBatched`|
|`cublasZtrttp`|  |  |  |`hipblasZtrttp`|


\* A - Added, D - Deprecated, R - Removed