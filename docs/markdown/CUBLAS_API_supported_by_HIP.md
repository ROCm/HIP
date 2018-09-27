# CUBLAS API supported by HIP

## **1. CUBLAS Data types**

| **type**     |   **CUDA**                                                    |   **HIP**                                                  |**HIP value** (if differs) |
|-------------:|---------------------------------------------------------------|------------------------------------------------------------|---------------------------|
| enum         |***`cublasStatus`***                                           |***`hipblasStatus_t`***                                     |
| enum         |***`cublasStatus_t`***                                         |***`hipblasStatus_t`***                                     |
|            0 |*`CUBLAS_STATUS_SUCCESS`*                                      |*`HIPBLAS_STATUS_SUCCESS`*                                  |
|            1 |*`CUBLAS_STATUS_NOT_INITIALIZED`*                              |*`HIPBLAS_STATUS_NOT_INITIALIZED`*                          |
|            3 |*`CUBLAS_STATUS_ALLOC_FAILED`*                                 |*`HIPBLAS_STATUS_ALLOC_FAILED`*                             | 2                         |
|            7 |*`CUBLAS_STATUS_INVALID_VALUE`*                                |*`HIPBLAS_STATUS_INVALID_VALUE`*                            | 3                         |
|            8 |*`CUBLAS_STATUS_ARCH_MISMATCH`*                                |*`HIPBLAS_STATUS_ARCH_MISMATCH`*                            |                           |
|           11 |*`CUBLAS_STATUS_MAPPING_ERROR`*                                |*`HIPBLAS_STATUS_MAPPING_ERROR`*                            | 4                         |
|           13 |*`CUBLAS_STATUS_EXECUTION_FAILED`*                             |*`HIPBLAS_STATUS_EXECUTION_FAILED`*                         | 5                         |
|           14 |*`CUBLAS_STATUS_INTERNAL_ERROR`*                               |*`HIPBLAS_STATUS_INTERNAL_ERROR`*                           | 6                         |
|           15 |*`CUBLAS_STATUS_NOT_SUPPORTED`*                                |*`HIPBLAS_STATUS_NOT_SUPPORTED`*                            | 7                         |
|           16 |*`CUBLAS_STATUS_LICENSE_ERROR`*                                |                                                            |
| enum         |***`cublasOperation_t`***                                      |***`hipblasOperation_t`***                                  |
|            0 |*`CUBLAS_OP_N`*                                                |*`HIPBLAS_OP_N`*                                            | 111                       |
|            1 |*`CUBLAS_OP_T`*                                                |*`HIPBLAS_OP_T`*                                            | 112                       |
|            2 |*`CUBLAS_OP_C`*                                                |*`HIPBLAS_OP_C`*                                            | 113                       |
| enum         |***`cublasFillMode_t`***                                       |***`hipblasFillMode_t`***                                   |
|            0 |*`CUBLAS_FILL_MODE_LOWER`*                                     |*`HIPBLAS_FILL_MODE_LOWER`*                                 | 121                       |
|            1 |*`CUBLAS_FILL_MODE_UPPER`*                                     |*`HIPBLAS_FILL_MODE_UPPER`*                                 | 122                       |
| enum         |***`cublasDiagType_t`***                                       |***`hipblasDiagType_t`***                                   |
|            0 |*`CUBLAS_DIAG_NON_UNIT`*                                       |*`HIPBLAS_DIAG_NON_UNIT`*                                   | 131                       |
|            1 |*`CUBLAS_DIAG_UNIT`*                                           |*`HIPBLAS_DIAG_UNIT`*                                       | 132                       |
| enum         |***`cublasSideMode_t`***                                       |***`hipblasSideMode_t`***                                   |
|            0 |*`CUBLAS_SIDE_LEFT`*                                           |*`HIPBLAS_SIDE_LEFT`*                                       | 141                       |
|            1 |*`CUBLAS_SIDE_RIGHT`*                                          |*`HIPBLAS_SIDE_RIGHT`*                                      | 142                       |
| enum         |***`cublasPointerMode_t`***                                    |***`hipblasPointerMode_t`***                                |
|            0 |*`CUBLAS_POINTER_MODE_HOST`*                                   |*`HIPBLAS_POINTER_MODE_HOST`*                               |
|            1 |*`CUBLAS_POINTER_MODE_DEVICE`*                                 |*`HIPBLAS_POINTER_MODE_DEVICE`*                             |
| enum         |***`cublasAtomicsMode_t`***                                    |                                                            |
|            0 |*`CUBLAS_ATOMICS_NOT_ALLOWED`*                                 |                                                            |
|            1 |*`CUBLAS_ATOMICS_ALLOWED`*                                     |                                                            |
| enum         |***`cublasGemmAlgo_t`***                                       |***`hipblasGemmAlgo_t`***                                   |
|           -1 |*`CUBLAS_GEMM_DFALT`*                                          |*`HIPBLAS_GEMM_DEFAULT`*                                    | 160                       |
|           -1 |*`CUBLAS_GEMM_DEFAULT`*                                        |*`HIPBLAS_GEMM_DEFAULT`*                                    | 160                       |
|            0 |*`CUBLAS_GEMM_ALGO0`*                                          |                                                            |
|            1 |*`CUBLAS_GEMM_ALGO1`*                                          |                                                            |
|            2 |*`CUBLAS_GEMM_ALGO2`*                                          |                                                            |
|            3 |*`CUBLAS_GEMM_ALGO3`*                                          |                                                            |
|            4 |*`CUBLAS_GEMM_ALGO4`*                                          |                                                            |
|            5 |*`CUBLAS_GEMM_ALGO5`*                                          |                                                            |
|            6 |*`CUBLAS_GEMM_ALGO6`*                                          |                                                            |
|            7 |*`CUBLAS_GEMM_ALGO7`*                                          |                                                            |
|            8 |*`CUBLAS_GEMM_ALGO8`*                                          |                                                            |
|            9 |*`CUBLAS_GEMM_ALGO9`*                                          |                                                            |
|           10 |*`CUBLAS_GEMM_ALGO10`*                                         |                                                            |
|           11 |*`CUBLAS_GEMM_ALGO11`*                                         |                                                            |
|           12 |*`CUBLAS_GEMM_ALGO12`*                                         |                                                            |
|           13 |*`CUBLAS_GEMM_ALGO13`*                                         |                                                            |
|           14 |*`CUBLAS_GEMM_ALGO14`*                                         |                                                            |
|           15 |*`CUBLAS_GEMM_ALGO15`*                                         |                                                            |
|           16 |*`CUBLAS_GEMM_ALGO16`*                                         |                                                            |
|           17 |*`CUBLAS_GEMM_ALGO17`*                                         |                                                            |
|           18 |*`CUBLAS_GEMM_ALGO18`*                                         |                                                            |
|           19 |*`CUBLAS_GEMM_ALGO19`*                                         |                                                            |
|           20 |*`CUBLAS_GEMM_ALGO20`*                                         |                                                            |
|           21 |*`CUBLAS_GEMM_ALGO21`*                                         |                                                            |
|           22 |*`CUBLAS_GEMM_ALGO22`*                                         |                                                            |
|           23 |*`CUBLAS_GEMM_ALGO23`*                                         |                                                            |
|           99 |*`CUBLAS_GEMM_DEFAULT_TENSOR_OP`*                              |                                                            |
|           99 |*`CUBLAS_GEMM_DFALT_TENSOR_OP`*                                |                                                            |
|          100 |*`CUBLAS_GEMM_ALGO0_TENSOR_OP`*                                |                                                            |
|          101 |*`CUBLAS_GEMM_ALGO1_TENSOR_OP`*                                |                                                            |
|          102 |*`CUBLAS_GEMM_ALGO2_TENSOR_OP`*                                |                                                            |
|          103 |*`CUBLAS_GEMM_ALGO3_TENSOR_OP`*                                |                                                            |
|          104 |*`CUBLAS_GEMM_ALGO4_TENSOR_OP`*                                |                                                            |
|          105 |*`CUBLAS_GEMM_ALGO5_TENSOR_OP`*                                |                                                            |
|          106 |*`CUBLAS_GEMM_ALGO6_TENSOR_OP`*                                |                                                            |
|          107 |*`CUBLAS_GEMM_ALGO7_TENSOR_OP`*                                |                                                            |
|          108 |*`CUBLAS_GEMM_ALGO8_TENSOR_OP`*                                |                                                            |
|          109 |*`CUBLAS_GEMM_ALGO9_TENSOR_OP`*                                |                                                            |
|          110 |*`CUBLAS_GEMM_ALGO10_TENSOR_OP`*                               |                                                            |
|          111 |*`CUBLAS_GEMM_ALGO11_TENSOR_OP`*                               |                                                            |
|          112 |*`CUBLAS_GEMM_ALGO12_TENSOR_OP`*                               |                                                            |
|          113 |*`CUBLAS_GEMM_ALGO13_TENSOR_OP`*                               |                                                            |
|          114 |*`CUBLAS_GEMM_ALGO14_TENSOR_OP`*                               |                                                            |
|          115 |*`CUBLAS_GEMM_ALGO15_TENSOR_OP`*                               |                                                            |
| enum         |***`cublasMath_t`***                                           |                                                            |
|            0 |*`CUBLAS_DEFAULT_MATH`*                                        |                                                            |
|            1 |*`CUBLAS_TENSOR_OP_MATH`*                                      |                                                            |
| enum*        |`cublasDataType_t`                                             |                                                            |
| struct       |`cublasContext`                                                |                                                            |
| struct*      |`cublasHandle_t`                                               |`hipblasHandle_t`                                           |

## **2. CUBLAS API functions**

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cublasCreate`                                             |`hipblasCreate`                                  |
|`cublasCreate_v2`                                          |`hipblasCreate`                                  |
|`cublasDestroy`                                            |`hipblasDestroy`                                 |
|`cublasDestroy_v2`                                         |`hipblasDestroy`                                 |
|`cublasGetVersion`                                         |                                                 |
|`cublasGetVersion_v2`                                      |                                                 |
|`cublasGetProperty`                                        |                                                 |
|`cublasGetStream`                                          |`hipblasGetStream`                               |
|`cublasGetStream_v2`                                       |`hipblasGetStream_v2`                            |
|`cublasSetStream`                                          |`hipblasSetStream`                               |
|`cublasSetStream_v2`                                       |`hipblasSetStream_v2`                            |
|`cublasGetPointerMode`                                     |`hipblasGetPointerMode`                          |
|`cublasGetPointerMode_v2`                                  |`hipblasGetPointerMode_v2`                       |
|`cublasSetPointerMode`                                     |`hipblasSetPointerMode`                          |
|`cublasSetPointerMode_v2`                                  |`hipblasSetPointerMode_v2`                       |
|`cublasGetAtomicsMode`                                     |                                                 |
|`cublasSetAtomicsMode`                                     |                                                 |
|`cublasGetMathMode`                                        |                                                 |
|`cublasSetMathMode`                                        |                                                 |
|`cublasLogCallback`                                        |                                                 |
|`cublasLoggerConfigure`                                    |                                                 |
|`cublasSetLoggerCallback`                                  |                                                 |
|`cublasGetLoggerCallback`                                  |                                                 |
|`cublasSetVector`                                          |`hipblasSetVector`                               |
|`cublasGetVector`                                          |`hipblasGetVector`                               |
|`cublasSetMatrix`                                          |`hipblasSetMatrix`                               |
|`cublasGetMatrix`                                          |`hipblasGetMatrix`                               |
|`cublasSetVectorAsync`                                     |                                                 |
|`cublasGetVectorAsync`                                     |                                                 |
|`cublasSetMatrixAsync`                                     |                                                 |
|`cublasGetMatrixAsync`                                     |                                                 |
|`cublasXerbla`                                             |                                                 |
|`cublasNrm2Ex`                                             |                                                 |
|`cublasSnrm2`                                              |`hipblasSnrm2`                                   |
|`cublasSnrm2_v2`                                           |`hipblasSnrm2`                                   |
|`cublasDnrm2`                                              |`hipblasDnrm2`                                   |
|`cublasDnrm2_v2`                                           |`hipblasDnrm2`                                   |
|`cublasScnrm2`                                             |                                                 |
|`cublasScnrm2_v2`                                          |                                                 |
|`cublasDznrm2`                                             |                                                 |
|`cublasDznrm2_v2`                                          |                                                 |
|`cublasDotEx`                                              |                                                 |
|`cublasDotcEx`                                             |                                                 |
|`cublasSdot`                                               |`hipblasSdot`                                    |
|`cublasSdot_v2`                                            |`hipblasSdot`                                    |
|`cublasDdot`                                               |`hipblasDdot`                                    |
|`cublasDdot_v2`                                            |`hipblasDdot`                                    |
|`cublasCdotu`                                              |                                                 |
|`cublasCdotu_v2`                                           |                                                 |
|`cublasCdotc`                                              |                                                 |
|`cublasCdotc_v2`                                           |                                                 |
|`cublasZdotu`                                              |                                                 |
|`cublasZdotu_v2`                                           |                                                 |
|`cublasZdotc`                                              |                                                 |
|`cublasZdotc_v2`                                           |                                                 |
|`cublasScalEx`                                             |                                                 |
|`cublasSscal`                                              |`hipblasSscal`                                   |
|`cublasSscal_v2`                                           |`hipblasSscal`                                   |
|`cublasDscal`                                              |`hipblasDscal`                                   |
|`cublasDscal_v2`                                           |`hipblasDscal`                                   |
|`cublasCscal`                                              |                                                 |
|`cublasCscal_v2`                                           |                                                 |
|`cublasCsscal`                                             |                                                 |
|`cublasCsscal_v2`                                          |                                                 |
|`cublasZscal`                                              |                                                 |
|`cublasZscal_v2`                                           |                                                 |
|`cublasZdscal`                                             |                                                 |
|`cublasZdscal_v2`                                          |                                                 |
|`cublasAxpyEx`                                             |                                                 |
|`cublasSaxpy`                                              |`hipblasSaxpy`                                   |
|`cublasSaxpy_v2`                                           |`hipblasSaxpy`                                   |
|`cublasDaxpy`                                              |`hipblasDaxpy`                                   |
|`cublasDaxpy_v2`                                           |`hipblasDaxpy`                                   |
|`cublasCaxpy`                                              |                                                 |
|`cublasCaxpy_v2`                                           |                                                 |
|`cublasZaxpy`                                              |                                                 |
|`cublasZaxpy_v2`                                           |                                                 |
|`cublasScopy`                                              |`hipblasScopy`                                   |
|`cublasScopy_v2`                                           |`hipblasScopy`                                   |
|`cublasDcopy`                                              |`hipblasDcopy`                                   |
|`cublasDcopy_v2`                                           |`hipblasDcopy`                                   |
|`cublasCcopy`                                              |                                                 |
|`cublasCcopy_v2`                                           |                                                 |
|`cublasZcopy`                                              |                                                 |
|`cublasZcopy_v2`                                           |                                                 |
|`cublasSswap`                                              |                                                 |
|`cublasSswap_v2`                                           |                                                 |
|`cublasDswap`                                              |                                                 |
|`cublasDswap_v2`                                           |                                                 |
|`cublasCswap`                                              |                                                 |
|`cublasCswap_v2`                                           |                                                 |
|`cublasZswap`                                              |                                                 |
|`cublasZswap_v2`                                           |                                                 |
|`cublasIsamax`                                             |`hipblasIsamax`                                  |
|`cublasIsamax_v2`                                          |`hipblasIsamax`                                  |
|`cublasIdamax`                                             |`hipblasIdamax`                                  |
|`cublasIdamax_v2`                                          |`hipblasIdamax`                                  |
|`cublasIcamax`                                             |                                                 |
|`cublasIcamax_v2`                                          |                                                 |
|`cublasIzamax`                                             |                                                 |
|`cublasIzamax_v2`                                          |                                                 |
|`cublasIsamin`                                             |                                                 |
|`cublasIsamin_v2`                                          |                                                 |
|`cublasIdamin`                                             |                                                 |
|`cublasIdamin_v2`                                          |                                                 |
|`cublasIcamin`                                             |                                                 |
|`cublasIcamin_v2`                                          |                                                 |
|`cublasIzamin`                                             |                                                 |
|`cublasIzamin_v2`                                          |                                                 |
|`cublasSasum`                                              |`hipblasSasum`                                   |
|`cublasSasum_v2`                                           |`hipblasSasum`                                   |
|`cublasDasum`                                              |`hipblasDasum`                                   |
|`cublasDasum_v2`                                           |`hipblasDasum`                                   |
|`cublasScasum`                                             |                                                 |
|`cublasScasum_v2`                                          |                                                 |
|`cublasDzasum`                                             |                                                 |
|`cublasDzasum_v2`                                          |                                                 |
|`cublasSrot`                                               |                                                 |
|`cublasSrot_v2`                                            |                                                 |
|`cublasDrot`                                               |                                                 |
|`cublasDrot_v2`                                            |                                                 |
|`cublasCrot`                                               |                                                 |
|`cublasCrot_v2`                                            |                                                 |
|`cublasZrot`                                               |                                                 |
|`cublasZrot_v2`                                            |                                                 |
|`cublasZdrot`                                              |                                                 |
|`cublasZdrot_v2`                                           |                                                 |
|`cublasSrotg`                                              |                                                 |
|`cublasSrotg_v2`                                           |                                                 |
|`cublasDrotg`                                              |                                                 |
|`cublasDrotg_v2`                                           |                                                 |
|`cublasCrotg`                                              |                                                 |
|`cublasCrotg_v2`                                           |                                                 |
|`cublasZrotg`                                              |                                                 |
|`cublasZrotg_v2`                                           |                                                 |
|`cublasSrotm`                                              |                                                 |
|`cublasSrotm_v2`                                           |                                                 |
|`cublasDrotm`                                              |                                                 |
|`cublasDrotm_v2`                                           |                                                 |
|`cublasSrotmg`                                             |                                                 |
|`cublasSrotmg_v2`                                          |                                                 |
|`cublasDrotmg`                                             |                                                 |
|`cublasDrotmg_v2`                                          |                                                 |
|`cublasSgemv`                                              |`hipblasSgemv`                                   |
|`cublasSgemv_v2`                                           |`hipblasSgemv`                                   |
|`cublasDgemv`                                              |`hipblasDgemv`                                   |
|`cublasDgemv_v2`                                           |`hipblasDgemv`                                   |
|`cublasCgemv`                                              |                                                 |
|`cublasCgemv_v2`                                           |                                                 |
|`cublasZgemv`                                              |                                                 |
|`cublasZgemv_v2`                                           |                                                 |
|`cublasSgbmv`                                              |                                                 |
|`cublasSgbmv_v2`                                           |                                                 |
|`cublasDgbmv`                                              |                                                 |
|`cublasDgbmv_v2`                                           |                                                 |
|`cublasCgbmv`                                              |                                                 |
|`cublasCgbmv_v2`                                           |                                                 |
|`cublasZgbmv`                                              |                                                 |
|`cublasZgbmv_v2`                                           |                                                 |
|`cublasStrmv`                                              |                                                 |
|`cublasStrmv_v2`                                           |                                                 |
|`cublasDtrmv`                                              |                                                 |
|`cublasDtrmv_v2`                                           |                                                 |
|`cublasCtrmv`                                              |                                                 |
|`cublasCtrmv_v2`                                           |                                                 |
|`cublasZtrmv`                                              |                                                 |
|`cublasZtrmv_v2`                                           |                                                 |
|`cublasStbmv`                                              |                                                 |
|`cublasStbmv_v2`                                           |                                                 |
|`cublasDtbmv`                                              |                                                 |
|`cublasDtbmv_v2`                                           |                                                 |
|`cublasCtbmv`                                              |                                                 |
|`cublasCtbmv_v2`                                           |                                                 |
|`cublasZtbmv`                                              |                                                 |
|`cublasZtbmv_v2`                                           |                                                 |
|`cublasStpmv`                                              |                                                 |
|`cublasStpmv_v2`                                           |                                                 |
|`cublasDtpmv`                                              |                                                 |
|`cublasDtpmv_v2`                                           |                                                 |
|`cublasCtpmv`                                              |                                                 |
|`cublasCtpmv_v2`                                           |                                                 |
|`cublasZtpmv`                                              |                                                 |
|`cublasZtpmv_v2`                                           |                                                 |
|`cublasStrsv`                                              |                                                 |
|`cublasStrsv_v2`                                           |                                                 |
|`cublasDtrsv`                                              |                                                 |
|`cublasDtrsv_v2`                                           |                                                 |
|`cublasCtrsv`                                              |                                                 |
|`cublasCtrsv_v2`                                           |                                                 |
|`cublasZtrsv`                                              |                                                 |
|`cublasZtrsv_v2`                                           |                                                 |
|`cublasStpsv`                                              |                                                 |
|`cublasStpsv_v2`                                           |                                                 |
|`cublasDtpsv`                                              |                                                 |
|`cublasDtpsv_v2`                                           |                                                 |
|`cublasCtpsv`                                              |                                                 |
|`cublasCtpsv_v2`                                           |                                                 |
|`cublasZtpsv`                                              |                                                 |
|`cublasZtpsv_v2`                                           |                                                 |
|`cublasStbsv`                                              |                                                 |
|`cublasStbsv_v2`                                           |                                                 |
|`cublasDtbsv`                                              |                                                 |
|`cublasDtbsv_v2`                                           |                                                 |
|`cublasCtbsv`                                              |                                                 |
|`cublasCtbsv_v2`                                           |                                                 |
|`cublasZtbsv`                                              |                                                 |
|`cublasZtbsv_v2`                                           |                                                 |
|`cublasSsymv`                                              |                                                 |
|`cublasSsymv_v2`                                           |                                                 |
|`cublasDsymv`                                              |                                                 |
|`cublasDsymv_v2`                                           |                                                 |
|`cublasCsymv`                                              |                                                 |
|`cublasCsymv_v2`                                           |                                                 |
|`cublasZsymv`                                              |                                                 |
|`cublasZsymv_v2`                                           |                                                 |
|`cublasChemv`                                              |                                                 |
|`cublasChemv_v2`                                           |                                                 |
|`cublasZhemv`                                              |                                                 |
|`cublasZhemv_v2`                                           |                                                 |
|`cublasSsbmv`                                              |                                                 |
|`cublasSsbmv_v2`                                           |                                                 |
|`cublasDsbmv`                                              |                                                 |
|`cublasDsbmv_v2`                                           |                                                 |
|`cublasChbmv`                                              |                                                 |
|`cublasChbmv_v2`                                           |                                                 |
|`cublasZhbmv`                                              |                                                 |
|`cublasZhbmv_v2`                                           |                                                 |
|`cublasSspmv`                                              |                                                 |
|`cublasSspmv_v2`                                           |                                                 |
|`cublasDspmv`                                              |                                                 |
|`cublasDspmv_v2`                                           |                                                 |
|`cublasChpmv`                                              |                                                 |
|`cublasChpmv_v2`                                           |                                                 |
|`cublasZhpmv`                                              |                                                 |
|`cublasZhpmv_v2`                                           |                                                 |
|`cublasSger`                                               |`hipblasSger`                                    |
|`cublasSger_v2`                                            |`hipblasSger`                                    |
|`cublasDger`                                               |`hipblasDger`                                    |
|`cublasDger_v2`                                            |`hipblasDger`                                    |
|`cublasCgeru`                                              |                                                 |
|`cublasCgeru_v2`                                           |                                                 |
|`cublasCgerc`                                              |                                                 |
|`cublasCgerc_v2`                                           |                                                 |
|`cublasZgeru`                                              |                                                 |
|`cublasZgeru_v2`                                           |                                                 |
|`cublasZgerc`                                              |                                                 |
|`cublasZgerc_v2`                                           |                                                 |
|`cublasSsyr`                                               |                                                 |
|`cublasSsyr_v2`                                            |                                                 |
|`cublasDsyr`                                               |                                                 |
|`cublasDsyr_v2`                                            |                                                 |
|`cublasCsyr`                                               |                                                 |
|`cublasCsyr_v2`                                            |                                                 |
|`cublasZsyr`                                               |                                                 |
|`cublasZsyr_v2`                                            |                                                 |
|`cublasCher`                                               |                                                 |
|`cublasCher_v2`                                            |                                                 |
|`cublasZher`                                               |                                                 |
|`cublasZher_v2`                                            |                                                 |
|`cublasSspr`                                               |                                                 |
|`cublasSspr_v2`                                            |                                                 |
|`cublasDspr`                                               |                                                 |
|`cublasDspr_v2`                                            |                                                 |
|`cublasChpr`                                               |                                                 |
|`cublasChpr_v2`                                            |                                                 |
|`cublasZhpr`                                               |                                                 |
|`cublasZhpr_v2`                                            |                                                 |
|`cublasSsyr2`                                              |                                                 |
|`cublasSsyr2_v2`                                           |                                                 |
|`cublasDsyr2`                                              |                                                 |
|`cublasDsyr2_v2`                                           |                                                 |
|`cublasCsyr2`                                              |                                                 |
|`cublasCsyr2_v2`                                           |                                                 |
|`cublasZsyr2`                                              |                                                 |
|`cublasZsyr2_v2`                                           |                                                 |
|`cublasCher2`                                              |                                                 |
|`cublasCher2_v2`                                           |                                                 |
|`cublasZher2`                                              |                                                 |
|`cublasZher2_v2`                                           |                                                 |
|`cublasSspr2`                                              |                                                 |
|`cublasSspr2_v2`                                           |                                                 |
|`cublasDspr2`                                              |                                                 |
|`cublasDspr2_v2`                                           |                                                 |
|`cublasChpr2`                                              |                                                 |
|`cublasChpr2_v2`                                           |                                                 |
|`cublasZhpr2`                                              |                                                 |
|`cublasZhpr2_v2`                                           |                                                 |
|`cublasSgemm`                                              |`hipblasSgemm`                                   |
|`cublasSgemm_v2`                                           |`hipblasSgemm`                                   |
|`cublasDgemm`                                              |`hipblasDgemm`                                   |
|`cublasDgemm_v2`                                           |`hipblasDgemm`                                   |
|`cublasCgemm`                                              |`hipblasCgemm`                                   |
|`cublasCgemm_v2`                                           |`hipblasCgemm`                                   |
|`cublasCgemm3m`                                            |                                                 |
|`cublasCgemm3mEx`                                          |                                                 |
|`cublasZgemm`                                              |                                                 |
|`cublasZgemm_v2`                                           |                                                 |
|`cublasZgemm3m`                                            |                                                 |
|`cublasHgemm`                                              |`hipblasHgemm`                                   |
|`cublasSgemmEx`                                            |                                                 |
|`cublasGemmEx`                                             |`hipblasGemmEx`                                  |
|`cublasCgemmEx`                                            |                                                 |
|`cublasUint8gemmBias`                                      |                                                 |
|`cublasSsyrk`                                              |                                                 |
|`cublasSsyrk_v2`                                           |                                                 |
|`cublasDsyrk`                                              |                                                 |
|`cublasDsyrk_v2`                                           |                                                 |
|`cublasCsyrk`                                              |                                                 |
|`cublasCsyrk_v2`                                           |                                                 |
|`cublasZsyrk`                                              |                                                 |
|`cublasZsyrk_v2`                                           |                                                 |
|`cublasCsyrkEx`                                            |                                                 |
|`cublasCsyrk3mEx`                                          |                                                 |
|`cublasCherk`                                              |                                                 |
|`cublasCherk_v2`                                           |                                                 |
|`cublasZherk`                                              |                                                 |
|`cublasZherk_v2`                                           |                                                 |
|`cublasCherkEx`                                            |                                                 |
|`cublasCherk3mEx`                                          |                                                 |
|`cublasSsyr2k`                                             |                                                 |
|`cublasSsyr2k_v2`                                          |                                                 |
|`cublasDsyr2k`                                             |                                                 |
|`cublasDsyr2k_v2`                                          |                                                 |
|`cublasCsyr2k`                                             |                                                 |
|`cublasCsyr2k_v2`                                          |                                                 |
|`cublasZsyr2k`                                             |                                                 |
|`cublasZsyr2k_v2`                                          |                                                 |
|`cublasCher2k`                                             |                                                 |
|`cublasCher2k_v2`                                          |                                                 |
|`cublasZher2k`                                             |                                                 |
|`cublasZher2k_v2`                                          |                                                 |
|`cublasSsyrkx`                                             |                                                 |
|`cublasDsyrkx`                                             |                                                 |
|`cublasCsyrkx`                                             |                                                 |
|`cublasZsyrkx`                                             |                                                 |
|`cublasCherkx`                                             |                                                 |
|`cublasZherkx`                                             |                                                 |
|`cublasSsymm`                                              |                                                 |
|`cublasSsymm_v2`                                           |                                                 |
|`cublasDsymm`                                              |                                                 |
|`cublasDsymm_v2`                                           |                                                 |
|`cublasCsymm`                                              |                                                 |
|`cublasCsymm_v2`                                           |                                                 |
|`cublasZsymm`                                              |                                                 |
|`cublasZsymm_v2`                                           |                                                 |
|`cublasChemm`                                              |                                                 |
|`cublasChemm_v2`                                           |                                                 |
|`cublasZhemm`                                              |                                                 |
|`cublasZhemm_v2`                                           |                                                 |
|`cublasStrsm`                                              |`hipblasStrsm`                                   |
|`cublasStrsm_v2`                                           |`hipblasStrsm`                                   |
|`cublasDtrsm`                                              |`hipblasDtrsm`                                   |
|`cublasDtrsm_v2`                                           |`hipblasDtrsm`                                   |
|`cublasCtrsm`                                              |                                                 |
|`cublasCtrsm_v2`                                           |                                                 |
|`cublasZtrsm`                                              |                                                 |
|`cublasZtrsm_v2`                                           |                                                 |
|`cublasStrmm`                                              |                                                 |
|`cublasStrmm_v2`                                           |                                                 |
|`cublasDtrmm`                                              |                                                 |
|`cublasDtrmm_v2`                                           |                                                 |
|`cublasCtrmm`                                              |                                                 |
|`cublasCtrmm_v2`                                           |                                                 |
|`cublasZtrmm`                                              |                                                 |
|`cublasZtrmm_v2`                                           |                                                 |
|`cublasHgemmBatched`                                       |                                                 |
|`cublasSgemmBatched`                                       |`hipblasSgemmBatched`                            |
|`cublasDgemmBatched`                                       |`hipblasDgemmBatched`                            |
|`cublasCgemmBatched`                                       |                                                 |
|`cublasCgemm3mBatched`                                     |                                                 |
|`cublasZgemmBatched`                                       |                                                 |
|`cublasGemmBatchedEx`                                      |                                                 |
|`cublasGemmStridedBatchedEx`                               |                                                 |
|`cublasSgemmStridedBatched`                                |                                                 |
|`cublasDgemmStridedBatched`                                |                                                 |
|`cublasCgemmStridedBatched`                                |                                                 |
|`cublasCgemm3mStridedBatched`                              |                                                 |
|`cublasZgemmStridedBatched`                                |                                                 |
|`cublasHgemmStridedBatched`                                |                                                 |
|`cublasSgeam`                                              |`hipblasSgeam`                                   |
|`cublasDgeam`                                              |`hipblasDgeam`                                   |
|`cublasCgeam`                                              |                                                 |
|`cublasZgeam`                                              |                                                 |
|`cublasSgetrfBatched`                                      |                                                 |
|`cublasDgetrfBatched`                                      |                                                 |
|`cublasCgetrfBatched`                                      |                                                 |
|`cublasZgetrfBatched`                                      |                                                 |
|`cublasSgetriBatched`                                      |                                                 |
|`cublasDgetriBatched`                                      |                                                 |
|`cublasCgetriBatched`                                      |                                                 |
|`cublasZgetriBatched`                                      |                                                 |
|`cublasSgetrsBatched`                                      |                                                 |
|`cublasDgetrsBatched`                                      |                                                 |
|`cublasCgetrsBatched`                                      |                                                 |
|`cublasZgetrsBatched`                                      |                                                 |
|`cublasStrsmBatched`                                       |                                                 |
|`cublasDtrsmBatched`                                       |                                                 |
|`cublasCtrsmBatched`                                       |                                                 |
|`cublasZtrsmBatched`                                       |                                                 |
|`cublasSmatinvBatched`                                     |                                                 |
|`cublasDmatinvBatched`                                     |                                                 |
|`cublasCmatinvBatched`                                     |                                                 |
|`cublasZmatinvBatched`                                     |                                                 |
|`cublasSgeqrfBatched`                                      |                                                 |
|`cublasDgeqrfBatched`                                      |                                                 |
|`cublasCgeqrfBatched`                                      |                                                 |
|`cublasZgeqrfBatched`                                      |                                                 |
|`cublasSgelsBatched`                                       |                                                 |
|`cublasDgelsBatched`                                       |                                                 |
|`cublasCgelsBatched`                                       |                                                 |
|`cublasZgelsBatched`                                       |                                                 |
|`cublasSdgmm`                                              |                                                 |
|`cublasDdgmm`                                              |                                                 |
|`cublasCdgmm`                                              |                                                 |
|`cublasZdgmm`                                              |                                                 |
|`cublasStpttr`                                             |                                                 |
|`cublasDtpttr`                                             |                                                 |
|`cublasCtpttr`                                             |                                                 |
|`cublasZtpttr`                                             |                                                 |
|`cublasStrttp`                                             |                                                 |
|`cublasDtrttp`                                             |                                                 |
|`cublasCtrttp`                                             |                                                 |
|`cublasZtrttp`                                             |                                                 |
