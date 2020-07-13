# CUBLAS API supported by HIP

## **1. CUBLAS Data types**

| **type**     |   **CUDA**                                                    |**CUDA version\***|   **HIP**                                                  |**HIP value** (if differs) |
|-------------:|---------------------------------------------------------------|:----------------:|------------------------------------------------------------|---------------------------|
| define       |`CUBLAS_VER_MAJOR`                                             | 10.1 Update 2    |                                                            |
| define       |`CUBLAS_VER_MINOR`                                             | 10.1 Update 2    |                                                            |
| define       |`CUBLAS_VER_PATCH`                                             | 10.1 Update 2    |                                                            |
| define       |`CUBLAS_VER_BUILD`                                             | 10.1 Update 2    |                                                            |
| define       |`CUBLAS_VERSION`                                               | 10.1 Update 2    |                                                            |
| enum         |***`cublasStatus`***                                           |                  |***`hipblasStatus_t`***                                     |
| enum         |***`cublasStatus_t`***                                         |                  |***`hipblasStatus_t`***                                     |
|            0 |*`CUBLAS_STATUS_SUCCESS`*                                      |                  |*`HIPBLAS_STATUS_SUCCESS`*                                  |
|            1 |*`CUBLAS_STATUS_NOT_INITIALIZED`*                              |                  |*`HIPBLAS_STATUS_NOT_INITIALIZED`*                          |
|            3 |*`CUBLAS_STATUS_ALLOC_FAILED`*                                 |                  |*`HIPBLAS_STATUS_ALLOC_FAILED`*                             | 2                         |
|            7 |*`CUBLAS_STATUS_INVALID_VALUE`*                                |                  |*`HIPBLAS_STATUS_INVALID_VALUE`*                            | 3                         |
|            8 |*`CUBLAS_STATUS_ARCH_MISMATCH`*                                |                  |*`HIPBLAS_STATUS_ARCH_MISMATCH`*                            |                           |
|           11 |*`CUBLAS_STATUS_MAPPING_ERROR`*                                |                  |*`HIPBLAS_STATUS_MAPPING_ERROR`*                            | 4                         |
|           13 |*`CUBLAS_STATUS_EXECUTION_FAILED`*                             |                  |*`HIPBLAS_STATUS_EXECUTION_FAILED`*                         | 5                         |
|           14 |*`CUBLAS_STATUS_INTERNAL_ERROR`*                               |                  |*`HIPBLAS_STATUS_INTERNAL_ERROR`*                           | 6                         |
|           15 |*`CUBLAS_STATUS_NOT_SUPPORTED`*                                |                  |*`HIPBLAS_STATUS_NOT_SUPPORTED`*                            | 7                         |
|           16 |*`CUBLAS_STATUS_LICENSE_ERROR`*                                |                  |                                                            |
| enum         |***`cublasOperation_t`***                                      |                  |***`hipblasOperation_t`***                                  |
|            0 |*`CUBLAS_OP_N`*                                                |                  |*`HIPBLAS_OP_N`*                                            | 111                       |
|            1 |*`CUBLAS_OP_T`*                                                |                  |*`HIPBLAS_OP_T`*                                            | 112                       |
|            2 |*`CUBLAS_OP_C`*                                                |                  |*`HIPBLAS_OP_C`*                                            | 113                       |
|            2 |*`CUBLAS_OP_HERMITAN`*                                         | 10.1             |*`HIPBLAS_OP_C`*                                            | 113                       |
|            3 |*`CUBLAS_OP_CONJG`*                                            | 10.1             |                                                            |
| enum         |***`cublasFillMode_t`***                                       |                  |***`hipblasFillMode_t`***                                   |
|            0 |*`CUBLAS_FILL_MODE_LOWER`*                                     |                  |*`HIPBLAS_FILL_MODE_LOWER`*                                 | 121                       |
|            1 |*`CUBLAS_FILL_MODE_UPPER`*                                     |                  |*`HIPBLAS_FILL_MODE_UPPER`*                                 | 122                       |
|            2 |*`CUBLAS_FILL_MODE_FULL`*                                      | 10.1             |*`HIPBLAS_FILL_MODE_FULL`*                                  | 123                       |
| enum         |***`cublasDiagType_t`***                                       |                  |***`hipblasDiagType_t`***                                   |
|            0 |*`CUBLAS_DIAG_NON_UNIT`*                                       |                  |*`HIPBLAS_DIAG_NON_UNIT`*                                   | 131                       |
|            1 |*`CUBLAS_DIAG_UNIT`*                                           |                  |*`HIPBLAS_DIAG_UNIT`*                                       | 132                       |
| enum         |***`cublasSideMode_t`***                                       |                  |***`hipblasSideMode_t`***                                   |
|            0 |*`CUBLAS_SIDE_LEFT`*                                           |                  |*`HIPBLAS_SIDE_LEFT`*                                       | 141                       |
|            1 |*`CUBLAS_SIDE_RIGHT`*                                          |                  |*`HIPBLAS_SIDE_RIGHT`*                                      | 142                       |
| enum         |***`cublasPointerMode_t`***                                    |                  |***`hipblasPointerMode_t`***                                |
|            0 |*`CUBLAS_POINTER_MODE_HOST`*                                   |                  |*`HIPBLAS_POINTER_MODE_HOST`*                               |
|            1 |*`CUBLAS_POINTER_MODE_DEVICE`*                                 |                  |*`HIPBLAS_POINTER_MODE_DEVICE`*                             |
| enum         |***`cublasAtomicsMode_t`***                                    |                  |                                                            |
|            0 |*`CUBLAS_ATOMICS_NOT_ALLOWED`*                                 |                  |                                                            |
|            1 |*`CUBLAS_ATOMICS_ALLOWED`*                                     |                  |                                                            |
| enum         |***`cublasGemmAlgo_t`***                                       | 8.0              |***`hipblasGemmAlgo_t`***                                   |
|           -1 |*`CUBLAS_GEMM_DFALT`*                                          | 8.0              |*`HIPBLAS_GEMM_DEFAULT`*                                    | 160                       |
|           -1 |*`CUBLAS_GEMM_DEFAULT`*                                        | 8.0              |*`HIPBLAS_GEMM_DEFAULT`*                                    | 160                       |
|            0 |*`CUBLAS_GEMM_ALGO0`*                                          | 8.0              |                                                            |
|            1 |*`CUBLAS_GEMM_ALGO1`*                                          | 8.0              |                                                            |
|            2 |*`CUBLAS_GEMM_ALGO2`*                                          | 8.0              |                                                            |
|            3 |*`CUBLAS_GEMM_ALGO3`*                                          | 8.0              |                                                            |
|            4 |*`CUBLAS_GEMM_ALGO4`*                                          | 8.0              |                                                            |
|            5 |*`CUBLAS_GEMM_ALGO5`*                                          | 8.0              |                                                            |
|            6 |*`CUBLAS_GEMM_ALGO6`*                                          | 8.0              |                                                            |
|            7 |*`CUBLAS_GEMM_ALGO7`*                                          | 8.0              |                                                            |
|            8 |*`CUBLAS_GEMM_ALGO8`*                                          | 9.0              |                                                            |
|            9 |*`CUBLAS_GEMM_ALGO9`*                                          | 9.0              |                                                            |
|           10 |*`CUBLAS_GEMM_ALGO10`*                                         | 9.0              |                                                            |
|           11 |*`CUBLAS_GEMM_ALGO11`*                                         | 9.0              |                                                            |
|           12 |*`CUBLAS_GEMM_ALGO12`*                                         | 9.0              |                                                            |
|           13 |*`CUBLAS_GEMM_ALGO13`*                                         | 9.0              |                                                            |
|           14 |*`CUBLAS_GEMM_ALGO14`*                                         | 9.0              |                                                            |
|           15 |*`CUBLAS_GEMM_ALGO15`*                                         | 9.0              |                                                            |
|           16 |*`CUBLAS_GEMM_ALGO16`*                                         | 9.0              |                                                            |
|           17 |*`CUBLAS_GEMM_ALGO17`*                                         | 9.0              |                                                            |
|           18 |*`CUBLAS_GEMM_ALGO18`*                                         | 9.2              |                                                            |
|           19 |*`CUBLAS_GEMM_ALGO19`*                                         | 9.2              |                                                            |
|           20 |*`CUBLAS_GEMM_ALGO20`*                                         | 9.2              |                                                            |
|           21 |*`CUBLAS_GEMM_ALGO21`*                                         | 9.2              |                                                            |
|           22 |*`CUBLAS_GEMM_ALGO22`*                                         | 9.2              |                                                            |
|           23 |*`CUBLAS_GEMM_ALGO23`*                                         | 9.2              |                                                            |
|           99 |*`CUBLAS_GEMM_DEFAULT_TENSOR_OP`*                              | 9.0              |                                                            |
|           99 |*`CUBLAS_GEMM_DFALT_TENSOR_OP`*                                | 9.0              |                                                            |
|          100 |*`CUBLAS_GEMM_ALGO0_TENSOR_OP`*                                | 9.0              |                                                            |
|          101 |*`CUBLAS_GEMM_ALGO1_TENSOR_OP`*                                | 9.0              |                                                            |
|          102 |*`CUBLAS_GEMM_ALGO2_TENSOR_OP`*                                | 9.0              |                                                            |
|          103 |*`CUBLAS_GEMM_ALGO3_TENSOR_OP`*                                | 9.0              |                                                            |
|          104 |*`CUBLAS_GEMM_ALGO4_TENSOR_OP`*                                | 9.0              |                                                            |
|          105 |*`CUBLAS_GEMM_ALGO5_TENSOR_OP`*                                | 9.2              |                                                            |
|          106 |*`CUBLAS_GEMM_ALGO6_TENSOR_OP`*                                | 9.2              |                                                            |
|          107 |*`CUBLAS_GEMM_ALGO7_TENSOR_OP`*                                | 9.2              |                                                            |
|          108 |*`CUBLAS_GEMM_ALGO8_TENSOR_OP`*                                | 9.2              |                                                            |
|          109 |*`CUBLAS_GEMM_ALGO9_TENSOR_OP`*                                | 9.2              |                                                            |
|          110 |*`CUBLAS_GEMM_ALGO10_TENSOR_OP`*                               | 9.2              |                                                            |
|          111 |*`CUBLAS_GEMM_ALGO11_TENSOR_OP`*                               | 9.2              |                                                            |
|          112 |*`CUBLAS_GEMM_ALGO12_TENSOR_OP`*                               | 9.2              |                                                            |
|          113 |*`CUBLAS_GEMM_ALGO13_TENSOR_OP`*                               | 9.2              |                                                            |
|          114 |*`CUBLAS_GEMM_ALGO14_TENSOR_OP`*                               | 9.2              |                                                            |
|          115 |*`CUBLAS_GEMM_ALGO15_TENSOR_OP`*                               | 9.2              |                                                            |
| enum         |***`cublasMath_t`***                                           | 9.0              |                                                            |
|            0 |*`CUBLAS_DEFAULT_MATH`*                                        | 9.0              |                                                            |
|            1 |*`CUBLAS_TENSOR_OP_MATH`*                                      | 9.0              |                                                            |
|            2 |*`CUBLAS_PEDANTIC_MATH`*                                       | 11.0             |                                                            |
|            3 |*`CUBLAS_TF32_TENSOR_OP_MATH`*                                 | 11.0             |                                                            |
|           16 |*`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`*           | 11.0             |                                                            |
| enum*        |`cublasDataType_t`                                             | 7.5              |                                                            |
| struct       |`cublasContext`                                                |                  |                                                            |
| struct*      |`cublasHandle_t`                                               |                  |`hipblasHandle_t`                                           |
| enum         |***`cublasComputeType_t`***                                    | 11.0             |                                                            |
|           64 |*`CUBLAS_COMPUTE_16F`*                                         | 11.0             |                                                            |
|           65 |*`CUBLAS_COMPUTE_16F_PEDANTIC`*                                | 11.0             |                                                            |
|           68 |*`CUBLAS_COMPUTE_32F`*                                         | 11.0             |                                                            |
|           69 |*`CUBLAS_COMPUTE_32F_PEDANTIC`*                                | 11.0             |                                                            |
|           74 |*`CUBLAS_COMPUTE_32F_FAST_16F`*                                | 11.0             |                                                            |
|           75 |*`CUBLAS_COMPUTE_32F_FAST_16BF`*                               | 11.0             |                                                            |
|           77 |*`CUBLAS_COMPUTE_32F_FAST_TF32`*                               | 11.0             |                                                            |
|           70 |*`CUBLAS_COMPUTE_64F`*                                         | 11.0             |                                                            |
|           71 |*`CUBLAS_COMPUTE_64F_PEDANTIC`*                                | 11.0             |                                                            |
|           72 |*`CUBLAS_COMPUTE_32I`*                                         | 11.0             |                                                            |
|           73 |*`CUBLAS_COMPUTE_32I_PEDANTIC`*                                | 11.0             |                                                            |

## **2. CUBLAS API functions**

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cublasCreate`                                             |`hipblasCreate`                                  |
|`cublasCreate_v2`                                          |`hipblasCreate`                                  |
|`cublasDestroy`                                            |`hipblasDestroy`                                 |
|`cublasDestroy_v2`                                         |`hipblasDestroy`                                 |
|`cublasGetVersion`                                         |                                                 |
|`cublasGetVersion_v2`                                      |                                                 |
|`cublasGetProperty`                                        |                                                 | 8.0              |
|`cublasGetCudartVersion`                                   |                                                 | 10.1             |
|`cublasGetStream`                                          |`hipblasGetStream`                               |
|`cublasGetStream_v2`                                       |`hipblasGetStream`                               |
|`cublasSetStream`                                          |`hipblasSetStream`                               |
|`cublasSetStream_v2`                                       |`hipblasSetStream`                               |
|`cublasGetPointerMode`                                     |`hipblasGetPointerMode`                          |
|`cublasGetPointerMode_v2`                                  |`hipblasGetPointerMode`                          |
|`cublasSetPointerMode`                                     |`hipblasSetPointerMode`                          |
|`cublasSetPointerMode_v2`                                  |`hipblasSetPointerMode`                          |
|`cublasGetAtomicsMode`                                     |                                                 |
|`cublasSetAtomicsMode`                                     |                                                 |
|`cublasGetMathMode`                                        |                                                 | 9.0              |
|`cublasSetMathMode`                                        |                                                 | 9.0              |
|`cublasLogCallback`                                        |                                                 | 9.2              |
|`cublasLoggerConfigure`                                    |                                                 | 9.2              |
|`cublasSetLoggerCallback`                                  |                                                 | 9.2              |
|`cublasGetLoggerCallback`                                  |                                                 | 9.2              |
|`cublasMigrateComputeType`                                 |                                                 | 11.0             |
|`cublasSetVector`                                          |`hipblasSetVector`                               |
|`cublasGetVector`                                          |`hipblasGetVector`                               |
|`cublasSetMatrix`                                          |`hipblasSetMatrix`                               |
|`cublasGetMatrix`                                          |`hipblasGetMatrix`                               |
|`cublasSetVectorAsync`                                     |`hipblasSetVectorAsync`                          |
|`cublasGetVectorAsync`                                     |`hipblasGetVectorAsync`                          |
|`cublasSetMatrixAsync`                                     |`hipblasSetMatrixAsync`                          |
|`cublasGetMatrixAsync`                                     |`hipblasGetMatrixAsync`                          |
|`cublasXerbla`                                             |                                                 |
|`cublasNrm2Ex`                                             |                                                 | 8.0              |
|`cublasSnrm2`                                              |`hipblasSnrm2`                                   |
|`cublasSnrm2_v2`                                           |`hipblasSnrm2`                                   |
|`cublasDnrm2`                                              |`hipblasDnrm2`                                   |
|`cublasDnrm2_v2`                                           |`hipblasDnrm2`                                   |
|`cublasScnrm2`                                             |`hipblasScnrm2`                                  |
|`cublasScnrm2_v2`                                          |`hipblasScnrm2`                                  |
|`cublasDznrm2`                                             |`hipblasDznrm2`                                  |
|`cublasDznrm2_v2`                                          |`hipblasDznrm2`                                  |
|`cublasDotEx`                                              |                                                 | 8.0              |
|`cublasDotcEx`                                             |                                                 | 8.0              |
|`cublasSdot`                                               |`hipblasSdot`                                    |
|`cublasSdot_v2`                                            |`hipblasSdot`                                    |
|`cublasDdot`                                               |`hipblasDdot`                                    |
|`cublasDdot_v2`                                            |`hipblasDdot`                                    |
|`cublasCdotu`                                              |`hipblasCdotu`                                   |
|`cublasCdotu_v2`                                           |`hipblasCdotu`                                   |
|`cublasCdotc`                                              |`hipblasCdotc`                                   |
|`cublasCdotc_v2`                                           |`hipblasCdotc`                                   |
|`cublasZdotu`                                              |`hipblasZdotu`                                   |
|`cublasZdotu_v2`                                           |`hipblasZdotu`                                   |
|`cublasZdotc`                                              |`hipblasZdotc`                                   |
|`cublasZdotc_v2`                                           |`hipblasZdotc`                                   |
|`cublasScalEx`                                             |                                                 | 8.0              |
|`cublasSscal`                                              |`hipblasSscal`                                   |
|`cublasSscal_v2`                                           |`hipblasSscal`                                   |
|`cublasDscal`                                              |`hipblasDscal`                                   |
|`cublasDscal_v2`                                           |`hipblasDscal`                                   |
|`cublasCscal`                                              |`hipblasCscal`                                   |
|`cublasCscal_v2`                                           |`hipblasCscal`                                   |
|`cublasCsscal`                                             |`hipblasCsscal`                                  |
|`cublasCsscal_v2`                                          |`hipblasCsscal`                                  |
|`cublasZscal`                                              |`hipblasZscal`                                   |
|`cublasZscal_v2`                                           |`hipblasZscal`                                   |
|`cublasZdscal`                                             |`hipblasZdscal`                                  |
|`cublasZdscal_v2`                                          |`hipblasZdscal`                                  |
|`cublasAxpyEx`                                             |                                                 | 8.0              |
|`cublasSaxpy`                                              |`hipblasSaxpy`                                   |
|`cublasSaxpy_v2`                                           |`hipblasSaxpy`                                   |
|`cublasDaxpy`                                              |`hipblasDaxpy`                                   |
|`cublasDaxpy_v2`                                           |`hipblasDaxpy`                                   |
|`cublasCaxpy`                                              |`hipblasCaxpy`                                   |
|`cublasCaxpy_v2`                                           |`hipblasCaxpy`                                   |
|`cublasZaxpy`                                              |`hipblasZaxpy`                                   |
|`cublasZaxpy_v2`                                           |`hipblasZaxpy`                                   |
|`cublasScopy`                                              |`hipblasScopy`                                   |
|`cublasScopy_v2`                                           |`hipblasScopy`                                   |
|`cublasDcopy`                                              |`hipblasDcopy`                                   |
|`cublasDcopy_v2`                                           |`hipblasDcopy`                                   |
|`cublasCcopy`                                              |`hipblasCcopy`                                   |
|`cublasCopyEx`                                             |                                                 | 10.1             |
|`cublasCcopy_v2`                                           |`hipblasCcopy`                                   |
|`cublasZcopy`                                              |`hipblasZcopy`                                   |
|`cublasZcopy_v2`                                           |`hipblasZcopy`                                   |
|`cublasSswap`                                              |`hipblasSswap`                                   |
|`cublasSswap_v2`                                           |`hipblasSswap`                                   |
|`cublasDswap`                                              |`hipblasDswap`                                   |
|`cublasDswap_v2`                                           |`hipblasDswap`                                   |
|`cublasCswap`                                              |`hipblasCswap`                                   |
|`cublasCswap_v2`                                           |`hipblasCswap`                                   |
|`cublasZswap`                                              |`hipblasZswap`                                   |
|`cublasZswap_v2`                                           |`hipblasZswap`                                   |
|`cublasIamaxEx`                                            |                                                 | 10.1             |
|`cublasIsamax`                                             |`hipblasIsamax`                                  |
|`cublasIsamax_v2`                                          |`hipblasIsamax`                                  |
|`cublasIdamax`                                             |`hipblasIdamax`                                  |
|`cublasIdamax_v2`                                          |`hipblasIdamax`                                  |
|`cublasIcamax`                                             |`hipblasIcamax`                                  |
|`cublasIcamax_v2`                                          |`hipblasIcamax`                                  |
|`cublasIzamax`                                             |`hipblasIzamax`                                  |
|`cublasIzamax_v2`                                          |`hipblasIzamax`                                  |
|`cublasIaminEx`                                            |                                                 | 10.1             |
|`cublasIsamin`                                             |`hipblasIsamin`                                  |
|`cublasIsamin_v2`                                          |`hipblasIsamin`                                  |
|`cublasIdamin`                                             |`hipblasIdamin`                                  |
|`cublasIdamin_v2`                                          |`hipblasIdamin`                                  |
|`cublasIcamin`                                             |`hipblasIcamin`                                  |
|`cublasIcamin_v2`                                          |`hipblasIcamin`                                  |
|`cublasIzamin`                                             |`hipblasIzamin`                                  |
|`cublasIzamin_v2`                                          |`hipblasIzamin`                                  |
|`cublasAsumEx`                                             |                                                 | 10.1             |
|`cublasSasum`                                              |`hipblasSasum`                                   |
|`cublasSasum_v2`                                           |`hipblasSasum`                                   |
|`cublasDasum`                                              |`hipblasDasum`                                   |
|`cublasDasum_v2`                                           |`hipblasDasum`                                   |
|`cublasScasum`                                             |`hipblasScasum`                                  |
|`cublasScasum_v2`                                          |`hipblasScasum`                                  |
|`cublasDzasum`                                             |`hipblasDzasum`                                  |
|`cublasDzasum_v2`                                          |`hipblasDzasum`                                  |
|`cublasRotEx`                                              |                                                 | 10.1             |
|`cublasSrot`                                               |`hipblasSrot`                                    |
|`cublasSrot_v2`                                            |`hipblasSrot`                                    |
|`cublasDrot`                                               |`hipblasDrot`                                    |
|`cublasDrot_v2`                                            |`hipblasDrot`                                    |
|`cublasCrot`                                               |`hipblasCrot`                                    |
|`cublasCrot_v2`                                            |`hipblasCrot`                                    |
|`cublasCsrot`                                              |`hipblasCsrot`                                   |
|`cublasCsrot_v2`                                           |`hipblasCsrot`                                   |
|`cublasZrot`                                               |`hipblasZrot`                                    |
|`cublasZrot_v2`                                            |`hipblasZrot`                                    |
|`cublasRotgEx`                                             |                                                 | 10.1             |
|`cublasZdrot`                                              |`hipblasZdrot`                                   |
|`cublasZdrot_v2`                                           |`hipblasZdrot`                                   |
|`cublasSrotg`                                              |`hipblasSrotg`                                   |
|`cublasSrotg_v2`                                           |`hipblasSrotg`                                   |
|`cublasDrotg`                                              |`hipblasDrotg`                                   |
|`cublasDrotg_v2`                                           |`hipblasDrotg`                                   |
|`cublasCrotg`                                              |`hipblasCrotg`                                   |
|`cublasCrotg_v2`                                           |`hipblasCrotg`                                   |
|`cublasZrotg`                                              |`hipblasZrotg`                                   |
|`cublasZrotg_v2`                                           |`hipblasZrotg`                                   |
|`cublasRotmEx`                                             |                                                 | 10.1             |
|`cublasSrotm`                                              |`hipblasSrotm`                                   |
|`cublasSrotm_v2`                                           |`hipblasSrotm`                                   |
|`cublasDrotm`                                              |`hipblasDrotm`                                   |
|`cublasDrotm_v2`                                           |`hipblasDrotm`                                   |
|`cublasRotmgEx`                                            |                                                 | 10.1             |
|`cublasSrotmg`                                             |`hipblasSrotmg`                                  |
|`cublasSrotmg_v2`                                          |`hipblasSrotmg`                                  |
|`cublasDrotmg`                                             |`hipblasDrotmg`                                  |
|`cublasDrotmg_v2`                                          |`hipblasDrotmg`                                  |
|`cublasSgemv`                                              |`hipblasSgemv`                                   |
|`cublasSgemv_v2`                                           |`hipblasSgemv`                                   |
|`cublasSgemvBatched`                                       |`hipblasSgemvBatched`                            |
|`cublasDgemv`                                              |`hipblasDgemv`                                   |
|`cublasDgemv_v2`                                           |`hipblasDgemv`                                   |
|`cublasCgemv`                                              |`hipblasCgemv`                                   |
|`cublasCgemv_v2`                                           |`hipblasCgemv`                                   |
|`cublasZgemv`                                              |`hipblasZgemv`                                   |
|`cublasZgemv_v2`                                           |`hipblasZgemv`                                   |
|`cublasSgbmv`                                              |`hipblasSgbmv`                                   |
|`cublasSgbmv_v2`                                           |`hipblasSgbmv`                                   |
|`cublasDgbmv`                                              |`hipblasDgbmv`                                   |
|`cublasDgbmv_v2`                                           |`hipblasDgbmv`                                   |
|`cublasCgbmv`                                              |`hipblasCgbmv`                                   |
|`cublasCgbmv_v2`                                           |`hipblasCgbmv`                                   |
|`cublasZgbmv`                                              |`hipblasZgbmv`                                   |
|`cublasZgbmv_v2`                                           |`hipblasZgbmv`                                   |
|`cublasStrmv`                                              |`hipblasStrmv`                                   |
|`cublasStrmv_v2`                                           |`hipblasStrmv`                                   |
|`cublasDtrmv`                                              |`hipblasDtrmv`                                   |
|`cublasDtrmv_v2`                                           |`hipblasDtrmv`                                   |
|`cublasCtrmv`                                              |`hipblasCtrmv`                                   |
|`cublasCtrmv_v2`                                           |`hipblasCtrmv`                                   |
|`cublasZtrmv`                                              |`hipblasZtrmv`                                   |
|`cublasZtrmv_v2`                                           |`hipblasZtrmv`                                   |
|`cublasStbmv`                                              |`hipblasStbmv`                                   |
|`cublasStbmv_v2`                                           |`hipblasStbmv`                                   |
|`cublasDtbmv`                                              |`hipblasDtbmv`                                   |
|`cublasDtbmv_v2`                                           |`hipblasDtbmv`                                   |
|`cublasCtbmv`                                              |`hipblasCtbmv`                                   |
|`cublasCtbmv_v2`                                           |`hipblasCtbmv`                                   |
|`cublasZtbmv`                                              |`hipblasZtbmv`                                   |
|`cublasZtbmv_v2`                                           |`hipblasZtbmv`                                   |
|`cublasStpmv`                                              |`hipblasStpmv`                                   |
|`cublasStpmv_v2`                                           |`hipblasStpmv`                                   |
|`cublasDtpmv`                                              |`hipblasDtpmv`                                   |
|`cublasDtpmv_v2`                                           |`hipblasDtpmv`                                   |
|`cublasCtpmv`                                              |`hipblasCtpmv`                                   |
|`cublasCtpmv_v2`                                           |`hipblasCtpmv`                                   |
|`cublasZtpmv`                                              |`hipblasZtpmv`                                   |
|`cublasZtpmv_v2`                                           |`hipblasZtpmv`                                   |
|`cublasStrsv`                                              |`hipblasStrsv`                                   |
|`cublasStrsv_v2`                                           |`hipblasStrsv`                                   |
|`cublasDtrsv`                                              |`hipblasDtrsv`                                   |
|`cublasDtrsv_v2`                                           |`hipblasDtrsv`                                   |
|`cublasCtrsv`                                              |`hipblasCtrsv`                                   |
|`cublasCtrsv_v2`                                           |`hipblasCtrsv`                                   |
|`cublasZtrsv`                                              |`hipblasZtrsv`                                   |
|`cublasZtrsv_v2`                                           |`hipblasZtrsv`                                   |
|`cublasStpsv`                                              |`hipblasStpsv`                                   |
|`cublasStpsv_v2`                                           |`hipblasStpsv`                                   |
|`cublasDtpsv`                                              |`hipblasDtpsv`                                   |
|`cublasDtpsv_v2`                                           |`hipblasDtpsv`                                   |
|`cublasCtpsv`                                              |`hipblasCtpsv`                                   |
|`cublasCtpsv_v2`                                           |`hipblasCtpsv`                                   |
|`cublasZtpsv`                                              |`hipblasZtpsv`                                   |
|`cublasZtpsv_v2`                                           |`hipblasZtpsv`                                   |
|`cublasStbsv`                                              |`hipblasStbsv`                                   |
|`cublasStbsv_v2`                                           |`hipblasStbsv`                                   |
|`cublasDtbsv`                                              |`hipblasDtbsv`                                   |
|`cublasDtbsv_v2`                                           |`hipblasDtbsv`                                   |
|`cublasCtbsv`                                              |`hipblasCtbsv`                                   |
|`cublasCtbsv_v2`                                           |`hipblasCtbsv`                                   |
|`cublasZtbsv`                                              |`hipblasZtbsv`                                   |
|`cublasZtbsv_v2`                                           |`hipblasZtbsv`                                   |
|`cublasSsymv`                                              |`hipblasSsymv`                                   |
|`cublasSsymv_v2`                                           |`hipblasSsymv`                                   |
|`cublasDsymv`                                              |`hipblasDsymv`                                   |
|`cublasDsymv_v2`                                           |`hipblasDsymv`                                   |
|`cublasCsymv`                                              |`hipblasCsymv`                                   |
|`cublasCsymv_v2`                                           |`hipblasCsymv`                                   |
|`cublasZsymv`                                              |`hipblasZsymv`                                   |
|`cublasZsymv_v2`                                           |`hipblasZsymv`                                   |
|`cublasChemv`                                              |`hipblasChemv`                                   |
|`cublasChemv_v2`                                           |`hipblasChemv`                                   |
|`cublasZhemv`                                              |`hipblasZhemv`                                   |
|`cublasZhemv_v2`                                           |`hipblasZhemv`                                   |
|`cublasSsbmv`                                              |`hipblasSsbmv`                                   |
|`cublasSsbmv_v2`                                           |`hipblasSsbmv`                                   |
|`cublasDsbmv`                                              |`hipblasDsbmv`                                   |
|`cublasDsbmv_v2`                                           |`hipblasDsbmv`                                   |
|`cublasChbmv`                                              |`hipblasChbmv`                                   |
|`cublasChbmv_v2`                                           |`hipblasChbmv`                                   |
|`cublasZhbmv`                                              |`hipblasZhbmv`                                   |
|`cublasZhbmv_v2`                                           |`hipblasZhbmv`                                   |
|`cublasSspmv`                                              |`hipblasSspmv`                                   |
|`cublasSspmv_v2`                                           |`hipblasSspmv`                                   |
|`cublasDspmv`                                              |`hipblasDspmv`                                   |
|`cublasDspmv_v2`                                           |`hipblasDspmv`                                   |
|`cublasChpmv`                                              |`hipblasChpmv`                                   |
|`cublasChpmv_v2`                                           |`hipblasChpmv`                                   |
|`cublasZhpmv`                                              |`hipblasZhpmv`                                   |
|`cublasZhpmv_v2`                                           |`hipblasZhpmv`                                   |
|`cublasSger`                                               |`hipblasSger`                                    |
|`cublasSger_v2`                                            |`hipblasSger`                                    |
|`cublasDger`                                               |`hipblasDger`                                    |
|`cublasDger_v2`                                            |`hipblasDger`                                    |
|`cublasCgeru`                                              |`hipblasCgeru`                                   |
|`cublasCgeru_v2`                                           |`hipblasCgeru`                                   |
|`cublasCgerc`                                              |`hipblasCgerc`                                   |
|`cublasCgerc_v2`                                           |`hipblasCgerc`                                   |
|`cublasZgeru`                                              |`hipblasZgeru`                                   |
|`cublasZgeru_v2`                                           |`hipblasZgeru`                                   |
|`cublasZgerc`                                              |`hipblasZgerc`                                   |
|`cublasZgerc_v2`                                           |`hipblasZgerc`                                   |
|`cublasSsyr`                                               |`hipblasSsyr`                                    |
|`cublasSsyr_v2`                                            |`hipblasSsyr`                                    |
|`cublasDsyr`                                               |`hipblasDsyr`                                    |
|`cublasDsyr_v2`                                            |`hipblasDsyr`                                    |
|`cublasCsyr`                                               |`hipblasCsyr`                                    |
|`cublasCsyr_v2`                                            |`hipblasCsyr`                                    |
|`cublasZsyr`                                               |`hipblasZsyr`                                    |
|`cublasZsyr_v2`                                            |`hipblasZsyr`                                    |
|`cublasCher`                                               |`hipblasCher`                                    |
|`cublasCher_v2`                                            |`hipblasCher`                                    |
|`cublasZher`                                               |`hipblasZher`                                    |
|`cublasZher_v2`                                            |`hipblasZher`                                    |
|`cublasSspr`                                               |`hipblasSspr`                                    |
|`cublasSspr_v2`                                            |`hipblasSspr`                                    |
|`cublasDspr`                                               |`hipblasDspr`                                    |
|`cublasDspr_v2`                                            |`hipblasDspr`                                    |
|`cublasChpr`                                               |`hipblasChpr`                                    |
|`cublasChpr_v2`                                            |`hipblasChpr`                                    |
|`cublasZhpr`                                               |`hipblasZhpr`                                    |
|`cublasZhpr_v2`                                            |`hipblasZhpr`                                    |
|`cublasSsyr2`                                              |`hipblasSsyr2`                                   |
|`cublasSsyr2_v2`                                           |`hipblasSsyr2`                                   |
|`cublasDsyr2`                                              |`hipblasDsyr2`                                   |
|`cublasDsyr2_v2`                                           |`hipblasDsyr2`                                   |
|`cublasCsyr2`                                              |`hipblasCsyr2`                                   |
|`cublasCsyr2_v2`                                           |`hipblasCsyr2`                                   |
|`cublasZsyr2`                                              |`hipblasZsyr2`                                   |
|`cublasZsyr2_v2`                                           |`hipblasZsyr2`                                   |
|`cublasCher2`                                              |`hipblasCher2`                                   |
|`cublasCher2_v2`                                           |`hipblasCher2`                                   |
|`cublasZher2`                                              |`hipblasZher2`                                   |
|`cublasZher2_v2`                                           |`hipblasZher2`                                   |
|`cublasSspr2`                                              |`hipblasSspr2`                                   |
|`cublasSspr2_v2`                                           |`hipblasSspr2`                                   |
|`cublasDspr2`                                              |`hipblasDspr2`                                   |
|`cublasDspr2_v2`                                           |`hipblasDspr2`                                   |
|`cublasChpr2`                                              |`hipblasChpr2`                                   |
|`cublasChpr2_v2`                                           |`hipblasChpr2`                                   |
|`cublasZhpr2`                                              |`hipblasZhpr2`                                   |
|`cublasZhpr2_v2`                                           |`hipblasZhpr2`                                   |
|`cublasSgemm`                                              |`hipblasSgemm`                                   |
|`cublasSgemm_v2`                                           |`hipblasSgemm`                                   |
|`cublasDgemm`                                              |`hipblasDgemm`                                   |
|`cublasDgemm_v2`                                           |`hipblasDgemm`                                   |
|`cublasCgemm`                                              |`hipblasCgemm`                                   |
|`cublasCgemm_v2`                                           |`hipblasCgemm`                                   |
|`cublasCgemm3m`                                            |                                                 | 8.0              |
|`cublasCgemm3mEx`                                          |                                                 | 8.0              |
|`cublasZgemm`                                              |`hipblasZgemm`                                   |
|`cublasZgemm_v2`                                           |`hipblasZgemm`                                   |
|`cublasZgemm3m`                                            |                                                 | 8.0              |
|`cublasHgemm`                                              |`hipblasHgemm`                                   | 7.5              |
|`cublasSgemmEx`                                            |                                                 | 7.5              |
|`cublasGemmEx`                                             |`hipblasGemmEx`                                  | 8.0              |
|`cublasCgemmEx`                                            |                                                 | 8.0              |
|`cublasUint8gemmBias`                                      |                                                 | 8.0              |
|`cublasSsyrk`                                              |`hipblasSsyrk`                                   |
|`cublasSsyrk_v2`                                           |`hipblasSsyrk`                                   |
|`cublasDsyrk`                                              |`hipblasDsyrk`                                   |
|`cublasDsyrk_v2`                                           |`hipblasDsyrk`                                   |
|`cublasCsyrk`                                              |`hipblasCsyrk`                                   |
|`cublasCsyrk_v2`                                           |`hipblasCsyrk`                                   |
|`cublasZsyrk`                                              |`hipblasZsyrk`                                   |
|`cublasZsyrk_v2`                                           |`hipblasZsyrk`                                   |
|`cublasCsyrkEx`                                            |                                                 | 8.0              |
|`cublasCsyrk3mEx`                                          |                                                 | 8.0              |
|`cublasCherk`                                              |`hipblasCherk`                                   |
|`cublasCherk_v2`                                           |`hipblasCherk`                                   |
|`cublasZherk`                                              |`hipblasZherk`                                   |
|`cublasZherk_v2`                                           |`hipblasZherk`                                   |
|`cublasCherkEx`                                            |                                                 | 8.0              |
|`cublasCherk3mEx`                                          |                                                 | 8.0              |
|`cublasSsyr2k`                                             |`hipblasSsyr2k`                                  |
|`cublasSsyr2k_v2`                                          |`hipblasSsyr2k`                                  |
|`cublasDsyr2k`                                             |`hipblasDsyr2k`                                  |
|`cublasDsyr2k_v2`                                          |`hipblasDsyr2k`                                  |
|`cublasCsyr2k`                                             |`hipblasCsyr2k`                                  |
|`cublasCsyr2k_v2`                                          |`hipblasCsyr2k`                                  |
|`cublasZsyr2k`                                             |`hipblasZsyr2k`                                  |
|`cublasZsyr2k_v2`                                          |`hipblasZsyr2k`                                  |
|`cublasCher2k`                                             |`hipblasCher2k`                                  |
|`cublasCher2k_v2`                                          |`hipblasCher2k`                                  |
|`cublasZher2k`                                             |`hipblasZher2k`                                  |
|`cublasZher2k_v2`                                          |`hipblasZher2k`                                  |
|`cublasSsyrkx`                                             |`hipblasSsyrkx`                                  |
|`cublasDsyrkx`                                             |`hipblasDsyrkx`                                  |
|`cublasCsyrkx`                                             |`hipblasCsyrkx`                                  |
|`cublasZsyrkx`                                             |`hipblasZsyrkx`                                  |
|`cublasCherkx`                                             |`hipblasCherkx`                                  |
|`cublasZherkx`                                             |`hipblasZherkx`                                  |
|`cublasSsymm`                                              |`hipblasSsymm`                                   |
|`cublasSsymm_v2`                                           |`hipblasSsymm`                                   |
|`cublasDsymm`                                              |`hipblasDsymm`                                   |
|`cublasDsymm_v2`                                           |`hipblasDsymm`                                   |
|`cublasCsymm`                                              |`hipblasCsymm`                                   |
|`cublasCsymm_v2`                                           |`hipblasCsymm`                                   |
|`cublasZsymm`                                              |`hipblasZsymm`                                   |
|`cublasZsymm_v2`                                           |`hipblasZsymm`                                   |
|`cublasChemm`                                              |`hipblasChemm`                                   |
|`cublasChemm_v2`                                           |`hipblasChemm`                                   |
|`cublasZhemm`                                              |`hipblasZhemm`                                   |
|`cublasZhemm_v2`                                           |`hipblasZhemm`                                   |
|`cublasStrsm`                                              |`hipblasStrsm`                                   |
|`cublasStrsm_v2`                                           |`hipblasStrsm`                                   |
|`cublasDtrsm`                                              |`hipblasDtrsm`                                   |
|`cublasDtrsm_v2`                                           |`hipblasDtrsm`                                   |
|`cublasCtrsm`                                              |`hipblasCtrsm`                                   |
|`cublasCtrsm_v2`                                           |`hipblasCtrsm`                                   |
|`cublasZtrsm`                                              |`hipblasZtrsm`                                   |
|`cublasZtrsm_v2`                                           |`hipblasZtrsm`                                   |
|`cublasStrmm`                                              |`hipblasStrmm`                                   |
|`cublasStrmm_v2`                                           |`hipblasStrmm`                                   |
|`cublasDtrmm`                                              |`hipblasDtrmm`                                   |
|`cublasDtrmm_v2`                                           |`hipblasDtrmm`                                   |
|`cublasCtrmm`                                              |`hipblasCtrmm`                                   |
|`cublasCtrmm_v2`                                           |`hipblasCtrmm`                                   |
|`cublasZtrmm`                                              |`hipblasZtrmm`                                   |
|`cublasZtrmm_v2`                                           |`hipblasZtrmm`                                   |
|`cublasHgemmBatched`                                       |`hipblasHgemmBatched`                            | 9.0              |
|`cublasSgemmBatched`                                       |`hipblasSgemmBatched`                            |
|`cublasDgemmBatched`                                       |`hipblasDgemmBatched`                            |
|`cublasCgemmBatched`                                       |`hipblasCgemmBatched`                            |
|`cublasCgemm3mBatched`                                     |                                                 | 8.0              |
|`cublasZgemmBatched`                                       |`hipblasZgemmBatched`                            |
|`cublasGemmBatchedEx`                                      |`hipblasGemmBatchedEx`                           | 9.1              |
|`cublasGemmStridedBatchedEx`                               |`hipblasGemmStridedBatchedEx`                    | 9.1              |
|`cublasSgemmStridedBatched`                                |`hipblasSgemmStridedBatched`                     | 8.0              |
|`cublasDgemmStridedBatched`                                |`hipblasDgemmStridedBatched`                     | 8.0              |
|`cublasCgemmStridedBatched`                                |`hipblasCgemmStridedBatched`                     | 8.0              |
|`cublasCgemm3mStridedBatched`                              |                                                 | 8.0              |
|`cublasZgemmStridedBatched`                                |`hipblasZgemmStridedBatched`                     | 8.0              |
|`cublasHgemmStridedBatched`                                |`hipblasHgemmStridedBatched`                     | 8.0              |
|`cublasSgeam`                                              |`hipblasSgeam`                                   |
|`cublasDgeam`                                              |`hipblasDgeam`                                   |
|`cublasCgeam`                                              |`hipblasCgeam`                                   |
|`cublasZgeam`                                              |`hipblasZgeam`                                   |
|`cublasSgetrfBatched`                                      |`hipblasSgetrfBatched`                           |
|`cublasDgetrfBatched`                                      |`hipblasDgetrfBatched`                           |
|`cublasCgetrfBatched`                                      |`hipblasCgetrfBatched`                           |
|`cublasZgetrfBatched`                                      |`hipblasZgetrfBatched`                           |
|`cublasSgetriBatched`                                      |`hipblasSgetriBatched`                           |
|`cublasDgetriBatched`                                      |`hipblasDgetriBatched`                           |
|`cublasCgetriBatched`                                      |`hipblasCgetriBatched`                           |
|`cublasZgetriBatched`                                      |`hipblasZgetriBatched`                           |
|`cublasSgetrsBatched`                                      |`hipblasSgetrsBatched`                           |
|`cublasDgetrsBatched`                                      |`hipblasDgetrsBatched`                           |
|`cublasCgetrsBatched`                                      |`hipblasCgetrsBatched`                           |
|`cublasZgetrsBatched`                                      |`hipblasZgetrsBatched`                           |
|`cublasStrsmBatched`                                       |`hipblasStrsmBatched`                            |
|`cublasDtrsmBatched`                                       |`hipblasDtrsmBatched`                            |
|`cublasCtrsmBatched`                                       |`hipblasCtrsmBatched`                            |
|`cublasZtrsmBatched`                                       |`hipblasZtrsmBatched`                            |
|`cublasSmatinvBatched`                                     |                                                 |
|`cublasDmatinvBatched`                                     |                                                 |
|`cublasCmatinvBatched`                                     |                                                 |
|`cublasZmatinvBatched`                                     |                                                 |
|`cublasSgeqrfBatched`                                      |`hipblasSgeqrfBatched`                           |
|`cublasDgeqrfBatched`                                      |`hipblasDgeqrfBatched`                           |
|`cublasCgeqrfBatched`                                      |`hipblasCgeqrfBatched`                           |
|`cublasZgeqrfBatched`                                      |`hipblasZgeqrfBatched`                           |
|`cublasSgelsBatched`                                       |                                                 |
|`cublasDgelsBatched`                                       |                                                 |
|`cublasCgelsBatched`                                       |                                                 |
|`cublasZgelsBatched`                                       |                                                 |
|`cublasSdgmm`                                              |`hipblasSdgmm`                                   |
|`cublasDdgmm`                                              |`hipblasDdgmm`                                   |
|`cublasCdgmm`                                              |`hipblasCdgmm`                                   |
|`cublasZdgmm`                                              |`hipblasZdgmm`                                   |
|`cublasStpttr`                                             |                                                 |
|`cublasDtpttr`                                             |                                                 |
|`cublasCtpttr`                                             |                                                 |
|`cublasZtpttr`                                             |                                                 |
|`cublasStrttp`                                             |                                                 |
|`cublasDtrttp`                                             |                                                 |
|`cublasCtrttp`                                             |                                                 |
|`cublasZtrttp`                                             |                                                 |

\* CUDA version, in which API has appeared and (optional) last version before abandoning it; no value in case of earlier versions < 7.5.
