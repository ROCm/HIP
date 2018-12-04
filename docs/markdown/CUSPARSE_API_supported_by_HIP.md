# CUSPARSE API supported by HIP

## **1. cuSPARSE Data types**

| **type**     |   **CUDA**                                                    |   **HIP**                                                  |
|-------------:|---------------------------------------------------------------|------------------------------------------------------------|
| enum         |***`cusparseAction_t`***                                       |***`hipsparseAction_t`***                                   |
|            0 |*`CUSPARSE_ACTION_SYMBOLIC`*                                   |*`HIPSPARSE_ACTION_SYMBOLIC`*                               |
|            1 |*`CUSPARSE_ACTION_NUMERIC`*                                    |*`HIPSPARSE_ACTION_NUMERIC`*                                |
| enum         |***`cusparseDirection_t`***                                    |                                                            |
|            0 |*`CUSPARSE_DIRECTION_ROW`*                                     |                                                            |
|            1 |*`CUSPARSE_DIRECTION_COLUMN`*                                  |                                                            |
| enum         |***`cusparseHybPartition_t`***                                 |***`hipsparseHybPartition_t`***                             |
|            0 |*`CUSPARSE_HYB_PARTITION_AUTO`*                                |*`HIPSPARSE_HYB_PARTITION_AUTO`*                            |
|            1 |*`CUSPARSE_HYB_PARTITION_USER`*                                |*`HIPSPARSE_HYB_PARTITION_USER`*                            |
|            2 |*`CUSPARSE_HYB_PARTITION_MAX`*                                 |*`HIPSPARSE_HYB_PARTITION_MAX`*                             |
| enum         |***`cusparseDiagType_t`***                                     |***`hipsparseDiagType_t`***                                 |
|            0 |*`CUSPARSE_DIAG_TYPE_NON_UNIT`*                                |*`HIPSPARSE_DIAG_TYPE_NON_UNIT`*                            |
|            1 |*`CUSPARSE_DIAG_TYPE_UNIT`*                                    |*`HIPSPARSE_DIAG_TYPE_UNIT`*                                |
| enum         |***`cusparseFillMode_t`***                                     |***`hipsparseFillMode_t`***                                 |
|            0 |*`CUSPARSE_FILL_MODE_LOWER`*                                   |*`HIPSPARSE_FILL_MODE_LOWER`*                               |
|            1 |*`CUSPARSE_FILL_MODE_UPPER`*                                   |*`HIPSPARSE_FILL_MODE_UPPER`*                               |
| enum         |***`cusparseIndexBase_t`***                                    |***`hipsparseIndexBase_t`***                                |
|            0 |*`CUSPARSE_INDEX_BASE_ZERO`*                                   |*`HIPSPARSE_INDEX_BASE_ZERO`*                               |
|            1 |*`CUSPARSE_INDEX_BASE_ONE`*                                    |*`HIPSPARSE_INDEX_BASE_ONE`*                                |
| enum         |***`cusparseMatrixType_t`***                                   |***`hipsparseMatrixType_t`***                               |
|            0 |*`CUSPARSE_MATRIX_TYPE_GENERAL`*                               |*`HIPSPARSE_MATRIX_TYPE_GENERAL`*                           |
|            1 |*`CUSPARSE_MATRIX_TYPE_SYMMETRIC`*                             |*`HIPSPARSE_MATRIX_TYPE_SYMMETRIC`*                         |
|            2 |*`CUSPARSE_MATRIX_TYPE_HERMITIAN`*                             |*`HIPSPARSE_MATRIX_TYPE_HERMITIAN`*                         |
|            3 |*`CUSPARSE_MATRIX_TYPE_TRIANGULAR`*                            |*`HIPSPARSE_MATRIX_TYPE_TRIANGULAR`*                        |
| enum         |***`cusparseOperation_t`***                                    |***`hipsparseOperation_t`***                                |
|            0 |*`CUSPARSE_OPERATION_NON_TRANSPOSE`*                           |*`HIPSPARSE_OPERATION_NON_TRANSPOSE`*                       |
|            1 |*`CUSPARSE_OPERATION_TRANSPOSE`*                               |*`HIPSPARSE_OPERATION_TRANSPOSE`*                           |
|            2 |*`CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE`*                     |*`HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE`*                 |
| enum         |***`cusparsePointerMode_t`***                                  |***`hipsparsePointerMode_t`***                              |
|            0 |*`CUSPARSE_POINTER_MODE_HOST`*                                 |*`HIPSPARSE_POINTER_MODE_HOST`*                             |
|            1 |*`CUSPARSE_POINTER_MODE_DEVICE`*                               |*`HIPSPARSE_POINTER_MODE_DEVICE`*                           |
| enum         |***`cusparseAlgMode_t`***                                      |                                                            |
|            0 |*`CUSPARSE_ALG0`*                                              |                                                            |
|            1 |*`CUSPARSE_ALG1`*                                              |                                                            |
|            0 |*`CUSPARSE_ALG_NAIVE`*                                         |                                                            |
|            1 |*`CUSPARSE_ALG_MERGE_PATH`*                                    |                                                            |
| enum         |***`cusparseSolvePolicy_t`***                                  |***`hipsparseSolvePolicy_t`***                              |
|            0 |*`CUSPARSE_SOLVE_POLICY_NO_LEVEL`*                             |*`HIPSPARSE_SOLVE_POLICY_NO_LEVEL`*                         |
|            1 |*`CUSPARSE_SOLVE_POLICY_USE_LEVEL`*                            |*`HIPSPARSE_SOLVE_POLICY_USE_LEVEL`*                        |
| enum         |***`cusparseStatus_t`***                                       |***`hipsparseMatrixType_t`***                               |
|            0 |*`CUSPARSE_STATUS_SUCCESS`*                                    |*`HIPSPARSE_STATUS_SUCCESS`*                                |
|            1 |*`CUSPARSE_STATUS_NOT_INITIALIZED`*                            |*`HIPSPARSE_STATUS_NOT_INITIALIZED`*                        |
|            2 |*`CUSPARSE_STATUS_ALLOC_FAILED`*                               |*`HIPSPARSE_STATUS_ALLOC_FAILED`*                           |
|            3 |*`CUSPARSE_STATUS_INVALID_VALUE`*                              |*`HIPSPARSE_STATUS_INVALID_VALUE`*                          |
|            4 |*`CUSPARSE_STATUS_ARCH_MISMATCH`*                              |*`HIPSPARSE_STATUS_ARCH_MISMATCH`*                          |
|            5 |*`CUSPARSE_STATUS_MAPPING_ERROR`*                              |*`HIPSPARSE_STATUS_MAPPING_ERROR`*                          |
|            6 |*`CUSPARSE_STATUS_EXECUTION_FAILED`*                           |*`HIPSPARSE_STATUS_EXECUTION_FAILED`*                       |
|            7 |*`CUSPARSE_STATUS_INTERNAL_ERROR`*                             |*`HIPSPARSE_STATUS_INTERNAL_ERROR`*                         |
|            8 |*`CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`*                  |*`HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`*              |
|            9 |*`CUSPARSE_STATUS_ZERO_PIVOT`*                                 |*`HIPSPARSE_STATUS_ZERO_PIVOT`*                             |
| struct       |`cusparseContext`                                              |                                                            |
| typedef      |`cusparseHandle_t`                                             |`hipsparseHandle_t`                                         |
| struct       |`cusparseHybMat`                                               |                                                            |
| typedef      |`cusparseHybMat_t`                                             |`hipsparseHybMat_t`                                         |
| struct       |`cusparseMatDescr`                                             |                                                            |
| typedef      |`cusparseMatDescr_t`                                           |`hipsparseMatDescr_t`                                       |
| struct       |`cusparseSolveAnalysisInfo`                                    |                                                            |
| typedef      |`cusparseSolveAnalysisInfo_t`                                  |                                                            |
| struct       |`csrsv2Info`                                                   |                                                            |
| typedef      |`csrsv2Info_t`                                                 |`csrsv2Info_t`                                              |
| struct       |`csrsm2Info`                                                   |                                                            |
| typedef      |`csrsm2Info_t`                                                 |                                                            |
| struct       |`bsrsv2Info`                                                   |                                                            |
| typedef      |`bsrsv2Info_t`                                                 |                                                            |
| struct       |`bsrsm2Info`                                                   |                                                            |
| typedef      |`bsrsm2Info_t`                                                 |                                                            |
| struct       |`bsric02Info`                                                  |                                                            |
| typedef      |`bsric02Info_t`                                                |                                                            |
| struct       |`csrilu02Info`                                                 |                                                            |
| typedef      |`csrilu02Info_t`                                               |`csrilu02Info_t`                                            |
| struct       |`bsrilu02Info`                                                 |                                                            |
| typedef      |`bsrilu02Info_t`                                               |                                                            |
| struct       |`csru2csrInfo`                                                 |                                                            |
| typedef      |`csru2csrInfo_t`                                               |                                                            |
| struct       |`cusparseColorInfo`                                            |                                                            |
| typedef      |`cusparseColorInfo_t`                                          |                                                            |
| struct       |`pruneInfo`                                                    |                                                            |
| typedef      |`pruneInfo_t`                                                  |                                                            |

## **2. cuSPARSE Helper Function Reference**

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseCreate`                                           |`hipsparseCreate`                                |
|`cusparseCreateSolveAnalysisInfo`                          |                                                 |
|`cusparseCreateHybMat`                                     |`hipsparseCreateHybMat`                          |
|`cusparseCreateMatDescr`                                   |`hipsparseCreateMatDescr`                        |
|`cusparseDestroy`                                          |`hipsparseDestroy`                               |
|`cusparseDestroySolveAnalysisInfo`                         |                                                 |
|`cusparseDestroyHybMat`                                    |`hipsparseDestroyHybMat`                         |
|`cusparseDestroyMatDescr`                                  |`hipsparseDestroyMatDescr`                       |
|`cusparseGetLevelInfo`                                     |                                                 |
|`cusparseGetMatDiagType`                                   |`hipsparseGetMatDiagType`                        |
|`cusparseGetMatFillMode`                                   |`hipsparseGetMatFillMode`                        |
|`cusparseGetMatIndexBase`                                  |`hipsparseGetMatIndexBase`                       |
|`cusparseGetMatType`                                       |`hipsparseGetMatType`                            |
|`cusparseGetPointerMode`                                   |`hipsparseGetPointerMode`                        |
|`cusparseGetVersion`                                       |`hipsparseGetVersion`                            |
|`cusparseSetMatDiagType`                                   |`hipsparseSetMatDiagType`                        |
|`cusparseSetMatFillMode`                                   |`hipsparseSetMatFillMode`                        |
|`cusparseSetMatType`                                       |`hipsparseSetMatType`                            |
|`cusparseSetPointerMode`                                   |`hipsparseSetPointerMode`                        |
|`cusparseSetStream`                                        |`hipsparseSetStream`                             |
|`cusparseGetStream`                                        |`hipsparseGetStream`                             |
|`cusparseCreateCsrsv2Info`                                 |`hipsparseCreateCsrsv2Info`                      |
|`cusparseDestroyCsrsv2Info`                                |`hipsparseDestroyCsrsv2Info`                     |
|`cusparseCreateCsrsm2Info`                                 |                                                 |
|`cusparseDestroyCsrsm2Info`                                |                                                 |
|`cusparseCreateCsric02Info`                                |                                                 |
|`cusparseDestroyCsric02Info`                               |                                                 |
|`cusparseCreateCsrilu02Info`                               |`hipsparseCreateCsrilu02Info`                    |
|`cusparseDestroyCsrilu02Info`                              |`hipsparseDestroyCsrilu02Info`                   |
|`cusparseCreateBsrsv2Info`                                 |                                                 |
|`cusparseDestroyBsrsv2Info`                                |                                                 |
|`cusparseCreateBsrsm2Info`                                 |                                                 |
|`cusparseDestroyBsrsm2Info`                                |                                                 |
|`cusparseCreateBsric02Info`                                |                                                 |
|`cusparseDestroyBsric02Info`                               |                                                 |
|`cusparseCreateBsrilu02Info`                               |                                                 |
|`cusparseDestroyBsrilu02Info`                              |                                                 |
|`cusparseCreateCsrgemm2Info`                               |                                                 |
|`cusparseDestroyCsrgemm2Info`                              |                                                 |
|`cusparseCreatePruneInfo`                                  |                                                 |
|`cusparseDestroyPruneInfo`                                 |                                                 |

## **3. cuSPARSE Level 1 Function Reference**

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseSaxpyi`                                           |`hipsparseSaxpyi`                                |
|`cusparseDaxpyi`                                           |`hipsparseDaxpyi`                                |
|`cusparseCaxpyi`                                           |                                                 |
|`cusparseZaxpyi`                                           |                                                 |
|`cusparseSdoti`                                            |`hipsparseSdoti`                                 |
|`cusparseDdoti`                                            |`hipsparseDdoti`                                 |
|`cusparseCdoti`                                            |                                                 |
|`cusparseZdoti`                                            |                                                 |
|`cusparseCdotci`                                           |                                                 |
|`cusparseZdotci`                                           |                                                 |
|`cusparseSgthr`                                            |`hipsparseSgthr`                                 |
|`cusparseDgthr`                                            |`hipsparseDgthr`                                 |
|`cusparseCgthr`                                            |                                                 |
|`cusparseZgthr`                                            |                                                 |
|`cusparseSgthrz`                                           |`hipsparseSgthrz`                                |
|`cusparseDgthrz`                                           |`hipsparseDgthrz`                                |
|`cusparseCgthrz`                                           |                                                 |
|`cusparseZgthrz`                                           |                                                 |
|`cusparseSroti`                                            |`hipsparseSroti`                                 |
|`cusparseDroti`                                            |`hipsparseDroti`                                 |
|`cusparseSsctr`                                            |`hipsparseSsctr`                                 |
|`cusparseDsctr`                                            |`hipsparseDsctr`                                 |
|`cusparseCsctr`                                            |                                                 |
|`cusparseZsctr`                                            |                                                 |

## **4. cuSPARSE Level 2 Function Reference**

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseSbsrmv`                                           |                                                 |
|`cusparseDbsrmv`                                           |                                                 |
|`cusparseCbsrmv`                                           |                                                 |
|`cusparseZbsrmv`                                           |                                                 |
|`cusparseSbsrxmv`                                          |                                                 |
|`cusparseDbsrxmv`                                          |                                                 |
|`cusparseCbsrxmv`                                          |                                                 |
|`cusparseZbsrxmv`                                          |                                                 |
|`cusparseScsrmv`                                           |`hipsparseScsrmv`                                |
|`cusparseDcsrmv`                                           |`hipsparseDcsrmv`                                |
|`cusparseCcsrmv`                                           |                                                 |
|`cusparseZcsrmv`                                           |                                                 |
|`cusparseCsrmvEx`                                          |                                                 |
|`cusparseCsrmvEx_bufferSize`                               |                                                 |
|`cusparseScsrmv_mp`                                        |                                                 |
|`cusparseDcsrmv_mp`                                        |                                                 |
|`cusparseCcsrmv_mp`                                        |                                                 |
|`cusparseZcsrmv_mp`                                        |                                                 |
|`cusparseSgemvi`                                           |                                                 |
|`cusparseDgemvi`                                           |                                                 |
|`cusparseCgemvi`                                           |                                                 |
|`cusparseZgemvi`                                           |                                                 |
|`cusparseSgemvi_bufferSize`                                |                                                 |
|`cusparseDgemvi_bufferSize`                                |                                                 |
|`cusparseCgemvi_bufferSize`                                |                                                 |
|`cusparseZgemvi_bufferSize`                                |                                                 |
|`cusparseSbsrsv2_bufferSize`                               |                                                 |
|`cusparseDbsrsv2_bufferSize`                               |                                                 |
|`cusparseCbsrsv2_bufferSize`                               |                                                 |
|`cusparseZbsrsv2_bufferSize`                               |                                                 |
|`cusparseSbsrsv2_analysis`                                 |                                                 |
|`cusparseDbsrsv2_analysis`                                 |                                                 |
|`cusparseCbsrsv2_analysis`                                 |                                                 |
|`cusparseZbsrsv2_analysis`                                 |                                                 |
|`cusparseScsrsv_solve`                                     |                                                 |
|`cusparseDcsrsv_solve`                                     |                                                 |
|`cusparseCcsrsv_solve`                                     |                                                 |
|`cusparseZcsrsv_solve`                                     |                                                 |
|`cusparseXbsrsv2_zeroPivot`                                |                                                 |
|`cusparseScsrsv_analysis`                                  |                                                 |
|`cusparseDcsrsv_analysis`                                  |                                                 |
|`cusparseCcsrsv_analysis`                                  |                                                 |
|`cusparseZcsrsv_analysis`                                  |                                                 |
|`cusparseCsrsv_analysisEx`                                 |                                                 |
|`cusparseScsrsv_solve`                                     |                                                 |
|`cusparseDcsrsv_solve`                                     |                                                 |
|`cusparseCcsrsv_solve`                                     |                                                 |
|`cusparseZcsrsv_solve`                                     |                                                 |
|`cusparseCsrsv_solveEx`                                    |                                                 |
|`cusparseScsrsv2_bufferSize`                               |`hipsparseScsrsv2_bufferSize`                    |
|`cusparseDcsrsv2_bufferSize`                               |`hipsparseDcsrsv2_bufferSize`                    |
|`cusparseCcsrsv2_bufferSize`                               |                                                 |
|`cusparseZcsrsv2_bufferSize`                               |                                                 |
|`cusparseScsrsv2_analysis`                                 |`hipsparseScsrsv2_analysis`                      |
|`cusparseDcsrsv2_analysis`                                 |`hipsparseDcsrsv2_analysis`                      |
|`cusparseCcsrsv2_analysis`                                 |                                                 |
|`cusparseZcsrsv2_analysis`                                 |                                                 |
|`cusparseScsrsv2_solve`                                    |`hipsparseScsrsv2_solve`                         |
|`cusparseDcsrsv2_solve`                                    |`hipsparseDcsrsv2_solve`                         |
|`cusparseCcsrsv2_solve`                                    |                                                 |
|`cusparseZcsrsv2_solve`                                    |                                                 |
|`cusparseXcsrsv2_zeroPivot`                                |`hipsparseXcsrsv2_zeroPivot`                     |
|`cusparseShybmv`                                           |`hipsparseShybmv`                                |
|`cusparseDhybmv`                                           |`hipsparseDhybmv`                                |
|`cusparseChybmv`                                           |                                                 |
|`cusparseZhybmv`                                           |                                                 |
|`cusparseShybsv_analysis`                                  |                                                 |
|`cusparseDhybsv_analysis`                                  |                                                 |
|`cusparseChybsv_analysis`                                  |                                                 |
|`cusparseZhybsv_analysis`                                  |                                                 |
|`cusparseShybsv_solve`                                     |                                                 |
|`cusparseDhybsv_solve`                                     |                                                 |
|`cusparseChybsv_solve`                                     |                                                 |
|`cusparseZhybsv_solve`                                     |                                                 |

## **5. cuSPARSE Level 3 Function Reference**

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseScsrmm`                                           |`hipsparseScsrmm`                                |
|`cusparseDcsrmm`                                           |`hipsparseDcsrmm`                                |
|`cusparseCcsrmm`                                           |                                                 |
|`cusparseZcsrmm`                                           |                                                 |
|`cusparseScsrmm2`                                          |`hipsparseScsrmm2`                               |
|`cusparseDcsrmm2`                                          |`hipsparseDcsrmm2`                               |
|`cusparseCcsrmm2`                                          |                                                 |
|`cusparseZcsrmm2`                                          |                                                 |
|`cusparseScsrsm_analysis`                                  |                                                 |
|`cusparseDcsrsm_analysis`                                  |                                                 |
|`cusparseCcsrsm_analysis`                                  |                                                 |
|`cusparseZcsrsm_analysis`                                  |                                                 |
|`cusparseScsrsm_solve`                                     |                                                 |
|`cusparseDcsrsm_solve`                                     |                                                 |
|`cusparseCcsrsm_solve`                                     |                                                 |
|`cusparseZcsrsm_solve`                                     |                                                 |
|`cusparseScsrsm2_bufferSizeExt`                            |                                                 |
|`cusparseDcsrsm2_bufferSizeExt`                            |                                                 |
|`cusparseCcsrsm2_bufferSizeExt`                            |                                                 |
|`cusparseZcsrsm2_bufferSizeExt`                            |                                                 |
|`cusparseScsrsm2_analysis`                                 |                                                 |
|`cusparseDcsrsm2_analysis`                                 |                                                 |
|`cusparseCcsrsm2_analysis`                                 |                                                 |
|`cusparseZcsrsm2_analysis`                                 |                                                 |
|`cusparseScsrsm2_solve`                                    |                                                 |
|`cusparseDcsrsm2_solve`                                    |                                                 |
|`cusparseCcsrsm2_solve`                                    |                                                 |
|`cusparseZcsrsm2_solve`                                    |                                                 |
|`cusparseXcsrsm2_zeroPivot`                                |                                                 |
|`cusparseSbsrmm`                                           |                                                 |
|`cusparseDbsrmm`                                           |                                                 |
|`cusparseCbsrmm`                                           |                                                 |
|`cusparseZbsrmm`                                           |                                                 |
|`cusparseSbsrsm2_bufferSize`                               |                                                 |
|`cusparseDbsrsm2_bufferSize`                               |                                                 |
|`cusparseCbsrsm2_bufferSize`                               |                                                 |
|`cusparseZbsrsm2_bufferSize`                               |                                                 |
|`cusparseSbsrsm2_analysis`                                 |                                                 |
|`cusparseDbsrsm2_analysis`                                 |                                                 |
|`cusparseCbsrsm2_analysis`                                 |                                                 |
|`cusparseZbsrsm2_analysis`                                 |                                                 |
|`cusparseSbsrsm2_solve`                                    |                                                 |
|`cusparseDbsrsm2_solve`                                    |                                                 |
|`cusparseCbsrsm2_solve`                                    |                                                 |
|`cusparseZbsrsm2_solve`                                    |                                                 |
|`cusparseXbsrsm2_zeroPivot`                                |                                                 |
|`cusparseSgemmi`                                           |                                                 |
|`cusparseDgemmi`                                           |                                                 |
|`cusparseCgemmi`                                           |                                                 |
|`cusparseZgemmi`                                           |                                                 |

## **6. cuSPARSE Extra Function Reference**

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseXcsrgeamNnz`                                      |                                                 |
|`cusparseScsrgeam`                                         |                                                 |
|`cusparseDcsrgeam`                                         |                                                 |
|`cusparseCcsrgeam`                                         |                                                 |
|`cusparseZcsrgeam`                                         |                                                 |
|`cusparseScsrgeam2_bufferSizeExt`                          |                                                 |
|`cusparseDcsrgeam2_bufferSizeExt`                          |                                                 |
|`cusparseCcsrgeam2_bufferSizeExt`                          |                                                 |
|`cusparseZcsrgeam2_bufferSizeExt`                          |                                                 |
|`cusparseXcsrgemmNnz`                                      |                                                 |
|`cusparseScsrgemm`                                         |                                                 |
|`cusparseDcsrgemm`                                         |                                                 |
|`cusparseCcsrgemm`                                         |                                                 |
|`cusparseZcsrgemm`                                         |                                                 |
|`cusparseScsrgemm2_bufferSizeExt`                          |                                                 |
|`cusparseDcsrgemm2_bufferSizeExt`                          |                                                 |
|`cusparseCcsrgemm2_bufferSizeExt`                          |                                                 |
|`cusparseZcsrgemm2_bufferSizeExt`                          |                                                 |

## **7. cuSPARSE Preconditioners Reference**

## ***7.1. Incomplete Cholesky Factorization: level 0***

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseScsric0`                                          |                                                 |
|`cusparseDcsric0`                                          |                                                 |
|`cusparseCcsric0`                                          |                                                 |
|`cusparseZcsric0`                                          |                                                 |
|`cusparseScsric02_bufferSize`                              |                                                 |
|`cusparseDcsric02_bufferSize`                              |                                                 |
|`cusparseCcsric02_bufferSize`                              |                                                 |
|`cusparseZcsric02_bufferSize`                              |                                                 |
|`cusparseScsric02_analysis`                                |                                                 |
|`cusparseDcsric02_analysis`                                |                                                 |
|`cusparseCcsric02_analysis`                                |                                                 |
|`cusparseZcsric02_analysis`                                |                                                 |
|`cusparseScsric02`                                         |                                                 |
|`cusparseDcsric02`                                         |                                                 |
|`cusparseCcsric02`                                         |                                                 |
|`cusparseZcsric02`                                         |                                                 |
|`cusparseXcsric02_zeroPivot`                               |                                                 |
|`cusparseSbsric02_bufferSize`                              |                                                 |
|`cusparseDbsric02_bufferSize`                              |                                                 |
|`cusparseCbsric02_bufferSize`                              |                                                 |
|`cusparseZbsric02_bufferSize`                              |                                                 |
|`cusparseSbsric02_analysis`                                |                                                 |
|`cusparseDbsric02_analysis`                                |                                                 |
|`cusparseCbsric02_analysis`                                |                                                 |
|`cusparseZbsric02_analysis`                                |                                                 |
|`cusparseSbsric02`                                         |                                                 |
|`cusparseDbsric02`                                         |                                                 |
|`cusparseCbsric02`                                         |                                                 |
|`cusparseZbsric02`                                         |                                                 |
|`cusparseXbsric02_zeroPivot`                               |                                                 |

## ***7.2. Incomplete LU Factorization: level 0***

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseScsrilu0`                                         |                                                 |
|`cusparseDcsrilu0`                                         |                                                 |
|`cusparseCcsrilu0`                                         |                                                 |
|`cusparseZcsrilu0`                                         |                                                 |
|`cusparseCsrilu0Ex`                                        |                                                 |
|`cusparseScsrilu02_numericBoost`                           |                                                 |
|`cusparseDcsrilu02_numericBoost`                           |                                                 |
|`cusparseCcsrilu02_numericBoost`                           |                                                 |
|`cusparseZcsrilu02_numericBoost`                           |                                                 |
|`cusparseScsrilu02_bufferSize`                             |`hipsparseScsrilu02_bufferSize`                  |
|`cusparseDcsrilu02_bufferSize`                             |`hipsparseDcsrilu02_bufferSize`                  |
|`cusparseCcsrilu02_bufferSize`                             |                                                 |
|`cusparseZcsrilu02_bufferSize`                             |                                                 |
|`cusparseScsrilu02_analysis`                               |`hipsparseScsrilu02_analysis`                    |
|`cusparseDcsrilu02_analysis`                               |`hipsparseDcsrilu02_analysis`                    |
|`cusparseCcsrilu02_analysis`                               |                                                 |
|`cusparseZcsrilu02_analysis`                               |                                                 |
|`cusparseScsrilu02`                                        |`hipsparseScsrilu02`                             |
|`cusparseDcsrilu02`                                        |`hipsparseDcsrilu02`                             |
|`cusparseCcsrilu02`                                        |                                                 |
|`cusparseZcsrilu02`                                        |                                                 |
|`cusparseXbsric02_zeroPivot`                               |`hipsparseXcsrilu02_zeroPivot`                   |
|`cusparseSbsrilu02_numericBoost`                           |                                                 |
|`cusparseDbsrilu02_numericBoost`                           |                                                 |
|`cusparseCbsrilu02_numericBoost`                           |                                                 |
|`cusparseZbsrilu02_numericBoost`                           |                                                 |
|`cusparseSbsrilu02_bufferSize`                             |                                                 |
|`cusparseDbsrilu02_bufferSize`                             |                                                 |
|`cusparseCbsrilu02_bufferSize`                             |                                                 |
|`cusparseZbsrilu02_bufferSize`                             |                                                 |
|`cusparseSbsrilu02_analysis`                               |                                                 |
|`cusparseDbsrilu02_analysis`                               |                                                 |
|`cusparseCbsrilu02_analysis`                               |                                                 |
|`cusparseZbsrilu02_analysis`                               |                                                 |
|`cusparseSbsrilu02`                                        |                                                 |
|`cusparseDbsrilu02`                                        |                                                 |
|`cusparseCbsrilu02`                                        |                                                 |
|`cusparseZbsrilu02`                                        |                                                 |
|`cusparseXbsrilu02_zeroPivot`                              |                                                 |

## ***7.3. Tridiagonal Solve***

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseSgtsv`                                            |                                                 |
|`cusparseDgtsv`                                            |                                                 |
|`cusparseCgtsv`                                            |                                                 |
|`cusparseZgtsv`                                            |                                                 |
|`cusparseSgtsv_nopivot`                                    |                                                 |
|`cusparseDgtsv_nopivot`                                    |                                                 |
|`cusparseCgtsv_nopivot`                                    |                                                 |
|`cusparseZgtsv_nopivot`                                    |                                                 |
|`cusparseSgtsv2_bufferSizeExt`                             |                                                 |
|`cusparseDgtsv2_bufferSizeExt`                             |                                                 |
|`cusparseCgtsv2_bufferSizeExt`                             |                                                 |
|`cusparseZgtsv2_bufferSizeExt`                             |                                                 |
|`cusparseSgtsv2`                                           |                                                 |
|`cusparseDgtsv2`                                           |                                                 |
|`cusparseCgtsv2`                                           |                                                 |
|`cusparseZgtsv2`                                           |                                                 |
|`cusparseSgtsv2_nopivot_bufferSizeExt`                     |                                                 |
|`cusparseDgtsv2_nopivot_bufferSizeExt`                     |                                                 |
|`cusparseCgtsv2_nopivot_bufferSizeExt`                     |                                                 |
|`cusparseZgtsv2_nopivot_bufferSizeExt`                     |                                                 |
|`cusparseSgtsv2_nopivot`                                   |                                                 |
|`cusparseDgtsv2_nopivot`                                   |                                                 |
|`cusparseCgtsv2_nopivot`                                   |                                                 |
|`cusparseZgtsv2_nopivot`                                   |                                                 |

## ***7.4. Batched Tridiagonal Solve***

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseSgtsvStridedBatch`                                |                                                 |
|`cusparseDgtsvStridedBatch`                                |                                                 |
|`cusparseCgtsvStridedBatch`                                |                                                 |
|`cusparseZgtsvStridedBatch`                                |                                                 |
|`cusparseSgtsv2StridedBatch_bufferSizeExt`                 |                                                 |
|`cusparseDgtsv2StridedBatch_bufferSizeExt`                 |                                                 |
|`cusparseCgtsv2StridedBatch_bufferSizeExt`                 |                                                 |
|`cusparseZgtsv2StridedBatch_bufferSizeExt`                 |                                                 |
|`cusparseSgtsv2StridedBatch`                               |                                                 |
|`cusparseDgtsv2StridedBatch`                               |                                                 |
|`cusparseCgtsv2StridedBatch`                               |                                                 |
|`cusparseZgtsv2StridedBatch`                               |                                                 |
|`cusparseSgtsvInterleavedBatch_bufferSizeExt`              |                                                 |
|`cusparseDgtsvInterleavedBatch_bufferSizeExt`              |                                                 |
|`cusparseCgtsvInterleavedBatch_bufferSizeExt`              |                                                 |
|`cusparseZgtsvInterleavedBatch_bufferSizeExt`              |                                                 |
|`cusparseSgtsvInterleavedBatch`                            |                                                 |
|`cusparseDgtsvInterleavedBatch`                            |                                                 |
|`cusparseCgtsvInterleavedBatch`                            |                                                 |
|`cusparseZgtsvInterleavedBatch`                            |                                                 |

## ***7.5. Batched Pentadiagonal Solve***

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseSgpsvInterleavedBatch_bufferSizeExt`              |                                                 |
|`cusparseDgpsvInterleavedBatch_bufferSizeExt`              |                                                 |
|`cusparseCgpsvInterleavedBatch_bufferSizeExt`              |                                                 |
|`cusparseZgpsvInterleavedBatch_bufferSizeExt`              |                                                 |
|`cusparseSgpsvInterleavedBatch`                            |                                                 |
|`cusparseDgpsvInterleavedBatch`                            |                                                 |
|`cusparseCgpsvInterleavedBatch`                            |                                                 |
|`cusparseZgpsvInterleavedBatch`                            |                                                 |

## **8. cuSPARSE Matrix Reorderings Reference**

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseScsrcolor`                                        |                                                 |
|`cusparseDcsrcolor`                                        |                                                 |
|`cusparseCcsrcolor`                                        |                                                 |
|`cusparseZcsrcolor`                                        |                                                 |

## **9. cuSPARSE Format Conversion Reference**

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cusparseSbsr2csr`                                         |                                                 |
|`cusparseDbsr2csr`                                         |                                                 |
|`cusparseCbsr2csr`                                         |                                                 |
|`cusparseZbsr2csr`                                         |                                                 |
|`cusparseSgebsr2gebsc_bufferSize`                          |                                                 |
|`cusparseDgebsr2gebsc_bufferSize`                          |                                                 |
|`cusparseCgebsr2gebsc_bufferSize`                          |                                                 |
|`cusparseZgebsr2gebsc_bufferSize`                          |                                                 |
|`cusparseSgebsr2gebsc`                                     |                                                 |
|`cusparseDgebsr2gebsc`                                     |                                                 |
|`cusparseCgebsr2gebsc`                                     |                                                 |
|`cusparseZgebsr2gebsc`                                     |                                                 |
|`cusparseSgebsr2gebsr_bufferSize`                          |                                                 |
|`cusparseDgebsr2gebsr_bufferSize`                          |                                                 |
|`cusparseCgebsr2gebsr_bufferSize`                          |                                                 |
|`cusparseZgebsr2gebsr_bufferSize`                          |                                                 |
|`cusparseXgebsr2gebsrNnz`                                  |                                                 |
|`cusparseSgebsr2gebsr`                                     |                                                 |
|`cusparseDgebsr2gebsr`                                     |                                                 |
|`cusparseCgebsr2gebsr`                                     |                                                 |
|`cusparseZgebsr2gebsr`                                     |                                                 |
|`cusparseSgebsr2csr`                                       |                                                 |
|`cusparseDgebsr2csr`                                       |                                                 |
|`cusparseCgebsr2csr`                                       |                                                 |
|`cusparseZgebsr2csr`                                       |                                                 |
|`cusparseScsr2gebsr_bufferSize`                            |                                                 |
|`cusparseDcsr2gebsr_bufferSize`                            |                                                 |
|`cusparseCcsr2gebsr_bufferSize`                            |                                                 |
|`cusparseZcsr2gebsr_bufferSize`                            |                                                 |
|`cusparseXcsr2gebsrNnz`                                    |                                                 |
|`cusparseScsr2gebsr`                                       |                                                 |
|`cusparseDcsr2gebsr`                                       |                                                 |
|`cusparseCcsr2gebsr`                                       |                                                 |
|`cusparseZcsr2gebsr`                                       |                                                 |
|`cusparseXcoo2csr`                                         |`hipsparseXcoo2csr`                              |
|`cusparseScsc2dense`                                       |                                                 |
|`cusparseDcsc2dense`                                       |                                                 |
|`cusparseCcsc2dense`                                       |                                                 |
|`cusparseZcsc2dense`                                       |                                                 |
|`cusparseScsc2hyb`                                         |                                                 |
|`cusparseDcsc2hyb`                                         |                                                 |
|`cusparseCcsc2hyb`                                         |                                                 |
|`cusparseZcsc2hyb`                                         |                                                 |
|`cusparseXcsr2bsrNnz`                                      |                                                 |
|`cusparseScsr2bsr`                                         |                                                 |
|`cusparseDcsr2bsr`                                         |                                                 |
|`cusparseCcsr2bsr`                                         |                                                 |
|`cusparseZcsr2bsr`                                         |                                                 |
|`cusparseXcsr2coo`                                         |`hipsparseXcsr2coo`                              |
|`cusparseScsr2csc`                                         |`hipsparseScsr2csc`                              |
|`cusparseDcsr2csc`                                         |`hipsparseDcsr2csc`                              |
|`cusparseCcsr2csc`                                         |                                                 |
|`cusparseZcsr2csc`                                         |                                                 |
|`cusparseCsr2cscEx`                                        |                                                 |
|`cusparseScsr2dense`                                       |                                                 |
|`cusparseDcsr2dense`                                       |                                                 |
|`cusparseCcsr2dense`                                       |                                                 |
|`cusparseZcsr2dense`                                       |                                                 |
|`cusparseScsr2csr_compress`                                |                                                 |
|`cusparseDcsr2csr_compress`                                |                                                 |
|`cusparseCcsr2csr_compress`                                |                                                 |
|`cusparseZcsr2csr_compress`                                |                                                 |
|`cusparseScsr2hyb`                                         |`hipsparseScsr2hyb`                              |
|`cusparseDcsr2hyb`                                         |`hipsparseDcsr2hyb`                              |
|`cusparseCcsr2hyb`                                         |                                                 |
|`cusparseZcsr2hyb`                                         |                                                 |
|`cusparseSdense2csc`                                       |                                                 |
|`cusparseDdense2csc`                                       |                                                 |
|`cusparseCdense2csc`                                       |                                                 |
|`cusparseZdense2csc`                                       |                                                 |
|`cusparseSdense2csr`                                       |                                                 |
|`cusparseDdense2csr`                                       |                                                 |
|`cusparseCdense2csr`                                       |                                                 |
|`cusparseZdense2csr`                                       |                                                 |
|`cusparseSdense2hyb`                                       |                                                 |
|`cusparseDdense2hyb`                                       |                                                 |
|`cusparseCdense2hyb`                                       |                                                 |
|`cusparseZdense2hyb`                                       |                                                 |
|`cusparseShyb2csc`                                         |                                                 |
|`cusparseDhyb2csc`                                         |                                                 |
|`cusparseChyb2csc`                                         |                                                 |
|`cusparseZhyb2csc`                                         |                                                 |
|`cusparseShyb2csr`                                         |                                                 |
|`cusparseDhyb2csr`                                         |                                                 |
|`cusparseChyb2csr`                                         |                                                 |
|`cusparseZhyb2csr`                                         |                                                 |
|`cusparseShyb2dense`                                       |                                                 |
|`cusparseDhyb2dense`                                       |                                                 |
|`cusparseChyb2dense`                                       |                                                 |
|`cusparseZhyb2dense`                                       |                                                 |
|`cusparseSnnz`                                             |                                                 |
|`cusparseDnnz`                                             |                                                 |
|`cusparseCnnz`                                             |                                                 |
|`cusparseZnnz`                                             |                                                 |
|`cusparseCreateIdentityPermutation`                        |`hipsparseCreateIdentityPermutation`             |
|`cusparseXcoosort_bufferSizeExt`                           |`hipsparseXcoosort_bufferSizeExt`                |
|`cusparseXcoosortByRow`                                    |`hipsparseXcoosortByRow`                         |
|`cusparseXcoosortByColumn`                                 |`hipsparseXcoosortByColumn`                      |
|`cusparseXcsrsort_bufferSizeExt`                           |`hipsparseXcsrsort_bufferSizeExt`                |
|`cusparseXcsrsort`                                         |`hipsparseXcsrsort`                              |
|`cusparseXcscsort_bufferSizeExt`                           |                                                 |
|`cusparseXcscsort`                                         |                                                 |
|`cusparseCreateCsru2csrInfo`                               |                                                 |
|`cusparseDestroyCsru2csrInfo`                              |                                                 |
|`cusparseScsru2csr_bufferSizeExt`                          |                                                 |
|`cusparseDcsru2csr_bufferSizeExt`                          |                                                 |
|`cusparseCcsru2csr_bufferSizeExt`                          |                                                 |
|`cusparseZcsru2csr_bufferSizeExt`                          |                                                 |
|`cusparseScsru2csr`                                        |                                                 |
|`cusparseDcsru2csr`                                        |                                                 |
|`cusparseCcsru2csr`                                        |                                                 |
|`cusparseZcsru2csr`                                        |                                                 |
|`cusparseHpruneDense2csr_bufferSizeExt`                    |                                                 |
|`cusparseSpruneDense2csr_bufferSizeExt`                    |                                                 |
|`cusparseDpruneDense2csr_bufferSizeExt`                    |                                                 |
|`cusparseHpruneDense2csrNnz`                               |                                                 |
|`cusparseSpruneDense2csrNnz`                               |                                                 |
|`cusparseDpruneDense2csrNnz`                               |                                                 |
|`cusparseHpruneCsr2csr_bufferSizeExt`                      |                                                 |
|`cusparseSpruneCsr2csr_bufferSizeExt`                      |                                                 |
|`cusparseDpruneCsr2csr_bufferSizeExt`                      |                                                 |
|`cusparseHpruneCsr2csrNnz`                                 |                                                 |
|`cusparseSpruneCsr2csrNnz`                                 |                                                 |
|`cusparseDpruneCsr2csrNnz`                                 |                                                 |
|`cusparseHpruneDense2csrByPercentage_bufferSizeExt`        |                                                 |
|`cusparseSpruneDense2csrByPercentage_bufferSizeExt`        |                                                 |
|`cusparseDpruneDense2csrByPercentage_bufferSizeExt`        |                                                 |
|`cusparseHpruneDense2csrNnzByPercentage`                   |                                                 |
|`cusparseSpruneDense2csrNnzByPercentage`                   |                                                 |
|`cusparseDpruneDense2csrNnzByPercentage`                   |                                                 |
|`cusparseHpruneCsr2csrByPercentage_bufferSizeExt`          |                                                 |
|`cusparseSpruneCsr2csrByPercentage_bufferSizeExt`          |                                                 |
|`cusparseDpruneCsr2csrByPercentage_bufferSizeExt`          |                                                 |
|`cusparseHpruneCsr2csrNnzByPercentage`                     |                                                 |
|`cusparseSpruneCsr2csrNnzByPercentage`                     |                                                 |
|`cusparseDpruneCsr2csrNnzByPercentage`                     |                                                 |
|`cusparseSnnz_compress`                                    |                                                 |
|`cusparseDnnz_compress`                                    |                                                 |
|`cusparseCnnz_compress`                                    |                                                 |
|`cusparseZnnz_compress`                                    |                                                 |
