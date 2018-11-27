# CUSPARSE API supported by HIP

## **1. CUSPARSE Data types**

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

## **2.cuSPARSE Helper Function Reference**

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

## **3.cuSPARSE Level 1 Function Reference**

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

## **4.cuSPARSE Level 2 Function Reference**

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
