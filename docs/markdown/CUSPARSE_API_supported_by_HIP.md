# CUSPARSE API supported by HIP

## **1. cuSPARSE Data types**

| **type**     |   **CUDA**                                                    |**CUDA version\***|   **HIP**                                                  |
|-------------:|---------------------------------------------------------------|:-----------------|------------------------------------------------------------|
| define       |`CUSPARSE_VER_MAJOR`                                           | 10.1 Update 2    |                                                            |
| define       |`CUSPARSE_VER_MINOR`                                           | 10.1 Update 2    |                                                            |
| define       |`CUSPARSE_VER_PATCH`                                           | 10.1 Update 2    |                                                            |
| define       |`CUSPARSE_VER_BUILD`                                           | 10.1 Update 2    |                                                            |
| define       |`CUSPARSE_VERSION`                                             | 10.1 Update 2    |                                                            |
| enum         |***`cusparseAction_t`***                                       |                  |***`hipsparseAction_t`***                                   |
|            0 |*`CUSPARSE_ACTION_SYMBOLIC`*                                   |                  |*`HIPSPARSE_ACTION_SYMBOLIC`*                               |
|            1 |*`CUSPARSE_ACTION_NUMERIC`*                                    |                  |*`HIPSPARSE_ACTION_NUMERIC`*                                |
| enum         |***`cusparseDirection_t`***                                    |                  |***`hipsparseDirection_t`***                                |
|            0 |*`CUSPARSE_DIRECTION_ROW`*                                     |                  |*`HIPSPARSE_DIRECTION_ROW`*                                 |
|            1 |*`CUSPARSE_DIRECTION_COLUMN`*                                  |                  |*`HIPSPARSE_DIRECTION_COLUMN`*                              |
| enum         |***`cusparseHybPartition_t`***                                 |                  |***`hipsparseHybPartition_t`***                             |
|            0 |*`CUSPARSE_HYB_PARTITION_AUTO`*                                |                  |*`HIPSPARSE_HYB_PARTITION_AUTO`*                            |
|            1 |*`CUSPARSE_HYB_PARTITION_USER`*                                |                  |*`HIPSPARSE_HYB_PARTITION_USER`*                            |
|            2 |*`CUSPARSE_HYB_PARTITION_MAX`*                                 |                  |*`HIPSPARSE_HYB_PARTITION_MAX`*                             |
| enum         |***`cusparseDiagType_t`***                                     |                  |***`hipsparseDiagType_t`***                                 |
|            0 |*`CUSPARSE_DIAG_TYPE_NON_UNIT`*                                |                  |*`HIPSPARSE_DIAG_TYPE_NON_UNIT`*                            |
|            1 |*`CUSPARSE_DIAG_TYPE_UNIT`*                                    |                  |*`HIPSPARSE_DIAG_TYPE_UNIT`*                                |
| enum         |***`cusparseFillMode_t`***                                     |                  |***`hipsparseFillMode_t`***                                 |
|            0 |*`CUSPARSE_FILL_MODE_LOWER`*                                   |                  |*`HIPSPARSE_FILL_MODE_LOWER`*                               |
|            1 |*`CUSPARSE_FILL_MODE_UPPER`*                                   |                  |*`HIPSPARSE_FILL_MODE_UPPER`*                               |
| enum         |***`cusparseIndexBase_t`***                                    |                  |***`hipsparseIndexBase_t`***                                |
|            0 |*`CUSPARSE_INDEX_BASE_ZERO`*                                   |                  |*`HIPSPARSE_INDEX_BASE_ZERO`*                               |
|            1 |*`CUSPARSE_INDEX_BASE_ONE`*                                    |                  |*`HIPSPARSE_INDEX_BASE_ONE`*                                |
| enum         |***`cusparseMatrixType_t`***                                   |                  |***`hipsparseMatrixType_t`***                               |
|            0 |*`CUSPARSE_MATRIX_TYPE_GENERAL`*                               |                  |*`HIPSPARSE_MATRIX_TYPE_GENERAL`*                           |
|            1 |*`CUSPARSE_MATRIX_TYPE_SYMMETRIC`*                             |                  |*`HIPSPARSE_MATRIX_TYPE_SYMMETRIC`*                         |
|            2 |*`CUSPARSE_MATRIX_TYPE_HERMITIAN`*                             |                  |*`HIPSPARSE_MATRIX_TYPE_HERMITIAN`*                         |
|            3 |*`CUSPARSE_MATRIX_TYPE_TRIANGULAR`*                            |                  |*`HIPSPARSE_MATRIX_TYPE_TRIANGULAR`*                        |
| enum         |***`cusparseOperation_t`***                                    |                  |***`hipsparseOperation_t`***                                |
|            0 |*`CUSPARSE_OPERATION_NON_TRANSPOSE`*                           |                  |*`HIPSPARSE_OPERATION_NON_TRANSPOSE`*                       |
|            1 |*`CUSPARSE_OPERATION_TRANSPOSE`*                               |                  |*`HIPSPARSE_OPERATION_TRANSPOSE`*                           |
|            2 |*`CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE`*                     |                  |*`HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE`*                 |
| enum         |***`cusparsePointerMode_t`***                                  |                  |***`hipsparsePointerMode_t`***                              |
|            0 |*`CUSPARSE_POINTER_MODE_HOST`*                                 |                  |*`HIPSPARSE_POINTER_MODE_HOST`*                             |
|            1 |*`CUSPARSE_POINTER_MODE_DEVICE`*                               |                  |*`HIPSPARSE_POINTER_MODE_DEVICE`*                           |
| enum         |***`cusparseAlgMode_t`***                                      | 8.0              |                                                            |
|            0 |*`CUSPARSE_ALG0`*                                              | 8.0              |                                                            |
|            1 |*`CUSPARSE_ALG1`*                                              | 8.0              |                                                            |
|            0 |*`CUSPARSE_ALG_NAIVE`*                                         | 9.2              |                                                            |
|            1 |*`CUSPARSE_ALG_MERGE_PATH`*                                    | 9.2              |                                                            |
| enum         |***`cusparseSolvePolicy_t`***                                  |                  |***`hipsparseSolvePolicy_t`***                              |
|            0 |*`CUSPARSE_SOLVE_POLICY_NO_LEVEL`*                             |                  |*`HIPSPARSE_SOLVE_POLICY_NO_LEVEL`*                         |
|            1 |*`CUSPARSE_SOLVE_POLICY_USE_LEVEL`*                            |                  |*`HIPSPARSE_SOLVE_POLICY_USE_LEVEL`*                        |
| enum         |***`cusparseStatus_t`***                                       |                  |***`hipsparseMatrixType_t`***                               |
|            0 |*`CUSPARSE_STATUS_SUCCESS`*                                    |                  |*`HIPSPARSE_STATUS_SUCCESS`*                                |
|            1 |*`CUSPARSE_STATUS_NOT_INITIALIZED`*                            |                  |*`HIPSPARSE_STATUS_NOT_INITIALIZED`*                        |
|            2 |*`CUSPARSE_STATUS_ALLOC_FAILED`*                               |                  |*`HIPSPARSE_STATUS_ALLOC_FAILED`*                           |
|            3 |*`CUSPARSE_STATUS_INVALID_VALUE`*                              |                  |*`HIPSPARSE_STATUS_INVALID_VALUE`*                          |
|            4 |*`CUSPARSE_STATUS_ARCH_MISMATCH`*                              |                  |*`HIPSPARSE_STATUS_ARCH_MISMATCH`*                          |
|            5 |*`CUSPARSE_STATUS_MAPPING_ERROR`*                              |                  |*`HIPSPARSE_STATUS_MAPPING_ERROR`*                          |
|            6 |*`CUSPARSE_STATUS_EXECUTION_FAILED`*                           |                  |*`HIPSPARSE_STATUS_EXECUTION_FAILED`*                       |
|            7 |*`CUSPARSE_STATUS_INTERNAL_ERROR`*                             |                  |*`HIPSPARSE_STATUS_INTERNAL_ERROR`*                         |
|            8 |*`CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`*                  |                  |*`HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`*              |
|            9 |*`CUSPARSE_STATUS_ZERO_PIVOT`*                                 |                  |*`HIPSPARSE_STATUS_ZERO_PIVOT`*                             |
| struct       |`cusparseContext`                                              |                  |                                                            |
| typedef      |`cusparseHandle_t`                                             |                  |`hipsparseHandle_t`                                         |
| struct       |`cusparseHybMat`                                               |                  |                                                            |
| typedef      |`cusparseHybMat_t`                                             |                  |`hipsparseHybMat_t`                                         |
| struct       |`cusparseMatDescr`                                             |                  |                                                            |
| typedef      |`cusparseMatDescr_t`                                           |                  |`hipsparseMatDescr_t`                                       |
| struct       |`cusparseSolveAnalysisInfo`                                    |                  |                                                            |
| typedef      |`cusparseSolveAnalysisInfo_t`                                  |                  |                                                            |
| struct       |`csrsv2Info`                                                   |                  |                                                            |
| typedef      |`csrsv2Info_t`                                                 |                  |`csrsv2Info_t`                                              |
| struct       |`csrsm2Info`                                                   | 9.2              |`csrsm2Info`                                                |
| typedef      |`csrsm2Info_t`                                                 |                  |`csrsm2Info_t`                                              |
| struct       |`bsrsv2Info`                                                   |                  |                                                            |
| typedef      |`bsrsv2Info_t`                                                 |                  |                                                            |
| struct       |`bsrsm2Info`                                                   |                  |                                                            |
| typedef      |`bsrsm2Info_t`                                                 |                  |                                                            |
| struct       |`bsric02Info`                                                  |                  |                                                            |
| typedef      |`bsric02Info_t`                                                |                  |                                                            |
| struct       |`csrilu02Info`                                                 |                  |                                                            |
| typedef      |`csrilu02Info_t`                                               |                  |`csrilu02Info_t`                                            |
| struct       |`bsrilu02Info`                                                 |                  |                                                            |
| typedef      |`bsrilu02Info_t`                                               |                  |                                                            |
| struct       |`csru2csrInfo`                                                 |                  |                                                            |
| typedef      |`csru2csrInfo_t`                                               |                  |                                                            |
| struct       |`csrgemm2Info`                                                 |                  |`csrgemm2Info`                                              |
| typedef      |`csrgemm2Info_t`                                               |                  |`csrgemm2Info_t`                                            |
| struct       |`cusparseColorInfo`                                            |                  |                                                            |
| typedef      |`cusparseColorInfo_t`                                          |                  |                                                            |
| struct       |`pruneInfo`                                                    | 9.0              |                                                            |
| typedef      |`pruneInfo_t`                                                  | 9.0              |                                                            |
| enum         |***`cusparseCsr2CscAlg_t`***                                   | 10.1             |                                                            |
|            1 |*`CUSPARSE_CSR2CSC_ALG1`*                                      | 10.1             |                                                            |
|            2 |*`CUSPARSE_CSR2CSC_ALG2`*                                      | 10.1             |                                                            |
| enum         |***`cusparseFormat_t`***                                       | 10.1             |                                                            |
|            1 |*`CUSPARSE_FORMAT_CSR`*                                        | 10.1             |                                                            |
|            2 |*`CUSPARSE_FORMAT_CSC`*                                        | 10.1             |                                                            |
|            3 |*`CUSPARSE_FORMAT_COO`*                                        | 10.1             |                                                            |
|            4 |*`CUSPARSE_FORMAT_COO_AOS`*                                    | 10.1             |                                                            |
| enum         |***`cusparseOrder_t`***                                        | 10.1             |                                                            |
|            1 |*`CUSPARSE_ORDER_COL`*                                         | 10.1             |                                                            |
|            2 |*`CUSPARSE_ORDER_ROW`*                                         | 10.1             |                                                            |
| enum         |***`cusparseSpMVAlg_t`***                                      | 10.1             |                                                            |
|            0 |*`CUSPARSE_MV_ALG_DEFAULT`*                                    | 10.1             |                                                            |
|            1 |*`CUSPARSE_COOMV_ALG`*                                         | 10.1             |                                                            |
|            2 |*`CUSPARSE_CSRMV_ALG1`*                                        | 10.1             |                                                            |
|            3 |*`CUSPARSE_CSRMV_ALG2`*                                        | 10.1             |                                                            |
| enum         |***`cusparseSpMMAlg_t`***                                      | 10.1             |                                                            |
|            0 |*`CUSPARSE_MM_ALG_DEFAULT`*                                    | 10.1             |                                                            |
|            1 |*`CUSPARSE_COOMM_ALG1`*                                        | 10.1             |                                                            |
|            2 |*`CUSPARSE_COOMM_ALG2`*                                        | 10.1             |                                                            |
|            3 |*`CUSPARSE_COOMM_ALG3`*                                        | 10.1             |                                                            |
|            4 |*`CUSPARSE_CSRMM_ALG1`*                                        | 10.1             |                                                            |
| enum         |***`cusparseIndexType_t`***                                    | 10.1             |                                                            |
|            1 |*`CUSPARSE_INDEX_16U`*                                         | 10.1             |                                                            |
|            2 |*`CUSPARSE_INDEX_32I`*                                         | 10.1             |                                                            |
|            3 |*`CUSPARSE_INDEX_64I`*                                         | 10.1             |                                                            |
| struct       |`cusparseSpMatDescr`                                           | 10.1             |                                                            |
| typedef      |`cusparseSpMatDescr_t`                                         | 10.1             |                                                            |
| struct       |`cusparseDnMatDescr`                                           | 10.1             |                                                            |
| typedef      |`cusparseDnMatDescr_t`                                         | 10.1             |                                                            |
| struct       |`cusparseSpVecDescr`                                           | 10.1             |                                                            |
| typedef      |`cusparseSpVecDescr_t`                                         | 10.1             |                                                            |
| struct       |`cusparseDnVecDescr`                                           | 10.1             |                                                            |
| typedef      |`cusparseDnVecDescr_t`                                         | 10.1             |                                                            |

## **2. cuSPARSE Helper Function Reference**

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
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
|`cusparseGetStream`                                        |`hipsparseGetStream`                             | 8.0              |
|`cusparseCreateCsrsv2Info`                                 |`hipsparseCreateCsrsv2Info`                      |
|`cusparseDestroyCsrsv2Info`                                |`hipsparseDestroyCsrsv2Info`                     |
|`cusparseCreateCsrsm2Info`                                 |`hipsparseCreateCsrsm2Info`                      | 9.2              |
|`cusparseDestroyCsrsm2Info`                                |`hipsparseDestroyCsrsm2Info`                     | 9.2              |
|`cusparseCreateCsric02Info`                                |`hipsparseCreateCsric02Info`                     |
|`cusparseDestroyCsric02Info`                               |`hipsparseDestroyCsric02Info`                    |
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
|`cusparseCreateCsrgemm2Info`                               |`hipsparseCreateCsrgemm2Info`                    |
|`cusparseDestroyCsrgemm2Info`                              |`hipsparseDestroyCsrgemm2Info`                   |
|`cusparseCreatePruneInfo`                                  |                                                 | 9.0              |
|`cusparseDestroyPruneInfo`                                 |                                                 | 9.0              |

## **3. cuSPARSE Level 1 Function Reference**

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseSaxpyi`                                           |`hipsparseSaxpyi`                                |
|`cusparseDaxpyi`                                           |`hipsparseDaxpyi`                                |
|`cusparseCaxpyi`                                           |`hipsparseCaxpyi`                                |
|`cusparseZaxpyi`                                           |`hipsparseZaxpyi`                                |
|`cusparseSdoti`                                            |`hipsparseSdoti`                                 |
|`cusparseDdoti`                                            |`hipsparseDdoti`                                 |
|`cusparseCdoti`                                            |`hipsparseCdoti`                                 |
|`cusparseZdoti`                                            |`hipsparseZdoti`                                 |
|`cusparseCdotci`                                           |`hipsparseCdotci`                                |
|`cusparseZdotci`                                           |`hipsparseZdotci`                                |
|`cusparseSgthr`                                            |`hipsparseSgthr`                                 |
|`cusparseDgthr`                                            |`hipsparseDgthr`                                 |
|`cusparseCgthr`                                            |`hipsparseCgthr`                                 |
|`cusparseZgthr`                                            |`hipsparseZgthr`                                 |
|`cusparseSgthrz`                                           |`hipsparseSgthrz`                                |
|`cusparseDgthrz`                                           |`hipsparseDgthrz`                                |
|`cusparseCgthrz`                                           |`hipsparseCgthrz`                                |
|`cusparseZgthrz`                                           |`hipsparseZgthrz`                                |
|`cusparseSroti`                                            |`hipsparseSroti`                                 |
|`cusparseDroti`                                            |`hipsparseDroti`                                 |
|`cusparseSsctr`                                            |`hipsparseSsctr`                                 |
|`cusparseDsctr`                                            |`hipsparseDsctr`                                 |
|`cusparseCsctr`                                            |`hipsparseCsctr`                                 |
|`cusparseZsctr`                                            |`hipsparseZsctr`                                 |

## **4. cuSPARSE Level 2 Function Reference**

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseSbsrmv`                                           |`hipsparseSbsrmv`                                |
|`cusparseDbsrmv`                                           |`hipsparseDbsrmv`                                |
|`cusparseCbsrmv`                                           |`hipsparseCbsrmv`                                |
|`cusparseZbsrmv`                                           |`hipsparseZbsrmv`                                |
|`cusparseSbsrxmv`                                          |                                                 |
|`cusparseDbsrxmv`                                          |                                                 |
|`cusparseCbsrxmv`                                          |                                                 |
|`cusparseZbsrxmv`                                          |                                                 |
|`cusparseScsrmv`                                           |`hipsparseScsrmv`                                |
|`cusparseDcsrmv`                                           |`hipsparseDcsrmv`                                |
|`cusparseCcsrmv`                                           |`hipsparseCcsrmv`                                |
|`cusparseZcsrmv`                                           |`hipsparseZcsrmv`                                |
|`cusparseCsrmvEx`                                          |                                                 | 8.0              |
|`cusparseCsrmvEx_bufferSize`                               |                                                 | 8.0              |
|`cusparseScsrmv_mp`                                        |                                                 | 8.0              |
|`cusparseDcsrmv_mp`                                        |                                                 | 8.0              |
|`cusparseCcsrmv_mp`                                        |                                                 | 8.0              |
|`cusparseZcsrmv_mp`                                        |                                                 | 8.0              |
|`cusparseSgemvi`                                           |                                                 | 7.5              |
|`cusparseDgemvi`                                           |                                                 | 7.5              |
|`cusparseCgemvi`                                           |                                                 | 7.5              |
|`cusparseZgemvi`                                           |                                                 | 7.5              |
|`cusparseSgemvi_bufferSize`                                |                                                 | 7.5              |
|`cusparseDgemvi_bufferSize`                                |                                                 | 7.5              |
|`cusparseCgemvi_bufferSize`                                |                                                 | 7.5              |
|`cusparseZgemvi_bufferSize`                                |                                                 | 7.5              |
|`cusparseSbsrsv2_bufferSize`                               |                                                 |
|`cusparseSbsrsv2_bufferSizeExt`                            |                                                 |
|`cusparseDbsrsv2_bufferSize`                               |                                                 |
|`cusparseDbsrsv2_bufferSizeExt`                            |                                                 |
|`cusparseCbsrsv2_bufferSize`                               |                                                 |
|`cusparseCbsrsv2_bufferSizeExt`                            |                                                 |
|`cusparseZbsrsv2_bufferSize`                               |                                                 |
|`cusparseZbsrsv2_bufferSizeExt`                            |                                                 |
|`cusparseSbsrsv2_analysis`                                 |                                                 |
|`cusparseDbsrsv2_analysis`                                 |                                                 |
|`cusparseCbsrsv2_analysis`                                 |                                                 |
|`cusparseZbsrsv2_analysis`                                 |                                                 |
|`cusparseXbsrsv2_zeroPivot`                                |                                                 |
|`cusparseSbsrsv2_solve                                     |                                                 |
|`cusparseDbsrsv2_solve                                     |                                                 |
|`cusparseCbsrsv2_solve                                     |                                                 |
|`cusparseZbsrsv2_solve                                     |                                                 |
|`cusparseScsrsv_analysis`                                  |                                                 |
|`cusparseDcsrsv_analysis`                                  |                                                 |
|`cusparseCcsrsv_analysis`                                  |                                                 |
|`cusparseZcsrsv_analysis`                                  |                                                 |
|`cusparseCsrsv_analysisEx`                                 |                                                 | 8.0              |
|`cusparseScsrsv_solve`                                     |                                                 |
|`cusparseDcsrsv_solve`                                     |                                                 |
|`cusparseCcsrsv_solve`                                     |                                                 |
|`cusparseZcsrsv_solve`                                     |                                                 |
|`cusparseCsrsv_solveEx`                                    |                                                 | 8.0              |
|`cusparseScsrsv2_bufferSize`                               |`hipsparseScsrsv2_bufferSize`                    |
|`cusparseScsrsv2_bufferSizeExt`                            |`hipsparseScsrsv2_bufferSizeExt`                 |
|`cusparseDcsrsv2_bufferSize`                               |`hipsparseDcsrsv2_bufferSize`                    |
|`cusparseDcsrsv2_bufferSizeExt`                            |`hipsparseDcsrsv2_bufferSizeExt`                 |
|`cusparseCcsrsv2_bufferSize`                               |`hipsparseCcsrsv2_bufferSize`                    |
|`cusparseCcsrsv2_bufferSizeExt`                            |`hipsparseCcsrsv2_bufferSizeExt`                 |
|`cusparseZcsrsv2_bufferSize`                               |`hipsparseZcsrsv2_bufferSize`                    |
|`cusparseZcsrsv2_bufferSizeExt`                            |`hipsparseZcsrsv2_bufferSizeExt`                 |
|`cusparseScsrsv2_analysis`                                 |`hipsparseScsrsv2_analysis`                      |
|`cusparseDcsrsv2_analysis`                                 |`hipsparseDcsrsv2_analysis`                      |
|`cusparseCcsrsv2_analysis`                                 |`hipsparseCcsrsv2_analysis`                      |
|`cusparseZcsrsv2_analysis`                                 |`hipsparseZcsrsv2_analysis`                      |
|`cusparseScsrsv2_solve`                                    |`hipsparseScsrsv2_solve`                         |
|`cusparseDcsrsv2_solve`                                    |`hipsparseDcsrsv2_solve`                         |
|`cusparseCcsrsv2_solve`                                    |`hipsparseCcsrsv2_solve`                         |
|`cusparseZcsrsv2_solve`                                    |`hipsparseZcsrsv2_solve`                         |
|`cusparseXcsrsv2_zeroPivot`                                |`hipsparseXcsrsv2_zeroPivot`                     |
|`cusparseShybmv`                                           |`hipsparseShybmv`                                |
|`cusparseDhybmv`                                           |`hipsparseDhybmv`                                |
|`cusparseChybmv`                                           |`hipsparseChybmv`                                |
|`cusparseZhybmv`                                           |`hipsparseZhybmv`                                |
|`cusparseShybsv_analysis`                                  |                                                 |
|`cusparseDhybsv_analysis`                                  |                                                 |
|`cusparseChybsv_analysis`                                  |                                                 |
|`cusparseZhybsv_analysis`                                  |                                                 |
|`cusparseShybsv_solve`                                     |                                                 |
|`cusparseDhybsv_solve`                                     |                                                 |
|`cusparseChybsv_solve`                                     |                                                 |
|`cusparseZhybsv_solve`                                     |                                                 |

## **5. cuSPARSE Level 3 Function Reference**

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseScsrmm`                                           |`hipsparseScsrmm`                                |
|`cusparseDcsrmm`                                           |`hipsparseDcsrmm`                                |
|`cusparseCcsrmm`                                           |`hipsparseCcsrmm`                                |
|`cusparseZcsrmm`                                           |`hipsparseZcsrmm`                                |
|`cusparseScsrmm2`                                          |`hipsparseScsrmm2`                               |
|`cusparseDcsrmm2`                                          |`hipsparseDcsrmm2`                               |
|`cusparseCcsrmm2`                                          |`hipsparseCcsrmm2`                               |
|`cusparseZcsrmm2`                                          |`hipsparseZcsrmm2`                               |
|`cusparseScsrsm_analysis`                                  |                                                 |
|`cusparseDcsrsm_analysis`                                  |                                                 |
|`cusparseCcsrsm_analysis`                                  |                                                 |
|`cusparseZcsrsm_analysis`                                  |                                                 |
|`cusparseScsrsm_solve`                                     |                                                 |
|`cusparseDcsrsm_solve`                                     |                                                 |
|`cusparseCcsrsm_solve`                                     |                                                 |
|`cusparseZcsrsm_solve`                                     |                                                 |
|`cusparseScsrsm2_bufferSizeExt`                            |`hipsparseScsrsm2_bufferSizeExt`                 | 9.2              |
|`cusparseDcsrsm2_bufferSizeExt`                            |`hipsparseDcsrsm2_bufferSizeExt`                 | 9.2              |
|`cusparseCcsrsm2_bufferSizeExt`                            |`hipsparseCcsrsm2_bufferSizeExt`                 | 9.2              |
|`cusparseZcsrsm2_bufferSizeExt`                            |`hipsparseZcsrsm2_bufferSizeExt`                 | 9.2              |
|`cusparseScsrsm2_analysis`                                 |`hipsparseScsrsm2_analysis`                      | 9.2              |
|`cusparseDcsrsm2_analysis`                                 |`hipsparseDcsrsm2_analysis`                      | 9.2              |
|`cusparseCcsrsm2_analysis`                                 |`hipsparseCcsrsm2_analysis`                      | 9.2              |
|`cusparseZcsrsm2_analysis`                                 |`hipsparseZcsrsm2_analysis`                      | 9.2              |
|`cusparseScsrsm2_solve`                                    |`hipsparseScsrsm2_solve`                         | 9.2              |
|`cusparseDcsrsm2_solve`                                    |`hipsparseDcsrsm2_solve`                         | 9.2              |
|`cusparseCcsrsm2_solve`                                    |`hipsparseCcsrsm2_solve`                         | 9.2              |
|`cusparseZcsrsm2_solve`                                    |`hipsparseZcsrsm2_solve`                         | 9.2              |
|`cusparseXcsrsm2_zeroPivot`                                |`hipsparseXcsrsm2_zeroPivot`                     | 9.2              |
|`cusparseSbsrmm`                                           |                                                 |
|`cusparseDbsrmm`                                           |                                                 |
|`cusparseCbsrmm`                                           |                                                 |
|`cusparseZbsrmm`                                           |                                                 |
|`cusparseSbsrsm2_bufferSize`                               |                                                 |
|`cusparseSbsrsm2_bufferSizeExt`                            |                                                 |
|`cusparseDbsrsm2_bufferSize`                               |                                                 |
|`cusparseDbsrsm2_bufferSizeExt`                            |                                                 |
|`cusparseCbsrsm2_bufferSize`                               |                                                 |
|`cusparseCbsrsm2_bufferSizeExt`                            |                                                 |
|`cusparseZbsrsm2_bufferSize`                               |                                                 |
|`cusparseZbsrsm2_bufferSizeExt`                            |                                                 |
|`cusparseSbsrsm2_analysis`                                 |                                                 |
|`cusparseDbsrsm2_analysis`                                 |                                                 |
|`cusparseCbsrsm2_analysis`                                 |                                                 |
|`cusparseZbsrsm2_analysis`                                 |                                                 |
|`cusparseSbsrsm2_solve`                                    |                                                 |
|`cusparseDbsrsm2_solve`                                    |                                                 |
|`cusparseCbsrsm2_solve`                                    |                                                 |
|`cusparseZbsrsm2_solve`                                    |                                                 |
|`cusparseXbsrsm2_zeroPivot`                                |                                                 |
|`cusparseSgemmi`                                           |                                                 | 8.0              |
|`cusparseDgemmi`                                           |                                                 | 8.0              |
|`cusparseCgemmi`                                           |                                                 | 8.0              |
|`cusparseZgemmi`                                           |                                                 | 8.0              |

## **6. cuSPARSE Extra Function Reference**

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseXcsrgeamNnz`                                      |`hipsparseXcsrgeamNnz`                           |
|`cusparseScsrgeam`                                         |`hipsparseScsrgeam`                              |
|`cusparseDcsrgeam`                                         |`hipsparseDcsrgeam`                              |
|`cusparseCcsrgeam`                                         |`hipsparseCcsrgeam`                              |
|`cusparseZcsrgeam`                                         |`hipsparseZcsrgeam`                              |
|`cusparseXcsrgeam2Nnz`                                     |`hipsparseXcsrgeam2Nnz`                          | 9.2              |
|`cusparseScsrgeam2`                                        |`hipsparseScsrgeam2`                             | 9.2              |
|`cusparseDcsrgeam2`                                        |`hipsparseDcsrgeam2`                             | 9.2              |
|`cusparseCcsrgeam2`                                        |`hipsparseCcsrgeam2`                             | 9.2              |
|`cusparseZcsrgeam2`                                        |`hipsparseZcsrgeam2`                             | 9.2              |
|`cusparseScsrgeam2_bufferSizeExt`                          |`hipsparseScsrgeam2_bufferSizeExt`               | 9.2              |
|`cusparseDcsrgeam2_bufferSizeExt`                          |`hipsparseDcsrgeam2_bufferSizeExt`               | 9.2              |
|`cusparseCcsrgeam2_bufferSizeExt`                          |`hipsparseCcsrgeam2_bufferSizeExt`               | 9.2              |
|`cusparseZcsrgeam2_bufferSizeExt`                          |`hipsparseZcsrgeam2_bufferSizeExt`               | 9.2              |
|`cusparseXcsrgemmNnz`                                      |`hipsparseXcsrgemmNnz`                           |
|`cusparseScsrgemm`                                         |`hipsparseScsrgemm`                              |
|`cusparseDcsrgemm`                                         |`hipsparseDcsrgemm`                              |
|`cusparseCcsrgemm`                                         |`hipsparseCcsrgemm`                              |
|`cusparseZcsrgemm`                                         |`hipsparseZcsrgemm`                              |
|`cusparseXcsrgemm2Nnz`                                     |`hipsparseXcsrgemm2Nnz`                          |
|`cusparseScsrgemm2`                                        |`hipsparseScsrgemm2`                             |
|`cusparseDcsrgemm2`                                        |`hipsparseDcsrgemm2`                             |
|`cusparseCcsrgemm2`                                        |`hipsparseCcsrgemm2`                             |
|`cusparseZcsrgemm2`                                        |`hipsparseZcsrgemm2`                             |
|`cusparseScsrgemm2_bufferSizeExt`                          |`hipsparseScsrgemm2_bufferSizeExt`               |
|`cusparseDcsrgemm2_bufferSizeExt`                          |`hipsparseDcsrgemm2_bufferSizeExt`               |
|`cusparseCcsrgemm2_bufferSizeExt`                          |`hipsparseCcsrgemm2_bufferSizeExt`               |
|`cusparseZcsrgemm2_bufferSizeExt`                          |`hipsparseZcsrgemm2_bufferSizeExt`               |

## **7. cuSPARSE Preconditioners Reference**

## ***7.1. Incomplete Cholesky Factorization: level 0***

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseScsric0`                                          |                                                 |
|`cusparseDcsric0`                                          |                                                 |
|`cusparseCcsric0`                                          |                                                 |
|`cusparseZcsric0`                                          |                                                 |
|`cusparseScsric02_bufferSize`                              |`hipsparseScsric02_bufferSize`                   |
|`cusparseScsric02_bufferSizeExt`                           |`hipsparseScsric02_bufferSizeExt`                |
|`cusparseDcsric02_bufferSize`                              |`hipsparseDcsric02_bufferSize`                   |
|`cusparseDcsric02_bufferSizeExt`                           |`hipsparseDcsric02_bufferSizeExt`                |
|`cusparseCcsric02_bufferSize`                              |`hipsparseCcsric02_bufferSize`                   |
|`cusparseCcsric02_bufferSizeExt`                           |`hipsparseCcsric02_bufferSizeExt`                |
|`cusparseZcsric02_bufferSize`                              |`hipsparseZcsric02_bufferSize`                   |
|`cusparseZcsric02_bufferSizeExt`                           |`hipsparseZcsric02_bufferSizeExt`                |
|`cusparseScsric02_analysis`                                |`hipsparseScsric02_analysis`                     |
|`cusparseDcsric02_analysis`                                |`hipsparseDcsric02_analysis`                     |
|`cusparseCcsric02_analysis`                                |`hipsparseCcsric02_analysis`                     |
|`cusparseZcsric02_analysis`                                |`hipsparseZcsric02_analysis`                     |
|`cusparseScsric02`                                         |`hipsparseScsric02`                              |
|`cusparseDcsric02`                                         |`hipsparseDcsric02`                              |
|`cusparseCcsric02`                                         |`hipsparseCcsric02`                              |
|`cusparseZcsric02`                                         |`hipsparseZcsric02`                              |
|`cusparseXcsric02_zeroPivot`                               |`hipsparseXcsric02_zeroPivot`                    |
|`cusparseSbsric02_bufferSize`                              |                                                 |
|`cusparseSbsric02_bufferSizeExt`                           |                                                 |
|`cusparseDbsric02_bufferSize`                              |                                                 |
|`cusparseDbsric02_bufferSizeExt`                           |                                                 |
|`cusparseCbsric02_bufferSize`                              |                                                 |
|`cusparseCbsric02_bufferSizeExt`                           |                                                 |
|`cusparseZbsric02_bufferSize`                              |                                                 |
|`cusparseZbsric02_bufferSizeExt`                           |                                                 |
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

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseScsrilu0`                                         |                                                 |
|`cusparseDcsrilu0`                                         |                                                 |
|`cusparseCcsrilu0`                                         |                                                 |
|`cusparseZcsrilu0`                                         |                                                 |
|`cusparseCsrilu0Ex`                                        |                                                 | 8.0              |
|`cusparseScsrilu02_numericBoost`                           |                                                 |
|`cusparseDcsrilu02_numericBoost`                           |                                                 |
|`cusparseCcsrilu02_numericBoost`                           |                                                 |
|`cusparseZcsrilu02_numericBoost`                           |                                                 |
|`cusparseXcsrilu02_zeroPivot`                              |`hipsparseXcsrilu02_zeroPivot`                   |
|`cusparseScsrilu02_bufferSize`                             |`hipsparseScsrilu02_bufferSize`                  |
|`cusparseScsrilu02_bufferSizeExt`                          |`hipsparseScsrilu02_bufferSizeExt`               |
|`cusparseDcsrilu02_bufferSize`                             |`hipsparseDcsrilu02_bufferSize`                  |
|`cusparseDcsrilu02_bufferSizeExt`                          |`hipsparseDcsrilu02_bufferSizeExt`               |
|`cusparseCcsrilu02_bufferSize`                             |`hipsparseCcsrilu02_bufferSize`                  |
|`cusparseCcsrilu02_bufferSizeExt`                          |`hipsparseCcsrilu02_bufferSizeExt`               |
|`cusparseZcsrilu02_bufferSize`                             |`hipsparseZcsrilu02_bufferSize`                  |
|`cusparseZcsrilu02_bufferSizeExt`                          |`hipsparseZcsrilu02_bufferSizeExt`               |
|`cusparseScsrilu02_analysis`                               |`hipsparseScsrilu02_analysis`                    |
|`cusparseDcsrilu02_analysis`                               |`hipsparseDcsrilu02_analysis`                    |
|`cusparseCcsrilu02_analysis`                               |`hipsparseCcsrilu02_analysis`                    |
|`cusparseZcsrilu02_analysis`                               |`hipsparseZcsrilu02_analysis`                    |
|`cusparseScsrilu02`                                        |`hipsparseScsrilu02`                             |
|`cusparseDcsrilu02`                                        |`hipsparseDcsrilu02`                             |
|`cusparseCcsrilu02`                                        |`hipsparseCcsrilu02`                             |
|`cusparseZcsrilu02`                                        |`hipsparseZcsrilu02`                             |
|`cusparseXbsric02_zeroPivot`                               |`hipsparseXcsrilu02_zeroPivot`                   |
|`cusparseSbsrilu02_numericBoost`                           |                                                 |
|`cusparseDbsrilu02_numericBoost`                           |                                                 |
|`cusparseCbsrilu02_numericBoost`                           |                                                 |
|`cusparseZbsrilu02_numericBoost`                           |                                                 |
|`cusparseSbsrilu02_bufferSize`                             |                                                 |
|`cusparseSbsrilu02_bufferSizeExt`                          |                                                 |
|`cusparseDbsrilu02_bufferSize`                             |                                                 |
|`cusparseDbsrilu02_bufferSizeExt`                          |                                                 |
|`cusparseCbsrilu02_bufferSize`                             |                                                 |
|`cusparseCbsrilu02_bufferSizeExt`                          |                                                 |
|`cusparseZbsrilu02_bufferSize`                             |                                                 |
|`cusparseZbsrilu02_bufferSizeExt`                          |                                                 |
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

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseSgtsv`                                            |                                                 |
|`cusparseDgtsv`                                            |                                                 |
|`cusparseCgtsv`                                            |                                                 |
|`cusparseZgtsv`                                            |                                                 |
|`cusparseSgtsv_nopivot`                                    |                                                 |
|`cusparseDgtsv_nopivot`                                    |                                                 |
|`cusparseCgtsv_nopivot`                                    |                                                 |
|`cusparseZgtsv_nopivot`                                    |                                                 |
|`cusparseSgtsv2_bufferSizeExt`                             |                                                 | 9.0              |
|`cusparseDgtsv2_bufferSizeExt`                             |                                                 | 9.0              |
|`cusparseCgtsv2_bufferSizeExt`                             |                                                 | 9.0              |
|`cusparseZgtsv2_bufferSizeExt`                             |                                                 | 9.0              |
|`cusparseSgtsv2`                                           |                                                 | 9.0              |
|`cusparseDgtsv2`                                           |                                                 | 9.0              |
|`cusparseCgtsv2`                                           |                                                 | 9.0              |
|`cusparseZgtsv2`                                           |                                                 | 9.0              |
|`cusparseSgtsv2_nopivot_bufferSizeExt`                     |                                                 | 9.0              |
|`cusparseDgtsv2_nopivot_bufferSizeExt`                     |                                                 | 9.0              |
|`cusparseCgtsv2_nopivot_bufferSizeExt`                     |                                                 | 9.0              |
|`cusparseZgtsv2_nopivot_bufferSizeExt`                     |                                                 | 9.0              |
|`cusparseSgtsv2_nopivot`                                   |                                                 | 9.0              |
|`cusparseDgtsv2_nopivot`                                   |                                                 | 9.0              |
|`cusparseCgtsv2_nopivot`                                   |                                                 | 9.0              |
|`cusparseZgtsv2_nopivot`                                   |                                                 | 9.0              |

## ***7.4. Batched Tridiagonal Solve***

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseSgtsvStridedBatch`                                |                                                 |
|`cusparseDgtsvStridedBatch`                                |                                                 |
|`cusparseCgtsvStridedBatch`                                |                                                 |
|`cusparseZgtsvStridedBatch`                                |                                                 |
|`cusparseSgtsv2StridedBatch_bufferSizeExt`                 |                                                 | 9.0              |
|`cusparseDgtsv2StridedBatch_bufferSizeExt`                 |                                                 | 9.0              |
|`cusparseCgtsv2StridedBatch_bufferSizeExt`                 |                                                 | 9.0              |
|`cusparseZgtsv2StridedBatch_bufferSizeExt`                 |                                                 | 9.0              |
|`cusparseSgtsv2StridedBatch`                               |                                                 | 9.0              |
|`cusparseDgtsv2StridedBatch`                               |                                                 | 9.0              |
|`cusparseCgtsv2StridedBatch`                               |                                                 | 9.0              |
|`cusparseZgtsv2StridedBatch`                               |                                                 | 9.0              |
|`cusparseSgtsvInterleavedBatch_bufferSizeExt`              |                                                 | 9.2              |
|`cusparseDgtsvInterleavedBatch_bufferSizeExt`              |                                                 | 9.2              |
|`cusparseCgtsvInterleavedBatch_bufferSizeExt`              |                                                 | 9.2              |
|`cusparseZgtsvInterleavedBatch_bufferSizeExt`              |                                                 | 9.2              |
|`cusparseSgtsvInterleavedBatch`                            |                                                 | 9.2              |
|`cusparseDgtsvInterleavedBatch`                            |                                                 | 9.2              |
|`cusparseCgtsvInterleavedBatch`                            |                                                 | 9.2              |
|`cusparseZgtsvInterleavedBatch`                            |                                                 | 9.2              |

## ***7.5. Batched Pentadiagonal Solve***

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseSgpsvInterleavedBatch_bufferSizeExt`              |                                                 | 9.2              |
|`cusparseDgpsvInterleavedBatch_bufferSizeExt`              |                                                 | 9.2              |
|`cusparseCgpsvInterleavedBatch_bufferSizeExt`              |                                                 | 9.2              |
|`cusparseZgpsvInterleavedBatch_bufferSizeExt`              |                                                 | 9.2              |
|`cusparseSgpsvInterleavedBatch`                            |                                                 | 9.2              |
|`cusparseDgpsvInterleavedBatch`                            |                                                 | 9.2              |
|`cusparseCgpsvInterleavedBatch`                            |                                                 | 9.2              |
|`cusparseZgpsvInterleavedBatch`                            |                                                 | 9.2              |

## **8. cuSPARSE Matrix Reorderings Reference**

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseScsrcolor`                                        |                                                 |
|`cusparseDcsrcolor`                                        |                                                 |
|`cusparseCcsrcolor`                                        |                                                 |
|`cusparseZcsrcolor`                                        |                                                 |

## **9. cuSPARSE Format Conversion Reference**

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseSbsr2csr`                                         |`hipsparseSbsr2csr`                              |
|`cusparseDbsr2csr`                                         |`hipsparseDbsr2csr`                              |
|`cusparseCbsr2csr`                                         |`hipsparseCbsr2csr`                              |
|`cusparseZbsr2csr`                                         |`hipsparseZbsr2csr`                              |
|`cusparseSgebsr2gebsc_bufferSize`                          |                                                 |
|`cusparseSgebsr2gebsc_bufferSizeExt`                       |                                                 |
|`cusparseDgebsr2gebsc_bufferSize`                          |                                                 |
|`cusparseDgebsr2gebsc_bufferSizeExt`                       |                                                 |
|`cusparseCgebsr2gebsc_bufferSize`                          |                                                 |
|`cusparseCgebsr2gebsc_bufferSizeExt`                       |                                                 |
|`cusparseZgebsr2gebsc_bufferSize`                          |                                                 |
|`cusparseZgebsr2gebsc_bufferSizeExt`                       |                                                 |
|`cusparseSgebsr2gebsc`                                     |                                                 |
|`cusparseDgebsr2gebsc`                                     |                                                 |
|`cusparseCgebsr2gebsc`                                     |                                                 |
|`cusparseZgebsr2gebsc`                                     |                                                 |
|`cusparseSgebsr2gebsr_bufferSize`                          |                                                 |
|`cusparseSgebsr2gebsr_bufferSizeExt`                       |                                                 |
|`cusparseDgebsr2gebsr_bufferSize`                          |                                                 |
|`cusparseDgebsr2gebsr_bufferSizeExt`                       |                                                 |
|`cusparseCgebsr2gebsr_bufferSize`                          |                                                 |
|`cusparseCgebsr2gebsr_bufferSizeExt`                       |                                                 |
|`cusparseZgebsr2gebsr_bufferSize`                          |                                                 |
|`cusparseZgebsr2gebsr_bufferSizeExt`                       |                                                 |
|`cusparseXgebsr2gebsrNnz`                                  |                                                 |
|`cusparseSgebsr2gebsr`                                     |                                                 |
|`cusparseDgebsr2gebsr`                                     |                                                 |
|`cusparseCgebsr2gebsr`                                     |                                                 |
|`cusparseZgebsr2gebsr`                                     |                                                 |
|`cusparseXgebsr2csr`                                       |                                                 |
|`cusparseSgebsr2csr`                                       |                                                 |
|`cusparseDgebsr2csr`                                       |                                                 |
|`cusparseCgebsr2csr`                                       |                                                 |
|`cusparseZgebsr2csr`                                       |                                                 |
|`cusparseScsr2gebsr_bufferSize`                            |                                                 |
|`cusparseScsr2gebsr_bufferSizeExt`                         |                                                 |
|`cusparseDcsr2gebsr_bufferSize`                            |                                                 |
|`cusparseDcsr2gebsr_bufferSizeExt`                         |                                                 |
|`cusparseCcsr2gebsr_bufferSize`                            |                                                 |
|`cusparseCcsr2gebsr_bufferSizeExt`                         |                                                 |
|`cusparseZcsr2gebsr_bufferSize`                            |                                                 |
|`cusparseZcsr2gebsr_bufferSizeExt`                         |                                                 |
|`cusparseXcsr2gebsrNnz`                                    |                                                 |
|`cusparseScsr2gebsr`                                       |                                                 |
|`cusparseDcsr2gebsr`                                       |                                                 |
|`cusparseCcsr2gebsr`                                       |                                                 |
|`cusparseZcsr2gebsr`                                       |                                                 |
|`cusparseXcoo2csr`                                         |`hipsparseXcoo2csr`                              |
|`cusparseScsc2dense`                                       |`hipsparseScsc2dense`                            |
|`cusparseDcsc2dense`                                       |`hipsparseDcsc2dense`                            |
|`cusparseCcsc2dense`                                       |`hipsparseCcsc2dense`                            |
|`cusparseZcsc2dense`                                       |`hipsparseZcsc2dense`                            |
|`cusparseScsc2hyb`                                         |                                                 |
|`cusparseDcsc2hyb`                                         |                                                 |
|`cusparseCcsc2hyb`                                         |                                                 |
|`cusparseZcsc2hyb`                                         |                                                 |
|`cusparseXcsr2bsrNnz`                                      |`hipsparseXcsr2bsrNnz`                           |
|`cusparseScsr2bsr`                                         |`hipsparseScsr2bsr`                              |
|`cusparseDcsr2bsr`                                         |`hipsparseDcsr2bsr`                              |
|`cusparseCcsr2bsr`                                         |`hipsparseCcsr2bsr`                              |
|`cusparseZcsr2bsr`                                         |`hipsparseZcsr2bsr`                              |
|`cusparseXcsr2coo`                                         |`hipsparseXcsr2coo`                              |
|`cusparseScsr2csc`                                         |`hipsparseScsr2csc`                              |
|`cusparseDcsr2csc`                                         |`hipsparseDcsr2csc`                              |
|`cusparseCcsr2csc`                                         |`hipsparseCcsr2csc`                              |
|`cusparseZcsr2csc`                                         |`hipsparseZcsr2csc`                              |
|`cusparseCsr2cscEx`                                        |                                                 | 8.0              |
|`cusparseCsr2cscEx2`                                       |                                                 | 10.1             |
|`cusparseCsr2cscEx2_bufferSize`                            |                                                 | 10.1             |
|`cusparseScsr2dense`                                       |`hipsparseScsr2dense`                            |
|`cusparseDcsr2dense`                                       |`hipsparseDcsr2dense`                            |
|`cusparseCcsr2dense`                                       |`hipsparseCcsr2dense`                            |
|`cusparseZcsr2dense`                                       |`hipsparseZcsr2dense`                            |
|`cusparseScsr2csr_compress`                                |`hipsparseScsr2csr_compress`                     | 8.0              |
|`cusparseDcsr2csr_compress`                                |`hipsparseDcsr2csr_compress`                     | 8.0              |
|`cusparseCcsr2csr_compress`                                |`hipsparseCcsr2csr_compress`                     | 8.0              |
|`cusparseZcsr2csr_compress`                                |`hipsparseZcsr2csr_compress`                     | 8.0              |
|`cusparseScsr2hyb`                                         |`hipsparseScsr2hyb`                              |
|`cusparseDcsr2hyb`                                         |`hipsparseDcsr2hyb`                              |
|`cusparseCcsr2hyb`                                         |`hipsparseCcsr2hyb`                              |
|`cusparseZcsr2hyb`                                         |`hipsparseZcsr2hyb`                              |
|`cusparseSdense2csc`                                       |`hipsparseSdense2csc`                            |
|`cusparseDdense2csc`                                       |`hipsparseDdense2csc`                            |
|`cusparseCdense2csc`                                       |`hipsparseCdense2csc`                            |
|`cusparseZdense2csc`                                       |`hipsparseZdense2csc`                            |
|`cusparseSdense2csr`                                       |`hipsparseSdense2csr`                            |
|`cusparseDdense2csr`                                       |`hipsparseDdense2csr`                            |
|`cusparseCdense2csr`                                       |`hipsparseCdense2csr`                            |
|`cusparseZdense2csr`                                       |`hipsparseZdense2csr`                            |
|`cusparseSdense2hyb`                                       |                                                 |
|`cusparseDdense2hyb`                                       |                                                 |
|`cusparseCdense2hyb`                                       |                                                 |
|`cusparseZdense2hyb`                                       |                                                 |
|`cusparseShyb2csc`                                         |                                                 |
|`cusparseDhyb2csc`                                         |                                                 |
|`cusparseChyb2csc`                                         |                                                 |
|`cusparseZhyb2csc`                                         |                                                 |
|`cusparseShyb2csr`                                         |`hipsparseShyb2csr`                              |
|`cusparseDhyb2csr`                                         |`hipsparseDhyb2csr`                              |
|`cusparseChyb2csr`                                         |`hipsparseChyb2csr`                              |
|`cusparseZhyb2csr`                                         |`hipsparseZhyb2csr`                              |
|`cusparseShyb2dense`                                       |                                                 |
|`cusparseDhyb2dense`                                       |                                                 |
|`cusparseChyb2dense`                                       |                                                 |
|`cusparseZhyb2dense`                                       |                                                 |
|`cusparseSnnz`                                             |`cusparseSnnz`                                   |
|`cusparseDnnz`                                             |`cusparseDnnz`                                   |
|`cusparseCnnz`                                             |`cusparseCnnz`                                   |
|`cusparseZnnz`                                             |`cusparseZnnz`                                   |
|`cusparseCreateIdentityPermutation`                        |`hipsparseCreateIdentityPermutation`             |
|`cusparseXcoosort_bufferSizeExt`                           |`hipsparseXcoosort_bufferSizeExt`                |
|`cusparseXcoosortByRow`                                    |`hipsparseXcoosortByRow`                         |
|`cusparseXcoosortByColumn`                                 |`hipsparseXcoosortByColumn`                      |
|`cusparseXcsrsort_bufferSizeExt`                           |`hipsparseXcsrsort_bufferSizeExt`                |
|`cusparseXcsrsort`                                         |`hipsparseXcsrsort`                              |
|`cusparseXcscsort_bufferSizeExt`                           |`hipsparseXcscsort_bufferSizeExt`                |
|`cusparseXcscsort`                                         |`hipsparseXcscsort`                              |
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
|`cusparseScsr2csru`                                        |                                                 |
|`cusparseDcsr2csru`                                        |                                                 |
|`cusparseCcsr2csru`                                        |                                                 |
|`cusparseZcsr2csru`                                        |                                                 |
|`cusparseHpruneDense2csr`                                  |                                                 | 9.0              |
|`cusparseSpruneDense2csr`                                  |                                                 | 9.0              |
|`cusparseDpruneDense2csr`                                  |                                                 | 9.0              |
|`cusparseHpruneDense2csr_bufferSizeExt`                    |                                                 | 9.0              |
|`cusparseSpruneDense2csr_bufferSizeExt`                    |                                                 | 9.0              |
|`cusparseDpruneDense2csr_bufferSizeExt`                    |                                                 | 9.0              |
|`cusparseHpruneDense2csrNnz`                               |                                                 | 9.0              |
|`cusparseSpruneDense2csrNnz`                               |                                                 | 9.0              |
|`cusparseDpruneDense2csrNnz`                               |                                                 | 9.0              |
|`cusparseHpruneCsr2csr`                                    |                                                 | 9.0              |
|`cusparseSpruneCsr2csr`                                    |                                                 | 9.0              |
|`cusparseDpruneCsr2csr`                                    |                                                 | 9.0              |
|`cusparseHpruneCsr2csr_bufferSizeExt`                      |                                                 | 9.0              |
|`cusparseSpruneCsr2csr_bufferSizeExt`                      |                                                 | 9.0              |
|`cusparseDpruneCsr2csr_bufferSizeExt`                      |                                                 | 9.0              |
|`cusparseHpruneCsr2csrNnz`                                 |                                                 | 9.0              |
|`cusparseSpruneCsr2csrNnz`                                 |                                                 | 9.0              |
|`cusparseDpruneCsr2csrNnz`                                 |                                                 | 9.0              |
|`cusparseHpruneDense2csrByPercentage`                      |                                                 | 9.0              |
|`cusparseSpruneDense2csrByPercentage`                      |                                                 | 9.0              |
|`cusparseDpruneDense2csrByPercentage`                      |                                                 | 9.0              |
|`cusparseHpruneDense2csrByPercentage_bufferSizeExt`        |                                                 | 9.0              |
|`cusparseSpruneDense2csrByPercentage_bufferSizeExt`        |                                                 | 9.0              |
|`cusparseDpruneDense2csrByPercentage_bufferSizeExt`        |                                                 | 9.0              |
|`cusparseHpruneDense2csrNnzByPercentage`                   |                                                 | 9.0              |
|`cusparseSpruneDense2csrNnzByPercentage`                   |                                                 | 9.0              |
|`cusparseDpruneDense2csrNnzByPercentage`                   |                                                 | 9.0              |
|`cusparseHpruneCsr2csrByPercentage`                        |                                                 | 9.0              |
|`cusparseSpruneCsr2csrByPercentage`                        |                                                 | 9.0              |
|`cusparseDpruneCsr2csrByPercentage`                        |                                                 | 9.0              |
|`cusparseHpruneCsr2csrByPercentage_bufferSizeExt`          |                                                 | 9.0              |
|`cusparseSpruneCsr2csrByPercentage_bufferSizeExt`          |                                                 | 9.0              |
|`cusparseDpruneCsr2csrByPercentage_bufferSizeExt`          |                                                 | 9.0              |
|`cusparseHpruneCsr2csrNnzByPercentage`                     |                                                 | 9.0              |
|`cusparseSpruneCsr2csrNnzByPercentage`                     |                                                 | 9.0              |
|`cusparseDpruneCsr2csrNnzByPercentage`                     |                                                 | 9.0              |
|`cusparseSnnz_compress`                                    |`hipsparseSnnz_compress`                         | 8.0              |
|`cusparseDnnz_compress`                                    |`hipsparseDnnz_compress`                         | 8.0              |
|`cusparseCnnz_compress`                                    |`hipsparseCnnz_compress`                         | 8.0              |
|`cusparseZnnz_compress`                                    |`hipsparseZnnz_compress`                         | 8.0              |

## **10. cuSPARSE Generic API Reference**

## ***10.1. Generic Sparse API helper functions***

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseCreateCoo`                                        |                                                 | 10.1             |
|`cusparseCreateCooAoS`                                     |                                                 | 10.1             |
|`cusparseCreateCsr`                                        |                                                 | 10.1             |
|`cusparseDestroySpMat`                                     |                                                 | 10.1             |
|`cusparseCooGet`                                           |                                                 | 10.1             |
|`cusparseCooAoSGet`                                        |                                                 | 10.1             |
|`cusparseCsrGet`                                           |                                                 | 10.1             |
|`cusparseSpMatGetFormat`                                   |                                                 | 10.1             |
|`cusparseSpMatGetIndexBase`                                |                                                 | 10.1             |
|`cusparseSpMatGetValues`                                   |                                                 | 10.1             |
|`cusparseSpMatSetValues`                                   |                                                 | 10.1             |
|`cusparseSpMatGetStridedBatch`                             |                                                 | 10.1             |
|`cusparseSpMatSetStridedBatch`                             |                                                 | 10.1             |
|`cusparseSpMatGetNumBatches`                               |                                                 | 10.1             |
|`cusparseSpMatSetNumBatches`                               |                                                 | 10.1             |
|`cusparseCreateSpVec`                                      |                                                 | 10.1             |
|`cusparseDestroySpVec`                                     |                                                 | 10.1             |
|`cusparseSpVecGet`                                         |                                                 | 10.1             |
|`cusparseSpVecGetIndexBase`                                |                                                 | 10.1             |
|`cusparseSpVecGetValues`                                   |                                                 | 10.1             |
|`cusparseSpVecSetValues`                                   |                                                 | 10.1             |

## ***10.2. Generic Dense API helper functions***

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseCreateDnMat`                                      |                                                 | 10.1             |
|`cusparseDestroyDnMat`                                     |                                                 | 10.1             |
|`cusparseDnMatGet`                                         |                                                 | 10.1             |
|`cusparseDnMatGetValues`                                   |                                                 | 10.1             |
|`cusparseDnMatSetValues`                                   |                                                 | 10.1             |
|`cusparseDnMatSetStridedBatch`                             |                                                 | 10.1             |
|`cusparseDnMatGetStridedBatch`                             |                                                 | 10.1             |
|`cusparseCreateDnVec`                                      |                                                 | 10.1             |
|`cusparseDestroyDnVec`                                     |                                                 | 10.1             |
|`cusparseDnVecGet`                                         |                                                 | 10.1             |
|`cusparseDnVecGetValues`                                   |                                                 | 10.1             |
|`cusparseDnVecSetValues`                                   |                                                 | 10.1             |

## ***10.3. Generic SpMM API functions***

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseSpMM`                                             |                                                 | 10.1             |
|`cusparseSpMM_bufferSize`                                  |                                                 | 10.1             |

## ***10.4. Generic SpVV API functions [Undocumented]***

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseSpVV`                                             |                                                 | 10.1             |
|`cusparseSpVV_bufferSize`                                  |                                                 | 10.1             |

## ***10.5. Generic SpMV API functions [Undocumented]***

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cusparseSpMV`                                             |                                                 | 10.1             |
|`cusparseSpMV_bufferSize`                                  |                                                 | 10.1             |

\* CUDA version, in which API has appeared and (optional) last version before abandoning it; no value in case of earlier versions < 7.5.
