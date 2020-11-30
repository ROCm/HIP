# CUSPARSE API supported by HIP

## **4. CUSPARSE Types References**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`CUSPARSE_ACTION_NUMERIC`|  |  |  |`HIPSPARSE_ACTION_NUMERIC`|
|`CUSPARSE_ACTION_SYMBOLIC`|  |  |  |`HIPSPARSE_ACTION_SYMBOLIC`|
|`CUSPARSE_ALG0`| 8.0 |  | 11.0 ||
|`CUSPARSE_ALG1`| 8.0 |  | 11.0 ||
|`CUSPARSE_ALG_MERGE_PATH`| 9.2 |  |  ||
|`CUSPARSE_ALG_NAIVE`| 9.2 |  | 11.0 ||
|`CUSPARSE_COOMM_ALG1`| 10.1 | 11.0 |  ||
|`CUSPARSE_COOMM_ALG2`| 10.1 | 11.0 |  ||
|`CUSPARSE_COOMM_ALG3`| 10.1 | 11.0 |  ||
|`CUSPARSE_COOMV_ALG`| 10.2 |  |  ||
|`CUSPARSE_CSR2CSC_ALG1`| 10.1 |  |  ||
|`CUSPARSE_CSR2CSC_ALG2`| 10.1 |  |  ||
|`CUSPARSE_CSRMM_ALG1`| 10.2 | 11.0 |  ||
|`CUSPARSE_CSRMV_ALG1`| 10.2 |  |  ||
|`CUSPARSE_CSRMV_ALG2`| 10.2 |  |  ||
|`CUSPARSE_DENSETOSPARSE_ALG_DEFAULT`| 11.1 |  |  ||
|`CUSPARSE_DIAG_TYPE_NON_UNIT`|  |  |  |`HIPSPARSE_DIAG_TYPE_NON_UNIT`|
|`CUSPARSE_DIAG_TYPE_UNIT`|  |  |  |`HIPSPARSE_DIAG_TYPE_UNIT`|
|`CUSPARSE_DIRECTION_COLUMN`|  |  |  |`HIPSPARSE_DIRECTION_COLUMN`|
|`CUSPARSE_DIRECTION_ROW`|  |  |  |`HIPSPARSE_DIRECTION_ROW`|
|`CUSPARSE_FILL_MODE_LOWER`|  |  |  |`HIPSPARSE_FILL_MODE_LOWER`|
|`CUSPARSE_FILL_MODE_UPPER`|  |  |  |`HIPSPARSE_FILL_MODE_UPPER`|
|`CUSPARSE_FORMAT_COO`| 10.1 |  |  ||
|`CUSPARSE_FORMAT_COO_AOS`| 10.2 |  |  ||
|`CUSPARSE_FORMAT_CSC`| 10.1 |  |  ||
|`CUSPARSE_FORMAT_CSR`| 10.1 |  |  ||
|`CUSPARSE_HYB_PARTITION_AUTO`|  | 10.2 | 11.0 |`HIPSPARSE_HYB_PARTITION_AUTO`|
|`CUSPARSE_HYB_PARTITION_MAX`|  | 10.2 | 11.0 |`HIPSPARSE_HYB_PARTITION_MAX`|
|`CUSPARSE_HYB_PARTITION_USER`|  | 10.2 | 11.0 |`HIPSPARSE_HYB_PARTITION_USER`|
|`CUSPARSE_INDEX_16U`| 10.1 |  |  ||
|`CUSPARSE_INDEX_32I`| 10.1 |  |  ||
|`CUSPARSE_INDEX_64I`| 10.2 |  |  ||
|`CUSPARSE_INDEX_BASE_ONE`|  |  |  |`HIPSPARSE_INDEX_BASE_ONE`|
|`CUSPARSE_INDEX_BASE_ZERO`|  |  |  |`HIPSPARSE_INDEX_BASE_ZERO`|
|`CUSPARSE_MATRIX_TYPE_GENERAL`|  |  |  |`HIPSPARSE_MATRIX_TYPE_GENERAL`|
|`CUSPARSE_MATRIX_TYPE_HERMITIAN`|  |  |  |`HIPSPARSE_MATRIX_TYPE_HERMITIAN`|
|`CUSPARSE_MATRIX_TYPE_SYMMETRIC`|  |  |  |`HIPSPARSE_MATRIX_TYPE_SYMMETRIC`|
|`CUSPARSE_MATRIX_TYPE_TRIANGULAR`|  |  |  |`HIPSPARSE_MATRIX_TYPE_TRIANGULAR`|
|`CUSPARSE_MM_ALG_DEFAULT`| 10.2 |  |  ||
|`CUSPARSE_MV_ALG_DEFAULT`| 10.2 |  |  ||
|`CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE`|  |  |  |`HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE`|
|`CUSPARSE_OPERATION_NON_TRANSPOSE`|  |  |  |`HIPSPARSE_OPERATION_NON_TRANSPOSE`|
|`CUSPARSE_OPERATION_TRANSPOSE`|  |  |  |`HIPSPARSE_OPERATION_TRANSPOSE`|
|`CUSPARSE_ORDER_COL`| 10.1 |  |  ||
|`CUSPARSE_ORDER_ROW`| 10.1 |  |  ||
|`CUSPARSE_POINTER_MODE_DEVICE`|  |  |  |`HIPSPARSE_POINTER_MODE_DEVICE`|
|`CUSPARSE_POINTER_MODE_HOST`|  |  |  |`HIPSPARSE_POINTER_MODE_HOST`|
|`CUSPARSE_SOLVE_POLICY_NO_LEVEL`|  |  |  |`HIPSPARSE_SOLVE_POLICY_NO_LEVEL`|
|`CUSPARSE_SOLVE_POLICY_USE_LEVEL`|  |  |  |`HIPSPARSE_SOLVE_POLICY_USE_LEVEL`|
|`CUSPARSE_SPARSETODENSE_ALG_DEFAULT`| 11.1 |  |  ||
|`CUSPARSE_SPGEMM_DEFAULT`| 11.0 |  |  ||
|`CUSPARSE_SPMMA_ALG1`| 11.1 |  |  ||
|`CUSPARSE_SPMMA_ALG2`| 11.1 |  |  ||
|`CUSPARSE_SPMMA_ALG3`| 11.1 |  |  ||
|`CUSPARSE_SPMMA_ALG4`| 11.1 |  |  ||
|`CUSPARSE_SPMMA_PREPROCESS`| 11.1 |  |  ||
|`CUSPARSE_SPMM_ALG_DEFAULT`| 11.0 |  |  ||
|`CUSPARSE_SPMM_COO_ALG1`| 11.0 |  |  ||
|`CUSPARSE_SPMM_COO_ALG2`| 11.0 |  |  ||
|`CUSPARSE_SPMM_COO_ALG3`| 11.0 |  |  ||
|`CUSPARSE_SPMM_COO_ALG4`| 11.0 |  |  ||
|`CUSPARSE_SPMM_CSR_ALG1`| 11.0 |  |  ||
|`CUSPARSE_SPMM_CSR_ALG2`| 11.0 |  |  ||
|`CUSPARSE_STATUS_ALLOC_FAILED`|  |  |  |`HIPSPARSE_STATUS_ALLOC_FAILED`|
|`CUSPARSE_STATUS_ARCH_MISMATCH`|  |  |  |`HIPSPARSE_STATUS_ARCH_MISMATCH`|
|`CUSPARSE_STATUS_EXECUTION_FAILED`|  |  |  |`HIPSPARSE_STATUS_EXECUTION_FAILED`|
|`CUSPARSE_STATUS_INSUFFICIENT_RESOURCES`| 11.0 |  |  ||
|`CUSPARSE_STATUS_INTERNAL_ERROR`|  |  |  |`HIPSPARSE_STATUS_INTERNAL_ERROR`|
|`CUSPARSE_STATUS_INVALID_VALUE`|  |  |  |`HIPSPARSE_STATUS_INVALID_VALUE`|
|`CUSPARSE_STATUS_MAPPING_ERROR`|  |  |  |`HIPSPARSE_STATUS_MAPPING_ERROR`|
|`CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`|  |  |  |`HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`|
|`CUSPARSE_STATUS_NOT_INITIALIZED`|  |  |  |`HIPSPARSE_STATUS_NOT_INITIALIZED`|
|`CUSPARSE_STATUS_NOT_SUPPORTED`| 10.2 |  |  ||
|`CUSPARSE_STATUS_SUCCESS`|  |  |  |`HIPSPARSE_STATUS_SUCCESS`|
|`CUSPARSE_STATUS_ZERO_PIVOT`|  |  |  |`HIPSPARSE_STATUS_ZERO_PIVOT`|
|`CUSPARSE_VERSION`| 10.2 |  |  ||
|`CUSPARSE_VER_BUILD`| 10.2 |  |  ||
|`CUSPARSE_VER_MAJOR`| 10.2 |  |  ||
|`CUSPARSE_VER_MINOR`| 10.2 |  |  ||
|`CUSPARSE_VER_PATCH`| 10.2 |  |  ||
|`bsric02Info`|  |  |  ||
|`bsric02Info_t`|  |  |  ||
|`bsrilu02Info`|  |  |  |`bsrilu02Info`|
|`bsrilu02Info_t`|  |  |  |`bsrilu02Info_t`|
|`bsrsm2Info`|  |  |  ||
|`bsrsm2Info_t`|  |  |  ||
|`bsrsv2Info`|  |  |  ||
|`bsrsv2Info_t`|  |  |  ||
|`csrgemm2Info`|  |  |  |`csrgemm2Info`|
|`csrgemm2Info_t`|  |  |  |`csrgemm2Info_t`|
|`csrilu02Info`|  |  |  ||
|`csrilu02Info_t`|  |  |  |`csrilu02Info_t`|
|`csrsm2Info`| 9.2 |  |  |`csrsm2Info`|
|`csrsm2Info_t`| 9.2 |  |  |`csrsm2Info_t`|
|`csrsv2Info`|  |  |  ||
|`csrsv2Info_t`|  |  |  |`csrsv2Info_t`|
|`csru2csrInfo`|  |  |  ||
|`csru2csrInfo_t`|  |  |  ||
|`cusparseAction_t`|  |  |  |`hipsparseAction_t`|
|`cusparseAlgMode_t`| 8.0 |  |  ||
|`cusparseColorInfo`|  |  |  ||
|`cusparseColorInfo_t`|  |  |  ||
|`cusparseContext`|  |  |  ||
|`cusparseCsr2CscAlg_t`| 10.1 |  |  ||
|`cusparseDenseToSparseAlg_t`| 11.1 |  |  ||
|`cusparseDiagType_t`|  |  |  |`hipsparseDiagType_t`|
|`cusparseDirection_t`|  |  |  |`hipsparseDirection_t`|
|`cusparseDnMatDescr`| 10.1 |  |  ||
|`cusparseDnMatDescr_t`| 10.1 |  |  ||
|`cusparseDnVecDescr`| 10.2 |  |  ||
|`cusparseDnVecDescr_t`| 10.2 |  |  ||
|`cusparseFillMode_t`|  |  |  |`hipsparseFillMode_t`|
|`cusparseFormat_t`| 10.1 |  |  ||
|`cusparseHandle_t`|  |  |  |`hipsparseHandle_t`|
|`cusparseHybMat`|  | 10.2 | 11.0 ||
|`cusparseHybMat_t`|  | 10.2 | 11.0 |`hipsparseHybMat_t`|
|`cusparseHybPartition_t`|  | 10.2 | 11.0 |`hipsparseHybPartition_t`|
|`cusparseIndexBase_t`|  |  |  |`hipsparseIndexBase_t`|
|`cusparseIndexType_t`| 10.1 |  |  ||
|`cusparseMatDescr`|  |  |  ||
|`cusparseMatDescr_t`|  |  |  |`hipsparseMatDescr_t`|
|`cusparseMatrixType_t`|  |  |  |`hipsparseMatrixType_t`|
|`cusparseOperation_t`|  |  |  |`hipsparseOperation_t`|
|`cusparseOrder_t`| 10.1 |  |  ||
|`cusparsePointerMode_t`|  |  |  |`hipsparsePointerMode_t`|
|`cusparseSolveAnalysisInfo`|  | 10.2 | 11.0 ||
|`cusparseSolveAnalysisInfo_t`|  | 10.2 | 11.0 ||
|`cusparseSolvePolicy_t`|  |  |  |`hipsparseSolvePolicy_t`|
|`cusparseSpGEMMAlg_t`| 11.0 |  |  ||
|`cusparseSpGEMMDescr`| 11.0 |  |  ||
|`cusparseSpGEMMDescr_t`| 11.0 |  |  ||
|`cusparseSpMMAlg_t`| 10.1 |  |  ||
|`cusparseSpMVAlg_t`| 10.2 |  |  ||
|`cusparseSpMatDescr`| 10.1 |  |  ||
|`cusparseSpMatDescr_t`| 10.1 |  |  ||
|`cusparseSpVecDescr`| 10.2 |  |  ||
|`cusparseSpVecDescr_t`| 10.2 |  |  ||
|`cusparseSparseToDenseAlg_t`| 11.1 |  |  ||
|`cusparseStatus_t`|  |  |  |`hipsparseStatus_t`|
|`pruneInfo`| 9.0 |  |  |`pruneInfo`|
|`pruneInfo_t`| 9.0 |  |  |`pruneInfo_t`|

## **5. CUSPARSE Management Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cusparseCreate`|  |  |  |`hipsparseCreate`|
|`cusparseDestroy`|  |  |  |`hipsparseDestroy`|
|`cusparseGetPointerMode`|  |  |  |`hipsparseGetPointerMode`|
|`cusparseGetStream`|  |  |  |`hipsparseGetStream`|
|`cusparseGetVersion`|  |  |  |`hipsparseGetVersion`|
|`cusparseSetPointerMode`|  |  |  |`hipsparseSetPointerMode`|
|`cusparseSetStream`|  |  |  |`hipsparseSetStream`|

## **6. CUSPARSE Helper Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cusparseCreateBsric02Info`|  |  |  |`hipsparseCreateBsric02Info`|
|`cusparseCreateBsrilu02Info`|  |  |  |`hipsparseCreateBsrilu02Info`|
|`cusparseCreateBsrsm2Info`|  |  |  ||
|`cusparseCreateBsrsv2Info`|  |  |  ||
|`cusparseCreateCsrgemm2Info`|  | 11.0 |  |`hipsparseCreateCsrgemm2Info`|
|`cusparseCreateCsric02Info`|  |  |  |`hipsparseCreateCsric02Info`|
|`cusparseCreateCsrilu02Info`|  |  |  |`hipsparseCreateCsrilu02Info`|
|`cusparseCreateCsrsm2Info`| 10.0 |  |  |`hipsparseCreateCsrsm2Info`|
|`cusparseCreateCsrsv2Info`|  |  |  |`hipsparseCreateCsrsv2Info`|
|`cusparseCreateHybMat`|  | 10.2 | 11.0 |`hipsparseCreateHybMat`|
|`cusparseCreateMatDescr`|  |  |  |`hipsparseCreateMatDescr`|
|`cusparseCreatePruneInfo`| 9.0 |  |  |`hipsparseCreatePruneInfo`|
|`cusparseCreateSolveAnalysisInfo`|  | 10.2 | 11.0 ||
|`cusparseDestroyBsric02Info`|  |  |  |`hipsparseDestroyBsric02Info`|
|`cusparseDestroyBsrilu02Info`|  |  |  |`hipsparseDestroyBsrilu02Info`|
|`cusparseDestroyBsrsm2Info`|  |  |  ||
|`cusparseDestroyBsrsv2Info`|  |  |  ||
|`cusparseDestroyCsrgemm2Info`|  | 11.0 |  |`hipsparseDestroyCsrgemm2Info`|
|`cusparseDestroyCsric02Info`|  |  |  |`hipsparseDestroyCsric02Info`|
|`cusparseDestroyCsrilu02Info`|  |  |  |`hipsparseDestroyCsrilu02Info`|
|`cusparseDestroyCsrsm2Info`| 10.0 |  |  |`hipsparseDestroyCsrsm2Info`|
|`cusparseDestroyCsrsv2Info`|  |  |  |`hipsparseDestroyCsrsv2Info`|
|`cusparseDestroyHybMat`|  | 10.2 | 11.0 |`hipsparseDestroyHybMat`|
|`cusparseDestroyMatDescr`|  |  |  |`hipsparseDestroyMatDescr`|
|`cusparseDestroyPruneInfo`| 9.0 |  |  |`hipsparseDestroyPruneInfo`|
|`cusparseDestroySolveAnalysisInfo`|  | 10.2 | 11.0 ||
|`cusparseGetLevelInfo`|  |  | 11.0 ||
|`cusparseGetMatDiagType`|  |  |  |`hipsparseGetMatDiagType`|
|`cusparseGetMatFillMode`|  |  |  |`hipsparseGetMatFillMode`|
|`cusparseGetMatIndexBase`|  |  |  |`hipsparseGetMatIndexBase`|
|`cusparseGetMatType`|  |  |  |`hipsparseGetMatType`|
|`cusparseSetMatDiagType`|  |  |  |`hipsparseSetMatDiagType`|
|`cusparseSetMatFillMode`|  |  |  |`hipsparseSetMatFillMode`|
|`cusparseSetMatIndexBase`|  |  |  |`hipsparseSetMatIndexBase`|
|`cusparseSetMatType`|  |  |  |`hipsparseSetMatType`|

## **7. CUSPARSE Level 1 Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cusparseCaxpyi`|  | 11.0 |  |`hipsparseCaxpyi`|
|`cusparseCdotci`|  | 10.2 | 11.0 |`hipsparseCdotci`|
|`cusparseCdoti`|  | 10.2 | 11.0 |`hipsparseCdoti`|
|`cusparseCgthr`|  | 11.0 |  |`hipsparseCgthr`|
|`cusparseCgthrz`|  | 11.0 |  |`hipsparseCgthrz`|
|`cusparseCsctr`|  | 11.0 |  |`hipsparseCsctr`|
|`cusparseDaxpyi`|  | 11.0 |  |`hipsparseDaxpyi`|
|`cusparseDdoti`|  | 10.2 | 11.0 |`hipsparseDdoti`|
|`cusparseDgthr`|  | 11.0 |  |`hipsparseDgthr`|
|`cusparseDgthrz`|  | 11.0 |  |`hipsparseDgthrz`|
|`cusparseDroti`|  | 11.0 |  |`hipsparseDroti`|
|`cusparseDsctr`|  | 11.0 |  |`hipsparseDsctr`|
|`cusparseSaxpyi`|  | 11.0 |  |`hipsparseSaxpyi`|
|`cusparseSdoti`|  | 10.2 | 11.0 |`hipsparseSdoti`|
|`cusparseSgthr`|  | 11.0 |  |`hipsparseSgthr`|
|`cusparseSgthrz`|  | 11.0 |  |`hipsparseSgthrz`|
|`cusparseSroti`|  | 11.0 |  |`hipsparseSroti`|
|`cusparseSsctr`|  | 11.0 |  |`hipsparseSsctr`|
|`cusparseZaxpyi`|  | 11.0 |  |`hipsparseZaxpyi`|
|`cusparseZdotci`|  | 10.2 | 11.0 |`hipsparseZdotci`|
|`cusparseZdoti`|  | 10.2 | 11.0 |`hipsparseZdoti`|
|`cusparseZgthr`|  | 11.0 |  |`hipsparseZgthr`|
|`cusparseZgthrz`|  | 11.0 |  |`hipsparseZgthrz`|
|`cusparseZsctr`|  | 11.0 |  |`hipsparseZsctr`|

## **8. CUSPARSE Level 2 Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cusparseCbsrmv`|  |  |  |`hipsparseCbsrmv`|
|`cusparseCbsrsv2_analysis`|  |  |  |`hipsparseCbsrsv2_analysis`|
|`cusparseCbsrsv2_bufferSize`|  |  |  |`hipsparseCbsrsv2_bufferSize`|
|`cusparseCbsrsv2_bufferSizeExt`|  |  |  |`hipsparseCbsrsv2_bufferSizeExt`|
|`cusparseCbsrsv2_solve`|  |  |  |`hipsparseCbsrsv2_solve`|
|`cusparseCbsrxmv`|  |  |  ||
|`cusparseCcsrmv`|  | 10.2 | 11.0 |`hipsparseCcsrmv`|
|`cusparseCcsrmv_mp`| 8.0 | 10.2 | 11.0 ||
|`cusparseCcsrsv2_analysis`|  |  |  |`hipsparseCcsrsv2_analysis`|
|`cusparseCcsrsv2_bufferSize`|  |  |  |`hipsparseCcsrsv2_bufferSize`|
|`cusparseCcsrsv2_bufferSizeExt`|  |  |  |`hipsparseCcsrsv2_bufferSizeExt`|
|`cusparseCcsrsv2_solve`|  |  |  |`hipsparseCcsrsv2_solve`|
|`cusparseCcsrsv_analysis`|  | 10.2 | 11.0 ||
|`cusparseCcsrsv_solve`|  | 10.2 | 11.0 ||
|`cusparseCgemvi`| 7.5 |  |  ||
|`cusparseCgemvi_bufferSize`| 7.5 |  |  ||
|`cusparseChybmv`|  | 10.2 | 11.0 |`hipsparseChybmv`|
|`cusparseChybsv_analysis`|  | 10.2 | 11.0 ||
|`cusparseChybsv_solve`|  | 10.2 | 11.0 ||
|`cusparseCsrmvEx`| 8.0 |  |  ||
|`cusparseCsrmvEx_bufferSize`| 8.0 |  |  ||
|`cusparseCsrsv_analysisEx`| 8.0 | 10.2 | 11.0 ||
|`cusparseCsrsv_solveEx`| 8.0 | 10.2 | 11.0 ||
|`cusparseDbsrmv`|  |  |  |`hipsparseDbsrmv`|
|`cusparseDbsrsv2_analysis`|  |  |  |`hipsparseDbsrsv2_analysis`|
|`cusparseDbsrsv2_bufferSize`|  |  |  |`hipsparseDbsrsv2_bufferSize`|
|`cusparseDbsrsv2_bufferSizeExt`|  |  |  |`hipsparseDbsrsv2_bufferSizeExt`|
|`cusparseDbsrsv2_solve`|  |  |  |`hipsparseDbsrsv2_solve`|
|`cusparseDbsrxmv`|  |  |  ||
|`cusparseDcsrmv`|  | 10.2 | 11.0 |`hipsparseDcsrmv`|
|`cusparseDcsrmv_mp`| 8.0 | 10.2 | 11.0 ||
|`cusparseDcsrsv2_analysis`|  |  |  |`hipsparseDcsrsv2_analysis`|
|`cusparseDcsrsv2_bufferSize`|  |  |  |`hipsparseDcsrsv2_bufferSize`|
|`cusparseDcsrsv2_bufferSizeExt`|  |  |  |`hipsparseDcsrsv2_bufferSizeExt`|
|`cusparseDcsrsv2_solve`|  |  |  |`hipsparseDcsrsv2_solve`|
|`cusparseDcsrsv_analysis`|  | 10.2 | 11.0 ||
|`cusparseDcsrsv_solve`|  | 10.2 | 11.0 ||
|`cusparseDgemvi`| 7.5 |  |  ||
|`cusparseDgemvi_bufferSize`| 7.5 |  |  ||
|`cusparseDhybmv`|  | 10.2 | 11.0 |`hipsparseDhybmv`|
|`cusparseDhybsv_analysis`|  | 10.2 | 11.0 ||
|`cusparseDhybsv_solve`|  | 10.2 | 11.0 ||
|`cusparseSbsrmv`|  |  |  |`hipsparseSbsrmv`|
|`cusparseSbsrsv2_analysis`|  |  |  |`hipsparseSbsrsv2_analysis`|
|`cusparseSbsrsv2_bufferSize`|  |  |  |`hipsparseSbsrsv2_bufferSize`|
|`cusparseSbsrsv2_bufferSizeExt`|  |  |  |`hipsparseSbsrsv2_bufferSizeExt`|
|`cusparseSbsrsv2_solve`|  |  |  |`hipsparseSbsrsv2_solve`|
|`cusparseSbsrxmv`|  |  |  ||
|`cusparseScsrmv`|  | 10.2 | 11.0 |`hipsparseScsrmv`|
|`cusparseScsrmv_mp`| 8.0 | 10.2 | 11.0 ||
|`cusparseScsrsv2_analysis`|  |  |  |`hipsparseScsrsv2_analysis`|
|`cusparseScsrsv2_bufferSize`|  |  |  |`hipsparseScsrsv2_bufferSize`|
|`cusparseScsrsv2_bufferSizeExt`|  |  |  |`hipsparseScsrsv2_bufferSizeExt`|
|`cusparseScsrsv2_solve`|  |  |  |`hipsparseScsrsv2_solve`|
|`cusparseScsrsv_analysis`|  | 10.2 | 11.0 ||
|`cusparseScsrsv_solve`|  | 10.2 | 11.0 ||
|`cusparseSgemvi`| 7.5 |  |  ||
|`cusparseSgemvi_bufferSize`| 7.5 |  |  ||
|`cusparseShybmv`|  | 10.2 | 11.0 |`hipsparseShybmv`|
|`cusparseShybsv_analysis`|  | 10.2 | 11.0 ||
|`cusparseShybsv_solve`|  | 10.2 | 11.0 ||
|`cusparseXbsrsv2_zeroPivot`|  |  |  |`hipsparseXbsrsv2_zeroPivot`|
|`cusparseXcsrsv2_zeroPivot`|  |  |  |`hipsparseXcsrsv2_zeroPivot`|
|`cusparseZbsrmv`|  |  |  |`hipsparseZbsrmv`|
|`cusparseZbsrsv2_analysis`|  |  |  |`hipsparseZbsrsv2_analysis`|
|`cusparseZbsrsv2_bufferSize`|  |  |  |`hipsparseZbsrsv2_bufferSize`|
|`cusparseZbsrsv2_bufferSizeExt`|  |  |  |`hipsparseZbsrsv2_bufferSizeExt`|
|`cusparseZbsrsv2_solve`|  |  |  |`hipsparseZbsrsv2_solve`|
|`cusparseZbsrxmv`|  |  |  ||
|`cusparseZcsrmv`|  | 10.2 | 11.0 |`hipsparseZcsrmv`|
|`cusparseZcsrmv_mp`| 8.0 | 10.2 | 11.0 ||
|`cusparseZcsrsv2_analysis`|  |  |  |`hipsparseZcsrsv2_analysis`|
|`cusparseZcsrsv2_bufferSize`|  |  |  |`hipsparseZcsrsv2_bufferSize`|
|`cusparseZcsrsv2_bufferSizeExt`|  |  |  |`hipsparseZcsrsv2_bufferSizeExt`|
|`cusparseZcsrsv2_solve`|  |  |  |`hipsparseZcsrsv2_solve`|
|`cusparseZcsrsv_analysis`|  | 10.2 | 11.0 ||
|`cusparseZcsrsv_solve`|  | 10.2 | 11.0 ||
|`cusparseZgemvi`| 7.5 |  |  ||
|`cusparseZgemvi_bufferSize`| 7.5 |  |  ||
|`cusparseZhybmv`|  | 10.2 | 11.0 |`hipsparseZhybmv`|
|`cusparseZhybsv_analysis`|  | 10.2 | 11.0 ||
|`cusparseZhybsv_solve`|  | 10.2 | 11.0 ||

## **9. CUSPARSE Level 3 Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cusparseCbsrmm`|  |  |  |`hipsparseCbsrmm`|
|`cusparseCbsrsm2_analysis`|  |  |  ||
|`cusparseCbsrsm2_bufferSize`|  |  |  ||
|`cusparseCbsrsm2_bufferSizeExt`|  |  |  ||
|`cusparseCbsrsm2_solve`|  |  |  ||
|`cusparseCcsrmm`|  | 10.2 | 11.0 |`hipsparseCcsrmm`|
|`cusparseCcsrmm2`|  | 10.2 | 11.0 |`hipsparseCcsrmm2`|
|`cusparseCcsrsm2_analysis`| 10.0 |  |  |`hipsparseCcsrsm2_analysis`|
|`cusparseCcsrsm2_bufferSizeExt`| 10.0 |  |  |`hipsparseCcsrsm2_bufferSizeExt`|
|`cusparseCcsrsm2_solve`| 10.0 |  |  |`hipsparseCcsrsm2_solve`|
|`cusparseCcsrsm_analysis`|  | 10.2 | 11.0 ||
|`cusparseCcsrsm_solve`|  | 10.2 | 11.0 |`hipsparseCcsrsm_solve`|
|`cusparseCgemmi`| 8.0 | 11.0 |  |`hipsparseCgemmi`|
|`cusparseDbsrmm`|  |  |  |`hipsparseDbsrmm`|
|`cusparseDbsrsm2_analysis`|  |  |  ||
|`cusparseDbsrsm2_bufferSize`|  |  |  ||
|`cusparseDbsrsm2_bufferSizeExt`|  |  |  ||
|`cusparseDbsrsm2_solve`|  |  |  ||
|`cusparseDcsrmm`|  | 10.2 | 11.0 |`hipsparseDcsrmm`|
|`cusparseDcsrmm2`|  | 10.2 | 11.0 |`hipsparseDcsrmm2`|
|`cusparseDcsrsm2_analysis`| 10.0 |  |  |`hipsparseDcsrsm2_analysis`|
|`cusparseDcsrsm2_bufferSizeExt`| 10.0 |  |  |`hipsparseDcsrsm2_bufferSizeExt`|
|`cusparseDcsrsm2_solve`| 10.0 |  |  |`hipsparseDcsrsm2_solve`|
|`cusparseDcsrsm_analysis`|  | 10.2 | 11.0 ||
|`cusparseDcsrsm_solve`|  | 10.2 | 11.0 |`hipsparseDcsrsm_solve`|
|`cusparseDgemmi`| 8.0 | 11.0 |  |`hipsparseDgemmi`|
|`cusparseSbsrmm`|  |  |  |`hipsparseSbsrmm`|
|`cusparseSbsrsm2_analysis`|  |  |  ||
|`cusparseSbsrsm2_bufferSize`|  |  |  ||
|`cusparseSbsrsm2_bufferSizeExt`|  |  |  ||
|`cusparseSbsrsm2_solve`|  |  |  ||
|`cusparseScsrmm`|  | 10.2 | 11.0 |`hipsparseScsrmm`|
|`cusparseScsrmm2`|  | 10.2 | 11.0 |`hipsparseScsrmm2`|
|`cusparseScsrsm2_analysis`| 10.0 |  |  |`hipsparseScsrsm2_analysis`|
|`cusparseScsrsm2_bufferSizeExt`| 10.0 |  |  |`hipsparseScsrsm2_bufferSizeExt`|
|`cusparseScsrsm2_solve`| 10.0 |  |  |`hipsparseScsrsm2_solve`|
|`cusparseScsrsm_analysis`|  | 10.2 | 11.0 ||
|`cusparseScsrsm_solve`|  | 10.2 | 11.0 |`hipsparseScsrsm_solve`|
|`cusparseSgemmi`| 8.0 | 11.0 |  |`hipsparseSgemmi`|
|`cusparseXbsrsm2_zeroPivot`|  |  |  ||
|`cusparseXcsrsm2_zeroPivot`| 10.0 |  |  |`hipsparseXcsrsm2_zeroPivot`|
|`cusparseZbsrmm`|  |  |  |`hipsparseZbsrmm`|
|`cusparseZbsrsm2_analysis`|  |  |  ||
|`cusparseZbsrsm2_bufferSize`|  |  |  ||
|`cusparseZbsrsm2_bufferSizeExt`|  |  |  ||
|`cusparseZbsrsm2_solve`|  |  |  ||
|`cusparseZcsrmm`|  | 10.2 | 11.0 |`hipsparseZcsrmm`|
|`cusparseZcsrmm2`|  | 10.2 | 11.0 |`hipsparseZcsrmm2`|
|`cusparseZcsrsm2_analysis`| 10.0 |  |  |`hipsparseZcsrsm2_analysis`|
|`cusparseZcsrsm2_bufferSizeExt`| 10.0 |  |  |`hipsparseZcsrsm2_bufferSizeExt`|
|`cusparseZcsrsm2_solve`| 10.0 |  |  |`hipsparseZcsrsm2_solve`|
|`cusparseZcsrsm_analysis`|  | 10.2 | 11.0 ||
|`cusparseZcsrsm_solve`|  | 10.2 | 11.0 |`hipsparseZcsrsm_solve`|
|`cusparseZgemmi`| 8.0 | 11.0 |  |`hipsparseZgemmi`|

## **10. CUSPARSE Extra Function Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cusparseCcsrgeam`|  | 10.2 | 11.0 |`hipsparseCcsrgeam`|
|`cusparseCcsrgeam2`| 10.0 |  |  |`hipsparseCcsrgeam2`|
|`cusparseCcsrgeam2_bufferSizeExt`| 10.0 |  |  |`hipsparseCcsrgeam2_bufferSizeExt`|
|`cusparseCcsrgemm`|  | 10.2 | 11.0 |`hipsparseCcsrgemm`|
|`cusparseCcsrgemm2`|  | 11.0 |  |`hipsparseCcsrgemm2`|
|`cusparseCcsrgemm2_bufferSizeExt`|  | 11.0 |  |`hipsparseCcsrgemm2_bufferSizeExt`|
|`cusparseDcsrgeam`|  | 10.2 | 11.0 |`hipsparseDcsrgeam`|
|`cusparseDcsrgeam2`| 10.0 |  |  |`hipsparseDcsrgeam2`|
|`cusparseDcsrgeam2_bufferSizeExt`| 10.0 |  |  |`hipsparseDcsrgeam2_bufferSizeExt`|
|`cusparseDcsrgemm`|  | 10.2 | 11.0 |`hipsparseDcsrgemm`|
|`cusparseDcsrgemm2`|  | 11.0 |  |`hipsparseDcsrgemm2`|
|`cusparseDcsrgemm2_bufferSizeExt`|  | 11.0 |  |`hipsparseDcsrgemm2_bufferSizeExt`|
|`cusparseScsrgeam`|  | 10.2 | 11.0 |`hipsparseScsrgeam`|
|`cusparseScsrgeam2`| 10.0 |  |  |`hipsparseScsrgeam2`|
|`cusparseScsrgeam2_bufferSizeExt`| 10.0 |  |  |`hipsparseScsrgeam2_bufferSizeExt`|
|`cusparseScsrgemm`|  | 10.2 | 11.0 |`hipsparseScsrgemm`|
|`cusparseScsrgemm2`|  | 11.0 |  |`hipsparseScsrgemm2`|
|`cusparseScsrgemm2_bufferSizeExt`|  | 11.0 |  |`hipsparseScsrgemm2_bufferSizeExt`|
|`cusparseXcsrgeam2Nnz`| 10.0 |  |  |`hipsparseXcsrgeam2Nnz`|
|`cusparseXcsrgeamNnz`|  | 10.2 | 11.0 |`hipsparseXcsrgeamNnz`|
|`cusparseXcsrgemm2Nnz`|  | 11.0 |  |`hipsparseXcsrgemm2Nnz`|
|`cusparseXcsrgemmNnz`|  | 10.2 | 11.0 |`hipsparseXcsrgemmNnz`|
|`cusparseZcsrgeam`|  | 10.2 | 11.0 |`hipsparseZcsrgeam`|
|`cusparseZcsrgeam2`| 10.0 |  |  |`hipsparseZcsrgeam2`|
|`cusparseZcsrgeam2_bufferSizeExt`| 10.0 |  |  |`hipsparseZcsrgeam2_bufferSizeExt`|
|`cusparseZcsrgemm`|  | 10.2 | 11.0 |`hipsparseZcsrgemm`|
|`cusparseZcsrgemm2`|  | 11.0 |  |`hipsparseZcsrgemm2`|
|`cusparseZcsrgemm2_bufferSizeExt`|  | 11.0 |  |`hipsparseZcsrgemm2_bufferSizeExt`|

## **11. CUSPARSE Preconditioners Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cusparseCbsric02`|  |  |  |`hipsparseCbsric02`|
|`cusparseCbsric02_analysis`|  |  |  |`hipsparseCbsric02_analysis`|
|`cusparseCbsric02_bufferSize`|  |  |  |`hipsparseCbsric02_bufferSize`|
|`cusparseCbsric02_bufferSizeExt`|  |  |  ||
|`cusparseCbsrilu02`|  |  |  |`hipsparseCbsrilu02`|
|`cusparseCbsrilu02_analysis`|  |  |  |`hipsparseCbsrilu02_analysis`|
|`cusparseCbsrilu02_bufferSize`|  |  |  |`hipsparseCbsrilu02_bufferSize`|
|`cusparseCbsrilu02_bufferSizeExt`|  |  |  ||
|`cusparseCbsrilu02_numericBoost`|  |  |  |`hipsparseCbsrilu02_numericBoost`|
|`cusparseCcsric0`|  | 10.2 | 11.0 ||
|`cusparseCcsric02`|  |  |  |`hipsparseCcsric02`|
|`cusparseCcsric02_analysis`|  |  |  |`hipsparseCcsric02_analysis`|
|`cusparseCcsric02_bufferSize`|  |  |  |`hipsparseCcsric02_bufferSize`|
|`cusparseCcsric02_bufferSizeExt`|  |  |  |`hipsparseCcsric02_bufferSizeExt`|
|`cusparseCcsrilu0`|  | 10.2 | 11.0 ||
|`cusparseCcsrilu02`|  |  |  |`hipsparseCcsrilu02`|
|`cusparseCcsrilu02_analysis`|  |  |  |`hipsparseCcsrilu02_analysis`|
|`cusparseCcsrilu02_bufferSize`|  |  |  |`hipsparseCcsrilu02_bufferSize`|
|`cusparseCcsrilu02_bufferSizeExt`|  |  |  |`hipsparseCcsrilu02_bufferSizeExt`|
|`cusparseCcsrilu02_numericBoost`|  |  |  |`hipsparseCcsrilu02_numericBoost`|
|`cusparseCgpsvInterleavedBatch`| 9.2 |  |  ||
|`cusparseCgpsvInterleavedBatch_bufferSizeExt`| 9.2 |  |  ||
|`cusparseCgtsv`|  | 10.2 | 11.0 ||
|`cusparseCgtsv2`| 9.0 |  |  ||
|`cusparseCgtsv2StridedBatch`|  |  |  ||
|`cusparseCgtsv2StridedBatch_bufferSizeExt`|  |  |  ||
|`cusparseCgtsv2_bufferSizeExt`| 9.0 |  |  ||
|`cusparseCgtsv2_nopivot`| 9.0 |  |  ||
|`cusparseCgtsv2_nopivot_bufferSizeExt`| 9.0 |  |  ||
|`cusparseCgtsvInterleavedBatch`| 9.2 |  |  ||
|`cusparseCgtsvInterleavedBatch_bufferSizeExt`| 9.2 |  |  ||
|`cusparseCgtsvStridedBatch`|  | 10.2 | 11.0 ||
|`cusparseCgtsv_nopivot`|  | 10.2 | 11.0 ||
|`cusparseCsrilu0Ex`| 8.0 | 10.2 | 11.0 ||
|`cusparseDbsric02`|  |  |  |`hipsparseDbsric02`|
|`cusparseDbsric02_analysis`|  |  |  |`hipsparseDbsric02_analysis`|
|`cusparseDbsric02_bufferSize`|  |  |  |`hipsparseDbsric02_bufferSize`|
|`cusparseDbsric02_bufferSizeExt`|  |  |  ||
|`cusparseDbsrilu02`|  |  |  |`hipsparseDbsrilu02`|
|`cusparseDbsrilu02_analysis`|  |  |  |`hipsparseDbsrilu02_analysis`|
|`cusparseDbsrilu02_bufferSize`|  |  |  |`hipsparseDbsrilu02_bufferSize`|
|`cusparseDbsrilu02_bufferSizeExt`|  |  |  ||
|`cusparseDbsrilu02_numericBoost`|  |  |  |`hipsparseDbsrilu02_numericBoost`|
|`cusparseDcsric0`|  | 10.2 | 11.0 ||
|`cusparseDcsric02`|  |  |  |`hipsparseDcsric02`|
|`cusparseDcsric02_analysis`|  |  |  |`hipsparseDcsric02_analysis`|
|`cusparseDcsric02_bufferSize`|  |  |  |`hipsparseDcsric02_bufferSize`|
|`cusparseDcsric02_bufferSizeExt`|  |  |  |`hipsparseDcsric02_bufferSizeExt`|
|`cusparseDcsrilu0`|  | 10.2 | 11.0 ||
|`cusparseDcsrilu02`|  |  |  |`hipsparseDcsrilu02`|
|`cusparseDcsrilu02_analysis`|  |  |  |`hipsparseDcsrilu02_analysis`|
|`cusparseDcsrilu02_bufferSize`|  |  |  |`hipsparseDcsrilu02_bufferSize`|
|`cusparseDcsrilu02_bufferSizeExt`|  |  |  |`hipsparseDcsrilu02_bufferSizeExt`|
|`cusparseDcsrilu02_numericBoost`|  |  |  |`hipsparseDcsrilu02_numericBoost`|
|`cusparseDgpsvInterleavedBatch`| 9.2 |  |  ||
|`cusparseDgpsvInterleavedBatch_bufferSizeExt`| 9.2 |  |  ||
|`cusparseDgtsv`|  | 10.2 | 11.0 ||
|`cusparseDgtsv2`| 9.0 |  |  ||
|`cusparseDgtsv2StridedBatch`|  |  |  ||
|`cusparseDgtsv2StridedBatch_bufferSizeExt`|  |  |  ||
|`cusparseDgtsv2_bufferSizeExt`| 9.0 |  |  ||
|`cusparseDgtsv2_nopivot`| 9.0 |  |  ||
|`cusparseDgtsv2_nopivot_bufferSizeExt`| 9.0 |  |  ||
|`cusparseDgtsvInterleavedBatch`| 9.2 |  |  ||
|`cusparseDgtsvInterleavedBatch_bufferSizeExt`| 9.2 |  |  ||
|`cusparseDgtsvStridedBatch`|  | 10.2 | 11.0 ||
|`cusparseDgtsv_nopivot`|  | 10.2 | 11.0 ||
|`cusparseSbsric02`|  |  |  |`hipsparseSbsric02`|
|`cusparseSbsric02_analysis`|  |  |  |`hipsparseSbsric02_analysis`|
|`cusparseSbsric02_bufferSize`|  |  |  |`hipsparseSbsric02_bufferSize`|
|`cusparseSbsric02_bufferSizeExt`|  |  |  ||
|`cusparseSbsrilu02`|  |  |  |`hipsparseSbsrilu02`|
|`cusparseSbsrilu02_analysis`|  |  |  |`hipsparseSbsrilu02_analysis`|
|`cusparseSbsrilu02_bufferSize`|  |  |  |`hipsparseSbsrilu02_bufferSize`|
|`cusparseSbsrilu02_bufferSizeExt`|  |  |  ||
|`cusparseSbsrilu02_numericBoost`|  |  |  |`hipsparseSbsrilu02_numericBoost`|
|`cusparseScsric0`|  | 10.2 | 11.0 ||
|`cusparseScsric02`|  |  |  |`hipsparseScsric02`|
|`cusparseScsric02_analysis`|  |  |  |`hipsparseScsric02_analysis`|
|`cusparseScsric02_bufferSize`|  |  |  |`hipsparseScsric02_bufferSize`|
|`cusparseScsric02_bufferSizeExt`|  |  |  |`hipsparseScsric02_bufferSizeExt`|
|`cusparseScsrilu0`|  | 10.2 | 11.0 ||
|`cusparseScsrilu02`|  |  |  |`hipsparseScsrilu02`|
|`cusparseScsrilu02_analysis`|  |  |  |`hipsparseScsrilu02_analysis`|
|`cusparseScsrilu02_bufferSize`|  |  |  |`hipsparseScsrilu02_bufferSize`|
|`cusparseScsrilu02_bufferSizeExt`|  |  |  |`hipsparseScsrilu02_bufferSizeExt`|
|`cusparseScsrilu02_numericBoost`|  |  |  |`hipsparseScsrilu02_numericBoost`|
|`cusparseSgpsvInterleavedBatch`| 9.2 |  |  ||
|`cusparseSgpsvInterleavedBatch_bufferSizeExt`| 9.2 |  |  ||
|`cusparseSgtsv`|  | 10.2 | 11.0 ||
|`cusparseSgtsv2`| 9.0 |  |  ||
|`cusparseSgtsv2StridedBatch`| 9.0 |  |  ||
|`cusparseSgtsv2StridedBatch_bufferSizeExt`| 9.0 |  |  ||
|`cusparseSgtsv2_bufferSizeExt`| 9.0 |  |  ||
|`cusparseSgtsv2_nopivot`| 9.0 |  |  ||
|`cusparseSgtsv2_nopivot_bufferSizeExt`| 9.0 |  |  ||
|`cusparseSgtsvInterleavedBatch`| 9.2 |  |  ||
|`cusparseSgtsvInterleavedBatch_bufferSizeExt`| 9.2 |  |  ||
|`cusparseSgtsvStridedBatch`|  | 10.2 | 11.0 ||
|`cusparseSgtsv_nopivot`|  | 10.2 | 11.0 ||
|`cusparseXbsric02_zeroPivot`|  |  |  |`hipsparseXbsric02_zeroPivot`|
|`cusparseXbsrilu02_zeroPivot`|  |  |  |`hipsparseXbsrilu02_zeroPivot`|
|`cusparseXcsric02_zeroPivot`|  |  |  |`hipsparseXcsric02_zeroPivot`|
|`cusparseXcsrilu02_zeroPivot`|  |  |  |`hipsparseXcsrilu02_zeroPivot`|
|`cusparseZbsric02`|  |  |  |`hipsparseZbsric02`|
|`cusparseZbsric02_analysis`|  |  |  |`hipsparseZbsric02_analysis`|
|`cusparseZbsric02_bufferSize`|  |  |  |`hipsparseZbsric02_bufferSize`|
|`cusparseZbsric02_bufferSizeExt`|  |  |  ||
|`cusparseZbsrilu02`|  |  |  |`hipsparseZbsrilu02`|
|`cusparseZbsrilu02_analysis`|  |  |  |`hipsparseZbsrilu02_analysis`|
|`cusparseZbsrilu02_bufferSize`|  |  |  |`hipsparseZbsrilu02_bufferSize`|
|`cusparseZbsrilu02_bufferSizeExt`|  |  |  ||
|`cusparseZbsrilu02_numericBoost`|  |  |  |`hipsparseZbsrilu02_numericBoost`|
|`cusparseZcsric0`|  | 10.2 | 11.0 ||
|`cusparseZcsric02`|  |  |  |`hipsparseZcsric02`|
|`cusparseZcsric02_analysis`|  |  |  |`hipsparseZcsric02_analysis`|
|`cusparseZcsric02_bufferSize`|  |  |  |`hipsparseZcsric02_bufferSize`|
|`cusparseZcsric02_bufferSizeExt`|  |  |  |`hipsparseZcsric02_bufferSizeExt`|
|`cusparseZcsrilu0`|  | 10.2 | 11.0 ||
|`cusparseZcsrilu02`|  |  |  |`hipsparseZcsrilu02`|
|`cusparseZcsrilu02_analysis`|  |  |  |`hipsparseZcsrilu02_analysis`|
|`cusparseZcsrilu02_bufferSize`|  |  |  |`hipsparseZcsrilu02_bufferSize`|
|`cusparseZcsrilu02_bufferSizeExt`|  |  |  |`hipsparseZcsrilu02_bufferSizeExt`|
|`cusparseZcsrilu02_numericBoost`|  |  |  |`hipsparseZcsrilu02_numericBoost`|
|`cusparseZgpsvInterleavedBatch`| 9.2 |  |  ||
|`cusparseZgpsvInterleavedBatch_bufferSizeExt`| 9.2 |  |  ||
|`cusparseZgtsv`|  | 10.2 | 11.0 ||
|`cusparseZgtsv2`| 9.0 |  |  ||
|`cusparseZgtsv2StridedBatch`|  |  |  ||
|`cusparseZgtsv2StridedBatch_bufferSizeExt`|  |  |  ||
|`cusparseZgtsv2_bufferSizeExt`| 9.0 |  |  ||
|`cusparseZgtsv2_nopivot`| 9.0 |  |  ||
|`cusparseZgtsv2_nopivot_bufferSizeExt`| 9.0 |  |  ||
|`cusparseZgtsvInterleavedBatch`| 9.2 |  |  ||
|`cusparseZgtsvInterleavedBatch_bufferSizeExt`| 9.2 |  |  ||
|`cusparseZgtsvStridedBatch`|  | 10.2 | 11.0 ||
|`cusparseZgtsv_nopivot`|  | 10.2 | 11.0 ||

## **12. CUSPARSE Reorderings Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cusparseCcsrcolor`|  |  |  ||
|`cusparseDcsrcolor`|  |  |  ||
|`cusparseScsrcolor`|  |  |  ||
|`cusparseZcsrcolor`|  |  |  ||

## **13. CUSPARSE Format Conversion Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cusparseCbsr2csr`|  |  |  |`hipsparseCbsr2csr`|
|`cusparseCcsc2dense`|  | 11.1 |  |`hipsparseCcsc2dense`|
|`cusparseCcsc2hyb`|  | 10.2 | 11.0 ||
|`cusparseCcsr2bsr`|  |  |  |`hipsparseCcsr2bsr`|
|`cusparseCcsr2csc`|  | 10.2 | 11.0 |`hipsparseCcsr2csc`|
|`cusparseCcsr2csr_compress`| 8.0 |  |  |`hipsparseCcsr2csr_compress`|
|`cusparseCcsr2csru`|  |  |  ||
|`cusparseCcsr2dense`|  | 11.1 |  |`hipsparseCcsr2dense`|
|`cusparseCcsr2gebsr`|  |  |  |`hipsparseCcsr2gebsr`|
|`cusparseCcsr2gebsr_bufferSize`|  |  |  |`hipsparseCcsr2gebsr_bufferSize`|
|`cusparseCcsr2gebsr_bufferSizeExt`|  |  |  ||
|`cusparseCcsr2hyb`|  | 10.2 | 11.0 |`hipsparseCcsr2hyb`|
|`cusparseCcsru2csr_bufferSizeExt`|  |  |  ||
|`cusparseCdense2csc`|  | 11.1 |  |`hipsparseCdense2csc`|
|`cusparseCdense2csr`|  | 11.1 |  |`hipsparseCdense2csr`|
|`cusparseCdense2hyb`|  | 10.2 | 11.0 ||
|`cusparseCgebsr2csr`|  |  |  |`hipsparseCgebsr2csr`|
|`cusparseCgebsr2gebsc`|  |  |  |`hipsparseCgebsr2gebsc`|
|`cusparseCgebsr2gebsc_bufferSize`|  |  |  |`hipsparseCgebsr2gebsc_bufferSize`|
|`cusparseCgebsr2gebsc_bufferSizeExt`|  |  |  ||
|`cusparseCgebsr2gebsr`|  |  |  |`hipsparseCgebsr2gebsr`|
|`cusparseCgebsr2gebsr_bufferSize`|  |  |  |`hipsparseCgebsr2gebsr_bufferSize`|
|`cusparseCgebsr2gebsr_bufferSizeExt`|  |  |  ||
|`cusparseChyb2csc`|  | 10.2 | 11.0 ||
|`cusparseChyb2csr`|  | 10.2 | 11.0 |`hipsparseChyb2csr`|
|`cusparseChyb2dense`|  | 10.2 | 11.0 ||
|`cusparseCnnz`|  |  |  |`hipsparseCnnz`|
|`cusparseCnnz_compress`| 8.0 |  |  |`hipsparseCnnz_compress`|
|`cusparseCreateCsru2csrInfo`|  |  |  ||
|`cusparseCreateIdentityPermutation`|  |  |  |`hipsparseCreateIdentityPermutation`|
|`cusparseCsr2cscEx`| 8.0 | 10.2 | 11.0 ||
|`cusparseCsr2cscEx2`| 10.1 |  |  ||
|`cusparseCsr2cscEx2_bufferSize`| 10.1 |  |  ||
|`cusparseDbsr2csr`|  |  |  |`hipsparseDbsr2csr`|
|`cusparseDcsc2dense`|  | 11.1 |  |`hipsparseDcsc2dense`|
|`cusparseDcsc2hyb`|  | 10.2 | 11.0 ||
|`cusparseDcsr2bsr`|  |  |  |`hipsparseDcsr2bsr`|
|`cusparseDcsr2csc`|  | 10.2 | 11.0 |`hipsparseDcsr2csc`|
|`cusparseDcsr2csr_compress`| 8.0 |  |  |`hipsparseDcsr2csr_compress`|
|`cusparseDcsr2csru`|  |  |  ||
|`cusparseDcsr2dense`|  | 11.1 |  |`hipsparseDcsr2dense`|
|`cusparseDcsr2gebsr`|  |  |  |`hipsparseDcsr2gebsr`|
|`cusparseDcsr2gebsr_bufferSize`|  |  |  |`hipsparseDcsr2gebsr_bufferSize`|
|`cusparseDcsr2gebsr_bufferSizeExt`|  |  |  ||
|`cusparseDcsr2hyb`|  | 10.2 | 11.0 |`hipsparseDcsr2hyb`|
|`cusparseDcsru2csr_bufferSizeExt`|  |  |  ||
|`cusparseDdense2csc`|  | 11.1 |  |`hipsparseDdense2csc`|
|`cusparseDdense2csr`|  | 11.1 |  |`hipsparseDdense2csr`|
|`cusparseDdense2hyb`|  | 10.2 | 11.0 ||
|`cusparseDestroyCsru2csrInfo`|  |  |  ||
|`cusparseDgebsr2csr`|  |  |  |`hipsparseDgebsr2csr`|
|`cusparseDgebsr2gebsc`|  |  |  |`hipsparseDgebsr2gebsc`|
|`cusparseDgebsr2gebsc_bufferSize`|  |  |  |`hipsparseDgebsr2gebsc_bufferSize`|
|`cusparseDgebsr2gebsc_bufferSizeExt`|  |  |  ||
|`cusparseDgebsr2gebsr`|  |  |  |`hipsparseDgebsr2gebsr`|
|`cusparseDgebsr2gebsr_bufferSize`|  |  |  |`hipsparseDgebsr2gebsr_bufferSize`|
|`cusparseDgebsr2gebsr_bufferSizeExt`|  |  |  ||
|`cusparseDhyb2csc`|  | 10.2 | 11.0 ||
|`cusparseDhyb2csr`|  | 10.2 | 11.0 |`hipsparseDhyb2csr`|
|`cusparseDhyb2dense`|  | 10.2 | 11.0 ||
|`cusparseDnnz`|  |  |  |`hipsparseDnnz`|
|`cusparseDnnz_compress`| 8.0 |  |  |`hipsparseDnnz_compress`|
|`cusparseDpruneCsr2csr`| 9.0 |  |  |`hipsparseDpruneCsr2csr`|
|`cusparseDpruneCsr2csrByPercentage`| 9.0 |  |  |`hipsparseDpruneCsr2csrByPercentage`|
|`cusparseDpruneCsr2csrByPercentage_bufferSizeExt`| 9.0 |  |  |`hipsparseDpruneCsr2csrByPercentage_bufferSizeExt`|
|`cusparseDpruneCsr2csrNnz`| 9.0 |  |  |`hipsparseDpruneCsr2csrNnz`|
|`cusparseDpruneCsr2csrNnzByPercentage`| 9.0 |  |  |`hipsparseDpruneCsr2csrNnzByPercentage`|
|`cusparseDpruneCsr2csr_bufferSizeExt`| 9.0 |  |  |`hipsparseDpruneCsr2csr_bufferSizeExt`|
|`cusparseDpruneDense2csr`| 9.0 |  |  |`hipsparseDpruneDense2csr`|
|`cusparseDpruneDense2csrByPercentage`| 9.0 |  |  |`hipsparseDpruneDense2csrByPercentage`|
|`cusparseDpruneDense2csrByPercentage_bufferSizeExt`| 9.0 |  |  |`hipsparseDpruneDense2csrByPercentage_bufferSizeExt`|
|`cusparseDpruneDense2csrNnz`| 9.0 |  |  |`hipsparseDpruneDense2csrNnz`|
|`cusparseDpruneDense2csrNnzByPercentage`| 9.0 |  |  |`hipsparseDpruneDense2csrNnzByPercentage`|
|`cusparseDpruneDense2csr_bufferSizeExt`| 9.0 |  |  |`hipsparseDpruneDense2csr_bufferSizeExt`|
|`cusparseHpruneCsr2csr`| 9.0 |  |  ||
|`cusparseHpruneCsr2csrByPercentage`| 9.0 |  |  ||
|`cusparseHpruneCsr2csrByPercentage_bufferSizeExt`| 9.0 |  |  ||
|`cusparseHpruneCsr2csrNnz`| 9.0 |  |  ||
|`cusparseHpruneCsr2csrNnzByPercentage`| 9.0 |  |  ||
|`cusparseHpruneCsr2csr_bufferSizeExt`| 9.0 |  |  ||
|`cusparseHpruneDense2csr`| 9.0 |  |  ||
|`cusparseHpruneDense2csrByPercentage`| 9.0 |  |  ||
|`cusparseHpruneDense2csrByPercentage_bufferSizeExt`| 9.0 |  |  ||
|`cusparseHpruneDense2csrNnz`| 9.0 |  |  ||
|`cusparseHpruneDense2csrNnzByPercentage`| 9.0 |  |  ||
|`cusparseHpruneDense2csr_bufferSizeExt`| 9.0 |  |  ||
|`cusparseSbsr2csr`|  |  |  |`hipsparseSbsr2csr`|
|`cusparseScsc2dense`|  | 11.1 |  |`hipsparseScsc2dense`|
|`cusparseScsc2hyb`|  | 10.2 | 11.0 ||
|`cusparseScsr2bsr`|  |  |  |`hipsparseScsr2bsr`|
|`cusparseScsr2csc`|  | 10.2 | 11.0 |`hipsparseScsr2csc`|
|`cusparseScsr2csr_compress`| 8.0 |  |  |`hipsparseScsr2csr_compress`|
|`cusparseScsr2csru`|  |  |  ||
|`cusparseScsr2dense`|  | 11.1 |  |`hipsparseScsr2dense`|
|`cusparseScsr2gebsr`|  |  |  |`hipsparseScsr2gebsr`|
|`cusparseScsr2gebsr_bufferSize`|  |  |  |`hipsparseScsr2gebsr_bufferSize`|
|`cusparseScsr2gebsr_bufferSizeExt`|  |  |  ||
|`cusparseScsr2hyb`|  | 10.2 | 11.0 |`hipsparseScsr2hyb`|
|`cusparseScsru2csr_bufferSizeExt`|  |  |  ||
|`cusparseSdense2csc`|  | 11.1 |  |`hipsparseSdense2csc`|
|`cusparseSdense2csr`|  | 11.1 |  |`hipsparseSdense2csr`|
|`cusparseSdense2hyb`|  | 10.2 | 11.0 ||
|`cusparseSgebsr2csr`|  |  |  |`hipsparseSgebsr2csr`|
|`cusparseSgebsr2gebsc`|  |  |  |`hipsparseSgebsr2gebsc`|
|`cusparseSgebsr2gebsc_bufferSize`|  |  |  |`hipsparseSgebsr2gebsc_bufferSize`|
|`cusparseSgebsr2gebsc_bufferSizeExt`|  |  |  ||
|`cusparseSgebsr2gebsr`|  |  |  |`hipsparseSgebsr2gebsr`|
|`cusparseSgebsr2gebsr_bufferSize`|  |  |  |`hipsparseSgebsr2gebsr_bufferSize`|
|`cusparseSgebsr2gebsr_bufferSizeExt`|  |  |  ||
|`cusparseShyb2csc`|  | 10.2 | 11.0 ||
|`cusparseShyb2csr`|  | 10.2 | 11.0 |`hipsparseShyb2csr`|
|`cusparseShyb2dense`|  | 10.2 | 11.0 ||
|`cusparseSnnz`|  |  |  |`hipsparseSnnz`|
|`cusparseSnnz_compress`| 8.0 |  |  |`hipsparseSnnz_compress`|
|`cusparseSpruneCsr2csr`| 9.0 |  |  |`hipsparseSpruneCsr2csr`|
|`cusparseSpruneCsr2csrByPercentage`| 9.0 |  |  |`hipsparseSpruneCsr2csrByPercentage`|
|`cusparseSpruneCsr2csrByPercentage_bufferSizeExt`| 9.0 |  |  |`hipsparseSpruneCsr2csrByPercentage_bufferSizeExt`|
|`cusparseSpruneCsr2csrNnz`| 9.0 |  |  |`hipsparseSpruneCsr2csrNnz`|
|`cusparseSpruneCsr2csrNnzByPercentage`| 9.0 |  |  |`hipsparseSpruneCsr2csrNnzByPercentage`|
|`cusparseSpruneCsr2csr_bufferSizeExt`| 9.0 |  |  |`hipsparseSpruneCsr2csr_bufferSizeExt`|
|`cusparseSpruneDense2csr`| 9.0 |  |  |`hipsparseSpruneDense2csr`|
|`cusparseSpruneDense2csrByPercentage`| 9.0 |  |  |`hipsparseSpruneDense2csrByPercentage`|
|`cusparseSpruneDense2csrByPercentage_bufferSizeExt`| 9.0 |  |  |`hipsparseSpruneDense2csrByPercentage_bufferSizeExt`|
|`cusparseSpruneDense2csrNnz`| 9.0 |  |  |`hipsparseSpruneDense2csrNnz`|
|`cusparseSpruneDense2csrNnzByPercentage`| 9.0 |  |  |`hipsparseSpruneDense2csrNnzByPercentage`|
|`cusparseSpruneDense2csr_bufferSizeExt`| 9.0 |  |  |`hipsparseSpruneDense2csr_bufferSizeExt`|
|`cusparseXcoo2csr`|  |  |  |`hipsparseXcoo2csr`|
|`cusparseXcoosortByColumn`|  |  |  |`hipsparseXcoosortByColumn`|
|`cusparseXcoosortByRow`|  |  |  |`hipsparseXcoosortByRow`|
|`cusparseXcoosort_bufferSizeExt`|  |  |  |`hipsparseXcoosort_bufferSizeExt`|
|`cusparseXcscsort`|  |  |  |`hipsparseXcscsort`|
|`cusparseXcscsort_bufferSizeExt`|  |  |  |`hipsparseXcscsort_bufferSizeExt`|
|`cusparseXcsr2bsrNnz`|  |  |  |`hipsparseXcsr2bsrNnz`|
|`cusparseXcsr2coo`|  |  |  |`hipsparseXcsr2coo`|
|`cusparseXcsr2gebsrNnz`|  |  |  |`hipsparseXcsr2gebsrNnz`|
|`cusparseXcsrsort`|  |  |  |`hipsparseXcsrsort`|
|`cusparseXcsrsort_bufferSizeExt`|  |  |  |`hipsparseXcsrsort_bufferSizeExt`|
|`cusparseXgebsr2csr`|  |  |  ||
|`cusparseXgebsr2gebsrNnz`|  |  |  |`hipsparseXgebsr2gebsrNnz`|
|`cusparseZbsr2csr`|  |  |  |`hipsparseZbsr2csr`|
|`cusparseZcsc2dense`|  | 11.1 |  |`hipsparseZcsc2dense`|
|`cusparseZcsc2hyb`|  | 10.2 | 11.0 ||
|`cusparseZcsr2bsr`|  |  |  |`hipsparseZcsr2bsr`|
|`cusparseZcsr2csc`|  | 10.2 | 11.0 |`hipsparseZcsr2csc`|
|`cusparseZcsr2csr_compress`| 8.0 |  |  |`hipsparseZcsr2csr_compress`|
|`cusparseZcsr2csru`|  |  |  ||
|`cusparseZcsr2dense`|  | 11.1 |  |`hipsparseZcsr2dense`|
|`cusparseZcsr2gebsr`|  |  |  |`hipsparseZcsr2gebsr`|
|`cusparseZcsr2gebsr_bufferSize`|  |  |  |`hipsparseZcsr2gebsr_bufferSize`|
|`cusparseZcsr2gebsr_bufferSizeExt`|  |  |  ||
|`cusparseZcsr2hyb`|  | 10.2 | 11.0 |`hipsparseZcsr2hyb`|
|`cusparseZcsru2csr_bufferSizeExt`|  |  |  ||
|`cusparseZdense2csc`|  | 11.1 |  |`hipsparseZdense2csc`|
|`cusparseZdense2csr`|  | 11.1 |  |`hipsparseZdense2csr`|
|`cusparseZdense2hyb`|  | 10.2 | 11.0 ||
|`cusparseZgebsr2csr`|  |  |  |`hipsparseZgebsr2csr`|
|`cusparseZgebsr2gebsc`|  |  |  |`hipsparseZgebsr2gebsc`|
|`cusparseZgebsr2gebsc_bufferSize`|  |  |  |`hipsparseZgebsr2gebsc_bufferSize`|
|`cusparseZgebsr2gebsc_bufferSizeExt`|  |  |  ||
|`cusparseZgebsr2gebsr`|  |  |  |`hipsparseZgebsr2gebsr`|
|`cusparseZgebsr2gebsr_bufferSize`|  |  |  |`hipsparseZgebsr2gebsr_bufferSize`|
|`cusparseZgebsr2gebsr_bufferSizeExt`|  |  |  ||
|`cusparseZhyb2csc`|  | 10.2 | 11.0 ||
|`cusparseZhyb2csr`|  | 10.2 | 11.0 |`hipsparseZhyb2csr`|
|`cusparseZhyb2dense`|  | 10.2 | 11.0 ||
|`cusparseZnnz`|  |  |  |`hipsparseZnnz`|
|`cusparseZnnz_compress`| 8.0 |  |  |`hipsparseZnnz_compress`|

## **14. CUSPARSE Generic API Reference**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cusparseAxpby`| 11.0 |  |  ||
|`cusparseConstrainedGeMM`| 10.2 |  |  ||
|`cusparseConstrainedGeMM_bufferSize`| 10.2 |  |  ||
|`cusparseCooAoSGet`| 10.2 |  |  ||
|`cusparseCooGet`| 10.1 |  |  ||
|`cusparseCooSetPointers`| 11.1 |  |  ||
|`cusparseCooSetStridedBatch`| 11.0 |  |  ||
|`cusparseCreateCoo`| 10.1 |  |  ||
|`cusparseCreateCooAoS`| 10.2 |  |  ||
|`cusparseCreateCsc`| 11.1 |  |  ||
|`cusparseCreateCsr`| 10.2 |  |  ||
|`cusparseCreateDnMat`| 10.1 |  |  ||
|`cusparseCreateDnVec`| 10.2 |  |  ||
|`cusparseCreateSpVec`| 10.2 |  |  ||
|`cusparseCscSetPointers`| 11.1 |  |  ||
|`cusparseCsrGet`| 10.2 |  |  ||
|`cusparseCsrSetPointers`| 11.0 |  |  ||
|`cusparseCsrSetStridedBatch`| 11.0 |  |  ||
|`cusparseDenseToSparse_analysis`| 11.1 |  |  ||
|`cusparseDenseToSparse_bufferSize`| 11.1 |  |  ||
|`cusparseDenseToSparse_convert`| 11.1 |  |  ||
|`cusparseDestroyDnMat`| 10.1 |  |  ||
|`cusparseDestroyDnVec`| 10.2 |  |  ||
|`cusparseDestroySpMat`| 10.1 |  |  ||
|`cusparseDestroySpVec`| 10.2 |  |  ||
|`cusparseDnMatGet`| 10.1 |  |  ||
|`cusparseDnMatGetStridedBatch`| 10.1 |  |  ||
|`cusparseDnMatGetValues`| 10.2 |  |  ||
|`cusparseDnMatSetStridedBatch`| 10.1 |  |  ||
|`cusparseDnMatSetValues`| 10.2 |  |  ||
|`cusparseDnVecGet`| 10.2 |  |  ||
|`cusparseDnVecGetValues`| 10.2 |  |  ||
|`cusparseDnVecSetValues`| 10.2 |  |  ||
|`cusparseGather`| 11.0 |  |  ||
|`cusparseRot`| 11.0 |  |  ||
|`cusparseScatter`| 11.0 |  |  ||
|`cusparseSpGEMM_compute`| 11.0 |  |  ||
|`cusparseSpGEMM_copy`| 11.0 |  |  ||
|`cusparseSpGEMM_createDescr`| 11.0 |  |  ||
|`cusparseSpGEMM_destroyDescr`| 11.0 |  |  ||
|`cusparseSpGEMM_workEstimation`|  |  |  ||
|`cusparseSpMM`| 10.1 |  |  ||
|`cusparseSpMM_bufferSize`| 10.1 |  |  ||
|`cusparseSpMV`| 10.2 |  |  ||
|`cusparseSpMV_bufferSize`| 10.2 |  |  ||
|`cusparseSpMatGetFormat`| 10.1 |  |  ||
|`cusparseSpMatGetIndexBase`| 10.1 |  |  ||
|`cusparseSpMatGetNumBatches`| 10.1 |  | 10.2 ||
|`cusparseSpMatGetSize`| 11.0 |  |  ||
|`cusparseSpMatGetStridedBatch`| 10.2 |  |  ||
|`cusparseSpMatGetValues`| 10.2 |  |  ||
|`cusparseSpMatSetNumBatches`| 10.1 |  | 10.2 ||
|`cusparseSpMatSetStridedBatch`| 10.2 |  |  ||
|`cusparseSpMatSetValues`| 10.2 |  |  ||
|`cusparseSpVV`| 10.2 |  |  ||
|`cusparseSpVV_bufferSize`| 10.2 |  |  ||
|`cusparseSpVecGet`| 10.2 |  |  ||
|`cusparseSpVecGetIndexBase`| 10.2 |  |  ||
|`cusparseSpVecGetValues`| 10.2 |  |  ||
|`cusparseSpVecSetValues`| 10.2 |  |  ||
|`cusparseSparseToDense`| 11.1 |  |  ||
|`cusparseSparseToDense_bufferSize`| 11.1 |  |  ||


\* A - Added, D - Deprecated, R - Removed