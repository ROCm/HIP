#include "CUDA2HIP.h"

// Maps the names of CUDA SPARSE API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_SPARSE_FUNCTION_MAP{
  // 5. cuSPARSE Helper Function Reference
  {"cusparseCreate",                         {"hipsparseCreate",                         CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCreateSolveAnalysisInfo",        {"hipsparseCreateSolveAnalysisInfo",        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateHybMat",                   {"hipsparseCreateHybMat",                   CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCreateMatDescr",                 {"hipsparseCreateMatDescr",                 CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDestroy",                        {"hipsparseDestroy",                        CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDestroySolveAnalysisInfo",       {"hipsparseDestroySolveAnalysisInfo",       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyHybMat",                  {"hipsparseDestroyHybMat",                  CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDestroyMatDescr",                {"hipsparseDestroyMatDescr",                CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetLevelInfo",                   {"hipsparseGetLevelInfo",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseGetMatDiagType",                 {"hipsparseGetMatDiagType",                 CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetMatFillMode",                 {"hipsparseGetMatFillMode",                 CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetMatIndexBase",                {"hipsparseGetMatIndexBase",                CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetMatType",                     {"hipsparseGetMatType",                     CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetPointerMode",                 {"hipsparseGetPointerMode",                 CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetVersion",                     {"hipsparseGetVersion",                     CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetMatDiagType",                 {"hipsparseSetMatDiagType",                 CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetMatFillMode",                 {"hipsparseSetMatFillMode",                 CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetMatIndexBase",                {"hipsparseSetMatIndexBase",                CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetMatType",                     {"hipsparseSetMatType",                     CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetPointerMode",                 {"hipsparseSetPointerMode",                 CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetStream",                      {"hipsparseSetStream",                      CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetStream",                      {"hipsparseGetStream",                      CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCreateCsrsv2Info",               {"hipsparseCreateCsrsv2Info",               CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDestroyCsrsv2Info",              {"hipsparseDestroyCsrsv2Info",              CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCreateCsrsm2Info",               {"hipsparseCreateCsrsm2Info",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyCsrsm2Info",              {"hipsparseDestroyCsrsm2Info",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateCsric02Info",              {"hipsparseCreateCsric02Info",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyCsric02Info",             {"hipsparseDestroyCsric02Info",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateCsrilu02Info",             {"hipsparseCreateCsrilu02Info",             CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDestroyCsrilu02Info",            {"hipsparseDestroyCsrilu02Info",            CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCreateBsrsv2Info",               {"hipsparseCreateBsrsv2Info",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyBsrsv2Info",              {"hipsparseDestroyBsrsv2Info",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateBsrsm2Info",               {"hipsparseCreateBsrsm2Info",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyBsrsm2Info",              {"hipsparseDestroyBsrsm2Info",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateBsric02Inf",               {"hipsparseCreateBsric02Inf",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyBsric02Info",             {"hipsparseDestroyBsric02Info",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateBsrilu02Info",             {"hipsparseCreateBsrilu02Info",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyBsrilu02Info",            {"hipsparseDestroyBsrilu02Info",            CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateCsrgemm2Info",             {"hipsparseCreateCsrgemm2Info",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyCsrgemm2Info",            {"hipsparseDestroyCsrgemm2Info",            CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreatePruneInfo",                {"hipsparseCreatePruneInfo",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyPruneInfo",               {"hipsparseDestroyPruneInfo",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 6. cuSPARSE Level 1 Function Reference
  {"cusparseSaxpyi",                         {"hipsparseSaxpyi",                         CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDaxpyi",                         {"hipsparseDaxpyi",                         CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCaxpyi",                         {"hipsparseCaxpyi",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZaxpyi",                         {"hipsparseZaxpyi",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSdoti",                          {"hipsparseSdoti",                          CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDdoti",                          {"hipsparseDdoti",                          CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCdoti",                          {"hipsparseCdoti",                          CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZdoti",                          {"hipsparseZdoti",                          CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCdotci",                         {"hipsparseCdotci",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZdotci",                         {"hipsparseZdotci",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgthr",                          {"hipsparseSgthr",                          CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDgthr",                          {"hipsparseDgthr",                          CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCgthr",                          {"hipsparseCgthr",                          CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgthr",                          {"hipsparseZgthr",                          CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgthrz",                         {"hipsparseSgthrz",                         CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDgthrz",                         {"hipsparseDgthrz",                         CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCgthrz",                         {"hipsparseCgthrz",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgthrz",                         {"hipsparseZgthrz",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSroti",                          {"hipsparseSroti",                          CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDroti",                          {"hipsparseDroti",                          CONV_LIB_FUNC, API_SPARSE}},

  {"cusparseSsctr",                          {"hipsparseSsctr",                          CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDsctr",                          {"hipsparseDsctr",                          CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCsctr",                          {"hipsparseCsctr",                          CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZsctr",                          {"hipsparseZsctr",                          CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 7. cuSPARSE Level 2 Function Reference
  {"cusparseSbsrmv",                         {"hipsparseSbsrmv",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrmv",                         {"hipsparseDbsrmv",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrmv",                         {"hipsparseCbsrmv",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrmv",                         {"hipsparseZbsrmv",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrxmv",                        {"hipsparseSbsrxmv",                        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrxmv",                        {"hipsparseDbsrxmv",                        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrxmv",                        {"hipsparseCbsrxmv",                        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrxmv",                        {"hipsparseZbsrxmv",                        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrmv",                         {"hipsparseScsrmv",                         CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrmv",                         {"hipsparseDcsrmv",                         CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrmv",                         {"hipsparseCcsrmv",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrmv",                         {"hipsparseZcsrmv",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCsrmvEx",                        {"hipsparseCsrmvEx",                        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCsrmvEx_bufferSize",             {"hipsparseCsrmvEx_bufferSize",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrmv_mp",                      {"hipsparseScsrmv_mp",                      CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrmv_mp",                      {"hipsparseDcsrmv_mp",                      CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrmv_mp",                      {"hipsparseCcsrmv_mp",                      CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrmv_mp",                      {"hipsparseZcsrmv_mp",                      CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgemvi",                         {"hipsparseSgemvi",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgemvi",                         {"hipsparseDgemvi",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgemvi",                         {"hipsparseCgemvi",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgemvi",                         {"hipsparseZgemvi",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgemvi_bufferSize",              {"hipsparseSgemvi_bufferSize",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgemvi_bufferSize",              {"hipsparseDgemvi_bufferSize",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgemvi_bufferSize",              {"hipsparseCgemvi_bufferSize",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgemvi_bufferSize",              {"hipsparseZgemvi_bufferSize",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrsv2_bufferSize",             {"hipsparseSbsrsv2_bufferSize",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrsv2_bufferSize",             {"hipsparseDbsrsv2_bufferSize",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrsv2_bufferSize",             {"hipsparseCbsrsv2_bufferSize",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrsv2_bufferSize",             {"hipsparseZbsrsv2_bufferSize",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrsv2_analysis",               {"hipsparseSbsrsv2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrsv2_analysis",               {"hipsparseDbsrsv2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrsv2_analysis",               {"hipsparseCbsrsv2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrsv2_analysis",               {"hipsparseZbsrsv2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv_solve",                   {"hipsparseScsrsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsv_solve",                   {"hipsparseDcsrsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsv_solve",                   {"hipsparseCcsrsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv_solve",                   {"hipsparseZcsrsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXbsrsv2_zeroPivot",              {"hipsparseXbsrsv2_zeroPivot",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv_analysis",                {"hipsparseScsrsv_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsv_analysis",                {"hipsparseDcsrsv_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsv_analysis",                {"hipsparseCcsrsv_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv_analysis",                {"hipsparseZcsrsv_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCsrsv_analysisEx",               {"hipsparseCsrsv_analysisEx",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv_solve",                   {"hipsparseScsrsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsv_solve",                   {"hipsparseDcsrsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsv_solve",                   {"hipsparseCcsrsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv_solve",                   {"hipsparseZcsrsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCsrsv_solveEx",                  {"hipsparseCsrsv_solveEx",                  CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv2_bufferSize",             {"hipsparseScsrsv2_bufferSize",             CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrsv2_bufferSize",             {"hipsparseDcsrsv2_bufferSize",             CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrsv2_bufferSize",             {"hipsparseCcsrsv2_bufferSize",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv2_bufferSize",             {"hipsparseZcsrsv2_bufferSize",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv2_analysis",               {"hipsparseScsrsv2_analysis",               CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrsv2_analysis",               {"hipsparseDcsrsv2_analysis",               CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrsv2_analysis",               {"hipsparseCcsrsv2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv2_analysis",               {"hipsparseZcsrsv2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv2_solve",                  {"hipsparseScsrsv2_solve",                  CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrsv2_solve",                  {"hipsparseDcsrsv2_solve",                  CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrsv2_solve",                  {"hipsparseCcsrsv2_solve",                  CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv2_solve",                  {"hipsparseZcsrsv2_solve",                  CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcsrsv2_zeroPivot",              {"hipsparseXcsrsv2_zeroPivot",              CONV_LIB_FUNC, API_SPARSE}},

  {"cusparseShybmv",                         {"hipsparseShybmv",                         CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDhybmv",                         {"hipsparseDhybmv",                         CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseChybmv",                         {"hipsparseChybmv",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZhybmv",                         {"hipsparseZhybmv",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseShybsv_analysis",                {"hipsparseShybsv_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDhybsv_analysis",                {"hipsparseDhybsv_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseChybsv_analysis",                {"hipsparseChybsv_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZhybsv_analysis",                {"hipsparseZhybsv_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseShybsv_solve",                   {"hipsparseShybsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDhybsv_solve",                   {"hipsparseDhybsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseChybsv_solve",                   {"hipsparseChybsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZhybsv_solve",                   {"hipsparseZhybsv_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
};
