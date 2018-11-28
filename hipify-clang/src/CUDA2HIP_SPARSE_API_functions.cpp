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

  // 8. cuSPARSE Level 3 Function Reference
  {"cusparseScsrmm",                         {"hipsparseScsrmm",                         CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrmm",                         {"hipsparseDcsrmm",                         CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrmm",                         {"hipsparseCcsrmm",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrmm",                         {"hipsparseZcsrmm",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrmm2",                        {"hipsparseScsrmm2",                        CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrmm2",                        {"hipsparseDcsrmm2",                        CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrmm2",                        {"hipsparseCcsrmm2",                        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrmm2",                        {"hipsparseZcsrmm2",                        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsm_analysis",                {"hipsparseScsrsm_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsm_analysis",                {"hipsparseDcsrsm_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsm_analysis",                {"hipsparseCcsrsm_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsm_analysis",                {"hipsparseZcsrsm_analysis",                CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsm_solve",                   {"hipsparseScsrsm_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsm_solve",                   {"hipsparseDcsrsm_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsm_solve",                   {"hipsparseCcsrsm_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsm_solve",                   {"hipsparseZcsrsm_solve",                   CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsm2_bufferSizeExt",          {"hipsparseScsrsm2_bufferSizeExt",          CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsm2_bufferSizeExt",          {"hipsparseDcsrsm2_bufferSizeExt",          CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsm2_bufferSizeExt",          {"hipsparseCcsrsm2_bufferSizeExt",          CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsm2_bufferSizeExt",          {"hipsparseZcsrsm2_bufferSizeExt",          CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsm2_analysis",               {"hipsparseScsrsm2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsm2_analysis",               {"hipsparseDcsrsm2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsm2_analysis",               {"hipsparseCcsrsm2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsm2_analysis",               {"hipsparseZcsrsm2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsm2_solve",                  {"hipsparseScsrsm2_solve",                  CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsm2_solve",                  {"hipsparseDcsrsm2_solve",                  CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsm2_solve",                  {"hipsparseCcsrsm2_solve",                  CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsm2_solve",                  {"hipsparseZcsrsm2_solve",                  CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcsrsm2_zeroPivot",              {"hipsparseXcsrsm2_zeroPivot",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrmm",                         {"hipsparseSbsrmm",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrmm",                         {"hipsparseDbsrmm",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrmm",                         {"hipsparseCbsrmm",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrmm",                         {"hipsparseZbsrmm",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrsm2_bufferSize",             {"hipsparseCbsrsm2_bufferSize",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrsm2_bufferSize",             {"hipsparseDbsrsm2_bufferSize",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrsm2_bufferSize",             {"hipsparseCbsrsm2_bufferSize",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrsm2_bufferSize",             {"hipsparseZbsrsm2_bufferSize",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrsm2_analysis",               {"hipsparseSbsrsm2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrsm2_analysis",               {"hipsparseDbsrsm2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrsm2_analysis",               {"hipsparseCbsrsm2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrsm2_analysis",               {"hipsparseZbsrsm2_analysis",               CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrsm2_solve",                  {"hipsparseSbsrsm2_solve",                  CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrsm2_solve",                  {"hipsparseDbsrsm2_solve",                  CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrsm2_solve",                  {"hipsparseCbsrsm2_solve",                  CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrsm2_solve",                  {"hipsparseZbsrsm2_solve",                  CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXbsrsm2_zeroPivot",              {"hipsparseXbsrsm2_zeroPivot",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgemmi",                         {"hipsparseSgemmi",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgemmi",                         {"hipsparseDgemmi",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgemmi",                         {"hipsparseCgemmi",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgemmi",                         {"hipsparseZgemmi",                         CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 9. cuSPARSE Extra Function Reference
  {"cusparseXcsrgeamNnz",                    {"hipsparseXcsrgeamNnz",                    CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseScsrgeam",                       {"hipsparseScsrgeam",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrgeam",                       {"hipsparseDcsrgeam",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrgeam",                       {"hipsparseCcsrgeam",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrgeam2_bufferSizeExt",        {"hipsparseScsrgeam2_bufferSizeExt",        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrgeam2_bufferSizeExt",        {"hipsparseDcsrgeam2_bufferSizeExt",        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrgeam2_bufferSizeExt",        {"hipsparseCcsrgeam2_bufferSizeExt",        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrgeam2_bufferSizeExt",        {"hipsparseZcsrgeam2_bufferSizeExt",        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcsrgemmNnz",                    {"hipsparseXcsrgemmNnz",                    CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseScsrgemm",                       {"hipsparseScsrgemm",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrgemm",                       {"hipsparseDcsrgemm",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrgemm",                       {"hipsparseCcsrgemm",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrgemm2_bufferSizeExt",        {"hipsparseScsrgemm2_bufferSizeExt",        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrgemm2_bufferSizeExt",        {"hipsparseDcsrgemm2_bufferSizeExt",        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrgemm2_bufferSizeExt",        {"hipsparseCcsrgemm2_bufferSizeExt",        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrgemm2_bufferSizeExt",        {"hipsparseZcsrgemm2_bufferSizeExt",        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 10. cuSPARSE Preconditioners Reference
  // 10.1. Incomplete Cholesky Factorization : level 0
  {"cusparseScsric0",                        {"hipsparseScsric0",                        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsric0",                        {"hipsparseDcsric0",                        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsric0",                        {"hipsparseCcsric0",                        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsric0",                        {"hipsparseZcsric0",                        CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsric02_bufferSize",            {"hipsparseScsric02_bufferSize",            CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsric02_bufferSize",            {"hipsparseDcsric02_bufferSize",            CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsric02_bufferSize",            {"hipsparseCcsric02_bufferSize",            CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsric02_bufferSize",            {"hipsparseZcsric02_bufferSize",            CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsric02_analysis",              {"hipsparseScsric02_analysis",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsric02_analysis",              {"hipsparseDcsric02_analysis",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsric02_analysis",              {"hipsparseCcsric02_analysis",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsric02_analysis",              {"hipsparseZcsric02_analysis",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsric02",                       {"hipsparseScsric02",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsric02",                       {"hipsparseDcsric02",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsric02",                       {"hipsparseCcsric02",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsric02",                       {"hipsparseZcsric02",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcsric02_zeroPivot",             {"hipsparseXcsric02_zeroPivot",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsric02_analysis",              {"hipsparseSbsric02_analysis",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsric02_analysis",              {"hipsparseDbsric02_analysis",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsric02_analysis",              {"hipsparseCbsric02_analysis",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsric02_analysis",              {"hipsparseZbsric02_analysis",              CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsric02",                       {"hipsparseSbsric02",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsric02",                       {"hipsparseDbsric02",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsric02",                       {"hipsparseCbsric02",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsric02",                       {"hipsparseZbsric02",                       CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXbsric02_zeroPivot",             {"hipsparseXbsric02_zeroPivot",             CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
};
