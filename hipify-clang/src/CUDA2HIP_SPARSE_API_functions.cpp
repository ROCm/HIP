/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "CUDA2HIP.h"

// Maps the names of CUDA SPARSE API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_SPARSE_FUNCTION_MAP{
  // 5. cuSPARSE Helper Function Reference
  {"cusparseCreate",                              {"hipsparseCreate",                              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCreateSolveAnalysisInfo",             {"hipsparseCreateSolveAnalysisInfo",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateHybMat",                        {"hipsparseCreateHybMat",                        "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCreateMatDescr",                      {"hipsparseCreateMatDescr",                      "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDestroy",                             {"hipsparseDestroy",                             "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDestroySolveAnalysisInfo",            {"hipsparseDestroySolveAnalysisInfo",            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyHybMat",                       {"hipsparseDestroyHybMat",                       "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDestroyMatDescr",                     {"hipsparseDestroyMatDescr",                     "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetLevelInfo",                        {"hipsparseGetLevelInfo",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseGetMatDiagType",                      {"hipsparseGetMatDiagType",                      "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetMatFillMode",                      {"hipsparseGetMatFillMode",                      "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetMatIndexBase",                     {"hipsparseGetMatIndexBase",                     "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetMatType",                          {"hipsparseGetMatType",                          "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetPointerMode",                      {"hipsparseGetPointerMode",                      "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetVersion",                          {"hipsparseGetVersion",                          "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetMatDiagType",                      {"hipsparseSetMatDiagType",                      "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetMatFillMode",                      {"hipsparseSetMatFillMode",                      "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetMatIndexBase",                     {"hipsparseSetMatIndexBase",                     "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetMatType",                          {"hipsparseSetMatType",                          "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetPointerMode",                      {"hipsparseSetPointerMode",                      "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseSetStream",                           {"hipsparseSetStream",                           "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseGetStream",                           {"hipsparseGetStream",                           "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCreateCsrsv2Info",                    {"hipsparseCreateCsrsv2Info",                    "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDestroyCsrsv2Info",                   {"hipsparseDestroyCsrsv2Info",                   "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCreateCsrsm2Info",                    {"hipsparseCreateCsrsm2Info",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyCsrsm2Info",                   {"hipsparseDestroyCsrsm2Info",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateCsric02Info",                   {"hipsparseCreateCsric02Info",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyCsric02Info",                  {"hipsparseDestroyCsric02Info",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateCsrilu02Info",                  {"hipsparseCreateCsrilu02Info",                  "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDestroyCsrilu02Info",                 {"hipsparseDestroyCsrilu02Info",                 "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCreateBsrsv2Info",                    {"hipsparseCreateBsrsv2Info",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyBsrsv2Info",                   {"hipsparseDestroyBsrsv2Info",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateBsrsm2Info",                    {"hipsparseCreateBsrsm2Info",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyBsrsm2Info",                   {"hipsparseDestroyBsrsm2Info",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateBsric02Inf",                    {"hipsparseCreateBsric02Inf",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyBsric02Info",                  {"hipsparseDestroyBsric02Info",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateBsrilu02Info",                  {"hipsparseCreateBsrilu02Info",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyBsrilu02Info",                 {"hipsparseDestroyBsrilu02Info",                 "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreateCsrgemm2Info",                  {"hipsparseCreateCsrgemm2Info",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyCsrgemm2Info",                 {"hipsparseDestroyCsrgemm2Info",                 "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCreatePruneInfo",                     {"hipsparseCreatePruneInfo",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyPruneInfo",                    {"hipsparseDestroyPruneInfo",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 6. cuSPARSE Level 1 Function Reference
  {"cusparseSaxpyi",                              {"hipsparseSaxpyi",                              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDaxpyi",                              {"hipsparseDaxpyi",                              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCaxpyi",                              {"hipsparseCaxpyi",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZaxpyi",                              {"hipsparseZaxpyi",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSdoti",                               {"hipsparseSdoti",                               "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDdoti",                               {"hipsparseDdoti",                               "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCdoti",                               {"hipsparseCdoti",                               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZdoti",                               {"hipsparseZdoti",                               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCdotci",                              {"hipsparseCdotci",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZdotci",                              {"hipsparseZdotci",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgthr",                               {"hipsparseSgthr",                               "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDgthr",                               {"hipsparseDgthr",                               "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCgthr",                               {"hipsparseCgthr",                               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgthr",                               {"hipsparseZgthr",                               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgthrz",                              {"hipsparseSgthrz",                              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDgthrz",                              {"hipsparseDgthrz",                              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCgthrz",                              {"hipsparseCgthrz",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgthrz",                              {"hipsparseZgthrz",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSroti",                               {"hipsparseSroti",                               "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDroti",                               {"hipsparseDroti",                               "", CONV_LIB_FUNC, API_SPARSE}},

  {"cusparseSsctr",                               {"hipsparseSsctr",                               "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDsctr",                               {"hipsparseDsctr",                               "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCsctr",                               {"hipsparseCsctr",                               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZsctr",                               {"hipsparseZsctr",                               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 7. cuSPARSE Level 2 Function Reference
  {"cusparseSbsrmv",                              {"hipsparseSbsrmv",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrmv",                              {"hipsparseDbsrmv",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrmv",                              {"hipsparseCbsrmv",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrmv",                              {"hipsparseZbsrmv",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrxmv",                             {"hipsparseSbsrxmv",                             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrxmv",                             {"hipsparseDbsrxmv",                             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrxmv",                             {"hipsparseCbsrxmv",                             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrxmv",                             {"hipsparseZbsrxmv",                             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrmv",                              {"hipsparseScsrmv",                              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrmv",                              {"hipsparseDcsrmv",                              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrmv",                              {"hipsparseCcsrmv",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrmv",                              {"hipsparseZcsrmv",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCsrmvEx",                             {"hipsparseCsrmvEx",                             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCsrmvEx_bufferSize",                  {"hipsparseCsrmvEx_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrmv_mp",                           {"hipsparseScsrmv_mp",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrmv_mp",                           {"hipsparseDcsrmv_mp",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrmv_mp",                           {"hipsparseCcsrmv_mp",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrmv_mp",                           {"hipsparseZcsrmv_mp",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgemvi",                              {"hipsparseSgemvi",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgemvi",                              {"hipsparseDgemvi",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgemvi",                              {"hipsparseCgemvi",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgemvi",                              {"hipsparseZgemvi",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgemvi_bufferSize",                   {"hipsparseSgemvi_bufferSize",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgemvi_bufferSize",                   {"hipsparseDgemvi_bufferSize",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgemvi_bufferSize",                   {"hipsparseCgemvi_bufferSize",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgemvi_bufferSize",                   {"hipsparseZgemvi_bufferSize",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrsv2_bufferSize",                  {"hipsparseSbsrsv2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrsv2_bufferSize",                  {"hipsparseDbsrsv2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrsv2_bufferSize",                  {"hipsparseCbsrsv2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrsv2_bufferSize",                  {"hipsparseZbsrsv2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrsv2_analysis",                    {"hipsparseSbsrsv2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrsv2_analysis",                    {"hipsparseDbsrsv2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrsv2_analysis",                    {"hipsparseCbsrsv2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrsv2_analysis",                    {"hipsparseZbsrsv2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv_solve",                        {"hipsparseScsrsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsv_solve",                        {"hipsparseDcsrsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsv_solve",                        {"hipsparseCcsrsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv_solve",                        {"hipsparseZcsrsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXbsrsv2_zeroPivot",                   {"hipsparseXbsrsv2_zeroPivot",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv_analysis",                     {"hipsparseScsrsv_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsv_analysis",                     {"hipsparseDcsrsv_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsv_analysis",                     {"hipsparseCcsrsv_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv_analysis",                     {"hipsparseZcsrsv_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCsrsv_analysisEx",                    {"hipsparseCsrsv_analysisEx",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv_solve",                        {"hipsparseScsrsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsv_solve",                        {"hipsparseDcsrsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsv_solve",                        {"hipsparseCcsrsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv_solve",                        {"hipsparseZcsrsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCsrsv_solveEx",                       {"hipsparseCsrsv_solveEx",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv2_bufferSize",                  {"hipsparseScsrsv2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrsv2_bufferSize",                  {"hipsparseDcsrsv2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrsv2_bufferSize",                  {"hipsparseCcsrsv2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv2_bufferSize",                  {"hipsparseZcsrsv2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv2_analysis",                    {"hipsparseScsrsv2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrsv2_analysis",                    {"hipsparseDcsrsv2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrsv2_analysis",                    {"hipsparseCcsrsv2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv2_analysis",                    {"hipsparseZcsrsv2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsv2_solve",                       {"hipsparseScsrsv2_solve",                       "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrsv2_solve",                       {"hipsparseDcsrsv2_solve",                       "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrsv2_solve",                       {"hipsparseCcsrsv2_solve",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsv2_solve",                       {"hipsparseZcsrsv2_solve",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcsrsv2_zeroPivot",                   {"hipsparseXcsrsv2_zeroPivot",                   "", CONV_LIB_FUNC, API_SPARSE}},

  {"cusparseShybmv",                              {"hipsparseShybmv",                              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDhybmv",                              {"hipsparseDhybmv",                              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseChybmv",                              {"hipsparseChybmv",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZhybmv",                              {"hipsparseZhybmv",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseShybsv_analysis",                     {"hipsparseShybsv_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDhybsv_analysis",                     {"hipsparseDhybsv_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseChybsv_analysis",                     {"hipsparseChybsv_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZhybsv_analysis",                     {"hipsparseZhybsv_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseShybsv_solve",                        {"hipsparseShybsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDhybsv_solve",                        {"hipsparseDhybsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseChybsv_solve",                        {"hipsparseChybsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZhybsv_solve",                        {"hipsparseZhybsv_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 8. cuSPARSE Level 3 Function Reference
  {"cusparseScsrmm",                              {"hipsparseScsrmm",                              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrmm",                              {"hipsparseDcsrmm",                              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrmm",                              {"hipsparseCcsrmm",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrmm",                              {"hipsparseZcsrmm",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrmm2",                             {"hipsparseScsrmm2",                             "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrmm2",                             {"hipsparseDcsrmm2",                             "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrmm2",                             {"hipsparseCcsrmm2",                             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrmm2",                             {"hipsparseZcsrmm2",                             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsm_analysis",                     {"hipsparseScsrsm_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsm_analysis",                     {"hipsparseDcsrsm_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsm_analysis",                     {"hipsparseCcsrsm_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsm_analysis",                     {"hipsparseZcsrsm_analysis",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsm_solve",                        {"hipsparseScsrsm_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsm_solve",                        {"hipsparseDcsrsm_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsm_solve",                        {"hipsparseCcsrsm_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsm_solve",                        {"hipsparseZcsrsm_solve",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsm2_bufferSizeExt",               {"hipsparseScsrsm2_bufferSizeExt",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsm2_bufferSizeExt",               {"hipsparseDcsrsm2_bufferSizeExt",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsm2_bufferSizeExt",               {"hipsparseCcsrsm2_bufferSizeExt",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsm2_bufferSizeExt",               {"hipsparseZcsrsm2_bufferSizeExt",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsm2_analysis",                    {"hipsparseScsrsm2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsm2_analysis",                    {"hipsparseDcsrsm2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsm2_analysis",                    {"hipsparseCcsrsm2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsm2_analysis",                    {"hipsparseZcsrsm2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrsm2_solve",                       {"hipsparseScsrsm2_solve",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrsm2_solve",                       {"hipsparseDcsrsm2_solve",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrsm2_solve",                       {"hipsparseCcsrsm2_solve",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrsm2_solve",                       {"hipsparseZcsrsm2_solve",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcsrsm2_zeroPivot",                   {"hipsparseXcsrsm2_zeroPivot",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrmm",                              {"hipsparseSbsrmm",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrmm",                              {"hipsparseDbsrmm",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrmm",                              {"hipsparseCbsrmm",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrmm",                              {"hipsparseZbsrmm",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrsm2_bufferSize",                  {"hipsparseCbsrsm2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrsm2_bufferSize",                  {"hipsparseDbsrsm2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrsm2_bufferSize",                  {"hipsparseCbsrsm2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrsm2_bufferSize",                  {"hipsparseZbsrsm2_bufferSize",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrsm2_analysis",                    {"hipsparseSbsrsm2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrsm2_analysis",                    {"hipsparseDbsrsm2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrsm2_analysis",                    {"hipsparseCbsrsm2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrsm2_analysis",                    {"hipsparseZbsrsm2_analysis",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrsm2_solve",                       {"hipsparseSbsrsm2_solve",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrsm2_solve",                       {"hipsparseDbsrsm2_solve",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrsm2_solve",                       {"hipsparseCbsrsm2_solve",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrsm2_solve",                       {"hipsparseZbsrsm2_solve",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXbsrsm2_zeroPivot",                   {"hipsparseXbsrsm2_zeroPivot",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgemmi",                              {"hipsparseSgemmi",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgemmi",                              {"hipsparseDgemmi",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgemmi",                              {"hipsparseCgemmi",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgemmi",                              {"hipsparseZgemmi",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 9. cuSPARSE Extra Function Reference
  {"cusparseXcsrgeamNnz",                         {"hipsparseXcsrgeamNnz",                         "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseScsrgeam",                            {"hipsparseScsrgeam",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrgeam",                            {"hipsparseDcsrgeam",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrgeam",                            {"hipsparseCcsrgeam",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrgeam",                            {"hipsparseZcsrgeam",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrgeam2_bufferSizeExt",             {"hipsparseScsrgeam2_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrgeam2_bufferSizeExt",             {"hipsparseDcsrgeam2_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrgeam2_bufferSizeExt",             {"hipsparseCcsrgeam2_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrgeam2_bufferSizeExt",             {"hipsparseZcsrgeam2_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcsrgemmNnz",                         {"hipsparseXcsrgemmNnz",                         "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseScsrgemm",                            {"hipsparseScsrgemm",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrgemm",                            {"hipsparseDcsrgemm",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrgemm",                            {"hipsparseCcsrgemm",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrgemm",                            {"hipsparseZcsrgemm",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrgemm2_bufferSizeExt",             {"hipsparseScsrgemm2_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrgemm2_bufferSizeExt",             {"hipsparseDcsrgemm2_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrgemm2_bufferSizeExt",             {"hipsparseCcsrgemm2_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrgemm2_bufferSizeExt",             {"hipsparseZcsrgemm2_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 10. cuSPARSE Preconditioners Reference
  // 10.1. Incomplete Cholesky Factorization : level 0
  {"cusparseScsric0",                             {"hipsparseScsric0",                             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsric0",                             {"hipsparseDcsric0",                             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsric0",                             {"hipsparseCcsric0",                             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsric0",                             {"hipsparseZcsric0",                             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsric02_bufferSize",                 {"hipsparseScsric02_bufferSize",                 "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsric02_bufferSize",                 {"hipsparseDcsric02_bufferSize",                 "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsric02_bufferSize",                 {"hipsparseCcsric02_bufferSize",                 "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsric02_bufferSize",                 {"hipsparseZcsric02_bufferSize",                 "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsric02_analysis",                   {"hipsparseScsric02_analysis",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsric02_analysis",                   {"hipsparseDcsric02_analysis",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsric02_analysis",                   {"hipsparseCcsric02_analysis",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsric02_analysis",                   {"hipsparseZcsric02_analysis",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsric02",                            {"hipsparseScsric02",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsric02",                            {"hipsparseDcsric02",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsric02",                            {"hipsparseCcsric02",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsric02",                            {"hipsparseZcsric02",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcsric02_zeroPivot",                  {"hipsparseXcsric02_zeroPivot",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsric02_analysis",                   {"hipsparseSbsric02_analysis",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsric02_analysis",                   {"hipsparseDbsric02_analysis",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsric02_analysis",                   {"hipsparseCbsric02_analysis",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsric02_analysis",                   {"hipsparseZbsric02_analysis",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsric02",                            {"hipsparseSbsric02",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsric02",                            {"hipsparseDbsric02",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsric02",                            {"hipsparseCbsric02",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsric02",                            {"hipsparseZbsric02",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXbsric02_zeroPivot",                  {"hipsparseXbsric02_zeroPivot",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 10.2. Incomplete LU Factorization: level 0
  {"cusparseScsrilu0",                            {"hipsparseScsrilu0",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrilu0",                            {"hipsparseDcsrilu0",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrilu0",                            {"hipsparseCcsrilu0",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrilu0",                            {"hipsparseZcsrilu0",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCsrilu0Ex",                           {"hipsparseCsrilu0Ex",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrilu02_numericBoost",              {"hipsparseScsrilu02_numericBoost",              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrilu02_numericBoost",              {"hipsparseDcsrilu02_numericBoost",              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrilu02_numericBoost",              {"hipsparseCcsrilu02_numericBoost",              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrilu02_numericBoost",              {"hipsparseZcsrilu02_numericBoost",              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrilu02_bufferSize",                {"hipsparseScsrilu02_bufferSize",                "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrilu02_bufferSize",                {"hipsparseDcsrilu02_bufferSize",                "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrilu02_bufferSize",                {"hipsparseCcsrilu02_bufferSize",                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrilu02_bufferSize",                {"hipsparseZcsrilu02_bufferSize",                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrilu02_analysis",                  {"hipsparseScsrilu02_analysis",                  "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrilu02_analysis",                  {"hipsparseDcsrilu02_analysis",                  "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrilu02_analysis",                  {"hipsparseCcsrilu02_analysis",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrilu02_analysis",                  {"hipsparseZcsrilu02_analysis",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsrilu02",                           {"hipsparseScsrilu02",                           "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsrilu02",                           {"hipsparseDcsrilu02",                           "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsrilu02",                           {"hipsparseCcsrilu02",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrilu02",                           {"hipsparseZcsrilu02",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXbsric02_zeroPivot",                  {"hipsparseXcsrilu02_zeroPivot",                 "", CONV_LIB_FUNC, API_SPARSE}},

  {"cusparseSbsrilu02_numericBoost",              {"hipsparseSbsrilu02_numericBoost",              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrilu02_numericBoost",              {"hipsparseDbsrilu02_numericBoost",              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrilu02_numericBoost",              {"hipsparseCbsrilu02_numericBoost",              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrilu02_numericBoost",              {"hipsparseZbsrilu02_numericBoost",              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrilu02_bufferSize",                {"hipsparseSbsrilu02_bufferSize",                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrilu02_bufferSize",                {"hipsparseDbsrilu02_bufferSize",                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrilu02_bufferSize",                {"hipsparseCbsrilu02_bufferSize",                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrilu02_bufferSize",                {"hipsparseZbsrilu02_bufferSize",                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrilu02_analysis",                  {"hipsparseSbsrilu02_analysis",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrilu02_analysis",                  {"hipsparseDbsrilu02_analysis",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrilu02_analysis",                  {"hipsparseCbsrilu02_analysis",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrilu02_analysis",                  {"hipsparseZbsrilu02_analysis",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSbsrilu02",                           {"hipsparseSbsrilu02",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsrilu02",                           {"hipsparseDbsrilu02",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsrilu02",                           {"hipsparseCbsrilu02",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsrilu02",                           {"hipsparseZbsrilu02",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXbsrilu02_zeroPivot",                 {"hipsparseXbsrilu02_zeroPivot",                 "", CONV_LIB_FUNC, API_SPARSE}},

  // 10.3. Tridiagonal Solve
  {"cusparseSgtsv",                               {"hipsparseSgtsv",                               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgtsv",                               {"hipsparseDgtsv",                               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgtsv",                               {"hipsparseCgtsv",                               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgtsv",                               {"hipsparseZgtsv",                               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgtsv_nopivot",                       {"hipsparseSgtsv_nopivot",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgtsv_nopivot",                       {"hipsparseDgtsv_nopivot",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgtsv_nopivot",                       {"hipsparseCgtsv_nopivot",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgtsv_nopivot",                       {"hipsparseZgtsv_nopivot",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgtsv2_bufferSizeExt",                {"hipsparseSgtsv2_bufferSizeExt",                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgtsv2_bufferSizeExt",                {"hipsparseDgtsv2_bufferSizeExt",                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgtsv2_bufferSizeExt",                {"hipsparseCgtsv2_bufferSizeExt",                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgtsv2_bufferSizeExt",                {"hipsparseZgtsv2_bufferSizeExt",                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgtsv2",                              {"hipsparseSgtsv2",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgtsv2",                              {"hipsparseDgtsv2",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgtsv2",                              {"hipsparseCgtsv2",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgtsv2",                              {"hipsparseZgtsv2",                              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgtsv2_nopivot_bufferSizeExt",        {"hipsparseSgtsv2_nopivot_bufferSizeExt",        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgtsv2_nopivot_bufferSizeExt",        {"hipsparseDgtsv2_nopivot_bufferSizeExt",        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgtsv2_nopivot_bufferSizeExt",        {"hipsparseCgtsv2_nopivot_bufferSizeExt",        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgtsv2_nopivot_bufferSizeExt",        {"hipsparseZgtsv2_nopivot_bufferSizeExt",        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgtsv2_nopivot",                      {"hipsparseSgtsv2_nopivot",                      "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgtsv2_nopivot",                      {"hipsparseDgtsv2_nopivot",                      "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgtsv2_nopivot",                      {"hipsparseCgtsv2_nopivot",                      "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgtsv2_nopivot",                      {"hipsparseZgtsv2_nopivot",                      "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 10.4. Batched Tridiagonal Solve
  {"cusparseSgtsvStridedBatch",                   {"hipsparseSgtsvStridedBatch",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgtsvStridedBatch",                   {"hipsparseDgtsvStridedBatch",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgtsvStridedBatch",                   {"hipsparseCgtsvStridedBatch",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgtsvStridedBatch",                   {"hipsparseZgtsvStridedBatch",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgtsv2StridedBatch_bufferSizeExt",    {"hipsparseSgtsv2StridedBatch_bufferSizeExt",    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgtsv2StridedBatch_bufferSizeExt",    {"hipsparseDgtsv2StridedBatch_bufferSizeExt",    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgtsv2StridedBatch_bufferSizeExt",    {"hipsparseCgtsv2StridedBatch_bufferSizeExt",    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgtsv2StridedBatch_bufferSizeExt",    {"hipsparseZgtsv2StridedBatch_bufferSizeExt",    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgtsv2StridedBatch",                  {"hipsparseSgtsv2StridedBatch",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgtsv2StridedBatch",                  {"hipsparseDgtsv2StridedBatch",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgtsv2StridedBatch",                  {"hipsparseCgtsv2StridedBatch",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgtsv2StridedBatch",                  {"hipsparseZgtsv2StridedBatch",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgtsvInterleavedBatch_bufferSizeExt", {"hipsparseSgtsvInterleavedBatch_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgtsvInterleavedBatch_bufferSizeExt", {"hipsparseDgtsvInterleavedBatch_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgtsvInterleavedBatch_bufferSizeExt", {"hipsparseCgtsvInterleavedBatch_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgtsvInterleavedBatch_bufferSizeExt", {"hipsparseZgtsvInterleavedBatch_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgtsvInterleavedBatch",               {"hipsparseSgtsvInterleavedBatch",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgtsvInterleavedBatch",               {"hipsparseDgtsvInterleavedBatch",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgtsvInterleavedBatch",               {"hipsparseCgtsvInterleavedBatch",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgtsvInterleavedBatch",               {"hipsparseZgtsvInterleavedBatch",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 10.5. Batched Pentadiagonal Solve
  {"cusparseSgpsvInterleavedBatch_bufferSizeExt", {"hipsparseSgpsvInterleavedBatch_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgpsvInterleavedBatch_bufferSizeExt", {"hipsparseDgpsvInterleavedBatch_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgpsvInterleavedBatch_bufferSizeExt", {"hipsparseCgpsvInterleavedBatch_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgpsvInterleavedBatch_bufferSizeExt", {"hipsparseZgpsvInterleavedBatch_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgpsvInterleavedBatch",               {"hipsparseSgpsvInterleavedBatch",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgpsvInterleavedBatch",               {"hipsparseDgpsvInterleavedBatch",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgpsvInterleavedBatch",               {"hipsparseCgpsvInterleavedBatch",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgpsvInterleavedBatch",               {"hipsparseZgpsvInterleavedBatch",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 11. cuSPARSE Matrix Reorderings Reference
  {"cusparseScsrcolor",                           {"hipsparseScsrcolor",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsrcolor",                           {"hipsparseDcsrcolor",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsrcolor",                           {"hipsparseCcsrcolor",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsrcolor",                           {"hipsparseZcsrcolor",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  // 12. cuSPARSE Format Conversion Reference
  {"cusparseSbsr2csr",                            {"hipsparseSbsr2csr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDbsr2csr",                            {"hipsparseDbsr2csr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCbsr2csr",                            {"hipsparseCbsr2csr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZbsr2csr",                            {"hipsparseZbsr2csr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgebsr2gebsc_bufferSize",             {"hipsparseSgebsr2gebsc_bufferSize",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgebsr2gebsc_bufferSize",             {"hipsparseDgebsr2gebsc_bufferSize",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgebsr2gebsc_bufferSize",             {"hipsparseCgebsr2gebsc_bufferSize",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgebsr2gebsc_bufferSize",             {"hipsparseZgebsr2gebsc_bufferSize",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgebsr2gebsc",                        {"hipsparseSgebsr2gebsc",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgebsr2gebsc",                        {"hipsparseDgebsr2gebsc",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgebsr2gebsc",                        {"hipsparseCgebsr2gebsc",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgebsr2gebsc",                        {"hipsparseZgebsr2gebsc",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgebsr2gebsr_bufferSize",             {"hipsparseSgebsr2gebsr_bufferSize",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgebsr2gebsr_bufferSize",             {"hipsparseDgebsr2gebsr_bufferSize",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgebsr2gebsr_bufferSize",             {"hipsparseCgebsr2gebsr_bufferSize",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgebsr2gebsr_bufferSize",             {"hipsparseZgebsr2gebsr_bufferSize",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSgebsr2csr",                          {"hipsparseSgebsr2csr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgebsr2csr",                          {"hipsparseDgebsr2csr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgebsr2csr",                          {"hipsparseCgebsr2csr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgebsr2csr",                          {"hipsparseZgebsr2csr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXgebsr2gebsrNnz",                     {"hipsparseXgebsr2gebsrNnz",                     "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseSgebsr2gebsr",                        {"hipsparseSgebsr2gebsr",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDgebsr2gebsr",                        {"hipsparseDgebsr2gebsr",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCgebsr2gebsr",                        {"hipsparseCgebsr2gebsr",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZgebsr2gebsr",                        {"hipsparseZgebsr2gebsr",                        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsr2gebsr_bufferSize",               {"hipsparseScsr2gebsr_bufferSize",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsr2gebsr_bufferSize",               {"hipsparseDcsr2gebsr_bufferSize",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsr2gebsr_bufferSize",               {"hipsparseCcsr2gebsr_bufferSize",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsr2gebsr_bufferSize",               {"hipsparseZcsr2gebsr_bufferSize",               "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcsr2gebsrNnz",                       {"hipsparseXcsr2gebsrNnz",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseScsr2gebsr",                          {"hipsparseScsr2gebsr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsr2gebsr",                          {"hipsparseDcsr2gebsr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsr2gebsr",                          {"hipsparseCcsr2gebsr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsr2gebsr",                          {"hipsparseZcsr2gebsr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcoo2csr",                            {"hipsparseXcoo2csr",                            "", CONV_LIB_FUNC, API_SPARSE}},

  {"cusparseScsc2dense",                          {"hipsparseScsc2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsc2dense",                          {"hipsparseDcsc2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsc2dense",                          {"hipsparseCcsc2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsc2dense",                          {"hipsparseZcsc2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsc2hyb",                            {"hipsparseScsc2hyb",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsc2hyb",                            {"hipsparseDcsc2hyb",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsc2hyb",                            {"hipsparseCcsc2hyb",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsc2hyb",                            {"hipsparseZcsc2hyb",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcsr2bsrNnz",                         {"hipsparseXcsr2bsrNnz",                         "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseScsr2bsr",                            {"hipsparseScsr2bsr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsr2bsr",                            {"hipsparseDcsr2bsr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsr2bsr",                            {"hipsparseCcsr2bsr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsr2bsr",                            {"hipsparseZcsr2bsr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseXcsr2coo",                            {"hipsparseXcsr2coo",                            "", CONV_LIB_FUNC, API_SPARSE}},

  {"cusparseScsr2csc",                            {"hipsparseScsr2csc",                            "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsr2csc",                            {"hipsparseDcsr2csc",                            "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsr2csc",                            {"hipsparseCcsr2csc",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsr2csc",                            {"hipsparseZcsr2csc",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCsr2cscEx",                           {"hipsparseCsr2cscEx",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsr2dense",                          {"hipsparseScsr2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsr2dense",                          {"hipsparseDcsr2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsr2dense",                          {"hipsparseCcsr2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsr2dense",                          {"hipsparseZcsr2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsr2csr_compress",                   {"hipsparseScsr2csr_compress",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsr2csr_compress",                   {"hipsparseDcsr2csr_compress",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsr2csr_compress",                   {"hipsparseDcsr2csr_compress",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsr2csr_compress",                   {"hipsparseZcsr2csr_compress",                   "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsr2hyb",                            {"hipsparseScsr2hyb",                            "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseDcsr2hyb",                            {"hipsparseDcsr2hyb",                            "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseCcsr2hyb",                            {"hipsparseCcsr2hyb",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsr2hyb",                            {"hipsparseZcsr2hyb",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSdense2csc",                          {"hipsparseSdense2csc",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDdense2csc",                          {"hipsparseDdense2csc",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCdense2csc",                          {"hipsparseCdense2csc",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZdense2csc",                          {"hipsparseZdense2csc",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSdense2csr",                          {"hipsparseSdense2csr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDdense2csr",                          {"hipsparseDdense2csr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCdense2csr",                          {"hipsparseCdense2csr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZdense2csr",                          {"hipsparseZdense2csr",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSdense2hyb",                          {"hipsparseSdense2hyb",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDdense2hyb",                          {"hipsparseDdense2hyb",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCdense2hyb",                          {"hipsparseCdense2hyb",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZdense2hyb",                          {"hipsparseZdense2hyb",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseShyb2csc",                            {"hipsparseShyb2csc",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDhyb2csc",                            {"hipsparseDhyb2csc",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseChyb2csc",                            {"hipsparseChyb2csc",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZhyb2csc",                            {"hipsparseZhyb2csc",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseShyb2csr",                            {"hipsparseShyb2csr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDhyb2csr",                            {"hipsparseDhyb2csr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseChyb2csr",                            {"hipsparseChyb2csr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZhyb2csr",                            {"hipsparseZhyb2csr",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseShyb2dense",                          {"hipsparseShyb2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDhyb2dense",                          {"hipsparseDhyb2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseChyb2dense",                          {"hipsparseChyb2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZhyb2dense",                          {"hipsparseZhyb2dense",                          "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSnnz",                                {"hipsparseSnnz",                                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDnnz",                                {"hipsparseDnnz",                                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCnnz",                                {"hipsparseCnnz",                                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZnnz",                                {"hipsparseZnnz",                                "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCreateIdentityPermutation",           {"hipsparseCreateIdentityPermutation",           "", CONV_LIB_FUNC, API_SPARSE}},

  {"cusparseXcoosort_bufferSizeExt",              {"hipsparseXcoosort_bufferSizeExt",              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseXcoosortByRow",                       {"hipsparseXcoosortByRow",                       "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseXcoosortByColumn",                    {"hipsparseXcoosortByColumn",                    "", CONV_LIB_FUNC, API_SPARSE}},

  {"cusparseXcsrsort_bufferSizeExt",              {"hipsparseXcsrsort_bufferSizeExt",              "", CONV_LIB_FUNC, API_SPARSE}},
  {"cusparseXcsrsort",                            {"hipsparseXcsrsort",                            "", CONV_LIB_FUNC, API_SPARSE}},

  {"cusparseXcscsort_bufferSizeExt",              {"hipsparseXcscsort_bufferSizeExt",              "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseXcscsort",                            {"hipsparseXcscsort",                            "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseCreateCsru2csrInfo",                  {"hipsparseCreateCsru2csrInfo",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDestroyCsru2csrInfo",                 {"hipsparseDestroyCsru2csrInfo",                 "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsru2csr_bufferSizeExt",             {"hipsparseScsru2csr_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsru2csr_bufferSizeExt",             {"hipsparseDcsru2csr_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsru2csr_bufferSizeExt",             {"hipsparseCcsru2csr_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsru2csr_bufferSizeExt",             {"hipsparseZcsru2csr_bufferSizeExt",             "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseScsru2csr",                           {"hipsparseScsru2csr",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDcsru2csr",                           {"hipsparseDcsru2csr",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCcsru2csr",                           {"hipsparseCcsru2csr",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZcsru2csr",                           {"hipsparseZcsru2csr",                           "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseHpruneDense2csr_bufferSizeExt",       {"hipsparseHpruneDense2csr_bufferSizeExt",       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseSpruneDense2csr_bufferSizeExt",       {"hipsparseSpruneDense2csr_bufferSizeExt",       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDpruneDense2csr_bufferSizeExt",       {"hipsparseDpruneDense2csr_bufferSizeExt",       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseHpruneDense2csrNnz",                  {"hipsparseHpruneDense2csrNnz",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseSpruneDense2csrNnz",                  {"hipsparseSpruneDense2csrNnz",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDpruneDense2csrNnz",                  {"hipsparseDpruneDense2csrNnz",                  "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseHpruneCsr2csr_bufferSizeExt",         {"hipsparseHpruneCsr2csr_bufferSizeExt",         "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseSpruneCsr2csr_bufferSizeExt",         {"hipsparseSpruneCsr2csr_bufferSizeExt",         "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDpruneCsr2csr_bufferSizeExt",         {"hipsparseDpruneCsr2csr_bufferSizeExt",         "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseHpruneCsr2csrNnz",                    {"hipsparseHpruneCsr2csrNnz",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseSpruneCsr2csrNnz",                    {"hipsparseSpruneCsr2csrNnz",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDpruneCsr2csrNnz",                    {"hipsparseDpruneCsr2csrNnz",                    "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseHpruneDense2csrByPercentage_bufferSizeExt", {"hipsparseHpruneDense2csrByPercentage_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseSpruneDense2csrByPercentage_bufferSizeExt", {"hipsparseSpruneDense2csrByPercentage_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDpruneDense2csrByPercentage_bufferSizeExt", {"hipsparseDpruneDense2csrByPercentage_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseHpruneDense2csrNnzByPercentage",      {"hipsparseHpruneDense2csrNnzByPercentage",      "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseSpruneDense2csrNnzByPercentage",      {"hipsparseSpruneDense2csrNnzByPercentage",      "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDpruneDense2csrNnzByPercentage",      {"hipsparseDpruneDense2csrNnzByPercentage",      "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseHpruneCsr2csrByPercentage_bufferSizeExt", {"hipsparseHpruneCsr2csrByPercentage_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseSpruneCsr2csrByPercentage_bufferSizeExt", {"hipsparseSpruneCsr2csrByPercentage_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDpruneCsr2csrByPercentage_bufferSizeExt", {"hipsparseDpruneCsr2csrByPercentage_bufferSizeExt", "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseHpruneCsr2csrNnzByPercentage",        {"hipsparseHpruneCsr2csrNnzByPercentage",        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseSpruneCsr2csrNnzByPercentage",        {"hipsparseSpruneCsr2csrNnzByPercentage",        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDpruneCsr2csrNnzByPercentage",        {"hipsparseDpruneCsr2csrNnzByPercentage",        "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},

  {"cusparseSnnz_compress",                       {"hipsparseSnnz_compress",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseDnnz_compress",                       {"hipsparseDnnz_compress",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseCnnz_compress",                       {"hipsparseCnnz_compress",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
  {"cusparseZnnz_compress",                       {"hipsparseZnnz_compress",                       "", CONV_LIB_FUNC, API_SPARSE, HIP_UNSUPPORTED}},
};
