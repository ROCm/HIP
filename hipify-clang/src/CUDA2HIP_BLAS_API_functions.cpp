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

// Map of all functions
const std::map<llvm::StringRef, hipCounter> CUDA_BLAS_FUNCTION_MAP{

  // Blas management functions
  {"cublasInit",                     {"hipblasInit",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasShutdown",                 {"hipblasShutdown",                 "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetVersion",               {"hipblasGetVersion",               "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetError",                 {"hipblasGetError",                 "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasAlloc",                    {"hipblasAlloc",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasFree",                     {"hipblasFree",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSetKernelStream",          {"hipblasSetKernelStream",          "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetAtomicsMode",           {"hipblasGetAtomicsMode",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSetAtomicsMode",           {"hipblasSetAtomicsMode",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetMathMode",              {"hipblasGetMathMode",              "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSetMathMode",              {"hipblasSetMathMode",              "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Blas logging
  {"cublasLogCallback",              {"hipblasLogCallback",              "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasLoggerConfigure",          {"hipblasLoggerConfigure",          "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSetLoggerCallback",        {"hipblasSetLoggerCallback",        "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetLoggerCallback",        {"hipblasGetLoggerCallback",        "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Blas1 (v1) Routines
  {"cublasCreate",                   {"hipblasCreate",                   "rocblas_create_handle",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasDestroy",                  {"hipblasDestroy",                  "rocblas_destroy_handle",                   CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetStream",                {"hipblasSetStream",                "rocblas_set_stream",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetStream",                {"hipblasGetStream",                "rocblas_get_stream",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetPointerMode",           {"hipblasSetPointerMode",           "rocblas_set_pointer_mode",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetPointerMode",           {"hipblasGetPointerMode",           "rocblas_get_pointer_mode",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetVector",                {"hipblasSetVector",                "rocblas_set_vector",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetVector",                {"hipblasGetVector",                "rocblas_get_vector",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetVectorAsync",           {"hipblasSetVectorAsync",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetVectorAsync",           {"hipblasGetVectorAsync",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSetMatrix",                {"hipblasSetMatrix",                "rocblas_set_matrix",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetMatrix",                {"hipblasGetMatrix",                "rocblas_get_matrix",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetMatrixAsync",           {"hipblasSetMatrixAsync",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetMatrixAsync",           {"hipblasGetMatrixAsync",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasXerbla",                   {"hipblasXerbla",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // NRM2
  {"cublasSnrm2",                    {"hipblasSnrm2",                    "rocblas_snrm2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDnrm2",                    {"hipblasDnrm2",                    "rocblas_dnrm2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasScnrm2",                   {"hipblasScnrm2",                   "rocblas_scnrm2",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDznrm2",                   {"hipblasDznrm2",                   "rocblas_dznrm2",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasNrm2Ex",                   {"hipblasNrm2Ex",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // DOT
  {"cublasSdot",                     {"hipblasSdot",                     "rocblas_sdot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDdot",                     {"hipblasDdot",                     "rocblas_ddot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCdotu",                    {"hipblasCdotu",                    "rocblas_cdotu",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCdotc",                    {"hipblasCdotc",                    "rocblas_cdotc",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdotu",                    {"hipblasZdotu",                    "rocblas_zdotu",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdotc",                    {"hipblasZdotc",                    "rocblas_zdotc",                            CONV_LIB_FUNC, API_BLAS}},

  // SCAL
  {"cublasSscal",                    {"hipblasSscal",                    "rocblas_sscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDscal",                    {"hipblasDscal",                    "rocblas_dscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCscal",                    {"hipblasCscal",                    "rocblas_cscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsscal",                   {"hipblasCsscal",                   "rocblas_csscal",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasZscal",                    {"hipblasZscal",                    "rocblas_zscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdscal",                   {"hipblasZdscal",                   "rocblas_zdscal",                           CONV_LIB_FUNC, API_BLAS}},

  // AXPY
  {"cublasSaxpy",                    {"hipblasSaxpy",                    "rocblas_saxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDaxpy",                    {"hipblasDaxpy",                    "rocblas_daxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCaxpy",                    {"hipblasCaxpy",                    "rocblas_caxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZaxpy",                    {"hipblasZaxpy",                    "rocblas_zaxpy",                            CONV_LIB_FUNC, API_BLAS}},

  // COPY
  {"cublasScopy",                    {"hipblasScopy",                    "rocblas_scopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDcopy",                    {"hipblasDcopy",                    "rocblas_dcopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCcopy",                    {"hipblasCcopy",                    "rocblas_ccopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZcopy",                    {"hipblasZcopy",                    "rocblas_zcopy",                            CONV_LIB_FUNC, API_BLAS}},

  // SWAP
  {"cublasSswap",                    {"hipblasSswap",                    "rocblas_sswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDswap",                    {"hipblasDswap",                    "rocblas_dswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCswap",                    {"hipblasCswap",                    "rocblas_cswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZswap",                    {"hipblasZswap",                    "rocblas_zswap",                            CONV_LIB_FUNC, API_BLAS}},

  // AMAX
  {"cublasIsamax",                   {"hipblasIsamax",                   "rocblas_isamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIdamax",                   {"hipblasIdamax",                   "rocblas_idamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIcamax",                   {"hipblasIcamax",                   "rocblas_icamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIzamax",                   {"hipblasIzamax",                   "rocblas_izamax",                           CONV_LIB_FUNC, API_BLAS}},

  // AMIN
  {"cublasIsamin",                   {"hipblasIsamin",                   "rocblas_isamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIdamin",                   {"hipblasIdamin",                   "rocblas_idamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIcamin",                   {"hipblasIcamin",                   "rocblas_icamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIzamin",                   {"hipblasIzamin",                   "rocblas_izamin",                           CONV_LIB_FUNC, API_BLAS}},

  // ASUM
  {"cublasSasum",                    {"hipblasSasum",                    "rocblas_sasum",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDasum",                    {"hipblasDasum",                    "rocblas_dasum",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasScasum",                   {"hipblasScasum",                   "rocblas_scasum",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDzasum",                   {"hipblasDzasum",                   "rocblas_dzasum",                           CONV_LIB_FUNC, API_BLAS}},

  // ROT
  {"cublasSrot",                     {"hipblasSrot",                     "rocblas_srot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrot",                     {"hipblasDrot",                     "rocblas_drot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCrot",                     {"hipblasCrot",                     "rocblas_crot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsrot",                    {"hipblasCsrot",                    "rocblas_csrot",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZrot",                     {"hipblasZrot",                     "rocblas_zrot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdrot",                    {"hipblasZdrot",                    "rocblas_zdrot",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTG
  {"cublasSrotg",                    {"hipblasSrotg",                    "rocblas_srotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotg",                    {"hipblasDrotg",                    "rocblas_drotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCrotg",                    {"hipblasCrotg",                    "rocblas_crotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZrotg",                    {"hipblasZrotg",                    "rocblas_zrotg",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTM
  {"cublasSrotm",                    {"hipblasSrotm",                    "rocblas_srotm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotm",                    {"hipblasDrotm",                    "rocblas_drotm",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTMG
  {"cublasSrotmg",                   {"hipblasSrotmg",                   "rocblas_srotmg",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotmg",                   {"hipblasDrotmg",                   "rocblas_drotmg",                           CONV_LIB_FUNC, API_BLAS}},

  // GEMV
  {"cublasSgemv",                    {"hipblasSgemv",                    "rocblas_sgemv",                            CONV_LIB_FUNC, API_BLAS}},
  // NOTE: there is no such a function in CUDA
  {"cublasSgemvBatched",             {"hipblasSgemvBatched",             "rocblas_sgemv_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemv",                    {"hipblasDgemv",                    "rocblas_dgemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemv",                    {"hipblasCgemv",                    "rocblas_cgemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgemv",                    {"hipblasZgemv",                    "rocblas_zgemv",                            CONV_LIB_FUNC, API_BLAS}},

  // GBMV
  {"cublasSgbmv",                    {"hipblasSgbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDgbmv",                    {"hipblasDgbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgbmv",                    {"hipblasCgbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgbmv",                    {"hipblasZgbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TRMV
  {"cublasStrmv",                    {"hipblasStrmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtrmv",                    {"hipblasDtrmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtrmv",                    {"hipblasCtrmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtrmv",                    {"hipblasZtrmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TBMV
  {"cublasStbmv",                    {"hipblasStbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtbmv",                    {"hipblasDtbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtbmv",                    {"hipblasCtbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtbmv",                    {"hipblasZtbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TPMV
  {"cublasStpmv",                    {"hipblasStpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtpmv",                    {"hipblasDtpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtpmv",                    {"hipblasCtpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtpmv",                    {"hipblasZtpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TRSV
  {"cublasStrsv",                    {"hipblasStrsv",                    "rocblas_strsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrsv",                    {"hipblasDtrsv",                    "rocblas_dtrsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrsv",                    {"hipblasCtrsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtrsv",                    {"hipblasZtrsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TPSV
  {"cublasStpsv",                    {"hipblasStpsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtpsv",                    {"hipblasDtpsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtpsv",                    {"hipblasCtpsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtpsv",                    {"hipblasZtpsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TBSV
  {"cublasStbsv",                    {"hipblasStbsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtbsv",                    {"hipblasDtbsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtbsv",                    {"hipblasCtbsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtbsv",                    {"hipblasZtbsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYMV/HEMV
  {"cublasSsymv",                    {"hipblasSsymv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsymv",                    {"hipblasDsymv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCsymv",                    {"hipblasCsymv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsymv",                    {"hipblasZsymv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasChemv",                    {"hipblasChemv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhemv",                    {"hipblasZhemv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SBMV/HBMV
  {"cublasSsbmv",                    {"hipblasSsbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsbmv",                    {"hpiblasDsbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasChbmv",                    {"hipblasChbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhbmv",                    {"hipblasZhbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SPMV/HPMV
  {"cublasSspmv",                    {"hipblasSspmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDspmv",                    {"hipblasDspmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasChpmv",                    {"hipblasChpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhpmv",                    {"hipblasZhpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // GER
  {"cublasSger",                     {"hipblasSger",                     "rocblas_sger",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDger",                     {"hipblasDger",                     "rocblas_dger",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgeru",                    {"hipblasCgeru",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgerc",                    {"hipblasCgerc",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgeru",                    {"hipblasZgeru",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgerc",                    {"hipblasZgerc",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYR/HER
  {"cublasSsyr",                     {"hipblasSsyr",                     "rocblas_ssyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsyr",                     {"hipblasDsyr",                     "rocblas_dsyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsyr",                     {"hipblasCsyr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsyr",                     {"hipblasZsyr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCher",                     {"hipblasCher",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZher",                     {"hipblasZher",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SPR/HPR
  {"cublasSspr",                     {"hipblasSspr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDspr",                     {"hipblasDspr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasChpr",                     {"hipblasChpr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhpr",                     {"hipblasZhpr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYR2/HER2
  {"cublasSsyr2",                    {"hipblasSsyr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsyr2",                    {"hipblasDsyr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCsyr2",                    {"hipblasCsyr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsyr2",                    {"hipblasZsyr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCher2",                    {"hipblasCher2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZher2",                    {"hipblasZher2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SPR2/HPR2
  {"cublasSspr2",                    {"hipblasSspr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDspr2",                    {"hipblasDspr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasChpr2",                    {"hipblasChpr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhpr2",                    {"hipblasZhpr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Blas3 (v1) Routines
  // GEMM
  {"cublasSgemm",                    {"hipblasSgemm",                    "rocblas_sgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemm",                    {"hipblasDgemm",                    "rocblas_dgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm",                    {"hipblasCgemm",                    "rocblas_cgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgemm",                    {"hipblasZgemm",                    "rocblas_zgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasHgemm",                    {"hipblasHgemm",                    "rocblas_hgemm",                            CONV_LIB_FUNC, API_BLAS}},

  // BATCH GEMM
  {"cublasSgemmBatched",             {"hipblasSgemmBatched",             "rocblas_sgemm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemmBatched",             {"hipblasDgemmBatched",             "rocblas_dgemm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasHgemmBatched",             {"hipblasHgemmBatched",             "rocblas_hgemm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasSgemmStridedBatched",      {"hipblasSgemmStridedBatched",      "rocblas_sgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemmStridedBatched",      {"hipblasDgemmStridedBatched",      "rocblas_dgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemmBatched",             {"hipblasCgemmBatched",             "rocblas_cgemm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm3mBatched",           {"hipblasCgemm3mBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgemmBatched",             {"hipblasZgemmBatched",             "rocblas_zgemm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemmStridedBatched",      {"hipblasCgemmStridedBatched",      "rocblas_cgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm3mStridedBatched",    {"hipblasCgemm3mStridedBatched",    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgemmStridedBatched",      {"hipblasZgemmStridedBatched",      "rocblas_zgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS}},
  {"cublasHgemmStridedBatched",      {"hipblasHgemmStridedBatched",      "rocblas_hgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS}},

  // SYRK
  {"cublasSsyrk",                    {"hipblasSsyrk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsyrk",                    {"hipblasDsyrk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCsyrk",                    {"hipblasCsyrk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsyrk",                    {"hipblasZsyrk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // HERK
  {"cublasCherk",                    {"hipblasCherk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZherk",                    {"hipblasZherk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYR2K
  {"cublasSsyr2k",                   {"hipblasSsyr2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsyr2k",                   {"hipblasDsyr2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCsyr2k",                   {"hipblasCsyr2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsyr2k",                   {"hipblasZsyr2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYRKX - eXtended SYRK
  {"cublasSsyrkx",                   {"hipblasSsyrkx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsyrkx",                   {"hipblasDsyrkx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCsyrkx",                   {"hipblasCsyrkx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsyrkx",                   {"hipblasZsyrkx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // HER2K
  {"cublasCher2k",                   {"hipblasCher2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZher2k",                   {"hipblasZher2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // HERKX - eXtended HERK
  {"cublasCherkx",                   {"hipblasCherkx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZherkx",                   {"hipblasZherkx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYMM
  {"cublasSsymm",                    {"hipblasSsymm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsymm",                    {"hipblasDsymm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCsymm",                    {"hipblasCsymm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsymm",                    {"hipblasZsymm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // HEMM
  {"cublasChemm",                    {"hipblasChemm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhemm",                    {"hipblasZhemm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TRSM
  {"cublasStrsm",                    {"hipblasStrsm",                    "rocblas_strsm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrsm",                    {"hipblasDtrsm",                    "rocblas_dtrsm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrsm",                    {"hipblasCtrsm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtrsm",                    {"hipblasZtrsm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TRMM
  {"cublasStrmm",                    {"hipblasStrmm",                    "rocblas_strmm",                            CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrmm",                    {"hipblasDtrmm",                    "rocblas_dtrmm",                            CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrmm",                    {"hipblasCtrmm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtrmm",                    {"hipblasZtrmm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // ------------------------ CUBLAS BLAS - like extension (cublas_api.h)
  // GEAM
  {"cublasSgeam",                    {"hipblasSgeam",                    "rocblas_sgeam",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgeam",                    {"hipblasDgeam",                    "rocblas_dgeam",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgeam",                    {"hipblasCgeam",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgeam",                    {"hipblasZgeam",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // GETRF - Batched LU
  {"cublasSgetrfBatched",            {"hipblasSgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDgetrfBatched",            {"hipblasDgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgetrfBatched",            {"hipblasCgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgetrfBatched",            {"hipblasZgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Batched inversion based on LU factorization from getrf
  {"cublasSgetriBatched",            {"hipblasSgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDgetriBatched",            {"hipblasDgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgetriBatched",            {"hipblasCgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgetriBatched",            {"hipblasZgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Batched solver based on LU factorization from getrf
  {"cublasSgetrsBatched",            {"hipblasSgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDgetrsBatched",            {"hipblasDgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgetrsBatched",            {"hipblasCgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgetrsBatched",            {"hipblasZgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TRSM - Batched Triangular Solver
  {"cublasStrsmBatched",             {"hipblasStrsmBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtrsmBatched",             {"hipblasDtrsmBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtrsmBatched",             {"hipblasCtrsmBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtrsmBatched",             {"hipblasZtrsmBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // MATINV - Batched
  {"cublasSmatinvBatched",           {"hipblasSmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDmatinvBatched",           {"hipblasDmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCmatinvBatched",           {"hipblasCmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZmatinvBatched",           {"hipblasZmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Batch QR Factorization
  {"cublasSgeqrfBatched",            {"hipblasSgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDgeqrfBatched",            {"hipblasDgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgeqrfBatched",            {"hipblasCgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgeqrfBatched",            {"hipblasZgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Least Square Min only m >= n and Non-transpose supported
  {"cublasSgelsBatched",             {"hipblasSgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDgelsBatched",             {"hipblasDgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgelsBatched",             {"hipblasCgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgelsBatched",             {"hipblasZgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // DGMM
  {"cublasSdgmm",                    {"hipblasSdgmm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDdgmm",                    {"hipblasDdgmm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCdgmm",                    {"hipblasCdgmm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZdgmm",                    {"hipblasZdgmm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TPTTR - Triangular Pack format to Triangular format
  {"cublasStpttr",                   {"hipblasStpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtpttr",                   {"hipblasDtpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtpttr",                   {"hipblasCtpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtpttr",                   {"hipblasZtpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TRTTP - Triangular format to Triangular Pack format
  {"cublasStrttp",                   {"hipblasStrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtrttp",                   {"hipblasDtrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtrttp",                   {"hipblasCtrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtrttp",                   {"hipblasZtrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Blas2 (v2) Routines
  {"cublasCreate_v2",                {"hipblasCreate",                   "rocblas_create_handle",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasDestroy_v2",               {"hipblasDestroy",                  "rocblas_destroy_handle",                   CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetVersion_v2",            {"hipblasGetVersion",               "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetProperty",              {"hipblasGetProperty",              "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSetStream_v2",             {"hipblasSetStream",                "rocblas_set_stream",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetStream_v2",             {"hipblasGetStream",                "rocblas_get_stream",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetPointerMode_v2",        {"hipblasGetPointerMode",           "rocblas_set_pointer_mode",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetPointerMode_v2",        {"hipblasSetPointerMode",           "rocblas_get_pointer_mode",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetCudartVersion",         {"hipblasGetCudartVersion",         "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // GEMV
  {"cublasSgemv_v2",                 {"hipblasSgemv",                    "rocblas_sgemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemv_v2",                 {"hipblasDgemv",                    "rocblas_dgemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemv_v2",                 {"hipblasCgemv",                    "rocblas_cgemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgemv_v2",                 {"hipblasZgemv",                    "rocblas_zgemv",                            CONV_LIB_FUNC, API_BLAS}},

  // GBMV
  {"cublasSgbmv_v2",                 {"hipblasSgbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDgbmv_v2",                 {"hipblasDgbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgbmv_v2",                 {"hipblasCgbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgbmv_v2",                 {"hipblasZgbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TRMV
  {"cublasStrmv_v2",                 {"hipblasStrmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtrmv_v2",                 {"hipblasDtrmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtrmv_v2",                 {"hipblasCtrmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtrmv_v2",                 {"hipblasZtrmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TBMV
  {"cublasStbmv_v2",                 {"hipblasStbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtbmv_v2",                 {"hipblasDtbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtbmv_v2",                 {"hipblasCtbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtbmv_v2",                 {"hipblasZtbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TPMV
  {"cublasStpmv_v2",                 {"hipblasStpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtpmv_v2",                 {"hipblasDtpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtpmv_v2",                 {"hipblasCtpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtpmv_v2",                 {"hipblasZtpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TRSV
  {"cublasStrsv_v2",                 {"hipblasStrsv",                    "rocblas_strsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrsv_v2",                 {"hipblasDtrsv",                    "rocblas_dtrsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrsv_v2",                 {"hipblasCtrsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtrsv_v2",                 {"hipblasZtrsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TPSV
  {"cublasStpsv_v2",                 {"hipblasStpsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtpsv_v2",                 {"hipblasDtpsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtpsv_v2",                 {"hipblasCtpsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtpsv_v2",                 {"hipblasZtpsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TBSV
  {"cublasStbsv_v2",                 {"hipblasStbsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtbsv_v2",                 {"hipblasDtbsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtbsv_v2",                 {"hipblasCtbsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtbsv_v2",                 {"hipblasZtbsv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYMV/HEMV
  {"cublasSsymv_v2",                 {"hipblasSsymv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsymv_v2",                 {"hipblasDsymv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCsymv_v2",                 {"hipblasCsymv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsymv_v2",                 {"hipblasZsymv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasChemv_v2",                 {"hipblasChemv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhemv_v2",                 {"hipblasZhemv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SBMV/HBMV
  {"cublasSsbmv_v2",                 {"hipblasSsbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsbmv_v2",                 {"hpiblasDsbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasChbmv_v2",                 {"hipblasChbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhbmv_v2",                 {"hipblasZhbmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SPMV/HPMV
  {"cublasSspmv_v2",                 {"hipblasSspmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDspmv_v2",                 {"hipblasDspmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasChpmv_v2",                 {"hipblasChpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhpmv_v2",                 {"hipblasZhpmv",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // GER
  {"cublasSger_v2",                  {"hipblasSger",                     "rocblas_sger",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDger_v2",                  {"hipblasDger",                     "rocblas_dger",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgeru_v2",                 {"hipblasCgeru",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgerc_v2",                 {"hipblasCgerc",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgeru_v2",                 {"hipblasZgeru",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgerc_v2",                 {"hipblasZgerc",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYR/HER
  {"cublasSsyr_v2",                  {"hipblasSsyr",                     "rocblas_ssyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsyr_v2",                  {"hipblasDsyr",                     "rocblas_dsyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsyr_v2",                  {"hipblasCsyr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsyr_v2",                  {"hipblasZsyr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCher_v2",                  {"hipblasCher",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZher_v2",                  {"hipblasZher",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SPR/HPR
  {"cublasSspr_v2",                  {"hipblasSspr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDspr_v2",                  {"hipblasDspr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasChpr_v2",                  {"hipblasChpr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhpr_v2",                  {"hipblasZhpr",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYR2/HER2
  {"cublasSsyr2_v2",                 {"hipblasSsyr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsyr2_v2",                 {"hipblasDsyr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCsyr2_v2",                 {"hipblasCsyr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsyr2_v2",                 {"hipblasZsyr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCher2_v2",                 {"hipblasCher2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZher2_v2",                 {"hipblasZher2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SPR2/HPR2
  {"cublasSspr2_v2",                 {"hipblasSspr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDspr2_v2",                 {"hipblasDspr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasChpr2_v2",                 {"hipblasChpr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhpr2_v2",                 {"hipblasZhpr2",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Blas3 (v2) Routines
  // GEMM
  {"cublasSgemm_v2",                 {"hipblasSgemm",                    "rocblas_sgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemm_v2",                 {"hipblasDgemm",                    "rocblas_dgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm_v2",                 {"hipblasCgemm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgemm3m",                  {"hipblasCgemm3m",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgemm3mEx",                {"hipblasCgemm3mEx",                "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgemm_v2",                 {"hipblasZgemm",                    "rocblas_zgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgemm3m",                  {"hipblasZgemm3m",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  //IO in FP16 / FP32, computation in float
  {"cublasSgemmEx",                  {"hipblasSgemmEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGemmEx",                   {"hipblasGemmEx",                   "rocblas_gemm_ex",                          CONV_LIB_FUNC, API_BLAS}},
  {"cublasGemmBatchedEx",            {"hipblasGemmBatchedEx",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGemmStridedBatchedEx",     {"hipblasGemmStridedBatchedEx",     "rocblas_gemm_strided_batched_ex",          CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCgemmEx",                  {"hipblasCgemmEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasUint8gemmBias",            {"hipblasUint8gemmBias",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYRK
  {"cublasSsyrk_v2",                 {"hipblasSsyrk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsyrk_v2",                 {"hipblasDsyrk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCsyrk_v2",                 {"hipblasCsyrk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsyrk_v2",                 {"hipblasZsyrk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCsyrkEx",                  {"hipblasCsyrkEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
  {"cublasCsyrk3mEx",                {"hipblasCsyrk3mEx",                "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // HERK
  {"cublasCherk_v2",                 {"hipblasCherk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCherkEx",                  {"hipblasCherkEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
  {"cublasCherk3mEx",                {"hipblasCherk3mEx",                "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZherk_v2",                 {"hipblasZherk",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYR2K
  {"cublasSsyr2k_v2",                {"hipblasSsyr2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsyr2k_v2",                {"hipblasDsyr2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCsyr2k_v2",                {"hipblasCsyr2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsyr2k_v2",                {"hipblasZsyr2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // HER2K
  {"cublasCher2k_v2",                {"hipblasCher2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZher2k_v2",                {"hipblasZher2k",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYMM
  {"cublasSsymm_v2",                 {"hipblasSsymm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDsymm_v2",                 {"hipblasDsymm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCsymm_v2",                 {"hipblasCsymm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZsymm_v2",                 {"hipblasZsymm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // HEMM
  {"cublasChemm_v2",                 {"hipblasChemm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZhemm_v2",                 {"hipblasZhemm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TRSM
  {"cublasStrsm_v2",                 {"hipblasStrsm",                    "rocblas_strsm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrsm_v2",                 {"hipblasDtrsm",                    "rocblas_dtrsm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrsm_v2",                 {"hipblasCtrsm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtrsm_v2",                 {"hipblasZtrsm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TRMM
  {"cublasStrmm_v2",                 {"hipblasStrmm",                    "rocblas_strmm",                            CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrmm_v2",                 {"hipblasDtrmm",                    "rocblas_dtrmm",                            CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrmm_v2",                 {"hipblasCtrmm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtrmm_v2",                 {"hipblasZtrmm",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // NRM2
  {"cublasSnrm2_v2",                 {"hipblasSnrm2",                    "rocblas_snrm2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDnrm2_v2",                 {"hipblasDnrm2",                    "rocblas_dnrm2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasScnrm2_v2",                {"hipblasScnrm2",                   "rocblas_scnrm2",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDznrm2_v2",                {"hipblasDznrm2",                   "rocblas_dznrm2",                           CONV_LIB_FUNC, API_BLAS}},

  // DOT
  {"cublasDotEx",                    {"hipblasDotEx",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDotcEx",                   {"hipblasDotcEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  {"cublasSdot_v2",                  {"hipblasSdot",                     "rocblas_sdot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDdot_v2",                  {"hipblasDdot",                     "rocblas_ddot",                             CONV_LIB_FUNC, API_BLAS}},

  {"cublasCdotu_v2",                 {"hipblasCdotu",                    "rocblas_cdotu",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCdotc_v2",                 {"hipblasCdotc",                    "rocblas_cdotc",                            CONV_LIB_FUNC, API_BLAS,}},
  {"cublasZdotu_v2",                 {"hipblasZdotu",                    "rocblas_zdotu",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdotc_v2",                 {"hipblasZdotc",                    "rocblas_zdotc",                            CONV_LIB_FUNC, API_BLAS}},

  // SCAL
  {"cublasScalEx",                   {"hipblasScalEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSscal_v2",                 {"hipblasSscal",                    "rocblas_sscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDscal_v2",                 {"hipblasDscal",                    "rocblas_dscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCscal_v2",                 {"hipblasCscal",                    "rocblas_cscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsscal_v2",                {"hipblasCsscal",                   "rocblas_csscal",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasZscal_v2",                 {"hipblasZscal",                    "rocblas_zscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdscal_v2",                {"hipblasZdscal",                   "rocblas_zdscal",                           CONV_LIB_FUNC, API_BLAS}},

  // AXPY
  {"cublasAxpyEx",                   {"hipblasAxpyEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSaxpy_v2",                 {"hipblasSaxpy",                    "rocblas_saxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDaxpy_v2",                 {"hipblasDaxpy",                    "rocblas_daxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCaxpy_v2",                 {"hipblasCaxpy",                    "rocblas_caxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZaxpy_v2",                 {"hipblasZaxpy",                    "rocblas_zaxpy",                            CONV_LIB_FUNC, API_BLAS}},

  // COPY
  {"cublasCopyEx",                   {"hipblasCopyEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasScopy_v2",                 {"hipblasScopy",                    "rocblas_scopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDcopy_v2",                 {"hipblasDcopy",                    "rocblas_dcopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCcopy_v2",                 {"hipblasCcopy",                    "rocblas_ccopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZcopy_v2",                 {"hipblasZcopy",                    "rocblas_zcopy",                            CONV_LIB_FUNC, API_BLAS}},

  // SWAP
  {"cublasSwapEx",                   {"hipblasSwapEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSswap_v2",                 {"hipblasSswap",                    "rocblas_sswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDswap_v2",                 {"hipblasDswap",                    "rocblas_dswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCswap_v2",                 {"hipblasCswap",                    "rocblas_cswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZswap_v2",                 {"hipblasZswap",                    "rocblas_zswap",                            CONV_LIB_FUNC, API_BLAS}},

  // AMAX
  {"cublasIamaxEx",                  {"hipblasIamaxEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasIsamax_v2",                {"hipblasIsamax",                   "rocblas_isamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIdamax_v2",                {"hipblasIdamax",                   "rocblas_idamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIcamax_v2",                {"hipblasIcamax",                   "rocblas_icamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIzamax_v2",                {"hipblasIzamax",                   "rocblas_izamax",                           CONV_LIB_FUNC, API_BLAS}},

  // AMIN
  {"cublasIaminEx",                  {"hipblasIaminEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasIsamin_v2",                {"hipblasIsamin",                   "rocblas_isamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIdamin_v2",                {"hipblasIdamin",                   "rocblas_idamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIcamin_v2",                {"hipblasIcamin",                   "rocblas_icamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIzamin_v2",                {"hipblasIzamin",                   "rocblas_izamin",                           CONV_LIB_FUNC, API_BLAS}},

  // ASUM
  {"cublasAsumEx",                   {"hipblasAsumEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSasum_v2",                 {"hipblasSasum",                    "rocblas_sasum",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDasum_v2",                 {"hipblasDasum",                    "rocblas_dasum",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasScasum_v2",                {"hipblasScasum",                   "rocblas_scasum",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDzasum_v2",                {"hipblasDzasum",                   "rocblas_dzasum",                           CONV_LIB_FUNC, API_BLAS}},

  // ROT
  {"cublasRotEx",                    {"hipblasRotEx",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSrot_v2",                  {"hipblasSrot",                     "rocblas_srot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrot_v2",                  {"hipblasDrot",                     "rocblas_drot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCrot_v2",                  {"hipblasCrot",                     "rocblas_crot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsrot_v2",                 {"hipblasCsrot",                    "rocblas_csrot",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZrot_v2",                  {"hipblasZrot",                     "rocblas_zrot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdrot_v2",                 {"hipblasZdrot",                    "rocblas_zdrot",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTG
  {"cublasRotgEx",                   {"hipblasRotgEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSrotg_v2",                 {"hipblasSrotg",                    "rocblas_srotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotg_v2",                 {"hipblasDrotg",                    "rocblas_drotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCrotg_v2",                 {"hipblasCrotg",                    "rocblas_crotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZrotg_v2",                 {"hipblasZrotg",                    "rocblas_zrotg",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTM
  {"cublasRotmEx",                   {"hipblasRotmEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSrotm_v2",                 {"hipblasSrotm",                    "rocblas_srotm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotm_v2",                 {"hipblasDrotm",                    "rocblas_drotm",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTMG
  {"cublasRotmgEx",                  {"hipblasRotmgEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSrotmg_v2",                {"hipblasSrotmg",                   "rocblas_srotmg",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotmg_v2",                {"hipblasDrotmg",                   "rocblas_drotmg",                           CONV_LIB_FUNC, API_BLAS}},
};
