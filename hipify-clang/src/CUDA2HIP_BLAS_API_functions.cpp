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
  {"cublasInit",                     {"hipblasInit",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasShutdown",                 {"hipblasShutdown",                 "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetVersion",               {"hipblasGetVersion",               "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetError",                 {"hipblasGetError",                 "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasAlloc",                    {"hipblasAlloc",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasFree",                     {"hipblasFree",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetKernelStream",          {"hipblasSetKernelStream",          "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetAtomicsMode",           {"hipblasGetAtomicsMode",           "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetAtomicsMode",           {"hipblasSetAtomicsMode",           "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetMathMode",              {"hipblasGetMathMode",              "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetMathMode",              {"hipblasSetMathMode",              "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Blas logging
  {"cublasLogCallback",              {"hipblasLogCallback",              "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasLoggerConfigure",          {"hipblasLoggerConfigure",          "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetLoggerCallback",        {"hipblasSetLoggerCallback",        "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetLoggerCallback",        {"hipblasGetLoggerCallback",        "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Blas1 (v1) Routines
  {"cublasCreate",                   {"hipblasCreate",                   "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDestroy",                  {"hipblasDestroy",                  "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetStream",                {"hipblasSetStream",                "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetStream",                {"hipblasGetStream",                "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetPointerMode",           {"hipblasGetPointerMode",           "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetPointerMode",           {"hipblasSetPointerMode",           "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetVector",                {"hipblasSetVector",                "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetVector",                {"hipblasGetVector",                "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetVectorAsync",           {"hipblasSetVectorAsync",           "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetVectorAsync",           {"hipblasGetVectorAsync",           "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetMatrix",                {"hipblasSetMatrix",                "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetMatrix",                {"hipblasGetMatrix",                "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetMatrixAsync",           {"hipblasGetMatrixAsync",           "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetMatrixAsync",           {"hipblasSetMatrixAsync",           "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasXerbla",                   {"hipblasXerbla",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // NRM2
  {"cublasSnrm2",                    {"hipblasSnrm2",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDnrm2",                    {"hipblasDnrm2",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasScnrm2",                   {"hipblasScnrm2",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDznrm2",                   {"hipblasDznrm2",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasNrm2Ex",                   {"hipblasNrm2Ex",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // DOT
  {"cublasSdot",                     {"hipblasSdot",                     "", CONV_LIB_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasSdotBatched",              {"hipblasSdotBatched",              "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDdot",                     {"hipblasDdot",                     "", CONV_LIB_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasDdotBatched",              {"hipblasDdotBatched",              "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCdotu",                    {"hipblasCdotu",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCdotc",                    {"hipblasCdotc",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdotu",                    {"hipblasZdotu",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdotc",                    {"hipblasZdotc",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SCAL
  {"cublasSscal",                    {"hipblasSscal",                    "", CONV_LIB_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasSscalBatched",             {"hipblasSscalBatched",             "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDscal",                    {"hipblasDscal",                    "", CONV_LIB_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasDscalBatched",             {"hipblasDscalBatched",             "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCscal",                    {"hipblasCscal",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsscal",                   {"hipblasCsscal",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZscal",                    {"hipblasZscal",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdscal",                   {"hipblasZdscal",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AXPY
  {"cublasSaxpy",                    {"hipblasSaxpy",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasSaxpyBatched",             {"hipblasSaxpyBatched",             "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDaxpy",                    {"hipblasDaxpy",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCaxpy",                    {"hipblasCaxpy",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZaxpy",                    {"hipblasZaxpy",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // COPY
  {"cublasScopy",                    {"hipblasScopy",                    "", CONV_LIB_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasScopyBatched",             {"hipblasScopyBatched",             "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDcopy",                    {"hipblasDcopy",                    "", CONV_LIB_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasDcopyBatched",             {"hipblasDcopyBatched",             "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCcopy",                    {"hipblasCcopy",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZcopy",                    {"hipblasZcopy",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SWAP
  {"cublasSswap",                    {"hipblasSswap",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDswap",                    {"hipblasDswap",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCswap",                    {"hipblasCswap",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZswap",                    {"hipblasZswap",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AMAX
  {"cublasIsamax",                   {"hipblasIsamax",                   "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasIdamax",                   {"hipblasIdamax",                   "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasIcamax",                   {"hipblasIcamax",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIzamax",                   {"hipblasIzamax",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AMIN
  {"cublasIsamin",                   {"hipblasIsamin",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIdamin",                   {"hipblasIdamin",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIcamin",                   {"hipblasIcamin",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIzamin",                   {"hipblasIzamin",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ASUM
  {"cublasSasum",                    {"hipblasSasum",                    "", CONV_LIB_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasSasumBatched",             {"hipblasSasumBatched",             "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDasum",                    {"hipblasDasum",                    "", CONV_LIB_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasDasumBatched",             {"hipblasDasumBatched",             "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasScasum",                   {"hipblasScasum",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDzasum",                   {"hipblasDzasum",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROT
  {"cublasSrot",                     {"hipblasSrot",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrot",                     {"hipblasDrot",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCrot",                     {"hipblasCrot",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsrot",                    {"hipblasCsrot",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZrot",                     {"hipblasZrot",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdrot",                    {"hipblasZdrot",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTG
  {"cublasSrotg",                    {"hipblasSrotg",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotg",                    {"hipblasDrotg",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCrotg",                    {"hipblasCrotg",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZrotg",                    {"hipblasZrotg",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTM
  {"cublasSrotm",                    {"hipblasSrotm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotm",                    {"hipblasDrotm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTMG
  {"cublasSrotmg",                   {"hipblasSrotmg",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotmg",                   {"hipblasDrotmg",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GEMV
  {"cublasSgemv",                    {"hipblasSgemv",                    "", CONV_LIB_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasSgemvBatched",             {"hipblasSgemvBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgemv",                    {"hipblasDgemv",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemv",                    {"hipblasCgemv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemv",                    {"hipblasZgemv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GBMV
  {"cublasSgbmv",                    {"hipblasSgbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgbmv",                    {"hipblasDgbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgbmv",                    {"hipblasCgbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgbmv",                    {"hipblasZgbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRMV
  {"cublasStrmv",                    {"hipblasStrmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrmv",                    {"hipblasDtrmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrmv",                    {"hipblasCtrmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrmv",                    {"hipblasZtrmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TBMV
  {"cublasStbmv",                    {"hipblasStbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtbmv",                    {"hipblasDtbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtbmv",                    {"hipblasCtbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtbmv",                    {"hipblasZtbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TPMV
  {"cublasStpmv",                    {"hipblasStpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtpmv",                    {"hipblasDtpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtpmv",                    {"hipblasCtpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtpmv",                    {"hipblasZtpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSV
  {"cublasStrsv",                    {"hipblasStrsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrsv",                    {"hipblasDtrsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrsv",                    {"hipblasCtrsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsv",                    {"hipblasZtrsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TPSV
  {"cublasStpsv",                    {"hipblasStpsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtpsv",                    {"hipblasDtpsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtpsv",                    {"hipblasCtpsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtpsv",                    {"hipblasZtpsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TBSV
  {"cublasStbsv",                    {"hipblasStbsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtbsv",                    {"hipblasDtbsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtbsv",                    {"hipblasCtbsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtbsv",                    {"hipblasZtbsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYMV/HEMV
  {"cublasSsymv",                    {"hipblasSsymv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsymv",                    {"hipblasDsymv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsymv",                    {"hipblasCsymv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsymv",                    {"hipblasZsymv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChemv",                    {"hipblasChemv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhemv",                    {"hipblasZhemv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SBMV/HBMV
  {"cublasSsbmv",                    {"hipblasSsbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsbmv",                    {"hpiblasDsbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChbmv",                    {"hipblasChbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhbmv",                    {"hipblasZhbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPMV/HPMV
  {"cublasSspmv",                    {"hipblasSspmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspmv",                    {"hipblasDspmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpmv",                    {"hipblasChpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpmv",                    {"hipblasZhpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GER
  {"cublasSger",                     {"hipblasSger",                     "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDger",                     {"hipblasDger",                     "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgeru",                    {"hipblasCgeru",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgerc",                    {"hipblasCgerc",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgeru",                    {"hipblasZgeru",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgerc",                    {"hipblasZgerc",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR/HER
  {"cublasSsyr",                     {"hipblasSsyr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr",                     {"hipblasDsyr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr",                     {"hipblasCsyr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr",                     {"hipblasZsyr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCher",                     {"hipblasCher",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher",                     {"hipblasZher",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPR/HPR
  {"cublasSspr",                     {"hipblasSspr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspr",                     {"hipblasDspr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpr",                     {"hipblasChpr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpr",                     {"hipblasZhpr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR2/HER2
  {"cublasSsyr2",                    {"hipblasSsyr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr2",                    {"hipblasDsyr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr2",                    {"hipblasCsyr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr2",                    {"hipblasZsyr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCher2",                    {"hipblasCher2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher2",                    {"hipblasZher2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPR2/HPR2
  {"cublasSspr2",                    {"hipblasSspr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspr2",                    {"hipblasDspr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpr2",                    {"hipblasChpr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpr2",                    {"hipblasZhpr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Blas3 (v1) Routines
  // GEMM
  {"cublasSgemm",                    {"hipblasSgemm",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemm",                    {"hipblasDgemm",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm",                    {"hipblasCgemm",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgemm",                    {"hipblasZgemm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasHgemm",                    {"hipblasHgemm",                    "", CONV_LIB_FUNC, API_BLAS}},

  // BATCH GEMM
  {"cublasSgemmBatched",             {"hipblasSgemmBatched",             "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemmBatched",             {"hipblasDgemmBatched",             "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasHgemmBatched",             {"hipblasHgemmBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSgemmStridedBatched",      {"hipblasSgemmStridedBatched",      "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgemmStridedBatched",      {"hipblasDgemmStridedBatched",      "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgemmBatched",             {"hipblasCgemmBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgemm3mBatched",           {"hipblasCgemm3mBatched",           "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemmBatched",             {"hipblasZgemmBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgemmStridedBatched",      {"hipblasCgemmStridedBatched",      "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgemm3mStridedBatched",    {"hipblasCgemm3mStridedBatched",    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemmStridedBatched",      {"hipblasZgemmStridedBatched",      "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasHgemmStridedBatched",      {"hipblasHgemmStridedBatched",      "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYRK
  {"cublasSsyrk",                    {"hipblasSsyrk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyrk",                    {"hipblasDsyrk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyrk",                    {"hipblasCsyrk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyrk",                    {"hipblasZsyrk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HERK
  {"cublasCherk",                    {"hipblasCherk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZherk",                    {"hipblasZherk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR2K
  {"cublasSsyr2k",                   {"hipblasSsyr2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr2k",                   {"hipblasDsyr2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr2k",                   {"hipblasCsyr2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr2k",                   {"hipblasZsyr2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYRKX - eXtended SYRK
  {"cublasSsyrkx",                   {"hipblasSsyrkx",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyrkx",                   {"hipblasDsyrkx",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyrkx",                   {"hipblasCsyrkx",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyrkx",                   {"hipblasZsyrkx",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HER2K
  {"cublasCher2k",                   {"hipblasCher2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher2k",                   {"hipblasZher2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HERKX - eXtended HERK
  {"cublasCherkx",                   {"hipblasCherkx",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZherkx",                   {"hipblasZherkx",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYMM
  {"cublasSsymm",                    {"hipblasSsymm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsymm",                    {"hipblasDsymm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsymm",                    {"hipblasCsymm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsymm",                    {"hipblasZsymm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HEMM
  {"cublasChemm",                    {"hipblasChemm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhemm",                    {"hipblasZhemm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSM
  {"cublasStrsm",                    {"hipblasStrsm",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrsm",                    {"hipblasDtrsm",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrsm",                    {"hipblasCtrsm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsm",                    {"hipblasZtrsm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSM - Batched Triangular Solver
  {"cublasStrsmBatched",             {"hipblasStrsmBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrsmBatched",             {"hipblasDtrsmBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrsmBatched",             {"hipblasCtrsmBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsmBatched",             {"hipblasZtrsmBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRMM
  {"cublasStrmm",                    {"hipblasStrmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrmm",                    {"hipblasDtrmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrmm",                    {"hipblasCtrmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrmm",                    {"hipblasZtrmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ------------------------ CUBLAS BLAS - like extension (cublas_api.h)
  // GEAM
  {"cublasSgeam",                    {"hipblasSgeam",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgeam",                    {"hipblasDgeam",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgeam",                    {"hipblasCgeam",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgeam",                    {"hipblasZgeam",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GETRF - Batched LU
  {"cublasSgetrfBatched",            {"hipblasSgetrfBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgetrfBatched",            {"hipblasDgetrfBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgetrfBatched",            {"hipblasCgetrfBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgetrfBatched",            {"hipblasZgetrfBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Batched inversion based on LU factorization from getrf
  {"cublasSgetriBatched",            {"hipblasSgetriBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgetriBatched",            {"hipblasDgetriBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgetriBatched",            {"hipblasCgetriBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgetriBatched",            {"hipblasZgetriBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Batched solver based on LU factorization from getrf
  {"cublasSgetrsBatched",            {"hipblasSgetrsBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgetrsBatched",            {"hipblasDgetrsBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgetrsBatched",            {"hipblasCgetrsBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgetrsBatched",            {"hipblasZgetrsBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSM - Batched Triangular Solver
  {"cublasStrsmBatched",             {"hipblasStrsmBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrsmBatched",             {"hipblasDtrsmBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrsmBatched",             {"hipblasCtrsmBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsmBatched",             {"hipblasZtrsmBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // MATINV - Batched
  {"cublasSmatinvBatched",           {"hipblasSmatinvBatched",           "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDmatinvBatched",           {"hipblasDmatinvBatched",           "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCmatinvBatched",           {"hipblasCmatinvBatched",           "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZmatinvBatched",           {"hipblasZmatinvBatched",           "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Batch QR Factorization
  {"cublasSgeqrfBatched",            {"hipblasSgeqrfBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgeqrfBatched",            {"hipblasDgeqrfBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgeqrfBatched",            {"hipblasCgeqrfBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgeqrfBatched",            {"hipblasZgeqrfBatched",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Least Square Min only m >= n and Non-transpose supported
  {"cublasSgelsBatched",             {"hipblasSgelsBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgelsBatched",             {"hipblasDgelsBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgelsBatched",             {"hipblasCgelsBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgelsBatched",             {"hipblasZgelsBatched",             "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // DGMM
  {"cublasSdgmm",                    {"hipblasSdgmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDdgmm",                    {"hipblasDdgmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCdgmm",                    {"hipblasCdgmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdgmm",                    {"hipblasZdgmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TPTTR - Triangular Pack format to Triangular format
  {"cublasStpttr",                   {"hipblasStpttr",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtpttr",                   {"hipblasDtpttr",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtpttr",                   {"hipblasCtpttr",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtpttr",                   {"hipblasZtpttr",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRTTP - Triangular format to Triangular Pack format
  {"cublasStrttp",                   {"hipblasStrttp",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrttp",                   {"hipblasDtrttp",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrttp",                   {"hipblasCtrttp",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrttp",                   {"hipblasZtrttp",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Blas2 (v2) Routines
  {"cublasCreate_v2",                {"hipblasCreate",                   "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDestroy_v2",               {"hipblasDestroy",                  "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetVersion_v2",            {"hipblasGetVersion",               "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetProperty",              {"hipblasGetProperty",              "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetStream_v2",             {"hipblasSetStream",                "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetStream_v2",             {"hipblasGetStream",                "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetPointerMode_v2",        {"hipblasGetPointerMode",           "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetPointerMode_v2",        {"hipblasSetPointerMode",           "", CONV_LIB_FUNC, API_BLAS}},

  // GEMV
  {"cublasSgemv_v2",                 {"hipblasSgemv",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemv_v2",                 {"hipblasDgemv",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemv_v2",                 {"hipblasCgemv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemv_v2",                 {"hipblasZgemv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GBMV
  {"cublasSgbmv_v2",                 {"hipblasSgbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgbmv_v2",                 {"hipblasDgbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgbmv_v2",                 {"hipblasCgbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgbmv_v2",                 {"hipblasZgbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRMV
  {"cublasStrmv_v2",                 {"hipblasStrmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrmv_v2",                 {"hipblasDtrmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrmv_v2",                 {"hipblasCtrmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrmv_v2",                 {"hipblasZtrmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TBMV
  {"cublasStbmv_v2",                 {"hipblasStbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtbmv_v2",                 {"hipblasDtbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtbmv_v2",                 {"hipblasCtbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtbmv_v2",                 {"hipblasZtbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TPMV
  {"cublasStpmv_v2",                 {"hipblasStpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtpmv_v2",                 {"hipblasDtpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtpmv_v2",                 {"hipblasCtpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtpmv_v2",                 {"hipblasZtpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSV
  {"cublasStrsv_v2",                 {"hipblasStrsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrsv_v2",                 {"hipblasDtrsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrsv_v2",                 {"hipblasCtrsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsv_v2",                 {"hipblasZtrsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TPSV
  {"cublasStpsv_v2",                 {"hipblasStpsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtpsv_v2",                 {"hipblasDtpsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtpsv_v2",                 {"hipblasCtpsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtpsv_v2",                 {"hipblasZtpsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TBSV
  {"cublasStbsv_v2",                 {"hipblasStbsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtbsv_v2",                 {"hipblasDtbsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtbsv_v2",                 {"hipblasCtbsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtbsv_v2",                 {"hipblasZtbsv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYMV/HEMV
  {"cublasSsymv_v2",                 {"hipblasSsymv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsymv_v2",                 {"hipblasDsymv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsymv_v2",                 {"hipblasCsymv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsymv_v2",                 {"hipblasZsymv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChemv_v2",                 {"hipblasChemv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhemv_v2",                 {"hipblasZhemv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SBMV/HBMV
  {"cublasSsbmv_v2",                 {"hipblasSsbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsbmv_v2",                 {"hpiblasDsbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChbmv_v2",                 {"hipblasChbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhbmv_v2",                 {"hipblasZhbmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPMV/HPMV
  {"cublasSspmv_v2",                 {"hipblasSspmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspmv_v2",                 {"hipblasDspmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpmv_v2",                 {"hipblasChpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpmv_v2",                 {"hipblasZhpmv",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GER
  {"cublasSger_v2",                  {"hipblasSger",                     "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDger_v2",                  {"hipblasDger",                     "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgeru_v2",                 {"hipblasCgeru",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgerc_v2",                 {"hipblasCgerc",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgeru_v2",                 {"hipblasZgeru",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgerc_v2",                 {"hipblasZgerc",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR/HER
  {"cublasSsyr_v2",                  {"hipblasSsyr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr_v2",                  {"hipblasDsyr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr_v2",                  {"hipblasCsyr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr_v2",                  {"hipblasZsyr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCher_v2",                  {"hipblasCher",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher_v2",                  {"hipblasZher",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPR/HPR
  {"cublasSspr_v2",                  {"hipblasSspr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspr_v2",                  {"hipblasDspr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpr_v2",                  {"hipblasChpr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpr_v2",                  {"hipblasZhpr",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR2/HER2
  {"cublasSsyr2_v2",                 {"hipblasSsyr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr2_v2",                 {"hipblasDsyr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr2_v2",                 {"hipblasCsyr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr2_v2",                 {"hipblasZsyr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCher2_v2",                 {"hipblasCher2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher2_v2",                 {"hipblasZher2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPR2/HPR2
  {"cublasSspr2_v2",                 {"hipblasSspr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspr2_v2",                 {"hipblasDspr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpr2_v2",                 {"hipblasChpr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpr2_v2",                 {"hipblasZhpr2",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Blas3 (v2) Routines
  // GEMM
  {"cublasSgemm_v2",                 {"hipblasSgemm",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemm_v2",                 {"hipblasDgemm",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm_v2",                 {"hipblasCgemm",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm3m",                  {"hipblasCgemm3m",                  "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgemm3mEx",                {"hipblasCgemm3mEx",                "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemm_v2",                 {"hipblasZgemm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemm3m",                  {"hipblasZgemm3m",                  "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  //IO in FP16 / FP32, computation in float
  {"cublasSgemmEx",                  {"hipblasSgemmEx",                  "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGemmEx",                   {"hipblasGemmEx",                   "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasGemmBatchedEx",            {"hipblasGemmBatchedEx",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGemmStridedBatchedEx",     {"hipblasGemmStridedBatchedEx",     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCgemmEx",                  {"hipblasCgemmEx",                  "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasUint8gemmBias",            {"hipblasUint8gemmBias",            "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYRK
  {"cublasSsyrk_v2",                 {"hipblasSsyrk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyrk_v2",                 {"hipblasDsyrk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyrk_v2",                 {"hipblasCsyrk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyrk_v2",                 {"hipblasZsyrk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCsyrkEx",                  {"hipblasCsyrkEx",                  "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
  {"cublasCsyrk3mEx",                {"hipblasCsyrk3mEx",                "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HERK
  {"cublasCherk_v2",                 {"hipblasCherk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCherkEx",                  {"hipblasCherkEx",                  "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
  {"cublasCherk3mEx",                {"hipblasCherk3mEx",                "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZherk_v2",                 {"hipblasZherk",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR2K
  {"cublasSsyr2k_v2",                {"hipblasSsyr2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr2k_v2",                {"hipblasDsyr2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr2k_v2",                {"hipblasCsyr2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr2k_v2",                {"hipblasZsyr2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HER2K
  {"cublasCher2k_v2",                {"hipblasCher2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher2k_v2",                {"hipblasZher2k",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYMM
  {"cublasSsymm_v2",                 {"hipblasSsymm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsymm_v2",                 {"hipblasDsymm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsymm_v2",                 {"hipblasCsymm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsymm_v2",                 {"hipblasZsymm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HEMM
  {"cublasChemm_v2",                 {"hipblasChemm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhemm_v2",                 {"hipblasZhemm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSM
  {"cublasStrsm_v2",                 {"hipblasStrsm",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrsm_v2",                 {"hipblasDtrsm",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrsm_v2",                 {"hipblasCtrsm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsm_v2",                 {"hipblasZtrsm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRMM
  {"cublasStrmm_v2",                 {"hipblasStrmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrmm_v2",                 {"hipblasDtrmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrmm_v2",                 {"hipblasCtrmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrmm_v2",                 {"hipblasZtrmm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // NRM2
  {"cublasSnrm2_v2",                 {"hipblasSnrm2",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDnrm2_v2",                 {"hipblasDnrm2",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasScnrm2_v2",                {"hipblasScnrm2",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDznrm2_v2",                {"hipblasDznrm2",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // DOT
  {"cublasDotEx",                    {"hipblasDotEx",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDotcEx",                   {"hipblasDotcEx",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  {"cublasSdot_v2",                  {"hipblasSdot",                     "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDdot_v2",                  {"hipblasDdot",                     "", CONV_LIB_FUNC, API_BLAS}},

  {"cublasCdotu_v2",                 {"hipblasCdotu",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCdotc_v2",                 {"hipblasCdotc",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdotu_v2",                 {"hipblasZdotu",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdotc_v2",                 {"hipblasZdotc",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SCAL
  {"cublasScalEx",                   {"hipblasScalEx",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSscal_v2",                 {"hipblasSscal",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDscal_v2",                 {"hipblasDscal",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCscal_v2",                 {"hipblasCscal",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsscal_v2",                {"hipblasCsscal",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZscal_v2",                 {"hipblasZscal",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdscal_v2",                {"hipblasZdscal",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AXPY
  {"cublasAxpyEx",                   {"hipblasAxpyEx",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSaxpy_v2",                 {"hipblasSaxpy",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDaxpy_v2",                 {"hipblasDaxpy",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCaxpy_v2",                 {"hipblasCaxpy",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZaxpy_v2",                 {"hipblasZaxpy",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // COPY
  {"cublasScopy_v2",                 {"hipblasScopy",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDcopy_v2",                 {"hipblasDcopy",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasCcopy_v2",                 {"hipblasCcopy",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZcopy_v2",                 {"hipblasZcopy",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SWAP
  {"cublasSswap_v2",                 {"hipblasSswap",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDswap_v2",                 {"hipblasDswap",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCswap_v2",                 {"hipblasCswap",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZswap_v2",                 {"hipblasZswap",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AMAX
  {"cublasIsamax_v2",                {"hipblasIsamax",                   "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasIdamax_v2",                {"hipblasIdamax",                   "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasIcamax_v2",                {"hipblasIcamax",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIzamax_v2",                {"hipblasIzamax",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AMIN
  {"cublasIsamin_v2",                {"hipblasIsamin",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIdamin_v2",                {"hipblasIdamin",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIcamin_v2",                {"hipblasIcamin",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIzamin_v2",                {"hipblasIzamin",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ASUM
  {"cublasSasum_v2",                 {"hipblasSasum",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasDasum_v2",                 {"hipblasDasum",                    "", CONV_LIB_FUNC, API_BLAS}},
  {"cublasScasum_v2",                {"hipblasScasum",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDzasum_v2",                {"hipblasDzasum",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROT
  {"cublasSrot_v2",                  {"hipblasSrot",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrot_v2",                  {"hipblasDrot",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCrot_v2",                  {"hipblasCrot",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsrot_v2",                 {"hipblasCsrot",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZrot_v2",                  {"hipblasZrot",                     "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdrot_v2",                 {"hipblasZdrot",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTG
  {"cublasSrotg_v2",                 {"hipblasSrotg",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotg_v2",                 {"hipblasDrotg",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCrotg_v2",                 {"hipblasCrotg",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZrotg_v2",                 {"hipblasZrotg",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTM
  {"cublasSrotm_v2",                 {"hipblasSrotm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotm_v2",                 {"hipblasDrotm",                    "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTMG
  {"cublasSrotmg_v2",                {"hipblasSrotmg",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotmg_v2",                {"hipblasDrotmg",                   "", CONV_LIB_FUNC, API_BLAS, HIP_UNSUPPORTED}},
};
