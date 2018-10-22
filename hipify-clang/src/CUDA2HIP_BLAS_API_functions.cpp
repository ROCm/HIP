#include "CUDA2HIP.h"

// Map of all functions
const std::map<llvm::StringRef, hipCounter> CUDA_BLAS_FUNCTION_MAP{

  // Blas management functions
  {"cublasInit",                     {"hipblasInit",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasShutdown",                 {"hipblasShutdown",                 CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetVersion",               {"hipblasGetVersion",               CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetError",                 {"hipblasGetError",                 CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasAlloc",                    {"hipblasAlloc",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasFree",                     {"hipblasFree",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetKernelStream",          {"hipblasSetKernelStream",          CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetAtomicsMode",           {"hipblasGetAtomicsMode",           CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetAtomicsMode",           {"hipblasSetAtomicsMode",           CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetMathMode",              {"hipblasGetMathMode",              CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetMathMode",              {"hipblasSetMathMode",              CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Blas logging
  {"cublasLogCallback",              {"hipblasLogCallback",              CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasLoggerConfigure",          {"hipblasLoggerConfigure",          CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetLoggerCallback",        {"hipblasSetLoggerCallback",        CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetLoggerCallback",        {"hipblasGetLoggerCallback",        CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Blas1 (v1) Routines
  {"cublasCreate",                   {"hipblasCreate",                   CONV_MATH_FUNC, API_BLAS}},
  {"cublasDestroy",                  {"hipblasDestroy",                  CONV_MATH_FUNC, API_BLAS}},
  {"cublasSetStream",                {"hipblasSetStream",                CONV_MATH_FUNC, API_BLAS}},
  {"cublasGetStream",                {"hipblasGetStream",                CONV_MATH_FUNC, API_BLAS}},
  {"cublasGetPointerMode",           {"hipblasGetPointerMode",           CONV_MATH_FUNC, API_BLAS}},
  {"cublasSetPointerMode",           {"hipblasSetPointerMode",           CONV_MATH_FUNC, API_BLAS}},
  {"cublasSetVector",                {"hipblasSetVector",                CONV_MATH_FUNC, API_BLAS}},
  {"cublasGetVector",                {"hipblasGetVector",                CONV_MATH_FUNC, API_BLAS}},
  {"cublasSetVectorAsync",           {"hipblasSetVectorAsync",           CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetVectorAsync",           {"hipblasGetVectorAsync",           CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetMatrix",                {"hipblasSetMatrix",                CONV_MATH_FUNC, API_BLAS}},
  {"cublasGetMatrix",                {"hipblasGetMatrix",                CONV_MATH_FUNC, API_BLAS}},
  {"cublasGetMatrixAsync",           {"hipblasGetMatrixAsync",           CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetMatrixAsync",           {"hipblasSetMatrixAsync",           CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasXerbla",                   {"hipblasXerbla",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // NRM2
  {"cublasSnrm2",                    {"hipblasSnrm2",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDnrm2",                    {"hipblasDnrm2",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasScnrm2",                   {"hipblasScnrm2",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDznrm2",                   {"hipblasDznrm2",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasNrm2Ex",                   {"hipblasNrm2Ex",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // DOT
  {"cublasSdot",                     {"hipblasSdot",                     CONV_MATH_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasSdotBatched",              {"hipblasSdotBatched",              CONV_MATH_FUNC, API_BLAS}},
  {"cublasDdot",                     {"hipblasDdot",                     CONV_MATH_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasDdotBatched",              {"hipblasDdotBatched",              CONV_MATH_FUNC, API_BLAS}},
  {"cublasCdotu",                    {"hipblasCdotu",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCdotc",                    {"hipblasCdotc",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdotu",                    {"hipblasZdotu",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdotc",                    {"hipblasZdotc",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SCAL
  {"cublasSscal",                    {"hipblasSscal",                    CONV_MATH_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasSscalBatched",             {"hipblasSscalBatched",             CONV_MATH_FUNC, API_BLAS}},
  {"cublasDscal",                    {"hipblasDscal",                    CONV_MATH_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasDscalBatched",             {"hipblasDscalBatched",             CONV_MATH_FUNC, API_BLAS}},
  {"cublasCscal",                    {"hipblasCscal",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsscal",                   {"hipblasCsscal",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZscal",                    {"hipblasZscal",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdscal",                   {"hipblasZdscal",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AXPY
  {"cublasSaxpy",                    {"hipblasSaxpy",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasSaxpyBatched",             {"hipblasSaxpyBatched",             CONV_MATH_FUNC, API_BLAS}},
  {"cublasDaxpy",                    {"hipblasDaxpy",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCaxpy",                    {"hipblasCaxpy",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZaxpy",                    {"hipblasZaxpy",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // COPY
  {"cublasScopy",                    {"hipblasScopy",                    CONV_MATH_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasScopyBatched",             {"hipblasScopyBatched",             CONV_MATH_FUNC, API_BLAS}},
  {"cublasDcopy",                    {"hipblasDcopy",                    CONV_MATH_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasDcopyBatched",             {"hipblasDcopyBatched",             CONV_MATH_FUNC, API_BLAS}},
  {"cublasCcopy",                    {"hipblasCcopy",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZcopy",                    {"hipblasZcopy",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SWAP
  {"cublasSswap",                    {"hipblasSswap",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDswap",                    {"hipblasDswap",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCswap",                    {"hipblasCswap",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZswap",                    {"hipblasZswap",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AMAX
  {"cublasIsamax",                   {"hipblasIsamax",                   CONV_MATH_FUNC, API_BLAS}},
  {"cublasIdamax",                   {"hipblasIdamax",                   CONV_MATH_FUNC, API_BLAS}},
  {"cublasIcamax",                   {"hipblasIcamax",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIzamax",                   {"hipblasIzamax",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AMIN
  {"cublasIsamin",                   {"hipblasIsamin",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIdamin",                   {"hipblasIdamin",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIcamin",                   {"hipblasIcamin",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIzamin",                   {"hipblasIzamin",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ASUM
  {"cublasSasum",                    {"hipblasSasum",                    CONV_MATH_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasSasumBatched",             {"hipblasSasumBatched",             CONV_MATH_FUNC, API_BLAS}},
  {"cublasDasum",                    {"hipblasDasum",                    CONV_MATH_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasDasumBatched",             {"hipblasDasumBatched",             CONV_MATH_FUNC, API_BLAS}},
  {"cublasScasum",                   {"hipblasScasum",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDzasum",                   {"hipblasDzasum",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROT
  {"cublasSrot",                     {"hipblasSrot",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrot",                     {"hipblasDrot",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCrot",                     {"hipblasCrot",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsrot",                    {"hipblasCsrot",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZrot",                     {"hipblasZrot",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdrot",                    {"hipblasZdrot",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTG
  {"cublasSrotg",                    {"hipblasSrotg",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotg",                    {"hipblasDrotg",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCrotg",                    {"hipblasCrotg",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZrotg",                    {"hipblasZrotg",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTM
  {"cublasSrotm",                    {"hipblasSrotm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotm",                    {"hipblasDrotm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTMG
  {"cublasSrotmg",                   {"hipblasSrotmg",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotmg",                   {"hipblasDrotmg",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GEMV
  {"cublasSgemv",                    {"hipblasSgemv",                    CONV_MATH_FUNC, API_BLAS}},
  // there is no such a function in CUDA
  {"cublasSgemvBatched",             {"hipblasSgemvBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgemv",                    {"hipblasDgemv",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCgemv",                    {"hipblasCgemv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemv",                    {"hipblasZgemv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GBMV
  {"cublasSgbmv",                    {"hipblasSgbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgbmv",                    {"hipblasDgbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgbmv",                    {"hipblasCgbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgbmv",                    {"hipblasZgbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRMV
  {"cublasStrmv",                    {"hipblasStrmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrmv",                    {"hipblasDtrmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrmv",                    {"hipblasCtrmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrmv",                    {"hipblasZtrmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TBMV
  {"cublasStbmv",                    {"hipblasStbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtbmv",                    {"hipblasDtbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtbmv",                    {"hipblasCtbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtbmv",                    {"hipblasZtbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TPMV
  {"cublasStpmv",                    {"hipblasStpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtpmv",                    {"hipblasDtpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtpmv",                    {"hipblasCtpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtpmv",                    {"hipblasZtpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSV
  {"cublasStrsv",                    {"hipblasStrsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrsv",                    {"hipblasDtrsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrsv",                    {"hipblasCtrsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsv",                    {"hipblasZtrsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TPSV
  {"cublasStpsv",                    {"hipblasStpsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtpsv",                    {"hipblasDtpsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtpsv",                    {"hipblasCtpsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtpsv",                    {"hipblasZtpsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TBSV
  {"cublasStbsv",                    {"hipblasStbsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtbsv",                    {"hipblasDtbsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtbsv",                    {"hipblasCtbsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtbsv",                    {"hipblasZtbsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYMV/HEMV
  {"cublasSsymv",                    {"hipblasSsymv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsymv",                    {"hipblasDsymv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsymv",                    {"hipblasCsymv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsymv",                    {"hipblasZsymv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChemv",                    {"hipblasChemv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhemv",                    {"hipblasZhemv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SBMV/HBMV
  {"cublasSsbmv",                    {"hipblasSsbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsbmv",                    {"hpiblasDsbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChbmv",                    {"hipblasChbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhbmv",                    {"hipblasZhbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPMV/HPMV
  {"cublasSspmv",                    {"hipblasSspmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspmv",                    {"hipblasDspmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpmv",                    {"hipblasChpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpmv",                    {"hipblasZhpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GER
  {"cublasSger",                     {"hipblasSger",                     CONV_MATH_FUNC, API_BLAS}},
  {"cublasDger",                     {"hipblasDger",                     CONV_MATH_FUNC, API_BLAS}},
  {"cublasCgeru",                    {"hipblasCgeru",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgerc",                    {"hipblasCgerc",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgeru",                    {"hipblasZgeru",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgerc",                    {"hipblasZgerc",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR/HER
  {"cublasSsyr",                     {"hipblasSsyr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr",                     {"hipblasDsyr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr",                     {"hipblasCsyr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr",                     {"hipblasZsyr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCher",                     {"hipblasCher",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher",                     {"hipblasZher",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPR/HPR
  {"cublasSspr",                     {"hipblasSspr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspr",                     {"hipblasDspr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpr",                     {"hipblasChpr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpr",                     {"hipblasZhpr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR2/HER2
  {"cublasSsyr2",                    {"hipblasSsyr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr2",                    {"hipblasDsyr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr2",                    {"hipblasCsyr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr2",                    {"hipblasZsyr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCher2",                    {"hipblasCher2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher2",                    {"hipblasZher2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPR2/HPR2
  {"cublasSspr2",                    {"hipblasSspr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspr2",                    {"hipblasDspr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpr2",                    {"hipblasChpr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpr2",                    {"hipblasZhpr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Blas3 (v1) Routines
  // GEMM
  {"cublasSgemm",                    {"hipblasSgemm",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDgemm",                    {"hipblasDgemm",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCgemm",                    {"hipblasCgemm",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasZgemm",                    {"hipblasZgemm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasHgemm",                    {"hipblasHgemm",                    CONV_MATH_FUNC, API_BLAS}},

  // BATCH GEMM
  {"cublasSgemmBatched",             {"hipblasSgemmBatched",             CONV_MATH_FUNC, API_BLAS}},
  {"cublasDgemmBatched",             {"hipblasDgemmBatched",             CONV_MATH_FUNC, API_BLAS}},
  {"cublasHgemmBatched",             {"hipblasHgemmBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSgemmStridedBatched",      {"hipblasSgemmStridedBatched",      CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgemmStridedBatched",      {"hipblasDgemmStridedBatched",      CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgemmBatched",             {"hipblasCgemmBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgemm3mBatched",           {"hipblasCgemm3mBatched",           CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemmBatched",             {"hipblasZgemmBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgemmStridedBatched",      {"hipblasCgemmStridedBatched",      CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgemm3mStridedBatched",    {"hipblasCgemm3mStridedBatched",    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemmStridedBatched",      {"hipblasZgemmStridedBatched",      CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasHgemmStridedBatched",      {"hipblasHgemmStridedBatched",      CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYRK
  {"cublasSsyrk",                    {"hipblasSsyrk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyrk",                    {"hipblasDsyrk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyrk",                    {"hipblasCsyrk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyrk",                    {"hipblasZsyrk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HERK
  {"cublasCherk",                    {"hipblasCherk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZherk",                    {"hipblasZherk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR2K
  {"cublasSsyr2k",                   {"hipblasSsyr2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr2k",                   {"hipblasDsyr2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr2k",                   {"hipblasCsyr2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr2k",                   {"hipblasZsyr2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYRKX - eXtended SYRK
  {"cublasSsyrkx",                   {"hipblasSsyrkx",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyrkx",                   {"hipblasDsyrkx",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyrkx",                   {"hipblasCsyrkx",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyrkx",                   {"hipblasZsyrkx",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HER2K
  {"cublasCher2k",                   {"hipblasCher2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher2k",                   {"hipblasZher2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HERKX - eXtended HERK
  {"cublasCherkx",                   {"hipblasCherkx",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZherkx",                   {"hipblasZherkx",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYMM
  {"cublasSsymm",                    {"hipblasSsymm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsymm",                    {"hipblasDsymm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsymm",                    {"hipblasCsymm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsymm",                    {"hipblasZsymm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HEMM
  {"cublasChemm",                    {"hipblasChemm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhemm",                    {"hipblasZhemm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSM
  {"cublasStrsm",                    {"hipblasStrsm",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDtrsm",                    {"hipblasDtrsm",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCtrsm",                    {"hipblasCtrsm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsm",                    {"hipblasZtrsm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSM - Batched Triangular Solver
  {"cublasStrsmBatched",             {"hipblasStrsmBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrsmBatched",             {"hipblasDtrsmBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrsmBatched",             {"hipblasCtrsmBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsmBatched",             {"hipblasZtrsmBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRMM
  {"cublasStrmm",                    {"hipblasStrmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrmm",                    {"hipblasDtrmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrmm",                    {"hipblasCtrmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrmm",                    {"hipblasZtrmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ------------------------ CUBLAS BLAS - like extension (cublas_api.h)
  // GEAM
  {"cublasSgeam",                    {"hipblasSgeam",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDgeam",                    {"hipblasDgeam",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCgeam",                    {"hipblasCgeam",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgeam",                    {"hipblasZgeam",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GETRF - Batched LU
  {"cublasSgetrfBatched",            {"hipblasSgetrfBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgetrfBatched",            {"hipblasDgetrfBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgetrfBatched",            {"hipblasCgetrfBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgetrfBatched",            {"hipblasZgetrfBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Batched inversion based on LU factorization from getrf
  {"cublasSgetriBatched",            {"hipblasSgetriBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgetriBatched",            {"hipblasDgetriBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgetriBatched",            {"hipblasCgetriBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgetriBatched",            {"hipblasZgetriBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Batched solver based on LU factorization from getrf
  {"cublasSgetrsBatched",            {"hipblasSgetrsBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgetrsBatched",            {"hipblasDgetrsBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgetrsBatched",            {"hipblasCgetrsBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgetrsBatched",            {"hipblasZgetrsBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSM - Batched Triangular Solver
  {"cublasStrsmBatched",             {"hipblasStrsmBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrsmBatched",             {"hipblasDtrsmBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrsmBatched",             {"hipblasCtrsmBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsmBatched",             {"hipblasZtrsmBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // MATINV - Batched
  {"cublasSmatinvBatched",           {"hipblasSmatinvBatched",           CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDmatinvBatched",           {"hipblasDmatinvBatched",           CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCmatinvBatched",           {"hipblasCmatinvBatched",           CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZmatinvBatched",           {"hipblasZmatinvBatched",           CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Batch QR Factorization
  {"cublasSgeqrfBatched",            {"hipblasSgeqrfBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgeqrfBatched",            {"hipblasDgeqrfBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgeqrfBatched",            {"hipblasCgeqrfBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgeqrfBatched",            {"hipblasZgeqrfBatched",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Least Square Min only m >= n and Non-transpose supported
  {"cublasSgelsBatched",             {"hipblasSgelsBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgelsBatched",             {"hipblasDgelsBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgelsBatched",             {"hipblasCgelsBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgelsBatched",             {"hipblasZgelsBatched",             CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // DGMM
  {"cublasSdgmm",                    {"hipblasSdgmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDdgmm",                    {"hipblasDdgmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCdgmm",                    {"hipblasCdgmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdgmm",                    {"hipblasZdgmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TPTTR - Triangular Pack format to Triangular format
  {"cublasStpttr",                   {"hipblasStpttr",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtpttr",                   {"hipblasDtpttr",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtpttr",                   {"hipblasCtpttr",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtpttr",                   {"hipblasZtpttr",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRTTP - Triangular format to Triangular Pack format
  {"cublasStrttp",                   {"hipblasStrttp",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrttp",                   {"hipblasDtrttp",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrttp",                   {"hipblasCtrttp",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrttp",                   {"hipblasZtrttp",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Blas2 (v2) Routines
  {"cublasCreate_v2",                {"hipblasCreate",                   CONV_MATH_FUNC, API_BLAS}},
  {"cublasDestroy_v2",               {"hipblasDestroy",                  CONV_MATH_FUNC, API_BLAS}},
  {"cublasGetVersion_v2",            {"hipblasGetVersion",               CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGetProperty",              {"hipblasGetProperty",              CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSetStream_v2",             {"hipblasSetStream",                CONV_MATH_FUNC, API_BLAS}},
  {"cublasGetStream_v2",             {"hipblasGetStream",                CONV_MATH_FUNC, API_BLAS}},
  {"cublasGetPointerMode_v2",        {"hipblasGetPointerMode",           CONV_MATH_FUNC, API_BLAS}},
  {"cublasSetPointerMode_v2",        {"hipblasSetPointerMode",           CONV_MATH_FUNC, API_BLAS}},

  // GEMV
  {"cublasSgemv_v2",                 {"hipblasSgemv",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDgemv_v2",                 {"hipblasDgemv",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCgemv_v2",                 {"hipblasCgemv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemv_v2",                 {"hipblasZgemv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GBMV
  {"cublasSgbmv_v2",                 {"hipblasSgbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDgbmv_v2",                 {"hipblasDgbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgbmv_v2",                 {"hipblasCgbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgbmv_v2",                 {"hipblasZgbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRMV
  {"cublasStrmv_v2",                 {"hipblasStrmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrmv_v2",                 {"hipblasDtrmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrmv_v2",                 {"hipblasCtrmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrmv_v2",                 {"hipblasZtrmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TBMV
  {"cublasStbmv_v2",                 {"hipblasStbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtbmv_v2",                 {"hipblasDtbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtbmv_v2",                 {"hipblasCtbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtbmv_v2",                 {"hipblasZtbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TPMV
  {"cublasStpmv_v2",                 {"hipblasStpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtpmv_v2",                 {"hipblasDtpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtpmv_v2",                 {"hipblasCtpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtpmv_v2",                 {"hipblasZtpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSV
  {"cublasStrsv_v2",                 {"hipblasStrsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrsv_v2",                 {"hipblasDtrsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrsv_v2",                 {"hipblasCtrsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsv_v2",                 {"hipblasZtrsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TPSV
  {"cublasStpsv_v2",                 {"hipblasStpsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtpsv_v2",                 {"hipblasDtpsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtpsv_v2",                 {"hipblasCtpsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtpsv_v2",                 {"hipblasZtpsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TBSV
  {"cublasStbsv_v2",                 {"hipblasStbsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtbsv_v2",                 {"hipblasDtbsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtbsv_v2",                 {"hipblasCtbsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtbsv_v2",                 {"hipblasZtbsv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYMV/HEMV
  {"cublasSsymv_v2",                 {"hipblasSsymv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsymv_v2",                 {"hipblasDsymv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsymv_v2",                 {"hipblasCsymv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsymv_v2",                 {"hipblasZsymv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChemv_v2",                 {"hipblasChemv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhemv_v2",                 {"hipblasZhemv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SBMV/HBMV
  {"cublasSsbmv_v2",                 {"hipblasSsbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsbmv_v2",                 {"hpiblasDsbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChbmv_v2",                 {"hipblasChbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhbmv_v2",                 {"hipblasZhbmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPMV/HPMV
  {"cublasSspmv_v2",                 {"hipblasSspmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspmv_v2",                 {"hipblasDspmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpmv_v2",                 {"hipblasChpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpmv_v2",                 {"hipblasZhpmv",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // GER
  {"cublasSger_v2",                  {"hipblasSger",                     CONV_MATH_FUNC, API_BLAS}},
  {"cublasDger_v2",                  {"hipblasDger",                     CONV_MATH_FUNC, API_BLAS}},
  {"cublasCgeru_v2",                 {"hipblasCgeru",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgerc_v2",                 {"hipblasCgerc",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgeru_v2",                 {"hipblasZgeru",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgerc_v2",                 {"hipblasZgerc",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR/HER
  {"cublasSsyr_v2",                  {"hipblasSsyr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr_v2",                  {"hipblasDsyr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr_v2",                  {"hipblasCsyr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr_v2",                  {"hipblasZsyr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCher_v2",                  {"hipblasCher",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher_v2",                  {"hipblasZher",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPR/HPR
  {"cublasSspr_v2",                  {"hipblasSspr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspr_v2",                  {"hipblasDspr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpr_v2",                  {"hipblasChpr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpr_v2",                  {"hipblasZhpr",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR2/HER2
  {"cublasSsyr2_v2",                 {"hipblasSsyr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr2_v2",                 {"hipblasDsyr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr2_v2",                 {"hipblasCsyr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr2_v2",                 {"hipblasZsyr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCher2_v2",                 {"hipblasCher2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher2_v2",                 {"hipblasZher2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SPR2/HPR2
  {"cublasSspr2_v2",                 {"hipblasSspr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDspr2_v2",                 {"hipblasDspr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasChpr2_v2",                 {"hipblasChpr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhpr2_v2",                 {"hipblasZhpr2",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // Blas3 (v2) Routines
  // GEMM
  {"cublasSgemm_v2",                 {"hipblasSgemm",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDgemm_v2",                 {"hipblasDgemm",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCgemm_v2",                 {"hipblasCgemm",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCgemm3m",                  {"hipblasCgemm3m",                  CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCgemm3mEx",                {"hipblasCgemm3mEx",                CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemm_v2",                 {"hipblasZgemm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZgemm3m",                  {"hipblasZgemm3m",                  CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  //IO in FP16 / FP32, computation in float
  {"cublasSgemmEx",                  {"hipblasSgemmEx",                  CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGemmEx",                   {"hipblasGemmEx",                   CONV_MATH_FUNC, API_BLAS}},
  {"cublasGemmBatchedEx",            {"hipblasGemmBatchedEx",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasGemmStridedBatchedEx",     {"hipblasGemmStridedBatchedEx",     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCgemmEx",                  {"hipblasCgemmEx",                  CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasUint8gemmBias",            {"hipblasUint8gemmBias",            CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYRK
  {"cublasSsyrk_v2",                 {"hipblasSsyrk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyrk_v2",                 {"hipblasDsyrk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyrk_v2",                 {"hipblasCsyrk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyrk_v2",                 {"hipblasZsyrk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCsyrkEx",                  {"hipblasCsyrkEx",                  CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
  {"cublasCsyrk3mEx",                {"hipblasCsyrk3mEx",                CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HERK
  {"cublasCherk_v2",                 {"hipblasCherk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCherkEx",                  {"hipblasCherkEx",                  CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
  {"cublasCherk3mEx",                {"hipblasCherk3mEx",                CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZherk_v2",                 {"hipblasZherk",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYR2K
  {"cublasSsyr2k_v2",                {"hipblasSsyr2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsyr2k_v2",                {"hipblasDsyr2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsyr2k_v2",                {"hipblasCsyr2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsyr2k_v2",                {"hipblasZsyr2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HER2K
  {"cublasCher2k_v2",                {"hipblasCher2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZher2k_v2",                {"hipblasZher2k",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SYMM
  {"cublasSsymm_v2",                 {"hipblasSsymm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDsymm_v2",                 {"hipblasDsymm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsymm_v2",                 {"hipblasCsymm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZsymm_v2",                 {"hipblasZsymm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // HEMM
  {"cublasChemm_v2",                 {"hipblasChemm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZhemm_v2",                 {"hipblasZhemm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRSM
  {"cublasStrsm_v2",                 {"hipblasStrsm",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDtrsm_v2",                 {"hipblasDtrsm",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCtrsm_v2",                 {"hipblasCtrsm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrsm_v2",                 {"hipblasZtrsm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // TRMM
  {"cublasStrmm_v2",                 {"hipblasStrmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDtrmm_v2",                 {"hipblasDtrmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCtrmm_v2",                 {"hipblasCtrmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZtrmm_v2",                 {"hipblasZtrmm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // NRM2
  {"cublasSnrm2_v2",                 {"hipblasSnrm2",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDnrm2_v2",                 {"hipblasDnrm2",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasScnrm2_v2",                {"hipblasScnrm2",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDznrm2_v2",                {"hipblasDznrm2",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // DOT
  {"cublasDotEx",                    {"hipblasDotEx",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDotcEx",                   {"hipblasDotcEx",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  {"cublasSdot_v2",                  {"hipblasSdot",                     CONV_MATH_FUNC, API_BLAS}},
  {"cublasDdot_v2",                  {"hipblasDdot",                     CONV_MATH_FUNC, API_BLAS}},

  {"cublasCdotu_v2",                 {"hipblasCdotu",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCdotc_v2",                 {"hipblasCdotc",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdotu_v2",                 {"hipblasZdotu",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdotc_v2",                 {"hipblasZdotc",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SCAL
  {"cublasScalEx",                   {"hipblasScalEx",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSscal_v2",                 {"hipblasSscal",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDscal_v2",                 {"hipblasDscal",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCscal_v2",                 {"hipblasCscal",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsscal_v2",                {"hipblasCsscal",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZscal_v2",                 {"hipblasZscal",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdscal_v2",                {"hipblasZdscal",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AXPY
  {"cublasAxpyEx",                   {"hipblasAxpyEx",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasSaxpy_v2",                 {"hipblasSaxpy",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDaxpy_v2",                 {"hipblasDaxpy",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCaxpy_v2",                 {"hipblasCaxpy",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZaxpy_v2",                 {"hipblasZaxpy",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // COPY
  {"cublasScopy_v2",                 {"hipblasScopy",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDcopy_v2",                 {"hipblasDcopy",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasCcopy_v2",                 {"hipblasCcopy",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZcopy_v2",                 {"hipblasZcopy",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // SWAP
  {"cublasSswap_v2",                 {"hipblasSswap",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDswap_v2",                 {"hipblasDswap",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCswap_v2",                 {"hipblasCswap",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZswap_v2",                 {"hipblasZswap",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AMAX
  {"cublasIsamax_v2",                {"hipblasIsamax",                   CONV_MATH_FUNC, API_BLAS}},
  {"cublasIdamax_v2",                {"hipblasIdamax",                   CONV_MATH_FUNC, API_BLAS}},
  {"cublasIcamax_v2",                {"hipblasIcamax",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIzamax_v2",                {"hipblasIzamax",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // AMIN
  {"cublasIsamin_v2",                {"hipblasIsamin",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIdamin_v2",                {"hipblasIdamin",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIcamin_v2",                {"hipblasIcamin",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasIzamin_v2",                {"hipblasIzamin",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ASUM
  {"cublasSasum_v2",                 {"hipblasSasum",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasDasum_v2",                 {"hipblasDasum",                    CONV_MATH_FUNC, API_BLAS}},
  {"cublasScasum_v2",                {"hipblasScasum",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDzasum_v2",                {"hipblasDzasum",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROT
  {"cublasSrot_v2",                  {"hipblasSrot",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrot_v2",                  {"hipblasDrot",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCrot_v2",                  {"hipblasCrot",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCsrot_v2",                 {"hipblasCsrot",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZrot_v2",                  {"hipblasZrot",                     CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZdrot_v2",                 {"hipblasZdrot",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTG
  {"cublasSrotg_v2",                 {"hipblasSrotg",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotg_v2",                 {"hipblasDrotg",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasCrotg_v2",                 {"hipblasCrotg",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasZrotg_v2",                 {"hipblasZrotg",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTM
  {"cublasSrotm_v2",                 {"hipblasSrotm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotm_v2",                 {"hipblasDrotm",                    CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

  // ROTMG
  {"cublasSrotmg_v2",                {"hipblasSrotmg",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
  {"cublasDrotmg_v2",                {"hipblasDrotmg",                   CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
};
