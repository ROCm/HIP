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
const std::map<llvm::StringRef, hipCounter> CUDA_BLAS_TYPE_NAME_MAP{

  ////////////////////// Blas Data Types

  // Blas operations
  {"cublasOperation_t",              {"hipblasOperation_t",              "", CONV_TYPE, API_BLAS}},
  {"CUBLAS_OP_N",                    {"HIPBLAS_OP_N",                    "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_OP_T",                    {"HIPBLAS_OP_T",                    "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_OP_C",                    {"HIPBLAS_OP_C",                    "", CONV_NUMERIC_LITERAL, API_BLAS}},

  // Blas statuses
  {"cublasStatus",                   {"hipblasStatus_t",                 "", CONV_TYPE, API_BLAS}},
  {"cublasStatus_t",                 {"hipblasStatus_t",                 "", CONV_TYPE, API_BLAS}},
  {"CUBLAS_STATUS_SUCCESS",          {"HIPBLAS_STATUS_SUCCESS",          "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_STATUS_NOT_INITIALIZED",  {"HIPBLAS_STATUS_NOT_INITIALIZED",  "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_STATUS_ALLOC_FAILED",     {"HIPBLAS_STATUS_ALLOC_FAILED",     "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_STATUS_INVALID_VALUE",    {"HIPBLAS_STATUS_INVALID_VALUE",    "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_STATUS_MAPPING_ERROR",    {"HIPBLAS_STATUS_MAPPING_ERROR",    "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_STATUS_EXECUTION_FAILED", {"HIPBLAS_STATUS_EXECUTION_FAILED", "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_STATUS_INTERNAL_ERROR",   {"HIPBLAS_STATUS_INTERNAL_ERROR",   "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_STATUS_NOT_SUPPORTED",    {"HIPBLAS_STATUS_NOT_SUPPORTED",    "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_STATUS_ARCH_MISMATCH",    {"HIPBLAS_STATUS_ARCH_MISMATCH",    "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_STATUS_LICENSE_ERROR",    {"HIPBLAS_STATUS_LICENSE_ERROR",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},

  // Blas Fill Modes
  {"cublasFillMode_t",               {"hipblasFillMode_t",               "", CONV_TYPE, API_BLAS}},
  {"CUBLAS_FILL_MODE_LOWER",         {"HIPBLAS_FILL_MODE_LOWER",         "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_FILL_MODE_UPPER",         {"HIPBLAS_FILL_MODE_UPPER",         "", CONV_NUMERIC_LITERAL, API_BLAS}},

  // Blas Diag Types
  {"cublasDiagType_t",               {"hipblasDiagType_t",               "", CONV_TYPE, API_BLAS}},
  {"CUBLAS_DIAG_NON_UNIT",           {"HIPBLAS_DIAG_NON_UNIT",           "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_DIAG_UNIT",               {"HIPBLAS_DIAG_UNIT",               "", CONV_NUMERIC_LITERAL, API_BLAS}},

  // Blas Side Modes
  {"cublasSideMode_t",               {"hipblasSideMode_t",               "", CONV_TYPE, API_BLAS}},
  {"CUBLAS_SIDE_LEFT",               {"HIPBLAS_SIDE_LEFT",               "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_SIDE_RIGHT",              {"HIPBLAS_SIDE_RIGHT",              "", CONV_NUMERIC_LITERAL, API_BLAS}},

  // Blas Pointer Modes
  {"cublasPointerMode_t",            {"hipblasPointerMode_t",            "", CONV_TYPE, API_BLAS}},
  {"CUBLAS_POINTER_MODE_HOST",       {"HIPBLAS_POINTER_MODE_HOST",       "", CONV_NUMERIC_LITERAL, API_BLAS}},
  {"CUBLAS_POINTER_MODE_DEVICE",     {"HIPBLAS_POINTER_MODE_DEVICE",     "", CONV_NUMERIC_LITERAL, API_BLAS}},

  // Blas Atomics Modes
  {"cublasAtomicsMode_t",            {"hipblasAtomicsMode_t",            "", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED}},
  {"CUBLAS_ATOMICS_NOT_ALLOWED",     {"HIPBLAS_ATOMICS_NOT_ALLOWED",     "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},
  {"CUBLAS_ATOMICS_ALLOWED",         {"HIPBLAS_ATOMICS_ALLOWED",         "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},

  // Blas Data Type
  {"cublasDataType_t",               {"hipblasDatatype_t",               "", CONV_TYPE, API_BLAS}},

  // Blas Math mode/tensor operation
  {"cublasMath_t",                   {"hipblasMath_t",                   "", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED}},
  {"CUBLAS_DEFAULT_MATH",            {"HIPBLAS_DEFAULT_MATH",            "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},
  {"CUBLAS_TENSOR_OP_MATH",          {"HIPBLAS_TENSOR_OP_MATH",          "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},

  // Blass different GEMM algorithms
  {"cublasGemmAlgo_t",               {"hipblasGemmAlgo_t",               "", CONV_TYPE, API_BLAS}},
  {"CUBLAS_GEMM_DFALT",              {"HIPBLAS_GEMM_DEFAULT",            "", CONV_NUMERIC_LITERAL, API_BLAS}},                   //  -1  // 160
  {"CUBLAS_GEMM_DEFAULT",            {"HIPBLAS_GEMM_DEFAULT",            "", CONV_NUMERIC_LITERAL, API_BLAS}},                   //  -1  // 160
  {"CUBLAS_GEMM_ALGO0",              {"HIPBLAS_GEMM_ALGO0",              "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //   0
  {"CUBLAS_GEMM_ALGO1",              {"HIPBLAS_GEMM_ALGO1",              "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //   1
  {"CUBLAS_GEMM_ALGO2",              {"HIPBLAS_GEMM_ALGO2",              "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //   2
  {"CUBLAS_GEMM_ALGO3",              {"HIPBLAS_GEMM_ALGO3",              "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //   3
  {"CUBLAS_GEMM_ALGO4",              {"HIPBLAS_GEMM_ALGO4",              "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //   4
  {"CUBLAS_GEMM_ALGO5",              {"HIPBLAS_GEMM_ALGO5",              "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //   5
  {"CUBLAS_GEMM_ALGO6",              {"HIPBLAS_GEMM_ALGO6",              "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //   6
  {"CUBLAS_GEMM_ALGO7",              {"HIPBLAS_GEMM_ALGO7",              "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //   7
  {"CUBLAS_GEMM_ALGO8",              {"HIPBLAS_GEMM_ALGO8",              "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //   8
  {"CUBLAS_GEMM_ALGO9",              {"HIPBLAS_GEMM_ALGO9",              "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //   9
  {"CUBLAS_GEMM_ALGO10",             {"HIPBLAS_GEMM_ALGO10",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  10
  {"CUBLAS_GEMM_ALGO11",             {"HIPBLAS_GEMM_ALGO11",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  11
  {"CUBLAS_GEMM_ALGO12",             {"HIPBLAS_GEMM_ALGO12",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  12
  {"CUBLAS_GEMM_ALGO13",             {"HIPBLAS_GEMM_ALGO13",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  13
  {"CUBLAS_GEMM_ALGO14",             {"HIPBLAS_GEMM_ALGO14",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  14
  {"CUBLAS_GEMM_ALGO15",             {"HIPBLAS_GEMM_ALGO15",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  15
  {"CUBLAS_GEMM_ALGO16",             {"HIPBLAS_GEMM_ALGO16",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  16
  {"CUBLAS_GEMM_ALGO17",             {"HIPBLAS_GEMM_ALGO17",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  17
  {"CUBLAS_GEMM_ALGO18",             {"HIPBLAS_GEMM_ALGO18",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  18
  {"CUBLAS_GEMM_ALGO19",             {"HIPBLAS_GEMM_ALGO19",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  19
  {"CUBLAS_GEMM_ALGO20",             {"HIPBLAS_GEMM_ALGO20",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  20
  {"CUBLAS_GEMM_ALGO21",             {"HIPBLAS_GEMM_ALGO21",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  21
  {"CUBLAS_GEMM_ALGO22",             {"HIPBLAS_GEMM_ALGO22",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  22
  {"CUBLAS_GEMM_ALGO23",             {"HIPBLAS_GEMM_ALGO23",             "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  23
  {"CUBLAS_GEMM_DEFAULT_TENSOR_OP",  {"HIPBLAS_GEMM_DEFAULT_TENSOR_OP",  "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  99
  {"CUBLAS_GEMM_DFALT_TENSOR_OP",    {"HIPBLAS_GEMM_DFALT_TENSOR_OP",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  //  99
  {"CUBLAS_GEMM_ALGO0_TENSOR_OP",    {"HIPBLAS_GEMM_ALGO0_TENSOR_OP",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 100
  {"CUBLAS_GEMM_ALGO1_TENSOR_OP",    {"HIPBLAS_GEMM_ALGO1_TENSOR_OP",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 101
  {"CUBLAS_GEMM_ALGO2_TENSOR_OP",    {"HIPBLAS_GEMM_ALGO2_TENSOR_OP",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 102
  {"CUBLAS_GEMM_ALGO3_TENSOR_OP",    {"HIPBLAS_GEMM_ALGO3_TENSOR_OP",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 103
  {"CUBLAS_GEMM_ALGO4_TENSOR_OP",    {"HIPBLAS_GEMM_ALGO4_TENSOR_OP",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 104
  {"CUBLAS_GEMM_ALGO5_TENSOR_OP",    {"HIPBLAS_GEMM_ALGO5_TENSOR_OP",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 105
  {"CUBLAS_GEMM_ALGO6_TENSOR_OP",    {"HIPBLAS_GEMM_ALGO6_TENSOR_OP",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 106
  {"CUBLAS_GEMM_ALGO7_TENSOR_OP",    {"HIPBLAS_GEMM_ALGO7_TENSOR_OP",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 107
  {"CUBLAS_GEMM_ALGO8_TENSOR_OP",    {"HIPBLAS_GEMM_ALGO8_TENSOR_OP",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 108
  {"CUBLAS_GEMM_ALGO9_TENSOR_OP",    {"HIPBLAS_GEMM_ALGO9_TENSOR_OP",    "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 109
  {"CUBLAS_GEMM_ALGO10_TENSOR_OP",   {"HIPBLAS_GEMM_ALGO10_TENSOR_OP",   "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 110
  {"CUBLAS_GEMM_ALGO11_TENSOR_OP",   {"HIPBLAS_GEMM_ALGO11_TENSOR_OP",   "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 111
  {"CUBLAS_GEMM_ALGO12_TENSOR_OP",   {"HIPBLAS_GEMM_ALGO12_TENSOR_OP",   "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 112
  {"CUBLAS_GEMM_ALGO13_TENSOR_OP",   {"HIPBLAS_GEMM_ALGO13_TENSOR_OP",   "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 113
  {"CUBLAS_GEMM_ALGO14_TENSOR_OP",   {"HIPBLAS_GEMM_ALGO14_TENSOR_OP",   "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 114
  {"CUBLAS_GEMM_ALGO15_TENSOR_OP",   {"HIPBLAS_GEMM_ALGO15_TENSOR_OP",   "", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},  // 115


  // TODO: rename hipblasDatatype_t to hipDataType_t and move from hipBLAS to HIP
  {"cudaDataType_t", {"hipblasDatatype_t", "", CONV_TYPE, API_RUNTIME}},
  {"cudaDataType",   {"hipblasDatatype_t", "", CONV_TYPE, API_RUNTIME}},
  {"CUDA_R_16F",     {"HIPBLAS_R_16F",     "", CONV_NUMERIC_LITERAL, API_RUNTIME}},                  //  2  // 150
  {"CUDA_C_16F",     {"HIPBLAS_C_16F",     "", CONV_NUMERIC_LITERAL, API_RUNTIME}},                  //  6  // 153
  {"CUDA_R_32F",     {"HIPBLAS_R_32F",     "", CONV_NUMERIC_LITERAL, API_RUNTIME}},                  //  0  // 151
  {"CUDA_C_32F",     {"HIPBLAS_C_32F",     "", CONV_NUMERIC_LITERAL, API_RUNTIME}},                  //  4  // 154
  {"CUDA_R_64F",     {"HIPBLAS_R_64F",     "", CONV_NUMERIC_LITERAL, API_RUNTIME}},                  //  1  // 152
  {"CUDA_C_64F",     {"HIPBLAS_C_64F",     "", CONV_NUMERIC_LITERAL, API_RUNTIME}},                  //  5  // 155
  {"CUDA_R_8I",      {"HIPBLAS_R_8I",      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, //  3  //
  {"CUDA_C_8I",      {"HIPBLAS_C_8I",      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, //  7  //
  {"CUDA_R_8U",      {"HIPBLAS_R_8U",      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, //  8  //
  {"CUDA_C_8U",      {"HIPBLAS_C_8U",      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, //  9  //
  {"CUDA_R_32I",     {"HIPBLAS_R_32I",     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 10  //
  {"CUDA_C_32I",     {"HIPBLAS_C_32I",     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 11  //
  {"CUDA_R_32U",     {"HIPBLAS_R_32U",     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 12  //
  {"CUDA_C_32U",     {"HIPBLAS_C_32U",     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 13  //


  {"cublasHandle_t", {"hipblasHandle_t", "", CONV_TYPE, API_BLAS}},
  // TODO: dereferencing: typedef struct cublasContext *cublasHandle_t;
  // {"cublasContext", {"hipblasHandle_t", "", CONV_TYPE, API_BLAS}},
};
