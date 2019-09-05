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
const std::map<llvm::StringRef, hipCounter> CUDA_DNN_TYPE_NAME_MAP{
  // cuDNN defines
  {"CUDNN_VERSION",                                       {"HIPDNN_VERSION",                                       "", CONV_NUMERIC_LITERAL, API_DNN}},    //  7000
  {"CUDNN_DIM_MAX",                                       {"HIPDNN_DIM_MAX",                                       "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    //  8
  {"CUDNN_LRN_MIN_N",                                     {"HIPDNN_LRN_MIN_N",                                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    //  1
  {"CUDNN_LRN_MAX_N",                                     {"HIPDNN_LRN_MAX_N",                                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 16
  {"CUDNN_LRN_MIN_K",                                     {"HIPDNN_LRN_MIN_K",                                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1e-5
  {"CUDNN_LRN_MIN_BETA",                                  {"HIPDNN_LRN_MIN_BETA",                                  "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0.01
  {"CUDNN_BN_MIN_EPSILON",                                {"HIPDNN_BN_MIN_EPSILON",                                "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1e-5
  {"CUDNN_SEV_ERROR_EN",                                  {"HIPDNN_SEV_ERROR_EN",                                  "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_SEV_WARNING_EN",                                {"HIPDNN_SEV_WARNING_EN",                                "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_SEV_INFO_EN",                                   {"HIPDNN_SEV_INFO_EN",                                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_SEQDATA_DIM_COUNT",                             {"HIPDNN_SEQDATA_DIM_COUNT",                             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 4

  // cuDNN enums
  {"cudnnStatus_t",                                       {"hipdnnStatus_t",                                       "", CONV_TYPE, API_DNN}},
  {"CUDNN_STATUS_SUCCESS",                                {"HIPDNN_STATUS_SUCCESS",                                "", CONV_NUMERIC_LITERAL, API_DNN}},    //  0
  {"CUDNN_STATUS_NOT_INITIALIZED",                        {"HIPDNN_STATUS_NOT_INITIALIZED",                        "", CONV_NUMERIC_LITERAL, API_DNN}},    //  1
  {"CUDNN_STATUS_ALLOC_FAILED",                           {"HIPDNN_STATUS_ALLOC_FAILED",                           "", CONV_NUMERIC_LITERAL, API_DNN}},    //  2
  {"CUDNN_STATUS_BAD_PARAM",                              {"HIPDNN_STATUS_BAD_PARAM",                              "", CONV_NUMERIC_LITERAL, API_DNN}},    //  3
  {"CUDNN_STATUS_INTERNAL_ERROR",                         {"HIPDNN_STATUS_INTERNAL_ERROR",                         "", CONV_NUMERIC_LITERAL, API_DNN}},    //  4
  {"CUDNN_STATUS_INVALID_VALUE",                          {"HIPDNN_STATUS_INVALID_VALUE",                          "", CONV_NUMERIC_LITERAL, API_DNN}},    //  5
  {"CUDNN_STATUS_ARCH_MISMATCH",                          {"HIPDNN_STATUS_ARCH_MISMATCH",                          "", CONV_NUMERIC_LITERAL, API_DNN}},    //  6
  {"CUDNN_STATUS_MAPPING_ERROR",                          {"HIPDNN_STATUS_MAPPING_ERROR",                          "", CONV_NUMERIC_LITERAL, API_DNN}},    //  7
  {"CUDNN_STATUS_EXECUTION_FAILED",                       {"HIPDNN_STATUS_EXECUTION_FAILED",                       "", CONV_NUMERIC_LITERAL, API_DNN}},    //  8
  {"CUDNN_STATUS_NOT_SUPPORTED",                          {"HIPDNN_STATUS_NOT_SUPPORTED",                          "", CONV_NUMERIC_LITERAL, API_DNN}},    //  9
  {"CUDNN_STATUS_LICENSE_ERROR",                          {"HIPDNN_STATUS_LICENSE_ERROR",                          "", CONV_NUMERIC_LITERAL, API_DNN}},    // 10
  {"CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING",           {"HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING",           "", CONV_NUMERIC_LITERAL, API_DNN}},    // 11
  {"CUDNN_STATUS_RUNTIME_IN_PROGRESS",                    {"HIPDNN_STATUS_RUNTIME_IN_PROGRESS",                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 12
  {"CUDNN_STATUS_RUNTIME_FP_OVERFLOW",                    {"HIPDNN_STATUS_RUNTIME_FP_OVERFLOW",                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 13
  {"cudnnRuntimeTag_t",                                   {"hipdnnRuntimeTag_t",                                   "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnConvolutionMode_t",                              {"hipdnnConvolutionMode_t",                              "", CONV_TYPE, API_DNN}},
  {"CUDNN_CONVOLUTION",                                   {"HIPDNN_CONVOLUTION",                                   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_CROSS_CORRELATION",                             {"HIPDNN_CROSS_CORRELATION",                             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"cudnnTensorFormat_t",                                 {"hipdnnTensorFormat_t",                                 "", CONV_TYPE, API_DNN}},
  {"CUDNN_TENSOR_NCHW",                                   {"HIPDNN_TENSOR_NCHW",                                   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_TENSOR_NHWC",                                   {"HIPDNN_TENSOR_NHWC",                                   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_TENSOR_NCHW_VECT_C",                            {"HIPDNN_TENSOR_NCHW_VECT_C",                            "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"cudnnFoldingDirection_t",                             {"hipdnnFoldingDirection_t",                             "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_TRANSFORM_FOLD",                                {"HIPDNN_TRANSFORM_FOLD",                                "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0U
  {"CUDNN_TRANSFORM_UNFOLD",                              {"HIPDNN_TRANSFORM_UNFOLD",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1U
  {"cudnnDataType_t",                                     {"hipdnnDataType_t",                                     "", CONV_TYPE, API_DNN}},
  {"CUDNN_DATA_FLOAT",                                    {"HIPDNN_DATA_FLOAT",                                    "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_DATA_DOUBLE",                                   {"HIPDNN_DATA_DOUBLE",                                   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_DATA_HALF",                                     {"HIPDNN_DATA_HALF",                                     "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"CUDNN_DATA_INT8",                                     {"HIPDNN_DATA_INT8",                                     "", CONV_NUMERIC_LITERAL, API_DNN}},    // 3
  {"CUDNN_DATA_INT32",                                    {"HIPDNN_DATA_INT32",                                    "", CONV_NUMERIC_LITERAL, API_DNN}},    // 4
  {"CUDNN_DATA_INT8x4",                                   {"HIPDNN_DATA_INT8x4",                                   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 5
  {"CUDNN_DATA_UINT8",                                    {"HIPDNN_DATA_UINT8",                                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},  // 6
  {"CUDNN_DATA_UINT8x4",                                  {"HIPDNN_DATA_UINT8x4",                                  "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},  // 7
  {"CUDNN_DATA_INT8x32",                                  {"HIPDNN_DATA_INT8x32",                                  "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},  // 8
  {"cudnnErrQueryMode_t",                                 {"hipdnnErrQueryMode_t",                                 "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_ERRQUERY_RAWCODE",                              {"HIPDNN_ERRQUERY_RAWCODE",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_ERRQUERY_NONBLOCKING",                          {"HIPDNN_ERRQUERY_NONBLOCKING",                          "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_ERRQUERY_BLOCKING",                             {"HIPDNN_ERRQUERY_BLOCKING",                             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 2
  {"cudnnSeverity_t",                                     {"hipdnnSeverity_t",                                     "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_SEV_FATAL",                                     {"HIPDNN_SEV_FATAL",                                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_SEV_ERROR",                                     {"HIPDNN_SEV_ERROR",                                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_SEV_WARNING",                                   {"HIPDNN_SEV_WARNING",                                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_SEV_INFO",                                      {"HIPDNN_SEV_INFO",                                      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 3
  {"cudnnConvolutionFwdAlgo_t",                           {"hipdnnConvolutionFwdAlgo_t",                           "", CONV_TYPE, API_DNN}},
  {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",            {"HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",            "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",    {"HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",    "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_CONVOLUTION_FWD_ALGO_GEMM",                     {"HIPDNN_CONVOLUTION_FWD_ALGO_GEMM",                     "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",                   {"HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT",                   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 3
  {"CUDNN_CONVOLUTION_FWD_ALGO_FFT",                      {"HIPDNN_CONVOLUTION_FWD_ALGO_FFT",                      "", CONV_NUMERIC_LITERAL, API_DNN}},    // 4
  {"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",               {"HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",               "", CONV_NUMERIC_LITERAL, API_DNN}},    // 5
  {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",                 {"HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 6
  {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",        {"HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",        "", CONV_NUMERIC_LITERAL, API_DNN}},    // 7
  {"CUDNN_CONVOLUTION_FWD_ALGO_COUNT",                    {"HIPDNN_CONVOLUTION_FWD_ALGO_COUNT",                    "", CONV_NUMERIC_LITERAL, API_DNN}},    // 8
  {"cudnnConvolutionFwdPreference_t",                     {"hipdnnConvolutionFwdPreference_t",                     "", CONV_TYPE, API_DNN}},
  {"CUDNN_CONVOLUTION_FWD_NO_WORKSPACE",                  {"HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE",                  "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_CONVOLUTION_FWD_PREFER_FASTEST",                {"HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST",                "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT",       {"HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT",       "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"cudnnDeterminism_t",                                  {"hipdnnDeterminism_t",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_NON_DETERMINISTIC",                             {"HIPDNN_NON_DETERMINISTIC",                             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_DETERMINISTIC",                                 {"HIPDNN_DETERMINISTIC",                                 "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"cudnnDivNormMode_t",                                  {"hipdnnDivNormMode_t",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_DIVNORM_PRECOMPUTED_MEANS",                     {"HIPDNN_DIVNORM_PRECOMPUTED_MEANS",                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"cudnnCTCLossAlgo_t",                                  {"hipdnnCTCLossAlgo_t",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_CTC_LOSS_ALGO_DETERMINISTIC",                   {"HIPDNN_CTC_LOSS_ALGO_DETERMINISTIC",                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC",               {"HIPDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC",               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"cudnnLRNMode_t",                                      {"hipdnnLRNMode_t",                                      "", CONV_TYPE, API_DNN}},
  {"CUDNN_LRN_CROSS_CHANNEL_DIM1",                        {"HIPDNN_LRN_CROSS_CHANNEL",                             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0 vs 1
  {"cudnnRNNInputMode_t",                                 {"hipdnnRNNInputMode_t",                                 "", CONV_TYPE, API_DNN}},
  {"CUDNN_LINEAR_INPUT",                                  {"HIPDNN_LINEAR_INPUT",                                  "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_SKIP_INPUT",                                    {"HIPDNN_SKIP_INPUT",                                    "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"cudnnDirectionMode_t",                                {"hipdnnDirectionMode_t",                                "", CONV_TYPE, API_DNN}},
  {"CUDNN_UNIDIRECTIONAL",                                {"HIPDNN_UNIDIRECTIONAL",                                "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_BIDIRECTIONAL",                                 {"HIPDNN_BIDIRECTIONAL",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"cudnnMathType_t",                                     {"hipdnnMathType_t",                                     "", CONV_TYPE, API_DNN}},
  {"CUDNN_DEFAULT_MATH",                                  {"HIPDNN_DEFAULT_MATH",                                  "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_TENSOR_OP_MATH",                                {"HIPDNN_TENSOR_OP_MATH",                                "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION",               {"HIPDNN_TENSOR_OP_MATH_ALLOW_CONVERSION",               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 2
  {"cudnnNanPropagation_t",                               {"hipdnnNanPropagation_t",                               "", CONV_TYPE, API_DNN}},
  {"CUDNN_NOT_PROPAGATE_NAN",                             {"HIPDNN_NOT_PROPAGATE_NAN",                             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_PROPAGATE_NAN",                                 {"HIPDNN_PROPAGATE_NAN",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"cudnnConvolutionBwdDataAlgo_t",                       {"hipdnnConvolutionBwdDataAlgo_t",                       "", CONV_TYPE, API_DNN}},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",                   {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0",                   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",                   {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1",                   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",                 {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",          {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",          "", CONV_NUMERIC_LITERAL, API_DNN}},    // 3
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",            {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",            "", CONV_NUMERIC_LITERAL, API_DNN}},    // 4
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",   {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 5
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT",               {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM",      "", CONV_NUMERIC_LITERAL, API_DNN}},    // 6
  {"cudnnConvolutionBwdFilterAlgo_t",                     {"hipdnnConvolutionBwdFilterAlgo_t",                     "", CONV_TYPE, API_DNN}},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",                 {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0",                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",                 {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1",                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",               {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",               "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",                 {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3",                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 3
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",          {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",          "", CONV_NUMERIC_LITERAL, API_DNN}},    // 4
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED", {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED", "", CONV_NUMERIC_LITERAL, API_DNN}},    // 5
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",        {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",        "", CONV_NUMERIC_LITERAL, API_DNN}},    // 6
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",             {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 7
  {"cudnnConvolutionBwdFilterPreference_t",               {"hipdnnConvolutionBwdFilterPreference_t",               "", CONV_TYPE, API_DNN}},
  {"CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE",           {"HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE",           "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST",         {"HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST",         "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT",{"HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT","", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"cudnnRNNAlgo_t",                                      {"hipdnnRNNAlgo_t",                                      "", CONV_TYPE, API_DNN}},
  {"CUDNN_RNN_ALGO_STANDARD",                             {"HIPDNN_RNN_ALGO_STANDARD",                             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_RNN_ALGO_PERSIST_STATIC",                       {"HIPDNN_RNN_ALGO_PERSIST_STATIC",                       "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_RNN_ALGO_PERSIST_DYNAMIC",                      {"HIPDNN_RNN_ALGO_PERSIST_DYNAMIC",                      "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"CUDNN_RNN_ALGO_COUNT",                                {"HIPDNN_RNN_ALGO_COUNT",                                "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 3
  {"cudnnRNNMode_t",                                      {"hipdnnRNNMode_t",                                      "", CONV_TYPE, API_DNN}},
  {"CUDNN_RNN_RELU",                                      {"HIPDNN_RNN_RELU",                                      "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_RNN_TANH",                                      {"HIPDNN_RNN_TANH",                                      "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_LSTM",                                          {"HIPDNN_LSTM",                                          "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"CUDNN_GRU",                                           {"HIPDNN_GRU",                                           "", CONV_NUMERIC_LITERAL, API_DNN}},    // 3
  {"cudnnRNNBiasMode_t",                                  {"hipdnnRNNBiasMode_t",                                  "", CONV_TYPE, API_DNN}},
  {"CUDNN_RNN_NO_BIAS",                                   {"HIPDNN_RNN_NO_BIAS",                                   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_RNN_SINGLE_INP_BIAS",                           {"HIPDNN_RNN_WITH_BIAS",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_RNN_DOUBLE_BIAS",                               {"HIPDNN_RNN_WITH_BIAS",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_RNN_SINGLE_REC_BIAS",                           {"HIPDNN_RNN_WITH_BIAS",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"cudnnOpTensorOp_t",                                   {"hipdnnOpTensorOp_t",                                   "", CONV_TYPE, API_DNN}},
  {"CUDNN_OP_TENSOR_ADD",                                 {"HIPDNN_OP_TENSOR_ADD",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_OP_TENSOR_MUL",                                 {"HIPDNN_OP_TENSOR_MUL",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_OP_TENSOR_MIN",                                 {"HIPDNN_OP_TENSOR_MIN",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"CUDNN_OP_TENSOR_MAX",                                 {"HIPDNN_OP_TENSOR_MAX",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 3
  {"CUDNN_OP_TENSOR_SQRT",                                {"HIPDNN_OP_TENSOR_SQRT",                                "", CONV_NUMERIC_LITERAL, API_DNN}},    // 4
  {"CUDNN_OP_TENSOR_NOT",                                 {"HIPDNN_OP_TENSOR_NOT",                                 "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 5
  {"cudnnReduceTensorOp_t",                               {"hipdnnReduceTensorOp_t",                               "", CONV_TYPE, API_DNN}},
  {"CUDNN_REDUCE_TENSOR_ADD",                             {"HIPDNN_REDUCE_TENSOR_ADD",                             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_REDUCE_TENSOR_MUL",                             {"HIPDNN_REDUCE_TENSOR_MUL",                             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_REDUCE_TENSOR_MIN",                             {"HIPDNN_REDUCE_TENSOR_MIN",                             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"CUDNN_REDUCE_TENSOR_MAX",                             {"HIPDNN_REDUCE_TENSOR_MAX",                             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 3
  {"CUDNN_REDUCE_TENSOR_AMAX",                            {"HIPDNN_REDUCE_TENSOR_AMAX",                            "", CONV_NUMERIC_LITERAL, API_DNN}},    // 4
  {"CUDNN_REDUCE_TENSOR_AVG",                             {"HIPDNN_REDUCE_TENSOR_AVG",                             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 5
  {"CUDNN_REDUCE_TENSOR_NORM1",                           {"HIPDNN_REDUCE_TENSOR_NORM1",                           "", CONV_NUMERIC_LITERAL, API_DNN}},    // 6
  {"CUDNN_REDUCE_TENSOR_NORM2",                           {"HIPDNN_REDUCE_TENSOR_NORM2",                           "", CONV_NUMERIC_LITERAL, API_DNN}},    // 7
  {"CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS",                    {"HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS",                    "", CONV_NUMERIC_LITERAL, API_DNN}},    // 8
  {"cudnnReduceTensorIndices_t",                          {"hipdnnReduceTensorIndices_t",                          "", CONV_TYPE, API_DNN}},
  {"CUDNN_REDUCE_TENSOR_NO_INDICES",                      {"HIPDNN_REDUCE_TENSOR_NO_INDICES",                      "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_REDUCE_TENSOR_FLATTENED_INDICES",               {"HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES",               "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"cudnnConvolutionBwdDataPreference_t",                 {"hipdnnConvolutionBwdDataPreference_t",                 "", CONV_TYPE, API_DNN}},
  {"CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE",             {"HIPDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE",             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST",           {"HIPDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST",           "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT",  {"HIPDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT",  "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"cudnnIndicesType_t",                                  {"hipdnnIndicesType_t",                                  "", CONV_TYPE, API_DNN}},
  {"CUDNN_32BIT_INDICES",                                 {"HIPDNN_32BIT_INDICES",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_64BIT_INDICES",                                 {"HIPDNN_64BIT_INDICES",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_16BIT_INDICES",                                 {"HIPDNN_16BIT_INDICES",                                 "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"CUDNN_8BIT_INDICES",                                  {"HIPDNN_8BIT_INDICES",                                  "", CONV_NUMERIC_LITERAL, API_DNN}},    // 3
  {"cudnnSoftmaxAlgorithm_t",                             {"hipdnnSoftmaxAlgorithm_t",                             "", CONV_TYPE, API_DNN}},
  {"CUDNN_SOFTMAX_FAST",                                  {"HIPDNN_SOFTMAX_FAST",                                  "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_SOFTMAX_ACCURATE",                              {"HIPDNN_SOFTMAX_ACCURATE",                              "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_SOFTMAX_LOG",                                   {"HIPDNN_SOFTMAX_LOG",                                   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"cudnnSoftmaxMode_t",                                  {"hipdnnSoftmaxMode_t",                                  "", CONV_TYPE, API_DNN}},
  {"CUDNN_SOFTMAX_MODE_INSTANCE",                         {"HIPDNN_SOFTMAX_MODE_INSTANCE",                         "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_SOFTMAX_MODE_CHANNEL",                          {"HIPDNN_SOFTMAX_MODE_CHANNEL",                          "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"cudnnPoolingMode_t",                                  {"hipdnnPoolingMode_t",                                  "", CONV_TYPE, API_DNN}},
  {"CUDNN_POOLING_MAX",                                   {"HIPDNN_POOLING_MAX",                                   "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",         {"HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",         "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",         {"HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",         "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"CUDNN_POOLING_MAX_DETERMINISTIC",                     {"HIPDNN_POOLING_MAX_DETERMINISTIC",                     "", CONV_NUMERIC_LITERAL, API_DNN}},    // 3
  {"cudnnActivationMode_t",                               {"hipdnnActivationMode_t",                               "", CONV_TYPE, API_DNN}},
  {"CUDNN_ACTIVATION_SIGMOID",                            {"HIPDNN_ACTIVATION_SIGMOID",                            "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_ACTIVATION_RELU",                               {"HIPDNN_ACTIVATION_RELU",                               "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_ACTIVATION_TANH",                               {"HIPDNN_ACTIVATION_TANH",                               "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"CUDNN_ACTIVATION_CLIPPED_RELU",                       {"HIPDNN_ACTIVATION_CLIPPED_RELU",                       "", CONV_NUMERIC_LITERAL, API_DNN}},    // 3
  {"CUDNN_ACTIVATION_ELU",                                {"HIPDNN_ACTIVATION_ELU",                                "", CONV_NUMERIC_LITERAL, API_DNN}},    // 4
  {"CUDNN_ACTIVATION_IDENTITY",                           {"HIPDNN_ACTIVATION_PATHTRU",                            "", CONV_NUMERIC_LITERAL, API_DNN}},    // 5
  {"cudnnBatchNormMode_t",                                {"hipdnnBatchNormMode_t",                                "", CONV_TYPE, API_DNN}},
  {"CUDNN_BATCHNORM_PER_ACTIVATION",                      {"HIPDNN_BATCHNORM_PER_ACTIVATION",                      "", CONV_NUMERIC_LITERAL, API_DNN}},    // 0
  {"CUDNN_BATCHNORM_SPATIAL",                             {"HIPDNN_BATCHNORM_SPATIAL",                             "", CONV_NUMERIC_LITERAL, API_DNN}},    // 1
  {"CUDNN_BATCHNORM_SPATIAL_PERSISTENT",                  {"HIPDNN_BATCHNORM_SPATIAL_PERSISTENT",                  "", CONV_NUMERIC_LITERAL, API_DNN}},    // 2
  {"cudnnSamplerType_t",                                  {"hipdnnSamplerType_t",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_SAMPLER_BILINEAR",                              {"HIPDNN_SAMPLER_BILINEAR",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"cudnnBatchNormOps_t",                                 {"hipdnnBatchNormOps_t",                                 "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_BATCHNORM_OPS_BN",                              {"HIPDNN_BATCHNORM_OPS_BN",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_BATCHNORM_OPS_BN_ACTIVATION",                   {"HIPDNN_BATCHNORM_OPS_BN_ACTIVATION",                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION",               {"HIPDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION",               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 2
  {"cudnnRNNClipMode_t",                                  {"hipdnnRNNClipMode_t",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_RNN_CLIP_NONE",                                 {"HIPDNN_RNN_CLIP_NONE",                                 "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_RNN_CLIP_MINMAX",                               {"HIPDNN_RNN_CLIP_MINMAX",                               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"cudnnRNNDataLayout_t",                                {"hipdnnRNNDataLayout_t",                                "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED",            {"HIPDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED",            "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED",              {"HIPDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED",              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED",          {"HIPDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED",          "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 2
  {"cudnnRNNPaddingMode_t",                               {"hipdnnRNNPaddingMode_t",                               "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_RNN_PADDED_IO_DISABLED",                        {"HIPDNN_RNN_PADDED_IO_DISABLED",                        "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_RNN_PADDED_IO_ENABLED",                         {"HIPDNN_RNN_PADDED_IO_ENABLED",                         "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"cudnnSeqDataAxis_t",                                  {"hipdnnSeqDataAxis_t",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_SEQDATA_TIME_DIM",                              {"HIPDNN_SEQDATA_TIME_DIM",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_SEQDATA_BATCH_DIM",                             {"HIPDNN_SEQDATA_BATCH_DIM",                             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_SEQDATA_BEAM_DIM",                              {"HIPDNN_SEQDATA_BEAM_DIM",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_SEQDATA_VECT_DIM",                              {"HIPDNN_SEQDATA_VECT_DIM",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 3
  {"cudnnAttnQueryMap_t",                                 {"hipdnnAttnQueryMap_t",                                 "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_ATTN_QUERYMAP_ALL_TO_ONE",                      {"HIPDNN_ATTN_QUERYMAP_ALL_TO_ONE",                      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_ATTN_QUERYMAP_ONE_TO_ONE",                      {"HIPDNN_ATTN_QUERYMAP_ONE_TO_ONE",                      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1U << 0
  {"CUDNN_ATTN_DISABLE_PROJ_BIASES",                      {"HIPDNN_ATTN_DISABLE_PROJ_BIASES",                      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_ATTN_ENABLE_PROJ_BIASES",                       {"HIPDNN_ATTN_ENABLE_PROJ_BIASES",                       "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1U << 1
  {"cudnnMultiHeadAttnWeightKind_t",                      {"hipdnnMultiHeadAttnWeightKind_t",                      "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_MH_ATTN_Q_WEIGHTS",                             {"HIPDNN_MH_ATTN_Q_WEIGHTS",                             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_MH_ATTN_K_WEIGHTS",                             {"HIPDNN_MH_ATTN_K_WEIGHTS",                             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_MH_ATTN_V_WEIGHTS",                             {"HIPDNN_MH_ATTN_V_WEIGHTS",                             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_MH_ATTN_O_WEIGHTS",                             {"HIPDNN_MH_ATTN_O_WEIGHTS",                             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 3
  {"CUDNN_MH_ATTN_Q_BIASES",                              {"HIPDNN_MH_ATTN_Q_BIASES",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 4
  {"CUDNN_MH_ATTN_K_BIASES",                              {"HIPDNN_MH_ATTN_K_BIASES",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 5
  {"CUDNN_MH_ATTN_V_BIASES",                              {"HIPDNN_MH_ATTN_V_BIASES",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 6
  {"CUDNN_MH_ATTN_O_BIASES",                              {"HIPDNN_MH_ATTN_O_BIASES",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 7
  {"CUDNN_ATTN_WKIND_COUNT",                              {"HIPDNN_ATTN_WKIND_COUNT",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 8
  {"cudnnWgradMode_t",                                    {"hipdnnWgradMode_t",                                    "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_WGRAD_MODE_ADD",                                {"HIPDNN_WGRAD_MODE_ADD",                                "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_WGRAD_MODE_SET",                                {"HIPDNN_WGRAD_MODE_SET",                                "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"cudnnReorderType_t",                                  {"hipdnnReorderType_t",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_DEFAULT_REORDER",                               {"HIPDNN_DEFAULT_REORDER",                               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_NO_REORDER",                                    {"HIPDNN_NO_REORDER",                                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"cudnnLossNormalizationMode_t",                        {"hipdnnLossNormalizationMode_t",                        "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_LOSS_NORMALIZATION_NONE",                       {"HIPDNN_LOSS_NORMALIZATION_NONE",                       "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_LOSS_NORMALIZATION_SOFTMAX",                    {"HIPDNN_LOSS_NORMALIZATION_SOFTMAX",                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"cudnnFusedOps_t",                                     {"hipdnnFusedOps_t",                                     "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS",      {"HIPDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS",      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD",             {"HIPDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD",             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING",         {"HIPDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING",         "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE",        {"HIPDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE",        "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 3
  {"CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION",          {"HIPDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION",          "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 4
  {"CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK",   {"HIPDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK",   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 5
  {"CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM",             {"HIPDNN_FUSED_DACTIVATION_FORK_DBATCHNORM",             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 6
  {"cudnnFusedOpsConstParamLabel_t",                      {"hipdnnFusedOpsConstParamLabel_t",                      "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_PARAM_XDESC",                                   {"HIPDNN_PARAM_XDESC",                                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_PARAM_XDATA_PLACEHOLDER",                       {"HIPDNN_PARAM_XDATA_PLACEHOLDER",                       "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_PARAM_BN_MODE",                                 {"HIPDNN_PARAM_BN_MODE",                                 "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_PARAM_BN_EQSCALEBIAS_DESC",                     {"HIPDNN_PARAM_BN_EQSCALEBIAS_DESC",                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 3
  {"CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER",                  {"HIPDNN_PARAM_BN_EQSCALE_PLACEHOLDER",                  "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 4
  {"CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER",                   {"HIPDNN_PARAM_BN_EQBIAS_PLACEHOLDER",                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 5
  {"CUDNN_PARAM_ACTIVATION_DESC",                         {"HIPDNN_PARAM_ACTIVATION_DESC",                         "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 6
  {"CUDNN_PARAM_CONV_DESC",                               {"HIPDNN_PARAM_CONV_DESC",                               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 7
  {"CUDNN_PARAM_WDESC",                                   {"HIPDNN_PARAM_WDESC",                                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 8
  {"CUDNN_PARAM_WDATA_PLACEHOLDER",                       {"HIPDNN_PARAM_WDATA_PLACEHOLDER",                       "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 9
  {"CUDNN_PARAM_DWDESC",                                  {"HIPDNN_PARAM_DWDESC",                                  "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 10
  {"CUDNN_PARAM_DWDATA_PLACEHOLDER",                      {"HIPDNN_PARAM_DWDATA_PLACEHOLDER",                      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 11
  {"CUDNN_PARAM_YDESC",                                   {"HIPDNN_PARAM_YDESC",                                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 12
  {"CUDNN_PARAM_YDATA_PLACEHOLDER",                       {"HIPDNN_PARAM_YDATA_PLACEHOLDER",                       "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 13
  {"CUDNN_PARAM_DYDESC",                                  {"HIPDNN_PARAM_DYDESC",                                  "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 14
  {"CUDNN_PARAM_DYDATA_PLACEHOLDER",                      {"HIPDNN_PARAM_DYDATA_PLACEHOLDER",                      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 15
  {"CUDNN_PARAM_YSTATS_DESC",                             {"HIPDNN_PARAM_YSTATS_DESC",                             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 16
  {"CUDNN_PARAM_YSUM_PLACEHOLDER",                        {"HIPDNN_PARAM_YSUM_PLACEHOLDER",                        "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 17
  {"CUDNN_PARAM_YSQSUM_PLACEHOLDER",                      {"HIPDNN_PARAM_YSQSUM_PLACEHOLDER",                      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 18
  {"CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC",               {"HIPDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC",               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 19
  {"CUDNN_PARAM_BN_SCALE_PLACEHOLDER",                    {"HIPDNN_PARAM_BN_SCALE_PLACEHOLDER",                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 20
  {"CUDNN_PARAM_BN_BIAS_PLACEHOLDER",                     {"HIPDNN_PARAM_BN_BIAS_PLACEHOLDER",                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 21
  {"CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER",               {"HIPDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER",               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 22
  {"CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER",             {"HIPDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER",             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 23
  {"CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER",             {"HIPDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER",             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 24
  {"CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER",              {"HIPDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER",              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 25
  {"CUDNN_PARAM_ZDESC",                                   {"HIPDNN_PARAM_ZDESC",                                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 26
  {"CUDNN_PARAM_ZDATA_PLACEHOLDER",                       {"HIPDNN_PARAM_ZDATA_PLACEHOLDER",                       "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 27
  {"CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC",                   {"HIPDNN_PARAM_BN_Z_EQSCALEBIAS_DESC",                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 28
  {"CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER",                {"HIPDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER",                "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 29
  {"CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER",                 {"HIPDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER",                 "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 30
  {"CUDNN_PARAM_ACTIVATION_BITMASK_DESC",                 {"HIPDNN_PARAM_ACTIVATION_BITMASK_DESC",                 "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 31
  {"CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER",          {"HIPDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER",          "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 32
  {"CUDNN_PARAM_DXDESC",                                  {"HIPDNN_PARAM_DXDESC",                                  "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 33
  {"CUDNN_PARAM_DXDATA_PLACEHOLDER",                      {"HIPDNN_PARAM_DXDATA_PLACEHOLDER",                      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 34
  {"CUDNN_PARAM_DZDESC",                                  {"HIPDNN_PARAM_DZDESC",                                  "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 35
  {"CUDNN_PARAM_DZDATA_PLACEHOLDER",                      {"HIPDNN_PARAM_DZDATA_PLACEHOLDER",                      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 36
  {"CUDNN_PARAM_BN_DSCALE_PLACEHOLDER",                   {"HIPDNN_PARAM_BN_DSCALE_PLACEHOLDER",                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 37
  {"CUDNN_PARAM_BN_DBIAS_PLACEHOLDER",                    {"HIPDNN_PARAM_BN_DBIAS_PLACEHOLDER",                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 38
  {"cudnnFusedOpsPointerPlaceHolder_t",                   {"hipdnnActivationMode_t",                               "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_PTR_NULL",                                      {"HIPDNN_ACTIVATION_SIGMOID",                            "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_PTR_ELEM_ALIGNED",                              {"HIPDNN_ACTIVATION_RELU",                               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_PTR_16B_ALIGNED",                               {"HIPDNN_ACTIVATION_TANH",                               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 2
  {"cudnnFusedOpsVariantParamLabel_t",                    {"hipdnnFusedOpsVariantParamLabel_t",                    "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"CUDNN_PTR_XDATA",                                     {"HIPDNN_PTR_XDATA",                                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_PTR_BN_EQSCALE",                                {"HIPDNN_PTR_BN_EQSCALE",                                "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_PTR_BN_EQBIAS",                                 {"HIPDNN_PTR_BN_EQBIAS",                                 "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_PTR_WDATA",                                     {"HIPDNN_PTR_WDATA",                                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 3
  {"CUDNN_PTR_DWDATA",                                    {"HIPDNN_PTR_DWDATA",                                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 4
  {"CUDNN_PTR_YDATA",                                     {"HIPDNN_PTR_YDATA",                                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 5
  {"CUDNN_PTR_DYDATA",                                    {"HIPDNN_PTR_DYDATA",                                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 6
  {"CUDNN_PTR_YSUM",                                      {"HIPDNN_PTR_YSUM",                                      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 7
  {"CUDNN_PTR_YSQSUM",                                    {"HIPDNN_PTR_YSQSUM",                                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 8
  {"CUDNN_PTR_WORKSPACE",                                 {"HIPDNN_PTR_WORKSPACE",                                 "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 9
  {"CUDNN_PTR_BN_SCALE",                                  {"HIPDNN_PTR_BN_SCALE",                                  "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 10
  {"CUDNN_PTR_BN_BIAS",                                   {"HIPDNN_PTR_BN_BIAS",                                   "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 11
  {"CUDNN_PTR_BN_SAVED_MEAN",                             {"HIPDNN_PTR_BN_SAVED_MEAN",                             "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 12
  {"CUDNN_PTR_BN_SAVED_INVSTD",                           {"HIPDNN_PTR_BN_SAVED_INVSTD",                           "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 13
  {"CUDNN_PTR_BN_RUNNING_MEAN",                           {"HIPDNN_PTR_BN_RUNNING_MEAN",                           "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 14
  {"CUDNN_PTR_BN_RUNNING_VAR",                            {"HIPDNN_PTR_BN_RUNNING_VAR",                            "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 15
  {"CUDNN_PTR_ZDATA",                                     {"HIPDNN_PTR_ZDATA",                                     "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 16
  {"CUDNN_PTR_BN_Z_EQSCALE",                              {"HIPDNN_PTR_BN_Z_EQSCALE",                              "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 17
  {"CUDNN_PTR_BN_Z_EQBIAS",                               {"HIPDNN_PTR_BN_Z_EQBIAS",                               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 18
  {"CUDNN_PTR_ACTIVATION_BITMASK",                        {"HIPDNN_PTR_ACTIVATION_BITMASK",                        "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 19
  {"CUDNN_PTR_DXDATA",                                    {"HIPDNN_PTR_DXDATA",                                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 20
  {"CUDNN_PTR_DZDATA",                                    {"HIPDNN_PTR_DZDATA",                                    "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 21
  {"CUDNN_PTR_BN_DSCALE",                                 {"HIPDNN_PTR_BN_DSCALE",                                 "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 22
  {"CUDNN_PTR_BN_DBIAS",                                  {"HIPDNN_PTR_BN_DBIAS",                                  "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 23
  {"CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES",         {"HIPDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES",         "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 100
  {"CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT",          {"HIPDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT",          "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 101
  {"CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR",               {"HIPDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR",               "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 102
  {"CUDNN_SCALAR_DOUBLE_BN_EPSILON",                      {"HIPDNN_SCALAR_DOUBLE_BN_EPSILON",                      "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    // 103

  // cuDNN types
  {"cudnnContext",                                        {"hipdnnContext",                                        "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnHandle_t",                                       {"hipdnnHandle_t",                                       "", CONV_TYPE, API_DNN}},
  {"cudnnTensorStruct",                                   {"hipdnnTensorStruct",                                   "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnTensorDescriptor_t",                             {"hipdnnTensorDescriptor_t",                             "", CONV_TYPE, API_DNN}},
  {"cudnnConvolutionStruct",                              {"hipdnnConvolutionStruct",                              "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnConvolutionDescriptor_t",                        {"hipdnnConvolutionDescriptor_t",                        "", CONV_TYPE, API_DNN}},
  {"cudnnPoolingStruct",                                  {"hipdnnPoolingStruct",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnPoolingDescriptor_t",                            {"hipdnnPoolingDescriptor_t",                            "", CONV_TYPE, API_DNN}},
  {"cudnnFilterStruct",                                   {"hipdnnFilterStruct",                                   "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFilterDescriptor_t",                             {"hipdnnFilterDescriptor_t",                             "", CONV_TYPE, API_DNN}},
  {"cudnnLRNStruct",                                      {"hipdnnLRNStruct",                                      "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnLRNDescriptor_t",                                {"hipdnnLRNDescriptor_t",                                "", CONV_TYPE, API_DNN}},
  {"cudnnActivationStruct",                               {"hipdnnActivationStruct",                               "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnActivationDescriptor_t",                         {"hipdnnActivationDescriptor_t",                         "", CONV_TYPE, API_DNN}},
  {"cudnnSpatialTransformerStruct",                       {"hipdnnSpatialTransformerStruct",                       "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSpatialTransformerDescriptor_t",                 {"hipdnnSpatialTransformerDescriptor_t",                 "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnOpTensorStruct",                                 {"hipdnnOpTensorStruct",                                 "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnOpTensorDescriptor_t",                           {"hipdnnOpTensorDescriptor_t",                           "", CONV_TYPE, API_DNN}},
  {"cudnnReduceTensorStruct",                             {"hipdnnReduceTensorStruct",                             "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnReduceTensorDescriptor_t",                       {"hipdnnReduceTensorDescriptor_t",                       "", CONV_TYPE, API_DNN}},
  {"cudnnCTCLossStruct",                                  {"hipdnnCTCLossStruct",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnCTCLossDescriptor_t",                            {"hipdnnCTCLossDescriptor_t",                            "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnTensorTransformStruct",                          {"hipdnnTensorTransformStruct",                          "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnTensorTransformDescriptor_t",                    {"hipdnnTensorTransformDescriptor_t",                    "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnConvolutionFwdAlgoPerf_t",                       {"hipdnnConvolutionFwdAlgoPerf_t",                       "", CONV_TYPE, API_DNN}},
  {"cudnnConvolutionBwdFilterAlgoPerf_t",                 {"hipdnnConvolutionBwdFilterAlgoPerf_t",                 "", CONV_TYPE, API_DNN}},
  {"cudnnConvolutionBwdDataAlgoPerf_t",                   {"hipdnnConvolutionBwdDataAlgoPerf_t",                   "", CONV_TYPE, API_DNN}},
  {"cudnnDropoutStruct",                                  {"hipdnnDropoutStruct",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnDropoutDescriptor_t",                            {"hipdnnDropoutDescriptor_t",                            "", CONV_TYPE, API_DNN}},
  {"cudnnAlgorithmStruct",                                {"hipdnnAlgorithmStruct",                                "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnAlgorithmDescriptor_t",                          {"hipdnnAlgorithmDescriptor_t",                          "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnAlgorithmPerformanceStruct",                     {"hipdnnAlgorithmPerformanceStruct",                     "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnAlgorithmPerformance_t",                         {"hipdnnAlgorithmPerformance_t",                         "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnRNNStruct",                                      {"hipdnnRNNStruct",                                      "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnRNNDescriptor_t",                                {"hipdnnRNNDescriptor_t",                                "", CONV_TYPE, API_DNN}},
  {"cudnnPersistentRNNPlan",                              {"hipdnnPersistentRNNPlan",                              "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnPersistentRNNPlan_t",                            {"hipdnnPersistentRNNPlan_t",                            "", CONV_TYPE, API_DNN}},
  {"cudnnAlgorithm_t",                                    {"hipdnnAlgorithm_t",                                    "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnDebug_t",                                        {"hipdnnDebug_t",                                        "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnCallback_t",                                     {"hipdnnCallback_t",                                     "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnRNNDataStruct",                                  {"hipdnnRNNDataStruct",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnRNNDataDescriptor_t",                            {"hipdnnRNNDataDescriptor_t",                            "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSeqDataStruct",                                  {"hipdnnSeqDataStruct",                                  "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSeqDataDescriptor_t",                            {"hipdnnSeqDataDescriptor_t",                            "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnAttnStruct",                                     {"hipdnnAttnStruct",                                     "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnAttnDescriptor_t",                               {"hipdnnAttnDescriptor_t",                               "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsConstParamStruct",                       {"hipdnnFusedOpsConstParamStruct",                       "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsConstParamPack_t",                       {"hipdnnFusedOpsConstParamPack_t",                       "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsVariantParamStruct",                     {"hipdnnFusedOpsVariantParamStruct",                     "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsVariantParamPack_t",                     {"hipdnnFusedOpsVariantParamPack_t",                     "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsPlanStruct",                             {"hipdnnFusedOpsPlanStruct",                             "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsPlan_t",                                 {"hipdnnFusedOpsPlan_t",                                 "", CONV_TYPE, API_DNN, HIP_UNSUPPORTED}},
};
