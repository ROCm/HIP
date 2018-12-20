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
const std::map<llvm::StringRef, hipCounter> CUDA_DNN_FUNCTION_MAP{

  {"cudnnGetVersion",                                     {"hipdnnGetVersion",                                     "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetCudartVersion",                               {"hipdnnGetCudartVersion",                               "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnQueryRuntimeError",                              {"hipdnnQueryRuntimeError",                              "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetProperty",                                    {"hipdnnGetProperty",                                    "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetErrorString",                                 {"hipdnnGetErrorString",                                 "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnIm2Col",                                         {"hipdnnIm2Col",                                         "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnCreate",                                         {"hipdnnCreate",                                         "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDestroy",                                        {"hipdnnDestroy",                                        "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetStream",                                      {"hipdnnSetStream",                                      "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetStream",                                      {"hipdnnGetStream",                                      "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetCallback",                                    {"hipdnnSetCallback",                                    "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetCallback",                                    {"hipdnnGetCallback",                                    "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},

  // cuDNN Tensor functions
  {"cudnnCreateTensorDescriptor",                         {"hipdnnCreateTensorDescriptor",                         "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetTensor4dDescriptor",                          {"hipdnnSetTensor4dDescriptor",                          "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetTensor4dDescriptorEx",                        {"hipdnnSetTensor4dDescriptorEx",                        "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetTensor4dDescriptor",                          {"hipdnnGetTensor4dDescriptor",                          "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetTensorNdDescriptor",                          {"hipdnnSetTensorNdDescriptor",                          "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetTensorNdDescriptorEx",                        {"hipdnnSetTensorNdDescriptorEx",                        "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetTensorNdDescriptor",                          {"hipdnnGetTensorNdDescriptor",                          "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetTensorSizeInBytes",                           {"hipdnnGetTensorSizeInBytes",                           "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnDestroyTensorDescriptor",                        {"hipdnnDestroyTensorDescriptor",                        "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnTransformTensor",                                {"hipdnnTransformTensor",                                "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnAddTensor",                                      {"hipdnnAddTensor",                                      "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnCreateOpTensorDescriptor",                       {"hipdnnCreateOpTensorDescriptor",                       "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetOpTensorDescriptor",                          {"hipdnnSetOpTensorDescriptor",                          "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetOpTensorDescriptor",                          {"hipdnnGetOpTensorDescriptor",                          "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDestroyOpTensorDescriptor",                      {"hipdnnDestroyOpTensorDescriptor",                      "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnOpTensor",                                       {"hipdnnOpTensor",                                       "", CONV_LIB_FUNC, API_DNN}},

  // cuDNN Reduce Tensor functions
  {"cudnnCreateReduceTensorDescriptor",                   {"hipdnnCreateReduceTensorDescriptor",                   "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetReduceTensorDescriptor",                      {"hipdnnSetReduceTensorDescriptor",                      "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetReduceTensorDescriptor",                      {"hipdnnGetReduceTensorDescriptor",                      "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDestroyReduceTensorDescriptor",                  {"hipdnnDestroyReduceTensorDescriptor",                  "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetReductionIndicesSize",                        {"hipdnnGetReductionIndicesSize",                        "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetReductionWorkspaceSize",                      {"hipdnnGetReductionWorkspaceSize",                      "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnReduceTensor",                                   {"hipdnnReduceTensor",                                   "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetTensor",                                      {"hipdnnSetTensor",                                      "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnScaleTensor",                                    {"hipdnnScaleTensor",                                    "", CONV_LIB_FUNC, API_DNN}},

  // cuDNN Filter functions
  {"cudnnCreateFilterDescriptor",                         {"hipdnnCreateFilterDescriptor",                         "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetFilter4dDescriptor",                          {"hipdnnSetFilter4dDescriptor",                          "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetFilter4dDescriptor",                          {"hipdnnGetFilter4dDescriptor",                          "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSetFilterNdDescriptor",                          {"hipdnnSetFilterNdDescriptor",                          "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetFilterNdDescriptor",                          {"hipdnnGetFilterNdDescriptor",                          "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDestroyFilterDescriptor",                        {"hipdnnDestroyFilterDescriptor",                        "", CONV_LIB_FUNC, API_DNN}},

  // cuDNN Convolution functions
  {"cudnnCreateConvolutionDescriptor",                    {"hipdnnCreateConvolutionDescriptor",                    "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetConvolutionMathType",                         {"hipdnnSetConvolutionMathType",                         "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolutionMathType",                         {"hipdnnGetConvolutionMathType",                         "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSetConvolutionGroupCount",                       {"hipdnnSetConvolutionGroupCount",                       "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetConvolutionGroupCount",                       {"hipdnnGetConvolutionGroupCount",                       "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSetConvolution2dDescriptor",                     {"hipdnnSetConvolution2dDescriptor",                     "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolution2dDescriptor",                     {"hipdnnGetConvolution2dDescriptor",                     "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolution2dForwardOutputDim",               {"hipdnnGetConvolution2dForwardOutputDim",               "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetConvolutionNdDescriptor",                     {"hipdnnSetConvolutionNdDescriptor",                     "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolutionNdDescriptor",                     {"hipdnnGetConvolutionNdDescriptor",                     "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetConvolutionNdForwardOutputDim",               {"hipdnnGetConvolutionNdForwardOutputDim",               "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnDestroyConvolutionDescriptor",                   {"hipdnnDestroyConvolutionDescriptor",                   "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolutionForwardAlgorithmMaxCount",         {"hipdnnGetConvolutionForwardAlgorithmMaxCount",         "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFindConvolutionForwardAlgorithm",                {"hipdnnFindConvolutionForwardAlgorithm",                "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnFindConvolutionForwardAlgorithmEx",              {"hipdnnFindConvolutionForwardAlgorithmEx",              "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolutionForwardAlgorithm",                 {"hipdnnGetConvolutionForwardAlgorithm",                 "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolutionForwardAlgorithm_v7",              {"hipdnnGetConvolutionForwardAlgorithm_v7",              "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetConvolutionForwardWorkspaceSize",             {"hipdnnGetConvolutionForwardWorkspaceSize",             "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnConvolutionForward",                             {"hipdnnConvolutionForward",                             "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnConvolutionBiasActivationForward",               {"hipdnnConvolutionBiasActivationForward",               "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnConvolutionBackwardBias",                        {"hipdnnConvolutionBackwardBias",                        "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolutionBackwardFilterAlgorithmMaxCount",  {"hipdnnGetConvolutionBackwardFilterAlgorithmMaxCount",  "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFindConvolutionBackwardFilterAlgorithm",         {"hipdnnFindConvolutionBackwardFilterAlgorithm",         "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnFindConvolutionBackwardFilterAlgorithmEx",       {"hipdnnFindConvolutionBackwardFilterAlgorithmEx",       "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolutionBackwardFilterAlgorithm",          {"hipdnnGetConvolutionBackwardFilterAlgorithm",          "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolutionBackwardFilterAlgorithm_v7",       {"hipdnnGetConvolutionBackwardFilterAlgorithm_v7",       "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetConvolutionBackwardFilterWorkspaceSize",      {"hipdnnGetConvolutionBackwardFilterWorkspaceSize",      "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnConvolutionBackwardFilter",                      {"hipdnnConvolutionBackwardFilter",                      "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolutionBackwardDataAlgorithmMaxCount",    {"hipdnnGetConvolutionBackwardDataAlgorithmMaxCount",    "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFindConvolutionBackwardDataAlgorithm",           {"hipdnnFindConvolutionBackwardDataAlgorithm",           "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnFindConvolutionBackwardDataAlgorithmEx",         {"hipdnnFindConvolutionBackwardDataAlgorithmEx",         "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolutionBackwardDataAlgorithm",            {"hipdnnGetConvolutionBackwardDataAlgorithm",            "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetConvolutionBackwardDataAlgorithm_v7",         {"hipdnnGetConvolutionBackwardDataAlgorithm_v7",         "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetConvolutionBackwardDataWorkspaceSize",        {"hipdnnGetConvolutionBackwardDataWorkspaceSize",        "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnConvolutionBackwardData",                        {"hipdnnConvolutionBackwardData",                        "", CONV_LIB_FUNC, API_DNN}},

  // cuDNN Sortmax functions
  {"cudnnSoftmaxForward",                                 {"hipdnnSoftmaxForward",                                 "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSoftmaxBackward",                                {"hipdnnSoftmaxBackward",                                "", CONV_LIB_FUNC, API_DNN}},

  // cuDNN Pooling functions
  {"cudnnCreatePoolingDescriptor",                        {"hipdnnCreatePoolingDescriptor",                        "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetPooling2dDescriptor",                         {"hipdnnSetPooling2dDescriptor",                         "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetPooling2dDescriptor",                         {"hipdnnGetPooling2dDescriptor",                         "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetPoolingNdDescriptor",                         {"hipdnnSetPoolingNdDescriptor",                         "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetPoolingNdDescriptor",                         {"hipdnnGetPoolingNdDescriptor",                         "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetPoolingNdForwardOutputDim",                   {"hipdnnGetPoolingNdForwardOutputDim",                   "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetPooling2dForwardOutputDim",                   {"hipdnnGetPooling2dForwardOutputDim",                   "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDestroyPoolingDescriptor",                       {"hipdnnDestroyPoolingDescriptor",                       "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnPoolingForward",                                 {"hipdnnPoolingForward",                                 "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnPoolingBackward",                                {"hipdnnPoolingBackward",                                "", CONV_LIB_FUNC, API_DNN}},

  // cuDNN Activation functions
  {"cudnnCreateActivationDescriptor",                     {"hipdnnCreateActivationDescriptor",                     "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetActivationDescriptor",                        {"hipdnnSetActivationDescriptor",                        "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetActivationDescriptor",                        {"hipdnnGetActivationDescriptor",                        "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDestroyActivationDescriptor",                    {"hipdnnDestroyActivationDescriptor",                    "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnActivationForward",                              {"hipdnnActivationForward",                              "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnActivationBackward",                             {"hipdnnActivationBackward",                             "", CONV_LIB_FUNC, API_DNN}},

  // cuDNN LRN functions
  {"cudnnCreateLRNDescriptor",                            {"hipdnnCreateLRNDescriptor",                            "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetLRNDescriptor",                               {"hipdnnSetLRNDescriptor",                               "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetLRNDescriptor",                               {"hipdnnGetLRNDescriptor",                               "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDestroyLRNDescriptor",                           {"hipdnnDestroyLRNDescriptor",                           "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnLRNCrossChannelForward",                         {"hipdnnLRNCrossChannelForward",                         "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnLRNCrossChannelBackward",                        {"hipdnnLRNCrossChannelBackward",                        "", CONV_LIB_FUNC, API_DNN}},

  // cuDNN Divisive Normalization functions
  {"cudnnDivisiveNormalizationForward",                   {"hipdnnDivisiveNormalizationForward",                   "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnDivisiveNormalizationBackward",                  {"hipdnnDivisiveNormalizationBackward",                  "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},

  // cuDNN Batch Normalization functions
  {"cudnnDeriveBNTensorDescriptor",                       {"hipdnnDeriveBNTensorDescriptor",                       "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnBatchNormalizationForwardTraining",              {"hipdnnBatchNormalizationForwardTraining",              "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnBatchNormalizationForwardInference",             {"hipdnnBatchNormalizationForwardInference",             "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnBatchNormalizationBackward",                     {"hipdnnBatchNormalizationBackward",                     "", CONV_LIB_FUNC, API_DNN}},

  // cuDNN Spatial Transformer functions
  {"cudnnCreateSpatialTransformerDescriptor",             {"hipdnnCreateSpatialTransformerDescriptor",             "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSetSpatialTransformerNdDescriptor",              {"hipdnnSetSpatialTransformerNdDescriptor",              "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnDestroySpatialTransformerDescriptor",            {"hipdnnDestroySpatialTransformerDescriptor",            "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSpatialTfGridGeneratorForward",                  {"hipdnnSpatialTfGridGeneratorForward",                  "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSpatialTfGridGeneratorBackward",                 {"hipdnnSpatialTfGridGeneratorBackward",                 "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSpatialTfSamplerForward",                        {"hipdnnSpatialTfSamplerForward",                        "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSpatialTfSamplerBackward",                       {"hipdnnSpatialTfSamplerBackward",                       "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},

  // cuDNN Dropout functions
  {"cudnnCreateDropoutDescriptor",                        {"hipdnnCreateDropoutDescriptor",                        "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDestroyDropoutDescriptor",                       {"hipdnnDestroyDropoutDescriptor",                       "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDropoutGetStatesSize",                           {"hipdnnDropoutGetStatesSize",                           "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDropoutGetReserveSpaceSize",                     {"hipdnnDropoutGetReserveSpaceSize",                     "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSetDropoutDescriptor",                           {"hipdnnSetDropoutDescriptor",                           "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetDropoutDescriptor",                           {"hipdnnGetDropoutDescriptor",                           "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnRestoreDropoutDescriptor",                       {"hipdnnRestoreDropoutDescriptor",                       "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnDropoutForward",                                 {"hipdnnDropoutForward",                                 "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnDropoutBackward",                                {"hipdnnDropoutBackward",                                "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},

  // cuDNN RNN functions
  {"cudnnCreateRNNDescriptor",                            {"hipdnnCreateRNNDescriptor",                            "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDestroyRNNDescriptor",                           {"hipdnnDestroyRNNDescriptor",                           "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetRNNForwardInferenceAlgorithmMaxCount",        {"hipdnnGetRNNForwardInferenceAlgorithmMaxCount",        "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFindRNNForwardInferenceAlgorithmEx",             {"hipdnnFindRNNForwardInferenceAlgorithmEx",             "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetRNNForwardTrainingAlgorithmMaxCount",         {"hipdnnGetRNNForwardTrainingAlgorithmMaxCount",         "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFindRNNForwardTrainingAlgorithmEx",              {"hipdnnFindRNNForwardTrainingAlgorithmEx",              "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetRNNBackwardDataAlgorithmMaxCount",            {"hipdnnGetRNNBackwardDataAlgorithmMaxCount",            "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFindRNNBackwardDataAlgorithmEx",                 {"hipdnnFindRNNBackwardDataAlgorithmEx",                 "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetRNNBackwardWeightsAlgorithmMaxCount",         {"hipdnnGetRNNBackwardWeightsAlgorithmMaxCount",         "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnFindRNNBackwardWeightsAlgorithmEx",              {"hipdnnFindRNNBackwardWeightsAlgorithmEx",              "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnCreatePersistentRNNPlan",                        {"hipdnnCreatePersistentRNNPlan",                        "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetPersistentRNNPlan",                           {"hipdnnSetPersistentRNNPlan",                           "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnDestroyPersistentRNNPlan",                       {"hipdnnDestroyPersistentRNNPlan",                       "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetRNNDescriptor",                               {"hipdnnSetRNNDescriptor",                               "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetRNNDescriptor",                               {"hipdnnGetRNNDescriptor",                               "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSetRNNProjectionLayers",                         {"hipdnnSetRNNProjectionLayers",                         "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetRNNProjectionLayers",                         {"hipdnnGetRNNProjectionLayers",                         "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSetRNNAlgorithmDescriptor",                      {"hipdnnSetRNNAlgorithmDescriptor",                      "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSetRNNMatrixMathType",                           {"hipdnnSetRNNMatrixMathType",                           "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetRNNMatrixMathType",                           {"hipdnnGetRNNMatrixMathType",                           "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetRNNWorkspaceSize",                            {"hipdnnGetRNNWorkspaceSize",                            "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetRNNTrainingReserveSize",                      {"hipdnnGetRNNTrainingReserveSize",                      "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetRNNParamsSize",                               {"hipdnnGetRNNParamsSize",                               "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetRNNLinLayerMatrixParams",                     {"hipdnnGetRNNLinLayerMatrixParams",                     "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnGetRNNLinLayerBiasParams",                       {"hipdnnGetRNNLinLayerBiasParams",                       "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnRNNForwardInference",                            {"hipdnnRNNForwardInference",                            "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnRNNForwardTraining",                             {"hipdnnRNNForwardTraining",                             "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnRNNBackwardData",                                {"hipdnnRNNBackwardData",                                "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnRNNBackwardWeights",                             {"hipdnnRNNBackwardWeights",                             "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetRNNDescriptor_v5",                            {"hipdnnSetRNNDescriptor_v5",                            "", CONV_LIB_FUNC, API_DNN}},
  {"cudnnSetRNNDescriptor_v6",                            {"hipdnnSetRNNDescriptor_v6",                            "", CONV_LIB_FUNC, API_DNN}},

  // cuDNN Connectionist Temporal Classification loss functions
  {"cudnnCreateCTCLossDescriptor",                        {"hipdnnCreateCTCLossDescriptor",                        "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSetCTCLossDescriptor",                           {"hipdnnSetCTCLossDescriptor",                           "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetCTCLossDescriptor",                           {"hipdnnGetCTCLossDescriptor",                           "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnDestroyCTCLossDescriptor",                       {"hipdnnDestroyCTCLossDescriptor",                       "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnCTCLoss",                                        {"hipdnnCTCLoss",                                        "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetCTCLossWorkspaceSize",                        {"hipdnnGetCTCLossWorkspaceSize",                        "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},

  // cuDNN Algorithm functions
  {"cudnnCreateAlgorithmDescriptor",                      {"hipdnnCreateAlgorithmDescriptor",                      "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSetAlgorithmDescriptor",                         {"hipdnnSetAlgorithmDescriptor",                         "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetAlgorithmDescriptor",                         {"hipdnnGetAlgorithmDescriptor",                         "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnCopyAlgorithmDescriptor",                        {"hipdnnCopyAlgorithmDescriptor",                        "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnDestroyAlgorithmDescriptor",                     {"hipdnnDestroyAlgorithmDescriptor",                     "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnCreateAlgorithmPerformance",                     {"hipdnnCreateAlgorithmPerformance",                     "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSetAlgorithmPerformance",                        {"hipdnnSetAlgorithmPerformance",                        "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetAlgorithmPerformance",                        {"hipdnnGetAlgorithmPerformance",                        "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnDestroyAlgorithmPerformance",                    {"hipdnnDestroyAlgorithmPerformance",                    "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnGetAlgorithmSpaceSize",                          {"hipdnnGetAlgorithmSpaceSize",                          "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnSaveAlgorithm",                                  {"hipdnnSaveAlgorithm",                                  "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
  {"cudnnRestoreAlgorithm",                               {"hipdnnRestoreAlgorithm",                               "", CONV_LIB_FUNC, API_DNN, HIP_UNSUPPORTED}},
};
