# CUDNN API supported by HIP

## **1. CUDNN Data types**

| **type**     |   **CUDA**                                                    |**CUDA version\***|   **HIP**                                                  |**HIP value** (if differs) |
|-------------:|---------------------------------------------------------------|:----------------:|------------------------------------------------------------|---------------------------|
| define       |`CUDNN_VERSION`                                                |                  |`HIPDNN_VERSION`                                            |
| struct       |`cudnnContext`                                                 |                  |                                                            |
| struct*      |`cudnnHandle_t`                                                |                  |`hipdnnHandle_t`                                            |
| enum         |***`cudnnStatus_t`***                                          |                  |***`hipdnnStatus_t`***                                      |
|            0 |*`CUDNN_STATUS_SUCCESS`*                                       |                  |*`HIPDNN_STATUS_SUCCESS`*                                   |
|            1 |*`CUDNN_STATUS_NOT_INITIALIZED`*                               |                  |*`HIPDNN_STATUS_NOT_INITIALIZED`*                           |
|            2 |*`CUDNN_STATUS_ALLOC_FAILED`*                                  |                  |*`HIPDNN_STATUS_ALLOC_FAILED`*                              |
|            3 |*`CUDNN_STATUS_BAD_PARAM`*                                     |                  |*`HIPDNN_STATUS_BAD_PARAM`*                                 |
|            4 |*`CUDNN_STATUS_INTERNAL_ERROR`*                                |                  |*`HIPDNN_STATUS_INTERNAL_ERROR`*                            |
|            5 |*`CUDNN_STATUS_INVALID_VALUE`*                                 |                  |*`HIPDNN_STATUS_INVALID_VALUE`*                             |
|            6 |*`CUDNN_STATUS_ARCH_MISMATCH`*                                 |                  |*`HIPDNN_STATUS_ARCH_MISMATCH`*                             |
|            7 |*`CUDNN_STATUS_MAPPING_ERROR`*                                 |                  |*`HIPDNN_STATUS_MAPPING_ERROR`*                             |
|            8 |*`CUDNN_STATUS_EXECUTION_FAILED`*                              |                  |*`HIPDNN_STATUS_EXECUTION_FAILED`*                          |
|            9 |*`CUDNN_STATUS_NOT_SUPPORTED`*                                 |                  |*`HIPDNN_STATUS_NOT_SUPPORTED`*                             |
|           10 |*`CUDNN_STATUS_LICENSE_ERROR`*                                 |                  |*`HIPDNN_STATUS_LICENSE_ERROR`*                             |
|           11 |*`CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING`*                  | 7.5              |*`HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING`*              |
|           12 |*`CUDNN_STATUS_RUNTIME_IN_PROGRESS`*                           | 8.0              |                                                            |
|           13 |*`CUDNN_STATUS_RUNTIME_FP_OVERFLOW`*                           | 8.0              |                                                            |
| struct       |`cudnnRuntimeTag_t`                                            | 8.0              |                                                            |
| enum         |***`cudnnErrQueryMode_t`***                                    | 8.0              |                                                            |
|            0 |*`CUDNN_ERRQUERY_RAWCODE`*                                     | 8.0              |                                                            |
|            1 |*`CUDNN_ERRQUERY_NONBLOCKING`*                                 | 8.0              |                                                            |
|            2 |*`CUDNN_ERRQUERY_BLOCKING`*                                    | 8.0              |                                                            |
| struct       |`cudnnTensorStruct`                                            |                  |                                                            |
| struct*      |`cudnnTensorDescriptor_t`                                      |                  |`hipdnnTensorDescriptor_t`                                  |
| struct       |`cudnnConvolutionStruct`                                       |                  |                                                            |
| struct*      |`cudnnConvolutionDescriptor_t`                                 |                  |`hipdnnConvolutionDescriptor_t`                             |
| struct       |`cudnnPoolingStruct`                                           |                  |                                                            |
| struct*      |`cudnnPoolingDescriptor_t`                                     |                  |`hipdnnPoolingDescriptor_t`                                 |
| struct       |`cudnnFilterStruct`                                            |                  |                                                            |
| struct*      |`cudnnFilterDescriptor_t`                                      |                  |`hipdnnFilterDescriptor_t`                                  |
| struct       |`cudnnLRNStruct`                                               |                  |                                                            |
| struct*      |`cudnnLRNDescriptor_t`                                         |                  |`hipdnnLRNDescriptor_t`                                     |
| struct       |`cudnnActivationStruct`                                        |                  |                                                            |
| struct*      |`cudnnActivationDescriptor_t`                                  |                  |`hipdnnActivationDescriptor_t`                              |
| struct       |`cudnnSpatialTransformerStruct`                                | 7.5              |                                                            |
| struct*      |`cudnnSpatialTransformerDescriptor_t`                          | 7.5              |                                                            |
| struct       |`cudnnOpTensorStruct`                                          | 7.5              |                                                            |
| struct*      |`cudnnOpTensorDescriptor_t`                                    | 7.5              |`hipdnnOpTensorDescriptor_t`                                |
| struct       |`cudnnReduceTensorStruct`                                      | 7.5              |                                                            |
| struct*      |`cudnnReduceTensorDescriptor_t`                                | 7.5              |`hipdnnReduceTensorDescriptor_t`                            |
| struct       |`cudnnCTCLossStruct`                                           | 8.0              |                                                            |
| struct*      |`cudnnCTCLossDescriptor_t`                                     | 8.0              |                                                            |
| struct       |`cudnnTensorTransformStruct`                                   | 9.0              |                                                            |
| struct*      |`cudnnTensorTransformDescriptor_t`                             | 9.0              |                                                            |
| enum         |***`cudnnDataType_t`***                                        |                  |***`hipdnnDataType_t`***                                    |
|            0 |*`CUDNN_DATA_FLOAT`*                                           |                  |*`HIPDNN_DATA_FLOAT`*                                       |
|            1 |*`CUDNN_DATA_DOUBLE`*                                          |                  |*`HIPDNN_DATA_DOUBLE`*                                      |
|            2 |*`CUDNN_DATA_HALF`*                                            |                  |*`HIPDNN_DATA_HALF`*                                        |
|            3 |*`CUDNN_DATA_INT8`*                                            | 7.5              |*`HIPDNN_DATA_INT8`*                                        |
|            4 |*`CUDNN_DATA_INT32`*                                           | 7.5              |*`HIPDNN_DATA_INT32`*                                       |
|            5 |*`CUDNN_DATA_INT8x4`*                                          | 7.5              |*`HIPDNN_DATA_INT8x4`*                                      |
|            6 |*`CUDNN_DATA_UINT8`*                                           | 8.0              |                                                            |
|            7 |*`CUDNN_DATA_UINT8x4`*                                         | 8.0              |                                                            |
|            8 |*`CUDNN_DATA_INT8x32`*                                         | 9.0              |                                                            |
| enum         |***`cudnnMathType_t`***                                        | 8.0              |***`hipdnnMathType_t`***                                    |
|            0 |*`CUDNN_DEFAULT_MATH`*                                         | 8.0              |*`HIPDNN_DEFAULT_MATH`*                                     |
|            1 |*`CUDNN_TENSOR_OP_MATH`*                                       | 8.0              |*`HIPDNN_TENSOR_OP_MATH`*                                   |
| enum         |***`cudnnNanPropagation_t`***                                  |                  |***`hipdnnNanPropagation_t`***                              |
|            0 |*`CUDNN_NOT_PROPAGATE_NAN`*                                    |                  |*`HIPDNN_NOT_PROPAGATE_NAN`*                                |
|            1 |*`CUDNN_PROPAGATE_NAN`*                                        |                  |*`HIPDNN_PROPAGATE_NAN`*                                    |
| enum         |***`cudnnDeterminism_t`***                                     | 7.5              |                                                            |
|            0 |*`CUDNN_NON_DETERMINISTIC`*                                    | 7.5              |                                                            |
|            1 |*`CUDNN_DETERMINISTIC`*                                        | 7.5              |                                                            |
| define       |`CUDNN_DIM_MAX`                                                |                  |                                                            |
| enum         |***`cudnnTensorFormat_t`***                                    |                  |***`hipdnnTensorFormat_t`***                                |
|            0 |*`CUDNN_TENSOR_NCHW`*                                          |                  |*`HIPDNN_TENSOR_NCHW`*                                      |
|            1 |*`CUDNN_TENSOR_NHWC`*                                          |                  |*`HIPDNN_TENSOR_NHWC`*                                      |
|            2 |*`CUDNN_TENSOR_NCHW_VECT_C`*                                   | 7.5              |*`HIPDNN_TENSOR_NCHW_VECT_C`*                               |
| enum         |***`cudnnFoldingDirection_t`***                                | 9.0              |                                                            |
|            0 |*`CUDNN_TRANSFORM_FOLD`*                                       | 9.0              |                                                            |
|            1 |*`CUDNN_TRANSFORM_UNFOLD`*                                     | 9.0              |                                                            |
| enum         |***`cudnnOpTensorOp_t`***                                      | 7.5              |***`hipdnnOpTensorOp_t`***                                  |
|            0 |*`CUDNN_OP_TENSOR_ADD`*                                        | 7.5              |*`HIPDNN_OP_TENSOR_ADD`*                                    |
|            1 |*`CUDNN_OP_TENSOR_MUL`*                                        | 7.5              |*`HIPDNN_OP_TENSOR_MUL`*                                    |
|            2 |*`CUDNN_OP_TENSOR_MIN`*                                        | 7.5              |*`HIPDNN_OP_TENSOR_MIN`*                                    |
|            3 |*`CUDNN_OP_TENSOR_MAX`*                                        | 7.5              |*`HIPDNN_OP_TENSOR_MAX`*                                    |
|            4 |*`CUDNN_OP_TENSOR_SQRT`*                                       | 7.5              |*`HIPDNN_OP_TENSOR_SQRT`*                                   |
|            5 |*`CUDNN_OP_TENSOR_NOT`*                                        | 8.0              |*`HIPDNN_OP_TENSOR_NOT`*                                    |
| enum         |***`cudnnReduceTensorOp_t`***                                  | 7.5              |***`hipdnnReduceTensorOp_t`***                              |
|            0 |*`CUDNN_REDUCE_TENSOR_ADD`*                                    | 7.5              |*`HIPDNN_REDUCE_TENSOR_ADD`*                                |
|            1 |*`CUDNN_REDUCE_TENSOR_MUL`*                                    | 7.5              |*`HIPDNN_REDUCE_TENSOR_MUL`*                                |
|            2 |*`CUDNN_REDUCE_TENSOR_MIN`*                                    | 7.5              |*`HIPDNN_REDUCE_TENSOR_MIN`*                                |
|            3 |*`CUDNN_REDUCE_TENSOR_MAX`*                                    | 7.5              |*`HIPDNN_REDUCE_TENSOR_MAX`*                                |
|            4 |*`CUDNN_REDUCE_TENSOR_AMAX`*                                   | 7.5              |*`HIPDNN_REDUCE_TENSOR_AMAX`*                               |
|            5 |*`CUDNN_REDUCE_TENSOR_AVG`*                                    | 7.5              |*`HIPDNN_REDUCE_TENSOR_AVG`*                                |
|            6 |*`CUDNN_REDUCE_TENSOR_NORM1`*                                  | 7.5              |*`HIPDNN_REDUCE_TENSOR_NORM1`*                              |
|            7 |*`CUDNN_REDUCE_TENSOR_NORM2`*                                  | 7.5              |*`HIPDNN_REDUCE_TENSOR_NORM2`*                              |
|            8 |*`CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS`*                           | 8.0              |*`HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS`*                       |
| enum         |***`cudnnReduceTensorIndices_t`***                             | 7.5              |***`hipdnnReduceTensorIndices_t`***                         |
|            0 |*`CUDNN_REDUCE_TENSOR_NO_INDICES`*                             | 7.5              |*`HIPDNN_REDUCE_TENSOR_NO_INDICES`*                         |
|            1 |*`CUDNN_REDUCE_TENSOR_FLATTENED_INDICES`*                      | 7.5              |*`HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES`*                  |
| enum         |***`cudnnIndicesType_t`***                                     | 7.5              |***`hipdnnIndicesType_t`***                                 |
|            0 |*`CUDNN_32BIT_INDICES`*                                        | 7.5              |*`HIPDNN_32BIT_INDICES`*                                    |
|            1 |*`CUDNN_64BIT_INDICES`*                                        | 7.5              |*`HIPDNN_64BIT_INDICES`*                                    |
|            2 |*`CUDNN_16BIT_INDICES`*                                        | 7.5              |*`HIPDNN_16BIT_INDICES`*                                    |
|            3 |*`CUDNN_8BIT_INDICES`*                                         | 7.5              |*`HIPDNN_8BIT_INDICES`*                                     |
| enum         |***`cudnnConvolutionMode_t`***                                 |                  |***`hipdnnConvolutionMode_t`***                             |
|            0 |*`CUDNN_CONVOLUTION`*                                          |                  |*`HIPDNN_CONVOLUTION`*                                      |
|            1 |*`CUDNN_CROSS_CORRELATION`*                                    |                  |*`HIPDNN_CROSS_CORRELATION`*                                |
| enum         |***`cudnnConvolutionFwdPreference_t`***                        |                  |***`hipdnnConvolutionFwdPreference_t`***                    |
|            0 |*`CUDNN_CONVOLUTION_FWD_NO_WORKSPACE`*                         |                  |*`HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE`*                     |
|            1 |*`CUDNN_CONVOLUTION_FWD_PREFER_FASTEST`*                       |                  |*`HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST`*                   |
|            2 |*`CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT`*              |                  |*`HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT`*          |
| enum         |***`cudnnConvolutionFwdAlgo_t`***                              |                  |***`hipdnnConvolutionFwdAlgo_t`***                          |
|            0 |*`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`*                   |                  |*`HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`*               |
|            1 |*`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`*           |                  |*`HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`*       |
|            2 |*`CUDNN_CONVOLUTION_FWD_ALGO_GEMM`*                            |                  |*`HIPDNN_CONVOLUTION_FWD_ALGO_GEMM`*                        |
|            3 |*`CUDNN_CONVOLUTION_FWD_ALGO_DIRECT`*                          |                  |*`HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT`*                      |
|            4 |*`CUDNN_CONVOLUTION_FWD_ALGO_FFT`*                             |                  |*`HIPDNN_CONVOLUTION_FWD_ALGO_FFT`*                         |
|            5 |*`CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING`*                      |                  |*`HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING`*                  |
|            6 |*`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD`*                        | 7.5              |*`HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD`*                    |
|            7 |*`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED`*               | 7.5              |*`HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED`*           |
|            8 |*`CUDNN_CONVOLUTION_FWD_ALGO_COUNT`*                           | 7.5              |*`HIPDNN_CONVOLUTION_FWD_ALGO_COUNT`*                       |
| struct       |`cudnnConvolutionFwdAlgoPerf_t`                                |                  |`hipdnnConvolutionFwdAlgoPerf_t`                            |
| enum         |***`cudnnConvolutionBwdFilterPreference_t`***                  |                  |***`hipdnnConvolutionBwdFilterPreference_t`***              |
|            0 |*`CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE`*                  |                  |*`HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE`*              |
|            1 |*`CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST`*                |                  |*`HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST`*            |
|            2 |*`CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT`*       |                  |*`HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT`*   |
| enum         |***`cudnnConvolutionBwdFilterAlgo_t`***                        |                  |***`hipdnnConvolutionBwdFilterAlgo_t`***                    |
|            0 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0`*                        |                  |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0`*                    |
|            1 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1`*                        |                  |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1`*                    |
|            2 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT`*                      |                  |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT`*                  |
|            3 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3`*                        |                  |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3`*                    |
|            4 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD`*                 | 7.5              |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD`*             |
|            5 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED`*        | 7.5              |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED`*    |
|            6 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING`*               | 7.5              |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING`*           |
|            7 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT`*                    | 7.5              |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT`*                |
| struct       |`cudnnConvolutionBwdDataAlgoPerf_t`                            |                  |`hipdnnConvolutionBwdDataAlgoPerf_t`                        |
| enum         |***`cudnnSoftmaxAlgorithm_t`***                                |                  |***`hipdnnSoftmaxAlgorithm_t`***                            |
|            0 |*`CUDNN_SOFTMAX_FAST`*                                         |                  |*`HIPDNN_SOFTMAX_FAST`*                                     |
|            1 |*`CUDNN_SOFTMAX_ACCURATE`*                                     |                  |*`HIPDNN_SOFTMAX_ACCURATE`*                                 |
|            2 |*`CUDNN_SOFTMAX_LOG`*                                          |                  |*`HIPDNN_SOFTMAX_LOG`*                                      |
| enum         |***`cudnnSoftmaxMode_t`***                                     |                  |***`hipdnnSoftmaxMode_t`***                                 |
|            0 |*`CUDNN_SOFTMAX_MODE_INSTANCE`*                                |                  |*`HIPDNN_SOFTMAX_MODE_INSTANCE`*                            |
|            1 |*`CUDNN_SOFTMAX_MODE_CHANNEL`*                                 |                  |*`HIPDNN_SOFTMAX_MODE_CHANNEL`*                             |
| enum         |***`cudnnPoolingMode_t`***                                     |                  |***`hipdnnPoolingMode_t`***                                 |
|            0 |*`CUDNN_POOLING_MAX`*                                          |                  |*`HIPDNN_POOLING_MAX`*                                      |
|            1 |*`CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING`*                |                  |*`HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING`*            |
|            2 |*`CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING`*                |                  |*`HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING`*            |
|            3 |*`CUDNN_POOLING_MAX_DETERMINISTIC`*                            | 7.5              |*`HIPDNN_POOLING_MAX_DETERMINISTIC`*                        |
| enum         |***`cudnnActivationMode_t`***                                  |                  |***`hipdnnActivationMode_t`***                              |
|            0 |*`CUDNN_ACTIVATION_SIGMOID`*                                   |                  |*`HIPDNN_ACTIVATION_SIGMOID`*                               |
|            1 |*`CUDNN_ACTIVATION_RELU`*                                      |                  |*`HIPDNN_ACTIVATION_RELU`*                                  |
|            2 |*`CUDNN_ACTIVATION_TANH`*                                      |                  |*`HIPDNN_ACTIVATION_TANH`*                                  |
|            3 |*`CUDNN_ACTIVATION_CLIPPED_RELU`*                              |                  |*`HIPDNN_ACTIVATION_CLIPPED_RELU`*                          |
|            4 |*`CUDNN_ACTIVATION_ELU`*                                       | 7.5              |*`HIPDNN_ACTIVATION_ELU`*                                   |
|            5 |*`CUDNN_ACTIVATION_IDENTITY`*                                  | 8.0              |*`HIPDNN_ACTIVATION_PATHTRU`*                               |
| define       |`CUDNN_LRN_MIN_N`                                              |                  |                                                            |
| define       |`CUDNN_LRN_MAX_N`                                              |                  |                                                            |
| define       |`CUDNN_LRN_MIN_K`                                              |                  |                                                            |
| define       |`CUDNN_LRN_MIN_BETA`                                           |                  |                                                            |
| enum         |***`cudnnLRNMode_t`***                                         |                  |***`hipdnnLRNMode_t`***                                     |
|            0 |*`CUDNN_LRN_CROSS_CHANNEL_DIM1`*                               |                  |*`HIPDNN_LRN_CROSS_CHANNEL`*                                |
| enum         |***`cudnnDivNormMode_t`***                                     |                  |                                                            |
|            0 |*`CUDNN_DIVNORM_PRECOMPUTED_MEANS`*                            |                  |                                                            |
| enum         |***`cudnnBatchNormMode_t`***                                   |                  |***`hipdnnBatchNormMode_t`***                               |
|            0 |*`CUDNN_BATCHNORM_PER_ACTIVATION`*                             |                  |*`HIPDNN_BATCHNORM_PER_ACTIVATION`*                         |
|            1 |*`CUDNN_BATCHNORM_SPATIAL`*                                    |                  |*`HIPDNN_BATCHNORM_SPATIAL`*                                |
|            2 |*`CUDNN_BATCHNORM_SPATIAL_PERSISTENT`*                         | 8.0              |*`HIPDNN_BATCHNORM_SPATIAL_PERSISTENT`*                     |
| define       |`CUDNN_BN_MIN_EPSILON`                                         |                  |`HIPDNN_BN_MIN_EPSILON`                                     |
| enum         |***`cudnnSamplerType_t`***                                     | 7.5              |                                                            |
|            0 |*`CUDNN_SAMPLER_BILINEAR`*                                     | 7.5              |                                                            |
| struct       |`cudnnDropoutStruct`                                           | 7.5              |                                                            |
| struct*      |`cudnnDropoutDescriptor_t`                                     | 7.5              |`hipdnnDropoutDescriptor_t`                                 |
| enum         |***`cudnnRNNMode_t`***                                         | 7.5              |***`hipdnnRNNMode_t`***                                     |
|            0 |*`CUDNN_RNN_RELU`*                                             | 7.5              |*`HIPDNN_RNN_RELU`*                                         |
|            1 |*`CUDNN_RNN_TANH`*                                             | 7.5              |*`HIPDNN_RNN_TANH`*                                         |
|            2 |*`CUDNN_LSTM`*                                                 | 7.5              |*`HIPDNN_LSTM`*                                             |
|            3 |*`CUDNN_GRU`*                                                  | 7.5              |*`HIPDNN_GRU`*                                              |
| enum         |***`cudnnRNNBiasMode_t`***                                     | 9.0              |***`hipdnnRNNBiasMode_t`***                                 |
|            0 |*`CUDNN_RNN_NO_BIAS`*                                          | 9.0              |*`HIPDNN_RNN_NO_BIAS`*                                      |
|            1 |*`CUDNN_RNN_SINGLE_INP_BIAS`*                                  | 9.0              |*`HIPDNN_RNN_WITH_BIAS`*                                    |
|            2 |*`CUDNN_RNN_DOUBLE_BIAS`*                                      | 9.0              |*`HIPDNN_RNN_WITH_BIAS`*                                    | 1                         |
|            3 |*`CUDNN_RNN_SINGLE_REC_BIAS`*                                  | 9.0              |*`HIPDNN_RNN_WITH_BIAS`*                                    | 1                         |
| enum         |***`cudnnDirectionMode_t`***                                   | 7.5              |***`hipdnnDirectionMode_t`***                               |
|            0 |*`CUDNN_UNIDIRECTIONAL`*                                       | 7.5              |*`HIPDNN_UNIDIRECTIONAL`*                                   |
|            1 |*`CUDNN_BIDIRECTIONAL`*                                        | 7.5              |*`HIPDNN_BIDIRECTIONAL`*                                    |
| enum         |***`cudnnRNNAlgo_t`***                                         | 7.5              |***`hipdnnRNNAlgo_t`***                                     |
|            0 |*`CUDNN_RNN_ALGO_STANDARD`*                                    | 7.5              |*`HIPDNN_RNN_ALGO_STANDARD`*                                |
|            1 |*`CUDNN_RNN_ALGO_PERSIST_STATIC`*                              | 7.5              |*`HIPDNN_RNN_ALGO_PERSIST_STATIC`*                          |
|            2 |*`CUDNN_RNN_ALGO_PERSIST_DYNAMIC`*                             | 7.5              |*`HIPDNN_RNN_ALGO_PERSIST_DYNAMIC`*                         |
|            3 |*`CUDNN_RNN_ALGO_COUNT`*                                       | 8.0              |                                                            |
| struct       |`cudnnAlgorithmStruct`                                         | 8.0              |                                                            |
| struct*      |`cudnnAlgorithmDescriptor_t`                                   | 8.0              |                                                            |
| struct       |`cudnnAlgorithmPerformanceStruct`                              | 8.0              |                                                            |
| struct*      |`cudnnAlgorithmPerformance_t`                                  | 8.0              |                                                            |
| struct       |`cudnnRNNStruct`                                               | 7.5              |                                                            |
| struct*      |`cudnnRNNDescriptor_t`                                         | 7.5              |`hipdnnRNNDescriptor_t`                                     |
| struct       |`cudnnPersistentRNNPlan`                                       | 7.5              |                                                            |
| struct*      |`cudnnPersistentRNNPlan_t`                                     | 7.5              |`hipdnnPersistentRNNPlan_t`                                 |
| enum         |***`cudnnCTCLossAlgo_t`***                                     | 8.0              |                                                            |
|            0 |*`CUDNN_CTC_LOSS_ALGO_DETERMINISTIC`*                          | 8.0              |                                                            |
|            1 |*`CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC`*                      | 8.0              |                                                            |
| struct       |`cudnnAlgorithm_t`                                             | 8.0              |                                                            |
| enum         |***`cudnnSeverity_t`***                                        | 8.0              |                                                            |
|            0 |*`CUDNN_SEV_FATAL`*                                            | 8.0              |                                                            |
|            1 |*`CUDNN_SEV_ERROR`*                                            | 8.0              |                                                            |
|            2 |*`CUDNN_SEV_WARNING`*                                          | 8.0              |                                                            |
|            3 |*`CUDNN_SEV_INFO`*                                             | 8.0              |                                                            |
| define       |`CUDNN_SEV_ERROR_EN`                                           | 8.0              |                                                            |
| define       |`CUDNN_SEV_WARNING_EN`                                         | 8.0              |                                                            |
| define       |`CUDNN_SEV_INFO_EN`                                            | 8.0              |                                                            |
| struct       |`cudnnDebug_t`                                                 | 8.0              |                                                            |
| struct       |`cudnnCallback_t`                                              | 8.0              |                                                            |
| enum         |***`cudnnBatchNormOps_t`***                                    | 9.0              |                                                            |
|            0 |*`CUDNN_BATCHNORM_OPS_BN`*                                     | 9.0              |                                                            |
|            1 |*`CUDNN_BATCHNORM_OPS_BN_ACTIVATION`*                          | 9.0              |                                                            |
|            2 |*`CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION`*                      | 9.0              |                                                            |
| enum         |***`cudnnRNNClipMode_t`***                                     | 9.0              |                                                            |
|            0 |*`CUDNN_RNN_CLIP_NONE`*                                        | 9.0              |                                                            |
|            1 |*`CUDNN_RNN_CLIP_MINMAX`*                                      | 9.0              |                                                            |
| struct       |`cudnnRNNDataStruct`                                           | 9.0              |                                                            |
| struct*      |`cudnnRNNDataDescriptor_t`                                     | 9.0              |                                                            |
| enum         |***`cudnnRNNDataLayout_t`***                                   | 9.0              |                                                            |
|            0 |*`CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED`*                   | 9.0              |                                                            |
|            1 |*`CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED`*                     | 9.0              |                                                            |
|            2 |*`CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED`*                 | 9.0              |                                                            |
| enum         |***`cudnnRNNPaddingMode_t`***                                  | 9.0              |                                                            |
|            0 |*`CUDNN_RNN_PADDED_IO_DISABLED`*                               | 9.0              |                                                            |
|            1 |*`CUDNN_RNN_PADDED_IO_ENABLED`*                                | 9.0              |                                                            |
| enum         |***`cudnnSeqDataAxis_t`***                                     | 9.0              |                                                            |
|            0 |*`CUDNN_SEQDATA_TIME_DIM`*                                     | 9.0              |                                                            |
|            1 |*`CUDNN_SEQDATA_BATCH_DIM`*                                    | 9.0              |                                                            |
|            2 |*`CUDNN_SEQDATA_BEAM_DIM`*                                     | 9.0              |                                                            |
|            3 |*`CUDNN_SEQDATA_VECT_DIM`*                                     | 9.0              |                                                            |
| define       |`CUDNN_SEQDATA_DIM_COUNT`                                      | 9.0              |                                                            |
| struct       |`cudnnSeqDataStruct`                                           | 9.0              |                                                            |
| struct*      |`cudnnSeqDataDescriptor_t`                                     | 9.0              |                                                            |
| unsigned     |***`cudnnAttnQueryMap_t`***                                    | 9.0              |                                                            |
|            0 |*`CUDNN_ATTN_QUERYMAP_ALL_TO_ONE`*                             | 9.0              |                                                            |
|      1U << 0 |*`CUDNN_ATTN_QUERYMAP_ONE_TO_ONE`*                             | 9.0              |                                                            |
|            1 |*`CUDNN_ATTN_DISABLE_PROJ_BIASES`*                             | 10.1 Update 2    |                                                            |
|      1U << 1 |*`CUDNN_ATTN_ENABLE_PROJ_BIASES`*                              | 10.1 Update 2    |                                                            |
| struct       |`cudnnAttnStruct`                                              | 9.0              |                                                            |
| struct*      |`cudnnAttnDescriptor_t`                                        | 9.0              |                                                            |
| enum         |***`cudnnMultiHeadAttnWeightKind_t`***                         | 9.0              |                                                            |
|            0 |*`CUDNN_MH_ATTN_Q_WEIGHTS`*                                    | 9.0              |                                                            |
|            1 |*`CUDNN_MH_ATTN_K_WEIGHTS`*                                    | 9.0              |                                                            |
|            2 |*`CUDNN_MH_ATTN_V_WEIGHTS`*                                    | 9.0              |                                                            |
|            3 |*`CUDNN_MH_ATTN_O_WEIGHTS`*                                    | 9.0              |                                                            |
|            4 |*`CUDNN_MH_ATTN_Q_BIASES`*                                     | 10.1 Update 2    |                                                            |
|            5 |*`CUDNN_MH_ATTN_K_BIASES`*                                     | 10.1 Update 2    |                                                            |
|            6 |*`CUDNN_MH_ATTN_V_BIASES`*                                     | 10.1 Update 2    |                                                            |
|            7 |*`CUDNN_MH_ATTN_O_BIASES`*                                     | 10.1 Update 2    |                                                            |
| define     8 |`CUDNN_ATTN_WKIND_COUNT`                                       | 10.1 Update 2    |                                                            |
| enum         |***`cudnnWgradMode_t`***                                       | 9.0              |                                                            |
|            0 |*`CUDNN_WGRAD_MODE_ADD`*                                       | 9.0              |                                                            |
|            1 |*`CUDNN_WGRAD_MODE_SET`*                                       | 9.0              |                                                            |
| enum         |***`cudnnReorderType_t`***                                     | 10.1             |                                                            |
|            0 |*`CUDNN_DEFAULT_REORDER`*                                      | 10.1             |                                                            |
|            1 |*`CUDNN_NO_REORDER`*                                           | 10.1             |                                                            |
| enum         |***`cudnnLossNormalizationMode_t`***                           | 10.1             |                                                            |
|            0 |*`CUDNN_LOSS_NORMALIZATION_NONE`*                              | 10.1             |                                                            |
|            1 |*`CUDNN_LOSS_NORMALIZATION_SOFTMAX`*                           | 10.1             |                                                            |
| struct       |`cudnnFusedOpsConstParamStruct`                                | 10.1             |                                                            |
| struct*      |`cudnnFusedOpsConstParamPack_t`                                | 10.1             |                                                            |
| struct       |`cudnnFusedOpsVariantParamStruct`                              | 10.1             |                                                            |
| struct*      |`cudnnFusedOpsVariantParamPack_t`                              | 10.1             |                                                            |
| struct       |`cudnnFusedOpsPlanStruct`                                      | 10.1             |                                                            |
| struct*      |`cudnnFusedOpsPlan_t`                                          | 10.1             |                                                            |
| enum         |***`cudnnFusedOps_t`***                                        | 10.1             |                                                            |
|            0 |*`CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS`*             | 10.1             |                                                            |
|            1 |*`CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD`*                    | 10.1             |                                                            |
|            2 |*`CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING`*                | 10.1             |                                                            |
|            3 |*`CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE`*               | 10.1             |                                                            |
|            4 |*`CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION`*                 | 10.1             |                                                            |
|            5 |*`CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK`*          | 10.1             |                                                            |
|            6 |*`CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM`*                    | 10.1             |                                                            |
| enum         |***`cudnnFusedOpsConstParamLabel_t`***                         | 10.1             |                                                            |
|            0 |*`CUDNN_PARAM_XDESC`*                                          | 10.1             |                                                            |
|            1 |*`CUDNN_PARAM_XDATA_PLACEHOLDER`*                              | 10.1             |                                                            |
|            2 |*`CUDNN_PARAM_BN_MODE`*                                        | 10.1             |                                                            |
|            3 |*`CUDNN_PARAM_BN_EQSCALEBIAS_DESC`*                            | 10.1             |                                                            |
|            4 |*`CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER`*                         | 10.1             |                                                            |
|            5 |*`CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER`*                          | 10.1             |                                                            |
|            6 |*`CUDNN_PARAM_ACTIVATION_DESC`*                                | 10.1             |                                                            |
|            7 |*`CUDNN_PARAM_CONV_DESC`*                                      | 10.1             |                                                            |
|            8 |*`CUDNN_PARAM_WDESC`*                                          | 10.1             |                                                            |
|            9 |*`CUDNN_PARAM_WDATA_PLACEHOLDER`*                              | 10.1             |                                                            |
|           10 |*`CUDNN_PARAM_DWDESC`*                                         | 10.1             |                                                            |
|           11 |*`CUDNN_PARAM_DWDATA_PLACEHOLDER`*                             | 10.1             |                                                            |
|           12 |*`CUDNN_PARAM_YDESC`*                                          | 10.1             |                                                            |
|           13 |*`CUDNN_PARAM_YDATA_PLACEHOLDER`*                              | 10.1             |                                                            |
|           14 |*`CUDNN_PARAM_DYDESC`*                                         | 10.1             |                                                            |
|           15 |*`CUDNN_PARAM_DYDATA_PLACEHOLDER`*                             | 10.1             |                                                            |
|           16 |*`CUDNN_PARAM_YSTATS_DESC`*                                    | 10.1             |                                                            |
|           17 |*`CUDNN_PARAM_YSUM_PLACEHOLDER`*                               | 10.1             |                                                            |
|           18 |*`CUDNN_PARAM_YSQSUM_PLACEHOLDER`*                             | 10.1             |                                                            |
|           19 |*`CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC`*                      | 10.1             |                                                            |
|           20 |*`CUDNN_PARAM_BN_SCALE_PLACEHOLDER`*                           | 10.1             |                                                            |
|           21 |*`CUDNN_PARAM_BN_BIAS_PLACEHOLDER`*                            | 10.1             |                                                            |
|           22 |*`CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER`*                      | 10.1             |                                                            |
|           23 |*`CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER`*                    | 10.1             |                                                            |
|           24 |*`CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER`*                    | 10.1             |                                                            |
|           25 |*`CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER`*                     | 10.1             |                                                            |
|           26 |*`CUDNN_PARAM_ZDESC`*                                          | 10.1             |                                                            |
|           27 |*`CUDNN_PARAM_ZDATA_PLACEHOLDER`*                              | 10.1             |                                                            |
|           28 |*`CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC`*                          | 10.1             |                                                            |
|           29 |*`CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER`*                       | 10.1             |                                                            |
|           30 |*`CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER`*                        | 10.1             |                                                            |
|           31 |*`CUDNN_PARAM_ACTIVATION_BITMASK_DESC`*                        | 10.1             |                                                            |
|           32 |*`CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER`*                 | 10.1             |                                                            |
|           33 |*`CUDNN_PARAM_DXDESC`*                                         | 10.1             |                                                            |
|           34 |*`CUDNN_PARAM_DXDATA_PLACEHOLDER`*                             | 10.1             |                                                            |
|           35 |*`CUDNN_PARAM_DZDESC`*                                         | 10.1             |                                                            |
|           36 |*`CUDNN_PARAM_DZDATA_PLACEHOLDER`*                             | 10.1             |                                                            |
|           37 |*`CUDNN_PARAM_BN_DSCALE_PLACEHOLDER`*                          | 10.1             |                                                            |
|           38 |*`CUDNN_PARAM_BN_DBIAS_PLACEHOLDER`*                           | 10.1             |                                                            |
| enum         |***`cudnnFusedOpsPointerPlaceHolder_t`***                      | 10.1             |                                                            |
|            0 |*`CUDNN_PTR_NULL`*                                             | 10.1             |                                                            |
|            1 |*`CUDNN_PTR_ELEM_ALIGNED`*                                     | 10.1             |                                                            |
|            2 |*`CUDNN_PTR_16B_ALIGNED`*                                      | 10.1             |                                                            |
| enum         |***`cudnnFusedOpsVariantParamLabel_t`***                       | 10.1             |                                                            |
|            0 |*`CUDNN_PTR_XDATA`*                                            | 10.1             |                                                            |
|            1 |*`CUDNN_PTR_BN_EQSCALE`*                                       | 10.1             |                                                            |
|            2 |*`CUDNN_PTR_BN_EQBIAS`*                                        | 10.1             |                                                            |
|            3 |*`CUDNN_PTR_WDATA`*                                            | 10.1             |                                                            |
|            4 |*`CUDNN_PTR_DWDATA`*                                           | 10.1             |                                                            |
|            5 |*`CUDNN_PTR_YDATA`*                                            | 10.1             |                                                            |
|            6 |*`CUDNN_PTR_DYDATA`*                                           | 10.1             |                                                            |
|            7 |*`CUDNN_PTR_YSUM`*                                             | 10.1             |                                                            |
|            8 |*`CUDNN_PTR_YSQSUM`*                                           | 10.1             |                                                            |
|            9 |*`CUDNN_PTR_WORKSPACE`*                                        | 10.1             |                                                            |
|           10 |*`CUDNN_PTR_BN_SCALE`*                                         | 10.1             |                                                            |
|           11 |*`CUDNN_PTR_BN_BIAS`*                                          | 10.1             |                                                            |
|           12 |*`CUDNN_PTR_BN_SAVED_MEAN`*                                    | 10.1             |                                                            |
|           13 |*`CUDNN_PTR_BN_SAVED_INVSTD`*                                  | 10.1             |                                                            |
|           14 |*`CUDNN_PTR_BN_RUNNING_MEAN`*                                  | 10.1             |                                                            |
|           15 |*`CUDNN_PTR_BN_RUNNING_VAR`*                                   | 10.1             |                                                            |
|           16 |*`CUDNN_PTR_ZDATA`*                                            | 10.1             |                                                            |
|           17 |*`CUDNN_PTR_BN_Z_EQSCALE`*                                     | 10.1             |                                                            |
|           18 |*`CUDNN_PTR_BN_Z_EQBIAS`*                                      | 10.1             |                                                            |
|           19 |*`CUDNN_PTR_ACTIVATION_BITMASK`*                               | 10.1             |                                                            |
|           20 |*`CUDNN_PTR_DXDATA`*                                           | 10.1             |                                                            |
|           21 |*`CUDNN_PTR_DZDATA`*                                           | 10.1             |                                                            |
|           22 |*`CUDNN_PTR_BN_DSCALE`*                                        | 10.1             |                                                            |
|           23 |*`CUDNN_PTR_BN_DBIAS`*                                         | 10.1             |                                                            |
|          100 |*`CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES`*                | 10.1             |                                                            |
|          101 |*`CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT`*                 | 10.1             |                                                            |
|          102 |*`CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR`*                      | 10.1             |                                                            |
|          103 |*`CUDNN_SCALAR_DOUBLE_BN_EPSILON`*                             | 10.1             |                                                            |

## **2. CUDNN API functions**

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:-----------------|
|`cudnnGetVersion`                                          |`hipdnnGetVersion`                               |
|`cudnnGetCudartVersion`                                    |                                                 | 7.5              |
|`cudnnGetErrorString`                                      |`hipdnnGetErrorString`                           |
|`cudnnQueryRuntimeError`                                   |                                                 | 8.0              |
|`cudnnGetProperty`                                         |                                                 | 7.5              |
|`cudnnCreate`                                              |`hipdnnCreate`                                   |
|`cudnnDestroy`                                             |`hipdnnDestroy`                                  |
|`cudnnSetStream`                                           |`hipdnnSetStream`                                |
|`cudnnGetStream`                                           |`hipdnnGetStream`                                |
|`cudnnCreateTensorDescriptor`                              |`hipdnnCreateTensorDescriptor`                   |
|`cudnnSetTensor4dDescriptor`                               |`hipdnnSetTensor4dDescriptor`                    |
|`cudnnSetTensor4dDescriptorEx`                             |`hipdnnSetTensor4dDescriptorEx`                  |
|`cudnnGetTensor4dDescriptor`                               |`hipdnnGetTensor4dDescriptor`                    |
|`cudnnSetTensorNdDescriptor`                               |`hipdnnSetTensorNdDescriptor`                    |
|`cudnnSetTensorNdDescriptorEx`                             |                                                 | 7.5              |
|`cudnnGetTensorNdDescriptor`                               |`hipdnnGetTensorNdDescriptor`                    |
|`cudnnGetTensorSizeInBytes`                                |                                                 | 7.5              |
|`cudnnDestroyTensorDescriptor`                             |`hipdnnDestroyTensorDescriptor`                  |
|`cudnnTransformTensor`                                     |                                                 |
|`cudnnTransformTensorEx`                                   |                                                 | 9.0              |
|`cudnnInitTransformDest`                                   |                                                 | 9.0              |
|`cudnnCreateTensorTransformDescriptor`                     |                                                 | 9.0              |
|`cudnnSetTensorTransformDescriptor`                        |                                                 | 9.0              |
|`cudnnGetTensorTransformDescriptor`                        |                                                 | 9.0              |
|`cudnnDestroyTensorTransformDescriptor`                    |                                                 | 9.0              |
|`cudnnAddTensor`                                           |`hipdnnAddTensor`                                |
|`cudnnCreateOpTensorDescriptor`                            |`hipdnnCreateOpTensorDescriptor`                 | 7.5              |
|`cudnnSetOpTensorDescriptor`                               |`hipdnnSetOpTensorDescriptor`                    | 7.5              |
|`cudnnGetOpTensorDescriptor`                               |`hipdnnGetOpTensorDescriptor`                    | 7.5              |
|`cudnnDestroyOpTensorDescriptor`                           |`hipdnnDestroyOpTensorDescriptor`                | 7.5              |
|`cudnnOpTensor`                                            |`hipdnnOpTensor`                                 | 7.5              |
|`cudnnGetFoldedConvBackwardDataDescriptors`                |                                                 | 10.1             |
|`cudnnCreateReduceTensorDescriptor`                        |`hipdnnCreateReduceTensorDescriptor`             | 7.5              |
|`cudnnSetReduceTensorDescriptor`                           |`hipdnnSetReduceTensorDescriptor`                | 7.5              |
|`cudnnGetReduceTensorDescriptor`                           |`hipdnnGetReduceTensorDescriptor`                | 7.5              |
|`cudnnDestroyReduceTensorDescriptor`                       |`hipdnnDestroyReduceTensorDescriptor`            | 7.5              |
|`cudnnGetReductionIndicesSize`                             |                                                 | 7.5              |
|`cudnnGetReductionWorkspaceSize`                           |`hipdnnGetReductionWorkspaceSize`                | 7.5              |
|`cudnnReduceTensor`                                        |`hipdnnReduceTensor`                             | 7.5              |
|`cudnnSetTensor`                                           |`hipdnnSetTensor`                                |
|`cudnnScaleTensor`                                         |`hipdnnScaleTensor`                              |
|`cudnnCreateFilterDescriptor`                              |`hipdnnCreateFilterDescriptor`                   |
|`cudnnSetFilter4dDescriptor`                               |`hipdnnSetFilter4dDescriptor`                    |
|`cudnnGetFilter4dDescriptor`                               |`hipdnnGetFilter4dDescriptor`                    |
|`cudnnSetFilterNdDescriptor`                               |`hipdnnSetFilterNdDescriptor`                    |
|`cudnnGetFilterNdDescriptor`                               |`hipdnnGetFilterNdDescriptor`                    |
|`cudnnGetFilterSizeInBytes`                                |                                                 | 10.1             |
|`cudnnTransformFilter`                                     |                                                 | 10.1             |
|`cudnnDestroyFilterDescriptor`                             |`hipdnnDestroyFilterDescriptor`                  |
|`cudnnReorderFilterAndBias`                                |                                                 | 10.1             |
|`cudnnCreateConvolutionDescriptor`                         |`hipdnnCreateConvolutionDescriptor`              |
|`cudnnSetConvolutionMathType`                              |`hipdnnSetConvolutionMathType`                   | 8.0              |
|`cudnnGetConvolutionMathType`                              |                                                 | 8.0              |
|`cudnnSetConvolutionGroupCount`                            |`hipdnnSetConvolutionGroupCount`                 | 8.0              |
|`cudnnGetConvolutionGroupCount`                            |                                                 | 8.0              |
|`cudnnSetConvolutionReorderType`                           |                                                 | 10.1             |
|`cudnnGetConvolutionReorderType`                           |                                                 | 10.1             |
|`cudnnSetConvolution2dDescriptor`                          |`hipdnnSetConvolution2dDescriptor`               |
|`cudnnGetConvolution2dDescriptor`                          |`hipdnnGetConvolution2dDescriptor`               |
|`cudnnGetConvolution2dForwardOutputDim`                    |`hipdnnGetConvolution2dForwardOutputDim`         |
|`cudnnSetConvolutionNdDescriptor`                          |`hipdnnSetConvolutionNdDescriptor`               |
|`cudnnGetConvolutionNdDescriptor`                          |                                                 |
|`cudnnGetConvolutionNdForwardOutputDim`                    |                                                 |
|`cudnnDestroyConvolutionDescriptor`                        |                                                 |
|`cudnnGetConvolutionForwardAlgorithmMaxCount`              |                                                 | 8.0              |
|`cudnnFindConvolutionForwardAlgorithm`                     |`hipdnnFindConvolutionForwardAlgorithm`          |
|`cudnnFindConvolutionForwardAlgorithmEx`                   |`hipdnnFindConvolutionForwardAlgorithmEx`        | 7.5              |
|`cudnnGetConvolutionForwardAlgorithm`                      |`hipdnnGetConvolutionForwardAlgorithm`           |
|`cudnnGetConvolutionForwardAlgorithm_v7`                   |                                                 | 8.0              |
|`cudnnGetConvolutionForwardWorkspaceSize`                  |`hipdnnGetConvolutionForwardWorkspaceSize`       |
|`cudnnConvolutionForward`                                  |`hipdnnConvolutionForward`                       |
|`cudnnConvolutionBiasActivationForward`                    |                                                 | 7.5              |
|`cudnnConvolutionBackwardBias`                             |`hipdnnConvolutionBackwardBias`                  |
|`cudnnGetConvolutionBackwardFilterAlgorithmMaxCount`       |                                                 | 8.0              |
|`cudnnFindConvolutionBackwardFilterAlgorithm`              |`hipdnnFindConvolutionBackwardFilterAlgorithm`   |
|`cudnnFindConvolutionBackwardFilterAlgorithmEx`            |`hipdnnFindConvolutionBackwardFilterAlgorithmEx` | 7.5              |
|`cudnnGetConvolutionBackwardFilterAlgorithm`               |`hipdnnGetConvolutionBackwardFilterAlgorithm`    |
|`cudnnGetConvolutionBackwardFilterAlgorithm_v7`            |                                                 | 8.0              |
|`cudnnGetConvolutionBackwardFilterWorkspaceSize`           |`hipdnnGetConvolutionBackwardFilterWorkspaceSize`|
|`cudnnConvolutionBackwardFilter`                           |`hipdnnConvolutionBackwardFilter`                |
|`cudnnGetConvolutionBackwardDataAlgorithmMaxCount`         |                                                 | 8.0              |
|`cudnnFindConvolutionBackwardDataAlgorithm`                |`hipdnnFindConvolutionBackwardDataAlgorithm`     |
|`cudnnFindConvolutionBackwardDataAlgorithmEx`              |`hipdnnFindConvolutionBackwardDataAlgorithmEx`   | 7.5              |
|`cudnnGetConvolutionBackwardDataAlgorithm`                 |`hipdnnGetConvolutionBackwardDataAlgorithm`      |
|`cudnnGetConvolutionBackwardDataAlgorithm_v7`              |                                                 | 8.0              |
|`cudnnGetConvolutionBackwardDataWorkspaceSize`             |`hipdnnGetConvolutionBackwardDataWorkspaceSize`  |
|`cudnnConvolutionBackwardData`                             |`hipdnnConvolutionBackwardData`                  |
|`cudnnIm2Col`                                              |                                                 |
|`cudnnSoftmaxForward`                                      |`hipdnnSoftmaxForward`                           |
|`cudnnSoftmaxBackward`                                     |`hipdnnSoftmaxBackward`                          |
|`cudnnCreatePoolingDescriptor`                             |`hipdnnCreatePoolingDescriptor`                  |
|`cudnnSetPooling2dDescriptor`                              |`hipdnnSetPooling2dDescriptor`                   |
|`cudnnGetPooling2dDescriptor`                              |`hipdnnGetPooling2dDescriptor`                   |
|`cudnnSetPoolingNdDescriptor`                              |`hipdnnSetPoolingNdDescriptor`                   |
|`cudnnGetPoolingNdDescriptor`                              |                                                 |
|`cudnnGetPoolingNdForwardOutputDim`                        |                                                 |
|`cudnnGetPooling2dForwardOutputDim`                        |`hipdnnGetPooling2dForwardOutputDim`             |
|`cudnnDestroyPoolingDescriptor`                            |`hipdnnDestroyPoolingDescriptor`                 |
|`cudnnPoolingForward`                                      |`hipdnnPoolingForward`                           |
|`cudnnPoolingBackward`                                     |`hipdnnPoolingBackward`                          |
|`cudnnCreateActivationDescriptor`                          |`hipdnnCreateActivationDescriptor`               |
|`cudnnSetActivationDescriptor`                             |`hipdnnSetActivationDescriptor`                  |
|`cudnnGetActivationDescriptor`                             |`hipdnnGetActivationDescriptor`                  |
|`cudnnDestroyActivationDescriptor`                         |`hipdnnDestroyActivationDescriptor`              |
|`cudnnActivationForward`                                   |`hipdnnActivationForward`                        |
|`cudnnActivationBackward`                                  |`hipdnnActivationBackward`                       |
|`cudnnCreateLRNDescriptor`                                 |`hipdnnCreateLRNDescriptor`                      |
|`cudnnSetLRNDescriptor`                                    |`hipdnnSetLRNDescriptor`                         |
|`cudnnGetLRNDescriptor`                                    |`hipdnnGetLRNDescriptor`                         |
|`cudnnDestroyLRNDescriptor`                                |`hipdnnDestroyLRNDescriptor`                     |
|`cudnnLRNCrossChannelForward`                              |`hipdnnLRNCrossChannelForward`                   |
|`cudnnLRNCrossChannelBackward`                             |`hipdnnLRNCrossChannelBackward`                  |
|`cudnnDivisiveNormalizationForward`                        |                                                 |
|`cudnnDivisiveNormalizationBackward`                       |                                                 |
|`cudnnDeriveBNTensorDescriptor`                            |`hipdnnDeriveBNTensorDescriptor`                 |
|`cudnnBatchNormalizationForwardTraining`                   |`hipdnnBatchNormalizationForwardTraining`        |
|`cudnnBatchNormalizationForwardTrainingEx`                 |                                                 | 9.0              |
|`cudnnBatchNormalizationForwardInference`                  |`hipdnnBatchNormalizationForwardInference`       |
|`cudnnBatchNormalizationBackward`                          |`hipdnnBatchNormalizationBackward`               |
|`cudnnBatchNormalizationBackwardEx`                        |                                                 | 9.0              |
|`cudnnCreateSpatialTransformerDescriptor`                  |                                                 | 7.5              |
|`cudnnSetSpatialTransformerNdDescriptor`                   |                                                 | 7.5              |
|`cudnnDestroySpatialTransformerDescriptor`                 |                                                 | 7.5              |
|`cudnnSpatialTfGridGeneratorForward`                       |                                                 | 7.5              |
|`cudnnSpatialTfGridGeneratorBackward`                      |                                                 | 7.5              |
|`cudnnSpatialTfSamplerForward`                             |                                                 | 7.5              |
|`cudnnSpatialTfSamplerBackward`                            |                                                 | 7.5              |
|`cudnnCreateDropoutDescriptor`                             |`hipdnnCreateDropoutDescriptor`                  | 7.5              |
|`cudnnDestroyDropoutDescriptor`                            |`hipdnnDestroyDropoutDescriptor`                 | 7.5              |
|`cudnnDropoutGetStatesSize`                                |`hipdnnDropoutGetStatesSize`                     | 7.5              |
|`cudnnDropoutGetReserveSpaceSize`                          |                                                 | 7.5              |
|`cudnnSetDropoutDescriptor`                                |`hipdnnSetDropoutDescriptor`                     | 7.5              |
|`cudnnGetDropoutDescriptor`                                |                                                 | 8.0              |
|`cudnnRestoreDropoutDescriptor`                            |                                                 | 8.0              |
|`cudnnDropoutForward`                                      |                                                 | 7.5              |
|`cudnnDropoutBackward`                                     |                                                 | 7.5              |
|`cudnnCreateRNNDescriptor`                                 |`hipdnnCreateRNNDescriptor`                      | 7.5              |
|`cudnnDestroyRNNDescriptor`                                |`hipdnnDestroyRNNDescriptor`                     | 7.5              |
|`cudnnGetRNNForwardInferenceAlgorithmMaxCount`             |                                                 | 8.0              |
|`cudnnFindRNNForwardInferenceAlgorithmEx`                  |                                                 | 8.0              |
|`cudnnGetRNNForwardTrainingAlgorithmMaxCount`              |                                                 | 8.0              |
|`cudnnFindRNNForwardTrainingAlgorithmEx`                   |                                                 | 8.0              |
|`cudnnGetRNNBackwardDataAlgorithmMaxCount`                 |                                                 | 8.0              |
|`cudnnFindRNNBackwardDataAlgorithmEx`                      |                                                 | 8.0              |
|`cudnnGetRNNBackwardWeightsAlgorithmMaxCount`              |                                                 | 8.0              |
|`cudnnFindRNNBackwardWeightsAlgorithmEx`                   |                                                 | 8.0              |
|`cudnnCreatePersistentRNNPlan`                             |`hipdnnCreatePersistentRNNPlan`                  | 7.5              |
|`cudnnSetPersistentRNNPlan`                                |`hipdnnSetPersistentRNNPlan`                     | 7.5              |
|`cudnnDestroyPersistentRNNPlan`                            |`hipdnnDestroyPersistentRNNPlan`                 | 7.5              |
|`cudnnSetRNNDescriptor`                                    |`hipdnnSetRNNDescriptor`                         | 7.5              |
|`cudnnGetRNNDescriptor`                                    |`hipdnnGetRNNDescriptor`                         | 8.0              |
|`cudnnSetRNNProjectionLayers`                              |                                                 | 8.0              |
|`cudnnGetRNNProjectionLayers`                              |                                                 | 8.0              |
|`cudnnSetRNNAlgorithmDescriptor`                           |                                                 | 8.0              |
|`cudnnSetRNNMatrixMathType`                                |                                                 | 8.0              |
|`cudnnGetRNNMatrixMathType`                                |                                                 | 8.0              |
|`cudnnGetRNNWorkspaceSize`                                 |`hipdnnGetRNNWorkspaceSize`                      | 7.5              |
|`cudnnGetRNNTrainingReserveSize`                           |`hipdnnGetRNNTrainingReserveSize`                | 7.5              |
|`cudnnGetRNNParamsSize`                                    |`hipdnnGetRNNParamsSize`                         | 7.5              |
|`cudnnGetRNNLinLayerMatrixParams`                          |`hipdnnGetRNNLinLayerMatrixParams`               | 7.5              |
|`cudnnGetRNNLinLayerBiasParams`                            |`hipdnnGetRNNLinLayerBiasParams`                 | 7.5              |
|`cudnnRNNForwardInference`                                 |`hipdnnRNNForwardInference`                      | 7.5              |
|`cudnnRNNForwardInferenceEx`                               |                                                 | 9.0              |
|`cudnnRNNForwardTraining`                                  |`hipdnnRNNForwardTraining`                       | 7.5              |
|`cudnnRNNForwardTrainingEx`                                |                                                 | 9.0              |
|`cudnnRNNBackwardData`                                     |`hipdnnRNNBackwardData`                          | 7.5              |
|`cudnnRNNBackwardDataEx`                                   |                                                 | 9.0              |
|`cudnnRNNBackwardWeights`                                  |`hipdnnRNNBackwardWeights`                       | 7.5              |
|`cudnnRNNBackwardWeightsEx`                                |                                                 | 9.0              |
|`cudnnSetRNNPaddingMode`                                   |                                                 | 9.0              |
|`cudnnGetRNNPaddingMode`                                   |                                                 | 9.0              |
|`cudnnCreateRNNDataDescriptor`                             |                                                 | 9.0              |
|`cudnnDestroyRNNDataDescriptor`                            |                                                 | 9.0              |
|`cudnnSetRNNDataDescriptor`                                |                                                 | 9.0              |
|`cudnnGetRNNDataDescriptor`                                |                                                 | 9.0              |
|`cudnnSetRNNBiasMode`                                      |                                                 | 9.0              |
|`cudnnGetRNNBiasMode`                                      |                                                 | 9.0              |
|`cudnnCreateCTCLossDescriptor`                             |                                                 | 8.0              |
|`cudnnSetCTCLossDescriptor`                                |                                                 | 8.0              |
|`cudnnSetCTCLossDescriptorEx`                              |                                                 | 10.1             |
|`cudnnGetCTCLossDescriptor`                                |                                                 | 8.0              |
|`cudnnGetCTCLossDescriptorEx`                              |                                                 | 10.1             |
|`cudnnDestroyCTCLossDescriptor`                            |                                                 | 8.0              |
|`cudnnCTCLoss`                                             |                                                 | 8.0              |
|`cudnnGetCTCLossWorkspaceSize`                             |                                                 | 8.0              |
|`cudnnCreateAlgorithmDescriptor`                           |                                                 | 8.0              |
|`cudnnSetAlgorithmDescriptor`                              |                                                 | 8.0              |
|`cudnnGetAlgorithmDescriptor`                              |                                                 | 8.0              |
|`cudnnCopyAlgorithmDescriptor`                             |                                                 | 8.0              |
|`cudnnDestroyAlgorithmDescriptor`                          |                                                 | 8.0              |
|`cudnnCreateAlgorithmPerformance`                          |                                                 | 8.0              |
|`cudnnSetAlgorithmPerformance`                             |                                                 | 8.0              |
|`cudnnGetAlgorithmPerformance`                             |                                                 | 8.0              |
|`cudnnDestroyAlgorithmPerformance`                         |                                                 | 8.0              |
|`cudnnGetAlgorithmSpaceSize`                               |                                                 | 8.0              |
|`cudnnSaveAlgorithm`                                       |                                                 | 8.0              |
|`cudnnRestoreAlgorithm`                                    |                                                 | 8.0              |
|`cudnnSetRNNDescriptor_v5`                                 |`hipdnnSetRNNDescriptor_v5`                      | 8.0              |
|`cudnnSetRNNDescriptor_v6`                                 |`hipdnnSetRNNDescriptor_v6`                      | 7.5              |
|`cudnnSetCallback`                                         |                                                 | 8.0              |
|`cudnnGetCallback`                                         |                                                 | 8.0              |
|`cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize` |                                                 | 9.0              |
|`cudnnGetBatchNormalizationBackwardExWorkspaceSize`        |                                                 | 9.0              |
|`cudnnGetBatchNormalizationTrainingExReserveSpaceSize`     |                                                 | 9.0              |
|`cudnnRNNSetClip`                                          |                                                 | 9.0              |
|`cudnnRNNGetClip`                                          |                                                 | 9.0              |
|`cudnnCreateSeqDataDescriptor`                             |                                                 | 9.0              |
|`cudnnDestroySeqDataDescriptor`                            |                                                 | 9.0              |
|`cudnnSetSeqDataDescriptor`                                |                                                 | 9.0              |
|`cudnnGetSeqDataDescriptor`                                |                                                 | 9.0              |
|`cudnnCreateAttnDescriptor`                                |                                                 | 9.0              |
|`cudnnDestroyAttnDescriptor`                               |                                                 | 9.0              |
|`cudnnSetAttnDescriptor`                                   |                                                 | 9.0              |
|`cudnnGetAttnDescriptor`                                   |                                                 | 9.0              |
|`cudnnGetMultiHeadAttnBuffers`                             |                                                 | 9.0              |
|`cudnnGetMultiHeadAttnWeights`                             |                                                 | 9.0              |
|`cudnnMultiHeadAttnForward`                                |                                                 | 9.0              |
|`cudnnMultiHeadAttnBackwardData`                           |                                                 | 9.0              |
|`cudnnMultiHeadAttnBackwardWeights`                        |                                                 | 9.0              |
|`cudnnCreateFusedOpsConstParamPack`                        |                                                 | 10.1             |
|`cudnnDestroyFusedOpsConstParamPack`                       |                                                 | 10.1             |
|`cudnnSetFusedOpsConstParamPackAttribute`                  |                                                 | 10.1             |
|`cudnnGetFusedOpsConstParamPackAttribute`                  |                                                 | 10.1             |
|`cudnnCreateFusedOpsVariantParamPack`                      |                                                 | 10.1             |
|`cudnnDestroyFusedOpsVariantParamPack`                     |                                                 | 10.1             |
|`cudnnSetFusedOpsVariantParamPackAttribute`                |                                                 | 10.1             |
|`cudnnGetFusedOpsVariantParamPackAttribute`                |                                                 | 10.1             |
|`cudnnCreateFusedOpsPlan`                                  |                                                 | 10.1             |
|`cudnnDestroyFusedOpsPlan`                                 |                                                 | 10.1             |
|`cudnnMakeFusedOpsPlan`                                    |                                                 | 10.1             |
|`cudnnFusedOpsExecute`                                     |                                                 | 10.1             |

\* CUDA version, in which API has appeared and (optional) last version before abandoning it; no value in case of earlier versions < 7.5.
