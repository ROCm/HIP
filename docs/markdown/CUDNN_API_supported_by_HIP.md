# CUDNN API supported by HIP

## **1. CUDNN Data types**

| **type**     |   **CUDA**                                                    |   **HIP**                                                  |
|-------------:|---------------------------------------------------------------|------------------------------------------------------------|
| define       |`CUDNN_VERSION`                                                |`HIPDNN_VERSION`                                            |
| struct       |`cudnnContext`                                                 |                                                            |
| struct*      |`cudnnHandle_t`                                                |`hipdnnHandle_t`                                            |
| enum         |***`cudnnStatus_t`***                                          |***`hipdnnStatus_t`***                                      |
|            0 |*`CUDNN_STATUS_SUCCESS`*                                       |*`HIPDNN_STATUS_SUCCESS`*                                   |
|            1 |*`CUDNN_STATUS_NOT_INITIALIZED`*                               |*`HIPDNN_STATUS_NOT_INITIALIZED`*                           |
|            2 |*`CUDNN_STATUS_ALLOC_FAILED`*                                  |*`HIPDNN_STATUS_ALLOC_FAILED`*                              |
|            3 |*`CUDNN_STATUS_BAD_PARAM`*                                     |*`HIPDNN_STATUS_BAD_PARAM`*                                 |
|            4 |*`CUDNN_STATUS_INTERNAL_ERROR`*                                |*`HIPDNN_STATUS_INTERNAL_ERROR`*                            |
|            5 |*`CUDNN_STATUS_INVALID_VALUE`*                                 |*`HIPDNN_STATUS_INVALID_VALUE`*                             |
|            6 |*`CUDNN_STATUS_ARCH_MISMATCH`*                                 |*`HIPDNN_STATUS_ARCH_MISMATCH`*                             |
|            7 |*`CUDNN_STATUS_MAPPING_ERROR`*                                 |*`HIPDNN_STATUS_MAPPING_ERROR`*                             |
|            8 |*`CUDNN_STATUS_EXECUTION_FAILED`*                              |*`HIPDNN_STATUS_EXECUTION_FAILED`*                          |
|            9 |*`CUDNN_STATUS_NOT_SUPPORTED`*                                 |*`HIPDNN_STATUS_NOT_SUPPORTED`*                             |
|           10 |*`CUDNN_STATUS_LICENSE_ERROR`*                                 |*`HIPDNN_STATUS_LICENSE_ERROR`*                             |
|           11 |*`CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING`*                  |*`HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING`*              |
|           12 |*`CUDNN_STATUS_RUNTIME_IN_PROGRESS`*                           |                                                            |
|           13 |*`CUDNN_STATUS_RUNTIME_FP_OVERFLOW`*                           |                                                            |
| struct       |`cudnnRuntimeTag_t`                                            |                                                            |
| enum         |***`cudnnErrQueryMode_t`***                                    |                                                            |
|            0 |*`CUDNN_ERRQUERY_RAWCODE`*                                     |                                                            |
|            1 |*`CUDNN_ERRQUERY_NONBLOCKING`*                                 |                                                            |
|            2 |*`CUDNN_ERRQUERY_BLOCKING`*                                    |                                                            |
| enum         |***`libraryPropertyType_t`***                                  |                                                            |
| struct       |`cudnnTensorStruct`                                            |                                                            |
| struct*      |`cudnnTensorDescriptor_t`                                      |`hipdnnTensorDescriptor_t`                                  |
| struct       |`cudnnConvolutionStruct`                                       |                                                            |
| struct*      |`cudnnConvolutionDescriptor_t`                                 |`hipdnnConvolutionDescriptor_t`                             |
| struct       |`cudnnPoolingStruct`                                           |                                                            |
| struct*      |`cudnnPoolingDescriptor_t`                                     |`hipdnnPoolingDescriptor_t`                                 |
| struct       |`cudnnFilterStruct`                                            |                                                            |
| struct*      |`cudnnFilterDescriptor_t`                                      |`hipdnnFilterDescriptor_t`                                  |
| struct       |`cudnnLRNStruct`                                               |                                                            |
| struct*      |`cudnnLRNDescriptor_t`                                         |`hipdnnLRNDescriptor_t`                                     |
| struct       |`cudnnActivationStruct`                                        |                                                            |
| struct*      |`cudnnActivationDescriptor_t`                                  |`hipdnnActivationDescriptor_t`                              |
| struct       |`cudnnSpatialTransformerStruct`                                |                                                            |
| struct*      |`cudnnSpatialTransformerDescriptor_t`                          |                                                            |
| struct       |`cudnnOpTensorStruct`                                          |                                                            |
| struct*      |`cudnnOpTensorDescriptor_t`                                    |`hipdnnOpTensorDescriptor_t`                                |
| struct       |`cudnnReduceTensorStruct`                                      |                                                            |
| struct*      |`cudnnReduceTensorDescriptor_t`                                |`hipdnnReduceTensorDescriptor_t`                            |
| struct       |`cudnnCTCLossStruct`                                           |                                                            |
| struct*      |`cudnnCTCLossDescriptor_t`                                     |                                                            |
| enum         |***`cudnnDataType_t`***                                        |***`hipdnnDataType_t`***                                    |
|            0 |*`CUDNN_DATA_FLOAT`*                                           |*`HIPDNN_DATA_FLOAT`*                                       |
|            1 |*`CUDNN_DATA_DOUBLE`*                                          |*`HIPDNN_DATA_DOUBLE`*                                      |
|            2 |*`CUDNN_DATA_HALF`*                                            |*`HIPDNN_DATA_HALF`*                                        |
|            3 |*`CUDNN_DATA_INT8`*                                            |*`HIPDNN_DATA_INT8`*                                        |
|            4 |*`CUDNN_DATA_INT32`*                                           |*`HIPDNN_DATA_INT32`*                                       |
|            5 |*`CUDNN_DATA_INT8x4`*                                          |*`HIPDNN_DATA_INT8x4`*                                      |
|            6 |*`CUDNN_DATA_UINT8`*                                           |*`HIPDNN_DATA_UINT8`*                                       |
|            7 |*`CUDNN_DATA_UINT8x4`*                                         |*`HIPDNN_DATA_UINT8x4`*                                     |
| enum         |***`cudnnMathType_t`***                                        |***`hipdnnMathType_t`***                                    |
|            0 |*`CUDNN_DEFAULT_MATH`*                                         |*`HIPDNN_DEFAULT_MATH`*                                     |
|            1 |*`CUDNN_TENSOR_OP_MATH`*                                       |*`HIPDNN_TENSOR_OP_MATH`*                                   |
| enum         |***`cudnnNanPropagation_t`***                                  |***`hipdnnNanPropagation_t`***                              |
|            0 |*`CUDNN_NOT_PROPAGATE_NAN`*                                    |*`HIPDNN_NOT_PROPAGATE_NAN`*                                |
|            1 |*`CUDNN_PROPAGATE_NAN`*                                        |*`HIPDNN_PROPAGATE_NAN`*                                    |
| enum         |***`cudnnDeterminism_t`***                                     |                                                            |
|            0 |*`CUDNN_NON_DETERMINISTIC`*                                    |                                                            |
|            1 |*`CUDNN_DETERMINISTIC`*                                        |                                                            |
| define       |`CUDNN_DIM_MAX`                                                |                                                            |
| enum         |***`cudnnTensorFormat_t`***                                    |***`hipdnnTensorFormat_t`***                                |
|            0 |*`CUDNN_TENSOR_NCHW`*                                          |*`HIPDNN_TENSOR_NCHW`*                                      |
|            1 |*`CUDNN_TENSOR_NHWC`*                                          |*`HIPDNN_TENSOR_NHWC`*                                      |
|            2 |*`CUDNN_TENSOR_NCHW_VECT_C`*                                   |*`HIPDNN_TENSOR_NCHW_VECT_C`*                               |
| enum         |***`cudnnOpTensorOp_t`***                                      |***`hipdnnOpTensorOp_t`***                                  |
|            0 |*`CUDNN_OP_TENSOR_ADD`*                                        |*`HIPDNN_OP_TENSOR_ADD`*                                    |
|            1 |*`CUDNN_OP_TENSOR_MUL`*                                        |*`HIPDNN_OP_TENSOR_MUL`*                                    |
|            2 |*`CUDNN_OP_TENSOR_MIN`*                                        |*`HIPDNN_OP_TENSOR_MIN`*                                    |
|            3 |*`CUDNN_OP_TENSOR_MAX`*                                        |*`HIPDNN_OP_TENSOR_MAX`*                                    |
|            4 |*`CUDNN_OP_TENSOR_SQRT`*                                       |*`HIPDNN_OP_TENSOR_SQRT`*                                   |
|            5 |*`CUDNN_OP_TENSOR_NOT`*                                        |                                                            |
| enum         |***`cudnnReduceTensorOp_t`***                                  |***`hipdnnReduceTensorOp_t`***                              |
|            0 |*`CUDNN_REDUCE_TENSOR_ADD`*                                    |*`HIPDNN_REDUCE_TENSOR_ADD`*                                |
|            1 |*`CUDNN_REDUCE_TENSOR_MUL`*                                    |*`HIPDNN_REDUCE_TENSOR_MUL`*                                |
|            2 |*`CUDNN_REDUCE_TENSOR_MIN`*                                    |*`HIPDNN_REDUCE_TENSOR_MIN`*                                |
|            3 |*`CUDNN_REDUCE_TENSOR_MAX`*                                    |*`HIPDNN_REDUCE_TENSOR_MAX`*                                |
|            4 |*`CUDNN_REDUCE_TENSOR_AMAX`*                                   |*`HIPDNN_REDUCE_TENSOR_AMAX`*                               |
|            5 |*`CUDNN_REDUCE_TENSOR_AVG`*                                    |*`HIPDNN_REDUCE_TENSOR_AVG`*                                |
|            6 |*`CUDNN_REDUCE_TENSOR_NORM1`*                                  |*`HIPDNN_REDUCE_TENSOR_NORM1`*                              |
|            7 |*`CUDNN_REDUCE_TENSOR_NORM2`*                                  |*`HIPDNN_REDUCE_TENSOR_NORM2`*                              |
|            8 |*`CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS`*                           |*`HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS`*                       |
| enum         |***`cudnnReduceTensorIndices_t`***                             |***`hipdnnReduceTensorIndices_t`***                         |
|            0 |*`CUDNN_REDUCE_TENSOR_NO_INDICES`*                             |*`HIPDNN_REDUCE_TENSOR_NO_INDICES`*                         |
|            1 |*`CUDNN_REDUCE_TENSOR_FLATTENED_INDICES`*                      |*`HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES`*                  |
| enum         |***`cudnnIndicesType_t`***                                     |***`hipdnnIndicesType_t`***                                 |
|            0 |*`CUDNN_32BIT_INDICES`*                                        |*`HIPDNN_32BIT_INDICES`*                                    |
|            1 |*`CUDNN_64BIT_INDICES`*                                        |*`HIPDNN_64BIT_INDICES`*                                    |
|            2 |*`CUDNN_16BIT_INDICES`*                                        |*`HIPDNN_16BIT_INDICES`*                                    |
|            3 |*`CUDNN_8BIT_INDICES`*                                         |*`HIPDNN_8BIT_INDICES`*                                     |
| enum         |***`cudnnConvolutionMode_t`***                                 |***`hipdnnConvolutionMode_t`***                             |
|            0 |*`CUDNN_CONVOLUTION`*                                          |*`HIPDNN_CONVOLUTION`*                                      |
|            1 |*`CUDNN_CROSS_CORRELATION`*                                    |*`HIPDNN_CROSS_CORRELATION`*                                |
| enum         |***`cudnnConvolutionFwdPreference_t`***                        |***`hipdnnConvolutionFwdPreference_t`***                    |
|            0 |*`CUDNN_CONVOLUTION_FWD_NO_WORKSPACE`*                         |*`HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE`*                     |
|            1 |*`CUDNN_CONVOLUTION_FWD_PREFER_FASTEST`*                       |*`HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST`*                   |
|            2 |*`CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT`*              |*`HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT`*          |
| enum         |***`cudnnConvolutionFwdAlgo_t`***                              |***`hipdnnConvolutionFwdAlgo_t`***                          |
|            0 |*`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`*                   |*`HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`*               |
|            1 |*`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`*           |*`HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`*       |
|            2 |*`CUDNN_CONVOLUTION_FWD_ALGO_GEMM`*                            |*`HIPDNN_CONVOLUTION_FWD_ALGO_GEMM`*                        |
|            3 |*`CUDNN_CONVOLUTION_FWD_ALGO_DIRECT`*                          |*`HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT`*                      |
|            4 |*`CUDNN_CONVOLUTION_FWD_ALGO_FFT`*                             |*`HIPDNN_CONVOLUTION_FWD_ALGO_FFT`*                         |
|            5 |*`CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING`*                      |*`HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING`*                  |
|            6 |*`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD`*                        |*`HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD`*                    |
|            7 |*`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED`*               |*`HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED`*           |
|            8 |*`CUDNN_CONVOLUTION_FWD_ALGO_COUNT`*                           |*`HIPDNN_CONVOLUTION_FWD_ALGO_COUNT`*                       |
| struct       |`cudnnConvolutionFwdAlgoPerf_t`                                |`hipdnnConvolutionFwdAlgoPerf_t`                            |
| enum         |***`cudnnConvolutionBwdFilterPreference_t`***                  |***`hipdnnConvolutionBwdFilterPreference_t`***              |
|            0 |*`CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE`*                  |*`HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE`*              |
|            1 |*`CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST`*                |*`HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST`*            |
|            2 |*`CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT`*       |*`HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT`*   |
| enum         |***`cudnnConvolutionBwdFilterAlgo_t`***                        |***`hipdnnConvolutionBwdFilterAlgo_t`***                    |
|            0 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0`*                        |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0`*                    |
|            1 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1`*                        |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1`*                    |
|            2 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT`*                      |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT`*                  |
|            3 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3`*                        |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3`*                    |
|            4 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD`*                 |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD`*             |
|            5 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED`*        |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED`*    |
|            6 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING`*               |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING`*           |
|            7 |*`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT`*                    |*`HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT`*                |
| struct       |`cudnnConvolutionBwdDataAlgoPerf_t`                            |`hipdnnConvolutionBwdDataAlgoPerf_t`                        |
| enum         |***`cudnnSoftmaxAlgorithm_t`***                                |***`hipdnnSoftmaxAlgorithm_t`***                            |
|            0 |*`CUDNN_SOFTMAX_FAST`*                                         |*`HIPDNN_SOFTMAX_FAST`*                                     |
|            1 |*`CUDNN_SOFTMAX_ACCURATE`*                                     |*`HIPDNN_SOFTMAX_ACCURATE`*                                 |
|            2 |*`CUDNN_SOFTMAX_LOG`*                                          |*`HIPDNN_SOFTMAX_LOG`*                                      |
| enum         |***`cudnnSoftmaxMode_t`***                                     |***`hipdnnSoftmaxMode_t`***                                 |
|            0 |*`CUDNN_SOFTMAX_MODE_INSTANCE`*                                |*`HIPDNN_SOFTMAX_MODE_INSTANCE`*                            |
|            1 |*`CUDNN_SOFTMAX_MODE_CHANNEL`*                                 |*`HIPDNN_SOFTMAX_MODE_CHANNEL`*                             |
| enum         |***`cudnnPoolingMode_t`***                                     |***`hipdnnPoolingMode_t`***                                 |
|            0 |*`CUDNN_POOLING_MAX`*                                          |*`HIPDNN_POOLING_MAX`*                                      |
|            1 |*`CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING`*                |*`HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING`*            |
|            2 |*`CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING`*                |*`HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING`*            |
|            3 |*`CUDNN_POOLING_MAX_DETERMINISTIC`*                            |*`HIPDNN_POOLING_MAX_DETERMINISTIC`*                        |
| enum         |***`cudnnActivationMode_t`***                                  |***`hipdnnActivationMode_t`***                              |
|            0 |*`CUDNN_ACTIVATION_SIGMOID`*                                   |*`HIPDNN_ACTIVATION_SIGMOID`*                               |
|            1 |*`CUDNN_ACTIVATION_RELU`*                                      |*`HIPDNN_ACTIVATION_RELU`*                                  |
|            2 |*`CUDNN_ACTIVATION_TANH`*                                      |*`HIPDNN_ACTIVATION_TANH`*                                  |
|            3 |*`CUDNN_ACTIVATION_CLIPPED_RELU`*                              |*`HIPDNN_ACTIVATION_CLIPPED_RELU`*                          |
|            4 |*`CUDNN_ACTIVATION_ELU`*                                       |*`HIPDNN_ACTIVATION_ELU`*                                   |
|            5 |*`CUDNN_ACTIVATION_IDENTITY`*                                  |*`HIPDNN_ACTIVATION_PATHTRU`*                               |
| define       |`CUDNN_LRN_MIN_N`                                              |                                                            |
| define       |`CUDNN_LRN_MAX_N`                                              |                                                            |
| define       |`CUDNN_LRN_MIN_K`                                              |                                                            |
| define       |`CUDNN_LRN_MIN_BETA`                                           |                                                            |
| enum         |***`cudnnLRNMode_t`***                                         |***`hipdnnLRNMode_t`***                                     |
|            0 |*`CUDNN_LRN_CROSS_CHANNEL_DIM1`*                               |*`HIPDNN_LRN_CROSS_CHANNEL`*                                |
| enum         |***`cudnnDivNormMode_t`***                                     |                                                            |
|            0 |*`CUDNN_DIVNORM_PRECOMPUTED_MEANS`*                            |                                                            |
| enum         |***`cudnnBatchNormMode_t`***                                   |***`hipdnnBatchNormMode_t`***                               |
|            0 |*`CUDNN_BATCHNORM_PER_ACTIVATION`*                             |*`HIPDNN_BATCHNORM_PER_ACTIVATION`*                         |
|            1 |*`CUDNN_BATCHNORM_SPATIAL`*                                    |*`HIPDNN_BATCHNORM_SPATIAL`*                                |
|            2 |*`CUDNN_BATCHNORM_SPATIAL_PERSISTENT`*                         |*`HIPDNN_BATCHNORM_SPATIAL_PERSISTENT`*                     |
| define       |`CUDNN_BN_MIN_EPSILON`                                         |`HIPDNN_BN_MIN_EPSILON`                                     |
| enum         |***`cudnnSamplerType_t`***                                     |                                                            |
|            0 |*`CUDNN_SAMPLER_BILINEAR`*                                     |                                                            |
| struct       |`cudnnDropoutStruct`                                           |                                                            |
| struct*      |`cudnnDropoutDescriptor_t`                                     |`hipdnnDropoutDescriptor_t`                                 |
| enum         |***`cudnnRNNMode_t`***                                         |***`hipdnnRNNMode_t`***                                     |
|            0 |*`CUDNN_RNN_RELU`*                                             |*`HIPDNN_RNN_RELU`*                                         |
|            1 |*`CUDNN_RNN_TANH`*                                             |*`HIPDNN_RNN_TANH`*                                         |
|            2 |*`CUDNN_LSTM`*                                                 |*`HIPDNN_LSTM`*                                             |
|            3 |*`CUDNN_GRU`*                                                  |*`HIPDNN_GRU`*                                              |
| enum         |***`cudnnDirectionMode_t`***                                   |***`hipdnnDirectionMode_t`***                               |
|            0 |*`CUDNN_UNIDIRECTIONAL`*                                       |*`HIPDNN_UNIDIRECTIONAL`*                                   |
|            1 |*`CUDNN_BIDIRECTIONAL`*                                        |*`HIPDNN_BIDIRECTIONAL`*                                    |
| enum         |***`cudnnRNNAlgo_t`***                                         |***`hipdnnRNNAlgo_t`***                                     |
|            0 |*`CUDNN_RNN_ALGO_STANDARD`*                                    |*`HIPDNN_RNN_ALGO_STANDARD`*                                |
|            1 |*`CUDNN_RNN_ALGO_PERSIST_STATIC`*                              |*`HIPDNN_RNN_ALGO_PERSIST_STATIC`*                          |
|            2 |*`CUDNN_RNN_ALGO_PERSIST_DYNAMIC`*                             |*`HIPDNN_RNN_ALGO_PERSIST_DYNAMIC`*                         |
|            3 |*`CUDNN_RNN_ALGO_COUNT`*                                       |                                                            |
| struct       |`cudnnAlgorithmStruct`                                         |                                                            |
| struct*      |`cudnnAlgorithmDescriptor_t`                                   |                                                            |
| struct       |`cudnnAlgorithmPerformanceStruct`                              |                                                            |
| struct*      |`cudnnAlgorithmPerformance_t`                                  |                                                            |
| struct       |`cudnnRNNStruct`                                               |                                                            |
| struct*      |`cudnnRNNDescriptor_t`                                         |`hipdnnRNNDescriptor_t`                                     |
| struct       |`cudnnPersistentRNNPlan`                                       |                                                            |
| struct*      |`cudnnPersistentRNNPlan_t`                                     |`hipdnnPersistentRNNPlan_t`                                 |
| enum         |***`cudnnCTCLossAlgo_t`***                                     |                                                            |
|            0 |*`CUDNN_CTC_LOSS_ALGO_DETERMINISTIC`*                          |                                                            |
|            1 |*`CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC`*                      |                                                            |
| struct       |`cudnnAlgorithm_t`                                             |                                                            |
| enum         |***`cudnnSeverity_t`***                                        |                                                            |
|            0 |*`CUDNN_SEV_FATAL`*                                            |                                                            |
|            1 |*`CUDNN_SEV_ERROR`*                                            |                                                            |
|            2 |*`CUDNN_SEV_WARNING`*                                          |                                                            |
|            3 |*`CUDNN_SEV_INFO`*                                             |                                                            |
| define       |`CUDNN_SEV_ERROR_EN`                                           |                                                            |
| define       |`CUDNN_SEV_WARNING_EN`                                         |                                                            |
| define       |`CUDNN_SEV_INFO_EN`                                            |                                                            |
| struct       |`cudnnDebug_t`                                                 |                                                            |
| struct       |`cudnnCallback_t`                                              |                                                            |

## **2. CUDNN API functions**

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cudnnGetVersion`                                          |`hipdnnGetVersion`                               |
|`cudnnGetCudartVersion`                                    |                                                 |
|`cudnnGetErrorString`                                      |`hipdnnGetErrorString`                           |
|`cudnnQueryRuntimeError`                                   |                                                 |
|`cudnnGetProperty`                                         |                                                 |
|`cudnnCreate`                                              |`hipdnnCreate`                                   |
|`cudnnDestroy`                                             |`hipdnnDestroy`                                  |
|`cudnnSetStream`                                           |`hipdnnSetStream`                                |
|`cudnnSetStream`                                           |`hipdnnGetStream`                                |
|`cudnnCreateTensorDescriptor`                              |`hipdnnCreateTensorDescriptor`                   |
|`cudnnSetTensor4dDescriptor`                               |`hipdnnSetTensor4dDescriptor`                    |
|`cudnnSetTensor4dDescriptorEx`                             |                                                 |
|`cudnnGetTensor4dDescriptor`                               |`hipdnnGetTensor4dDescriptor`                    |
|`cudnnSetTensorNdDescriptor`                               |`hipdnnSetTensorNdDescriptor`                    |
|`cudnnSetTensorNdDescriptorEx`                             |                                                 |
|`cudnnGetTensorNdDescriptor`                               |`hipdnnGetTensorNdDescriptor`                    |
|`cudnnGetTensorSizeInBytes`                                |                                                 |
|`cudnnDestroyTensorDescriptor`                             |`hipdnnDestroyTensorDescriptor`                  |
|`cudnnTransformTensor`                                     |                                                 |
|`cudnnAddTensor`                                           |`hipdnnAddTensor`                                |
|`cudnnCreateOpTensorDescriptor`                            |`hipdnnCreateOpTensorDescriptor`                 |
|`cudnnSetOpTensorDescriptor`                               |`hipdnnSetOpTensorDescriptor`                    |
|`cudnnGetOpTensorDescriptor`                               |`hipdnnGetOpTensorDescriptor`                    |
|`cudnnDestroyOpTensorDescriptor`                           |`hipdnnDestroyOpTensorDescriptor`                |
|`cudnnOpTensor`                                            |`hipdnnOpTensor`                                 |
|`cudnnCreateReduceTensorDescriptor`                        |`hipdnnCreateReduceTensorDescriptor`             |
|`cudnnSetReduceTensorDescriptor`                           |`hipdnnSetReduceTensorDescriptor`                |
|`cudnnGetReduceTensorDescriptor`                           |`hipdnnGetReduceTensorDescriptor`                |
|`cudnnDestroyReduceTensorDescriptor`                       |`hipdnnDestroyReduceTensorDescriptor`            |
|`cudnnGetReductionIndicesSize`                             |                                                 |
|`cudnnGetReductionWorkspaceSize`                           |`hipdnnGetReductionWorkspaceSize`                |
|`cudnnReduceTensor`                                        |`hipdnnReduceTensor`                             |
|`cudnnSetTensor`                                           |`hipdnnSetTensor`                                |
|`cudnnScaleTensor`                                         |`hipdnnScaleTensor`                              |
|`cudnnCreateFilterDescriptor`                              |`hipdnnCreateFilterDescriptor`                   |
|`cudnnSetFilter4dDescriptor`                               |`hipdnnSetFilter4dDescriptor`                    |
|`cudnnGetFilter4dDescriptor`                               |`hipdnnGetFilter4dDescriptor`                    |
|`cudnnSetFilterNdDescriptor`                               |`hipdnnSetFilterNdDescriptor`                    |
|`cudnnGetFilterNdDescriptor`                               |`hipdnnGetFilterNdDescriptor`                    |
|`cudnnDestroyFilterDescriptor`                             |`hipdnnDestroyFilterDescriptor`                  |
|`cudnnCreateConvolutionDescriptor`                         |`hipdnnCreateConvolutionDescriptor`              |
|`cudnnSetConvolutionMathType`                              |`hipdnnSetConvolutionMathType`                   |
|`cudnnGetConvolutionMathType`                              |                                                 |
|`cudnnSetConvolutionGroupCount`                            |                                                 |
|`cudnnGetConvolutionGroupCount`                            |                                                 |
|`cudnnSetConvolution2dDescriptor`                          |`hipdnnSetConvolution2dDescriptor`               |
|`cudnnGetConvolution2dDescriptor`                          |`hipdnnGetConvolution2dDescriptor`               |
|`cudnnGetConvolution2dForwardOutputDim`                    |`hipdnnGetConvolution2dForwardOutputDim`         |
|`cudnnSetConvolutionNdDescriptor`                          |`hipdnnSetConvolutionNdDescriptor`               |
|`cudnnGetConvolutionNdDescriptor`                          |                                                 |
|`cudnnGetConvolutionNdForwardOutputDim`                    |                                                 |
|`cudnnDestroyConvolutionDescriptor`                        |                                                 |
|`cudnnGetConvolutionForwardAlgorithmMaxCount`              |                                                 |
|`cudnnFindConvolutionForwardAlgorithm`                     |`hipdnnFindConvolutionForwardAlgorithm`          |
|`cudnnFindConvolutionForwardAlgorithmEx`                   |`hipdnnFindConvolutionForwardAlgorithmEx`        |
|`cudnnGetConvolutionForwardAlgorithm`                      |`hipdnnGetConvolutionForwardAlgorithm`           |
|`cudnnGetConvolutionForwardAlgorithm_v7`                   |                                                 |
|`cudnnGetConvolutionForwardWorkspaceSize`                  |`hipdnnGetConvolutionForwardWorkspaceSize`       |
|`cudnnConvolutionForward`                                  |`hipdnnConvolutionForward`                       |
|`cudnnConvolutionBiasActivationForward`                    |                                                 |
|`cudnnConvolutionBackwardBias`                             |`hipdnnConvolutionBackwardBias`                  |
|`cudnnGetConvolutionBackwardFilterAlgorithmMaxCount`       |                                                 |
|`cudnnFindConvolutionBackwardFilterAlgorithm`              |`hipdnnFindConvolutionBackwardFilterAlgorithm`   |
|`cudnnFindConvolutionBackwardFilterAlgorithmEx`            |`hipdnnFindConvolutionBackwardFilterAlgorithmEx` |
|`cudnnGetConvolutionBackwardFilterAlgorithm`               |`hipdnnGetConvolutionBackwardFilterAlgorithm`    |
|`cudnnGetConvolutionBackwardFilterAlgorithm_v7`            |                                                 |
|`cudnnGetConvolutionBackwardFilterWorkspaceSize`           |`hipdnnGetConvolutionBackwardFilterWorkspaceSize`|
|`cudnnConvolutionBackwardFilter`                           |`hipdnnConvolutionBackwardFilter`                |
|`cudnnGetConvolutionBackwardDataAlgorithmMaxCount`         |                                                 |
|`cudnnFindConvolutionBackwardDataAlgorithm`                |`hipdnnFindConvolutionBackwardDataAlgorithm`     |
|`cudnnFindConvolutionBackwardDataAlgorithmEx`              |`hipdnnFindConvolutionBackwardDataAlgorithmEx`   |
|`cudnnGetConvolutionBackwardDataAlgorithm`                 |`hipdnnGetConvolutionBackwardDataAlgorithm`      |
|`cudnnGetConvolutionBackwardDataAlgorithm_v7`              |                                                 |
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
|`cudnnBatchNormalizationForwardInference`                  |`hipdnnBatchNormalizationForwardInference`       |
|`cudnnBatchNormalizationBackward`                          |`hipdnnBatchNormalizationBackward`               |
|`cudnnCreateSpatialTransformerDescriptor`                  |                                                 |
|`cudnnSetSpatialTransformerNdDescriptor`                   |                                                 |
|`cudnnDestroySpatialTransformerDescriptor`                 |                                                 |
|`cudnnSpatialTfGridGeneratorForward`                       |                                                 |
|`cudnnSpatialTfGridGeneratorBackward`                      |                                                 |
|`cudnnSpatialTfSamplerForward`                             |                                                 |
|`cudnnSpatialTfSamplerBackward`                            |                                                 |
|`cudnnCreateDropoutDescriptor`                             |`hipdnnCreateDropoutDescriptor`                  |
|`cudnnDestroyDropoutDescriptor`                            |`hipdnnDestroyDropoutDescriptor`                 |
|`cudnnDropoutGetStatesSize`                                |`hipdnnDropoutGetStatesSize`                     |
|`cudnnDropoutGetReserveSpaceSize`                          |                                                 |
|`cudnnSetDropoutDescriptor`                                |`hipdnnSetDropoutDescriptor`                     |
|`cudnnGetDropoutDescriptor`                                |                                                 |
|`cudnnRestoreDropoutDescriptor`                            |                                                 |
|`cudnnDropoutForward`                                      |                                                 |
|`cudnnDropoutBackward`                                     |                                                 |
|`cudnnCreateRNNDescriptor`                                 |`hipdnnCreateRNNDescriptor`                      |
|`cudnnDestroyRNNDescriptor`                                |`hipdnnDestroyRNNDescriptor`                     |
|`cudnnGetRNNForwardInferenceAlgorithmMaxCount`             |                                                 |
|`cudnnFindRNNForwardInferenceAlgorithmEx`                  |                                                 |
|`cudnnGetRNNForwardTrainingAlgorithmMaxCount`              |                                                 |
|`cudnnFindRNNForwardTrainingAlgorithmEx`                   |                                                 |
|`cudnnGetRNNBackwardDataAlgorithmMaxCount`                 |                                                 |
|`cudnnFindRNNBackwardDataAlgorithmEx`                      |                                                 |
|`cudnnGetRNNBackwardWeightsAlgorithmMaxCount`              |                                                 |
|`cudnnFindRNNBackwardWeightsAlgorithmEx`                   |                                                 |
|`cudnnCreatePersistentRNNPlan`                             |`hipdnnCreatePersistentRNNPlan`                  |
|`cudnnSetPersistentRNNPlan`                                |`hipdnnSetPersistentRNNPlan`                     |
|`cudnnDestroyPersistentRNNPlan`                            |`hipdnnDestroyPersistentRNNPlan`                 |
|`cudnnSetRNNDescriptor`                                    |`hipdnnSetRNNDescriptor`                         |
|`cudnnGetRNNDescriptor`                                    |                                                 |
|`cudnnSetRNNProjectionLayers`                              |                                                 |
|`cudnnGetRNNProjectionLayers`                              |                                                 |
|`cudnnSetRNNAlgorithmDescriptor`                           |                                                 |
|`cudnnSetRNNMatrixMathType`                                |                                                 |
|`cudnnGetRNNMatrixMathType`                                |                                                 |
|`cudnnGetRNNWorkspaceSize`                                 |`hipdnnGetRNNWorkspaceSize`                      |
|`cudnnGetRNNTrainingReserveSize`                           |`hipdnnGetRNNTrainingReserveSize`                |
|`cudnnGetRNNParamsSize`                                    |`hipdnnGetRNNParamsSize`                         |
|`cudnnGetRNNLinLayerMatrixParams`                          |`hipdnnGetRNNLinLayerMatrixParams`               |
|`cudnnGetRNNLinLayerBiasParams`                            |`hipdnnGetRNNLinLayerBiasParams`                 |
|`cudnnRNNForwardInference`                                 |`hipdnnRNNForwardInference`                      |
|`cudnnRNNForwardTraining`                                  |`hipdnnRNNForwardTraining`                       |
|`cudnnRNNBackwardData`                                     |`hipdnnRNNBackwardData`                          |
|`cudnnRNNBackwardWeights`                                  |`hipdnnRNNBackwardWeights`                       |
|`cudnnCreateCTCLossDescriptor`                             |                                                 |
|`cudnnSetCTCLossDescriptor`                                |                                                 |
|`cudnnGetCTCLossDescriptor`                                |                                                 |
|`cudnnDestroyCTCLossDescriptor`                            |                                                 |
|`cudnnCTCLoss`                                             |                                                 |
|`cudnnGetCTCLossWorkspaceSize`                             |                                                 |
|`cudnnCreateAlgorithmDescriptor`                           |                                                 |
|`cudnnSetAlgorithmDescriptor`                              |                                                 |
|`cudnnGetAlgorithmDescriptor`                              |                                                 |
|`cudnnCopyAlgorithmDescriptor`                             |                                                 |
|`cudnnDestroyAlgorithmDescriptor`                          |                                                 |
|`cudnnCreateAlgorithmPerformance`                          |                                                 |
|`cudnnSetAlgorithmPerformance`                             |                                                 |
|`cudnnGetAlgorithmPerformance`                             |                                                 |
|`cudnnDestroyAlgorithmPerformance`                         |                                                 |
|`cudnnGetAlgorithmSpaceSize`                               |                                                 |
|`cudnnSaveAlgorithm`                                       |                                                 |
|`cudnnRestoreAlgorithm`                                    |                                                 |
|`cudnnSetRNNDescriptor_v5`                                 |`hipdnnSetRNNDescriptor_v5`                      |
|`cudnnSetRNNDescriptor_v6`                                 |`hipdnnSetRNNDescriptor_v6`                      |
|`cudnnSetCallback`                                         |                                                 |
|`cudnnGetCallback`                                         |                                                 |
