# CUDNN API supported by HIP

## **1. CUNN Data types**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`CUDNN_16BIT_INDICES`| 6.0.0 |  |  | `HIPDNN_16BIT_INDICES` |  |  |  | 
|`CUDNN_32BIT_INDICES`| 6.0.0 |  |  | `HIPDNN_32BIT_INDICES` |  |  |  | 
|`CUDNN_64BIT_INDICES`| 6.0.0 |  |  | `HIPDNN_64BIT_INDICES` |  |  |  | 
|`CUDNN_8BIT_INDICES`| 6.0.0 |  |  | `HIPDNN_8BIT_INDICES` |  |  |  | 
|`CUDNN_ACTIVATION_CLIPPED_RELU`| 4.0.0 |  |  | `HIPDNN_ACTIVATION_CLIPPED_RELU` |  |  |  | 
|`CUDNN_ACTIVATION_ELU`| 6.0.0 |  |  | `HIPDNN_ACTIVATION_ELU` |  |  |  | 
|`CUDNN_ACTIVATION_IDENTITY`| 7.1.3 |  |  | `HIPDNN_ACTIVATION_PATHTRU` |  |  |  | 
|`CUDNN_ACTIVATION_RELU`| 1.0.0 |  |  | `HIPDNN_ACTIVATION_RELU` |  |  |  | 
|`CUDNN_ACTIVATION_SIGMOID`| 1.0.0 |  |  | `HIPDNN_ACTIVATION_SIGMOID` |  |  |  | 
|`CUDNN_ACTIVATION_TANH`| 1.0.0 |  |  | `HIPDNN_ACTIVATION_TANH` |  |  |  | 
|`CUDNN_ADV_INFER_MAJOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ADV_INFER_MINOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ADV_INFER_PATCH`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ADV_TRAIN_MAJOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ADV_TRAIN_MINOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ADV_TRAIN_PATCH`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTN_DISABLE_PROJ_BIASES`| 7.6.3 |  |  |  |  |  |  | 
|`CUDNN_ATTN_ENABLE_PROJ_BIASES`| 7.6.3 |  |  |  |  |  |  | 
|`CUDNN_ATTN_QUERYMAP_ALL_TO_ONE`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_ATTN_QUERYMAP_ONE_TO_ONE`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_ATTN_WKIND_COUNT`| 7.6.3 |  |  |  |  |  |  | 
|`CUDNN_ATTR_CONVOLUTION_COMP_TYPE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_CONVOLUTION_CONV_MODE`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_ATTR_CONVOLUTION_DILATIONS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_CONVOLUTION_POST_PADDINGS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_ENGINECFG_ENGINE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_ENGINECFG_KNOB_CHOICES`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_ENGINEHEUR_MODE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_ENGINEHEUR_RESULTS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_ENGINE_GLOBAL_INDEX`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_ENGINE_KNOB_INFO`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_ATTR_ENGINE_LAYOUT_INFO`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_ATTR_ENGINE_NUMERICAL_NOTE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_ENGINE_OPERATION_GRAPH`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_EXECUTION_PLAN_HANDLE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_ATTR_INTERMEDIATE_INFO_SIZE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_KNOB_INFO_STRIDE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_KNOB_INFO_TYPE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_ATTR_LAYOUT_INFO_TYPES`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_ATTR_MATMUL_COMP_TYPE`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATIONGRAPH_HANDLE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATIONGRAPH_OPS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_GENSTATS_MODE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_GENSTATS_XDESC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_MATMUL_ADESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_MATMUL_BDESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_MATMUL_CDESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_MATMUL_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_POINTWISE_BDESC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_POINTWISE_DXDESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_POINTWISE_DYDESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_POINTWISE_XDESC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_POINTWISE_YDESC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_REDUCTION_DESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_REDUCTION_XDESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_OPERATION_REDUCTION_YDESC`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_POINTWISE_ELU_ALPHA`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_POINTWISE_MATH_PREC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_POINTWISE_MODE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_POINTWISE_NAN_PROPAGATION`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_POINTWISE_SWISH_BETA`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_REDUCTION_COMP_TYPE`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_REDUCTION_OPERATOR`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_TENSOR_DATA_TYPE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_TENSOR_DIMENSIONS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_TENSOR_IS_BY_VALUE`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_ATTR_TENSOR_IS_VIRTUAL`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_TENSOR_STRIDES`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_TENSOR_UNIQUE_ID`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_TENSOR_VECTOR_COUNT`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_ATTR_VARIANT_PACK_WORKSPACE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_ENGINECFG_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_ENGINE_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_MATMUL_DESCRIPTOR`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_POINTWISE_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_REDUCTION_DESCRIPTOR`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_TENSOR_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_BATCHNORM_OPS_BN`| 7.4.1 |  |  |  |  |  |  | 
|`CUDNN_BATCHNORM_OPS_BN_ACTIVATION`| 7.4.1 |  |  |  |  |  |  | 
|`CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION`| 7.4.1 |  |  |  |  |  |  | 
|`CUDNN_BATCHNORM_PER_ACTIVATION`| 4.0.0 |  |  | `HIPDNN_BATCHNORM_PER_ACTIVATION` |  |  |  | 
|`CUDNN_BATCHNORM_SPATIAL`| 4.0.0 |  |  | `HIPDNN_BATCHNORM_SPATIAL` |  |  |  | 
|`CUDNN_BATCHNORM_SPATIAL_PERSISTENT`| 7.0.5 |  |  | `HIPDNN_BATCHNORM_SPATIAL_PERSISTENT` |  |  |  | 
|`CUDNN_BIDIRECTIONAL`| 5.0.0 |  |  | `HIPDNN_BIDIRECTIONAL` |  |  |  | 
|`CUDNN_BN_FINALIZE_STATISTICS_INFERENCE`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_BN_FINALIZE_STATISTICS_TRAINING`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_BN_MIN_EPSILON`| 4.0.0 |  |  | `HIPDNN_BN_MIN_EPSILON` |  |  |  | 
|`CUDNN_CNN_INFER_MAJOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_CNN_INFER_MINOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_CNN_INFER_PATCH`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_CNN_TRAIN_MAJOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_CNN_TRAIN_MINOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_CNN_TRAIN_PATCH`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_CONVOLUTION`| 1.0.0 |  |  | `HIPDNN_CONVOLUTION` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_DATA_ALGO_0`| 3.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_DATA_ALGO_1`| 3.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT`| 6.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT`| 3.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING`| 4.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD`| 5.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED`| 5.1.0 |  |  | `HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE`| 3.0.0 | 7.6.5 | 8.0.1 | `HIPDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST`| 3.0.0 | 7.6.5 | 8.0.1 | `HIPDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT`| 3.0.0 | 7.6.5 | 8.0.1 | `HIPDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0`| 3.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1`| 3.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3`| 3.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT`| 6.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT`| 3.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING`| 6.0.0 |  |  | `HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD`| 5.1.0 |  |  | `HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED`| 5.1.0 |  |  | `HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE`| 3.0.0 | 7.6.5 | 8.0.1 | `HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST`| 3.0.0 | 7.6.5 | 8.0.1 | `HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST` |  |  |  | 
|`CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT`| 3.0.0 | 7.6.5 | 8.0.1 | `HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_ALGO_COUNT`| 6.0.0 |  |  | `HIPDNN_CONVOLUTION_FWD_ALGO_COUNT` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_ALGO_DIRECT`| 2.0.0 |  |  | `HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_ALGO_FFT`| 3.0.0 |  |  | `HIPDNN_CONVOLUTION_FWD_ALGO_FFT` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING`| 4.0.0 |  |  | `HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_ALGO_GEMM`| 2.0.0 |  |  | `HIPDNN_CONVOLUTION_FWD_ALGO_GEMM` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`| 2.0.0 |  |  | `HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`| 2.0.0 |  |  | `HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD`| 5.0.0 |  |  | `HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED`| 5.1.0 |  |  | `HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_NO_WORKSPACE`| 2.0.0 | 7.6.5 | 8.0.1 | `HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_PREFER_FASTEST`| 2.0.0 | 7.6.5 | 8.0.1 | `HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST` |  |  |  | 
|`CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT`| 2.0.0 | 7.6.5 | 8.0.1 | `HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT` |  |  |  | 
|`CUDNN_CROSS_CORRELATION`| 1.0.0 |  |  | `HIPDNN_CROSS_CORRELATION` |  |  |  | 
|`CUDNN_CTC_LOSS_ALGO_DETERMINISTIC`| 7.0.5 |  |  |  |  |  |  | 
|`CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC`| 7.0.5 |  |  |  |  |  |  | 
|`CUDNN_DATA_BFLOAT16`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_DATA_DOUBLE`| 1.0.0 |  |  | `HIPDNN_DATA_DOUBLE` |  |  |  | 
|`CUDNN_DATA_FLOAT`| 1.0.0 |  |  | `HIPDNN_DATA_FLOAT` |  |  |  | 
|`CUDNN_DATA_HALF`| 3.0.0 |  |  | `HIPDNN_DATA_HALF` |  |  |  | 
|`CUDNN_DATA_INT32`| 6.0.0 |  |  | `HIPDNN_DATA_INT32` |  |  |  | 
|`CUDNN_DATA_INT64`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_DATA_INT8`| 6.0.0 |  |  | `HIPDNN_DATA_INT8` |  |  |  | 
|`CUDNN_DATA_INT8x32`| 7.2.1 |  |  |  |  |  |  | 
|`CUDNN_DATA_INT8x4`| 6.0.0 |  |  | `HIPDNN_DATA_INT8x4` |  |  |  | 
|`CUDNN_DATA_UINT8`| 7.1.3 |  |  |  |  |  |  | 
|`CUDNN_DATA_UINT8x4`| 7.1.3 |  |  |  |  |  |  | 
|`CUDNN_DEFAULT_MATH`| 7.0.5 |  |  | `HIPDNN_DEFAULT_MATH` |  |  |  | 
|`CUDNN_DEFAULT_REORDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_DETERMINISTIC`| 6.0.0 |  |  |  |  |  |  | 
|`CUDNN_DIM_MAX`| 4.0.0 |  |  |  |  |  |  | 
|`CUDNN_DIVNORM_PRECOMPUTED_MEANS`| 3.0.0 |  |  |  |  |  |  | 
|`CUDNN_ERRQUERY_BLOCKING`| 7.0.5 |  |  |  |  |  |  | 
|`CUDNN_ERRQUERY_NONBLOCKING`| 7.0.5 |  |  |  |  |  |  | 
|`CUDNN_ERRQUERY_RAWCODE`| 7.0.5 |  |  |  |  |  |  | 
|`CUDNN_FMA_MATH`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_FWD_MODE_INFERENCE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_FWD_MODE_TRAINING`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_GENSTATS_SUM_SQSUM`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_GRU`| 5.0.0 |  |  | `HIPDNN_GRU` |  |  |  | 
|`CUDNN_HEUR_MODES_COUNT`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_HEUR_MODE_B`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_HEUR_MODE_INSTANT`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_CHUNK_K`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_COUNTS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_EDGE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_IDX_MODE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_KBLOCK`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_KERNEL_CFG`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_LDGA`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_LDGB`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_LDGC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_MULTIPLY`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_REDUCTION_MODE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_SINGLEBUFFER`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_SLICED`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_SPECFILT`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_SPLIT_H`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_SPLIT_K`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_SPLIT_K_BUF`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_SPLIT_K_SLC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_SPLIT_RS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_STAGES`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_SWIZZLE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_TILEK`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_TILE_SIZE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_USE_TEX`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_KNOB_TYPE_WINO_TILE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_LAYOUT_TYPE_COUNT`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_LAYOUT_TYPE_PREFERRED_NCHW`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_LAYOUT_TYPE_PREFERRED_NHWC`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_LINEAR_INPUT`| 5.0.0 |  |  | `HIPDNN_LINEAR_INPUT` |  |  |  | 
|`CUDNN_LOSS_NORMALIZATION_NONE`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_LOSS_NORMALIZATION_SOFTMAX`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_LRN_CROSS_CHANNEL_DIM1`| 3.0.0 |  |  | `HIPDNN_LRN_CROSS_CHANNEL` |  |  |  | 
|`CUDNN_LRN_MAX_N`| 3.0.0 |  |  |  |  |  |  | 
|`CUDNN_LRN_MIN_BETA`| 3.0.0 |  |  |  |  |  |  | 
|`CUDNN_LRN_MIN_K`| 3.0.0 |  |  |  |  |  |  | 
|`CUDNN_LRN_MIN_N`| 3.0.0 |  |  |  |  |  |  | 
|`CUDNN_LSTM`| 5.0.0 |  |  | `HIPDNN_LSTM` |  |  |  | 
|`CUDNN_MAJOR`| 3.0.0 |  |  |  |  |  |  | 
|`CUDNN_MH_ATTN_K_BIASES`| 7.6.3 |  |  |  |  |  |  | 
|`CUDNN_MH_ATTN_K_WEIGHTS`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_MH_ATTN_O_BIASES`| 7.6.3 |  |  |  |  |  |  | 
|`CUDNN_MH_ATTN_O_WEIGHTS`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_MH_ATTN_Q_BIASES`| 7.6.3 |  |  |  |  |  |  | 
|`CUDNN_MH_ATTN_Q_WEIGHTS`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_MH_ATTN_V_BIASES`| 7.6.3 |  |  |  |  |  |  | 
|`CUDNN_MH_ATTN_V_WEIGHTS`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_MINOR`| 3.0.0 |  |  |  |  |  |  | 
|`CUDNN_NON_DETERMINISTIC`| 6.0.0 |  |  |  |  |  |  | 
|`CUDNN_NORM_ALGO_PERSIST`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NORM_ALGO_STANDARD`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NORM_OPS_NORM`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NORM_OPS_NORM_ACTIVATION`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NORM_OPS_NORM_ADD_ACTIVATION`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NORM_PER_ACTIVATION`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NORM_PER_CHANNEL`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NOT_PROPAGATE_NAN`| 4.0.0 |  |  | `HIPDNN_NOT_PROPAGATE_NAN` |  |  |  | 
|`CUDNN_NO_REORDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NUMERICAL_NOTE_FFT`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NUMERICAL_NOTE_TENSOR_CORE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NUMERICAL_NOTE_TYPE_COUNT`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_NUMERICAL_NOTE_WINOGRAD`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_OPS_INFER_MAJOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_OPS_INFER_MINOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_OPS_INFER_PATCH`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_OPS_TRAIN_MAJOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_OPS_TRAIN_MINOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_OPS_TRAIN_PATCH`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_OP_TENSOR_ADD`| 5.0.0 |  |  | `HIPDNN_OP_TENSOR_ADD` |  |  |  | 
|`CUDNN_OP_TENSOR_MAX`| 5.0.0 |  |  | `HIPDNN_OP_TENSOR_MAX` |  |  |  | 
|`CUDNN_OP_TENSOR_MIN`| 5.0.0 |  |  | `HIPDNN_OP_TENSOR_MIN` |  |  |  | 
|`CUDNN_OP_TENSOR_MUL`| 5.0.0 |  |  | `HIPDNN_OP_TENSOR_MUL` |  |  |  | 
|`CUDNN_OP_TENSOR_NOT`| 7.0.5 |  |  |  |  |  |  | 
|`CUDNN_OP_TENSOR_SQRT`| 6.0.0 |  |  | `HIPDNN_OP_TENSOR_SQRT` |  |  |  | 
|`CUDNN_PARAM_ACTIVATION_BITMASK_DESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_ACTIVATION_DESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_BIAS_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_DBIAS_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_DSCALE_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_EQSCALEBIAS_DESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_MODE`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_SCALE_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_CONV_DESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_DWDATA_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_DWDESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_DXDATA_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_DXDESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_DYDATA_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_DYDESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_DZDATA_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_DZDESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_WDATA_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_WDESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_XDATA_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_XDESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_YDATA_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_YDESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_YSQSUM_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_YSTATS_DESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_YSUM_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_ZDATA_PLACEHOLDER`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PARAM_ZDESC`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PATCHLEVEL`| 3.0.0 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_ADD`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_ELU_BWD`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_ELU_FWD`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_GELU_BWD`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_GELU_FWD`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_MAX`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_MIN`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_MUL`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_RELU_BWD`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_RELU_FWD`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_SIGMOID_BWD`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_SIGMOID_FWD`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_SOFTPLUS_BWD`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_SOFTPLUS_FWD`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_SQRT`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_SWISH_BWD`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_SWISH_FWD`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_TANH_BWD`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_POINTWISE_TANH_FWD`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING`| 2.0.0 |  |  | `HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING` |  |  |  | 
|`CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING`| 2.0.0 |  |  | `HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING` |  |  |  | 
|`CUDNN_POOLING_MAX`| 1.0.0 |  |  | `HIPDNN_POOLING_MAX` |  |  |  | 
|`CUDNN_POOLING_MAX_DETERMINISTIC`| 6.0.0 |  |  | `HIPDNN_POOLING_MAX_DETERMINISTIC` |  |  |  | 
|`CUDNN_PROPAGATE_NAN`| 4.0.0 |  |  | `HIPDNN_PROPAGATE_NAN` |  |  |  | 
|`CUDNN_PTR_16B_ALIGNED`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_ACTIVATION_BITMASK`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_BIAS`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_DBIAS`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_DSCALE`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_EQBIAS`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_EQSCALE`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_RUNNING_MEAN`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_RUNNING_VAR`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_SAVED_INVSTD`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_SAVED_MEAN`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_SCALE`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_Z_EQBIAS`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_BN_Z_EQSCALE`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_DWDATA`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_DXDATA`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_DYDATA`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_DZDATA`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_ELEM_ALIGNED`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_NULL`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_WDATA`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_WORKSPACE`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_XDATA`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_YDATA`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_YSQSUM`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_YSUM`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_PTR_ZDATA`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_REDUCE_TENSOR_ADD`| 6.0.0 |  |  | `HIPDNN_REDUCE_TENSOR_ADD` |  |  |  | 
|`CUDNN_REDUCE_TENSOR_AMAX`| 6.0.0 |  |  | `HIPDNN_REDUCE_TENSOR_AMAX` |  |  |  | 
|`CUDNN_REDUCE_TENSOR_AVG`| 6.0.0 |  |  | `HIPDNN_REDUCE_TENSOR_AVG` |  |  |  | 
|`CUDNN_REDUCE_TENSOR_FLATTENED_INDICES`| 6.0.0 |  |  | `HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES` |  |  |  | 
|`CUDNN_REDUCE_TENSOR_MAX`| 6.0.0 |  |  | `HIPDNN_REDUCE_TENSOR_MAX` |  |  |  | 
|`CUDNN_REDUCE_TENSOR_MIN`| 6.0.0 |  |  | `HIPDNN_REDUCE_TENSOR_MIN` |  |  |  | 
|`CUDNN_REDUCE_TENSOR_MUL`| 6.0.0 |  |  | `HIPDNN_REDUCE_TENSOR_MUL` |  |  |  | 
|`CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS`| 7.0.5 |  |  | `HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS` |  |  |  | 
|`CUDNN_REDUCE_TENSOR_NORM1`| 6.0.0 |  |  | `HIPDNN_REDUCE_TENSOR_NORM1` |  |  |  | 
|`CUDNN_REDUCE_TENSOR_NORM2`| 6.0.0 |  |  | `HIPDNN_REDUCE_TENSOR_NORM2` |  |  |  | 
|`CUDNN_REDUCE_TENSOR_NO_INDICES`| 6.0.0 |  |  | `HIPDNN_REDUCE_TENSOR_NO_INDICES` |  |  |  | 
|`CUDNN_RNN_ALGO_COUNT`| 7.1.3 |  |  |  |  |  |  | 
|`CUDNN_RNN_ALGO_PERSIST_DYNAMIC`| 6.0.0 |  |  | `HIPDNN_RNN_ALGO_PERSIST_DYNAMIC` |  |  |  | 
|`CUDNN_RNN_ALGO_PERSIST_STATIC`| 6.0.0 |  |  | `HIPDNN_RNN_ALGO_PERSIST_STATIC` |  |  |  | 
|`CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H`| 8.1.0 |  |  | `HIPDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H` |  |  |  | 
|`CUDNN_RNN_ALGO_STANDARD`| 6.0.0 |  |  | `HIPDNN_RNN_ALGO_STANDARD` |  |  |  | 
|`CUDNN_RNN_CLIP_MINMAX`| 7.2.1 |  |  |  |  |  |  | 
|`CUDNN_RNN_CLIP_NONE`| 7.2.1 |  |  |  |  |  |  | 
|`CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED`| 7.2.1 |  |  |  |  |  |  | 
|`CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED`| 7.2.1 |  |  |  |  |  |  | 
|`CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED`| 7.2.1 |  |  |  |  |  |  | 
|`CUDNN_RNN_DOUBLE_BIAS`| 7.5.0 |  |  | `HIPDNN_RNN_WITH_BIAS` |  |  |  | 
|`CUDNN_RNN_NO_BIAS`| 7.5.0 |  |  | `HIPDNN_RNN_NO_BIAS` |  |  |  | 
|`CUDNN_RNN_PADDED_IO_DISABLED`| 7.2.1 |  |  |  |  |  |  | 
|`CUDNN_RNN_PADDED_IO_ENABLED`| 7.2.1 |  |  |  |  |  |  | 
|`CUDNN_RNN_RELU`| 5.0.0 |  |  | `HIPDNN_RNN_RELU` |  |  |  | 
|`CUDNN_RNN_SINGLE_INP_BIAS`| 7.5.0 |  |  | `HIPDNN_RNN_WITH_BIAS` |  |  |  | 
|`CUDNN_RNN_SINGLE_REC_BIAS`| 7.5.0 |  |  | `HIPDNN_RNN_WITH_BIAS` |  |  |  | 
|`CUDNN_RNN_TANH`| 5.0.0 |  |  | `HIPDNN_RNN_TANH` |  |  |  | 
|`CUDNN_SAMPLER_BILINEAR`| 5.0.0 |  |  |  |  |  |  | 
|`CUDNN_SCALAR_DOUBLE_BN_EPSILON`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES`| 7.6.0 |  |  |  |  |  |  | 
|`CUDNN_SEQDATA_BATCH_DIM`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_SEQDATA_BEAM_DIM`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_SEQDATA_DIM_COUNT`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_SEQDATA_TIME_DIM`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_SEQDATA_VECT_DIM`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_SEV_ERROR`| 7.1.3 |  |  |  |  |  |  | 
|`CUDNN_SEV_ERROR_EN`| 7.1.3 |  |  |  |  |  |  | 
|`CUDNN_SEV_FATAL`| 7.1.3 |  |  |  |  |  |  | 
|`CUDNN_SEV_INFO`| 7.1.3 |  |  |  |  |  |  | 
|`CUDNN_SEV_INFO_EN`| 7.1.3 |  |  |  |  |  |  | 
|`CUDNN_SEV_WARNING`| 7.1.3 |  |  |  |  |  |  | 
|`CUDNN_SEV_WARNING_EN`| 7.1.3 |  |  |  |  |  |  | 
|`CUDNN_SKIP_INPUT`| 5.0.0 |  |  | `HIPDNN_SKIP_INPUT` |  |  |  | 
|`CUDNN_SOFTMAX_ACCURATE`| 1.0.0 |  |  | `HIPDNN_SOFTMAX_ACCURATE` |  |  |  | 
|`CUDNN_SOFTMAX_FAST`| 1.0.0 |  |  | `HIPDNN_SOFTMAX_FAST` |  |  |  | 
|`CUDNN_SOFTMAX_LOG`| 3.0.0 |  |  | `HIPDNN_SOFTMAX_LOG` |  |  |  | 
|`CUDNN_SOFTMAX_MODE_CHANNEL`| 1.0.0 |  |  | `HIPDNN_SOFTMAX_MODE_CHANNEL` |  |  |  | 
|`CUDNN_SOFTMAX_MODE_INSTANCE`| 1.0.0 |  |  | `HIPDNN_SOFTMAX_MODE_INSTANCE` |  |  |  | 
|`CUDNN_STATUS_ALLOC_FAILED`| 1.0.0 |  |  | `HIPDNN_STATUS_ALLOC_FAILED` |  |  |  | 
|`CUDNN_STATUS_ARCH_MISMATCH`| 1.0.0 |  |  | `HIPDNN_STATUS_ARCH_MISMATCH` |  |  |  | 
|`CUDNN_STATUS_BAD_PARAM`| 1.0.0 |  |  | `HIPDNN_STATUS_BAD_PARAM` |  |  |  | 
|`CUDNN_STATUS_EXECUTION_FAILED`| 1.0.0 |  |  | `HIPDNN_STATUS_EXECUTION_FAILED` |  |  |  | 
|`CUDNN_STATUS_INTERNAL_ERROR`| 1.0.0 |  |  | `HIPDNN_STATUS_INTERNAL_ERROR` |  |  |  | 
|`CUDNN_STATUS_INVALID_VALUE`| 1.0.0 |  |  | `HIPDNN_STATUS_INVALID_VALUE` |  |  |  | 
|`CUDNN_STATUS_LICENSE_ERROR`| 1.0.0 |  |  | `HIPDNN_STATUS_LICENSE_ERROR` |  |  |  | 
|`CUDNN_STATUS_MAPPING_ERROR`| 1.0.0 |  |  | `HIPDNN_STATUS_MAPPING_ERROR` |  |  |  | 
|`CUDNN_STATUS_NOT_INITIALIZED`| 1.0.0 |  |  | `HIPDNN_STATUS_NOT_INITIALIZED` |  |  |  | 
|`CUDNN_STATUS_NOT_SUPPORTED`| 1.0.0 |  |  | `HIPDNN_STATUS_NOT_SUPPORTED` |  |  |  | 
|`CUDNN_STATUS_RUNTIME_FP_OVERFLOW`| 7.0.5 |  |  |  |  |  |  | 
|`CUDNN_STATUS_RUNTIME_IN_PROGRESS`| 7.0.5 |  |  |  |  |  |  | 
|`CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING`| 6.0.0 |  |  | `HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING` |  |  |  | 
|`CUDNN_STATUS_SUCCESS`| 1.0.0 |  |  | `HIPDNN_STATUS_SUCCESS` |  |  |  | 
|`CUDNN_STATUS_VERSION_MISMATCH`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TENSOR_NCHW`| 1.0.0 |  |  | `HIPDNN_TENSOR_NCHW` |  |  |  | 
|`CUDNN_TENSOR_NCHW_VECT_C`| 6.0.0 |  |  | `HIPDNN_TENSOR_NCHW_VECT_C` |  |  |  | 
|`CUDNN_TENSOR_NHWC`| 1.0.0 |  |  | `HIPDNN_TENSOR_NHWC` |  |  |  | 
|`CUDNN_TENSOR_OP_MATH`| 7.0.5 |  |  | `HIPDNN_TENSOR_OP_MATH` |  |  |  | 
|`CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION`| 7.2.1 |  |  |  |  |  |  | 
|`CUDNN_TRANSFORM_FOLD`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_TRANSFORM_UNFOLD`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_TYPE_ATTRIB_NAME`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_BACKEND_DESCRIPTOR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_BN_FINALIZE_STATS_MODE`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_TYPE_BOOLEAN`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_CONVOLUTION_MODE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_DATA_TYPE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_DOUBLE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_FLOAT`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_GENSTATS_MODE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_HANDLE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_HEUR_MODE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_INT64`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_KNOB_TYPE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_LAYOUT_TYPE`| 8.0.2 |  |  |  |  |  |  | 
|`CUDNN_TYPE_NAN_PROPOGATION`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_NUMERICAL_NOTE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_POINTWISE_MODE`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_TYPE_REDUCTION_OPERATOR_TYPE`| 8.1.0 |  |  |  |  |  |  | 
|`CUDNN_TYPE_VOID_PTR`| 8.0.1 |  |  |  |  |  |  | 
|`CUDNN_UNIDIRECTIONAL`| 5.0.0 |  |  | `HIPDNN_UNIDIRECTIONAL` |  |  |  | 
|`CUDNN_VERSION`| 2.0.0 |  |  | `HIPDNN_VERSION` |  |  |  | 
|`CUDNN_WGRAD_MODE_ADD`| 7.5.0 |  |  |  |  |  |  | 
|`CUDNN_WGRAD_MODE_SET`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnActivationDescriptor_t`| 4.0.0 |  |  | `hipdnnActivationDescriptor_t` |  |  |  | 
|`cudnnActivationMode_t`| 1.0.0 |  |  | `hipdnnActivationMode_t` |  |  |  | 
|`cudnnActivationStruct`| 4.0.0 |  |  |  |  |  |  | 
|`cudnnAlgorithmDescriptor_t`| 7.1.3 |  |  |  |  |  |  | 
|`cudnnAlgorithmPerformanceStruct`| 7.1.3 |  |  |  |  |  |  | 
|`cudnnAlgorithmPerformance_t`| 7.1.3 |  |  |  |  |  |  | 
|`cudnnAlgorithmStruct`| 7.1.3 |  |  |  |  |  |  | 
|`cudnnAlgorithm_t`| 7.1.3 |  |  |  |  |  |  | 
|`cudnnAttnDescriptor_t`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnAttnQueryMap_t`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnAttnStruct`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnBackendAttributeName_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendAttributeType_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendDescriptorType_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendDescriptor_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendHeurMode_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendKnobType_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendLayoutType_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendNumericalNote_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBatchNormMode_t`| 4.0.0 |  |  | `hipdnnBatchNormMode_t` |  |  |  | 
|`cudnnBatchNormOps_t`| 7.4.1 |  |  |  |  |  |  | 
|`cudnnBnFinalizeStatsMode_t`| 8.1.0 |  |  |  |  |  |  | 
|`cudnnCTCLossAlgo_t`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnCTCLossDescriptor_t`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnCTCLossStruct`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnCallback_t`| 7.1.3 |  |  |  |  |  |  | 
|`cudnnContext`| 1.0.0 |  |  |  |  |  |  | 
|`cudnnConvolutionBwdDataAlgoPerf_t`| 3.0.0 |  |  | `hipdnnConvolutionBwdDataAlgoPerf_t` |  |  |  | 
|`cudnnConvolutionBwdDataAlgo_t`| 3.0.0 |  |  | `hipdnnConvolutionBwdDataAlgo_t` |  |  |  | 
|`cudnnConvolutionBwdDataPreference_t`| 3.0.0 | 7.6.5 | 8.0.1 | `hipdnnConvolutionBwdDataPreference_t` |  |  |  | 
|`cudnnConvolutionBwdFilterAlgoPerf_t`| 3.0.0 |  |  | `hipdnnConvolutionBwdFilterAlgoPerf_t` |  |  |  | 
|`cudnnConvolutionBwdFilterAlgo_t`| 3.0.0 |  |  | `hipdnnConvolutionBwdFilterAlgo_t` |  |  |  | 
|`cudnnConvolutionBwdFilterPreference_t`| 3.0.0 | 7.6.5 | 8.0.1 | `hipdnnConvolutionBwdFilterPreference_t` |  |  |  | 
|`cudnnConvolutionDescriptor_t`| 1.0.0 |  |  | `hipdnnConvolutionDescriptor_t` |  |  |  | 
|`cudnnConvolutionFwdAlgoPerf_t`| 3.0.0 |  |  | `hipdnnConvolutionFwdAlgoPerf_t` |  |  |  | 
|`cudnnConvolutionFwdAlgo_t`| 2.0.0 |  |  | `hipdnnConvolutionFwdAlgo_t` |  |  |  | 
|`cudnnConvolutionFwdPreference_t`| 2.0.0 | 7.6.5 | 8.0.1 | `hipdnnConvolutionFwdPreference_t` |  |  |  | 
|`cudnnConvolutionMode_t`| 1.0.0 |  |  | `hipdnnConvolutionMode_t` |  |  |  | 
|`cudnnConvolutionStruct`| 1.0.0 |  |  |  |  |  |  | 
|`cudnnDataType_t`| 1.0.0 |  |  | `hipdnnDataType_t` |  |  |  | 
|`cudnnDebug_t`| 7.1.3 |  |  |  |  |  |  | 
|`cudnnDeterminism_t`| 6.0.0 |  |  |  |  |  |  | 
|`cudnnDirectionMode_t`| 5.0.0 |  |  | `hipdnnDirectionMode_t` |  |  |  | 
|`cudnnDivNormMode_t`| 3.0.0 |  |  |  |  |  |  | 
|`cudnnDropoutDescriptor_t`| 5.0.0 |  |  | `hipdnnDropoutDescriptor_t` |  |  |  | 
|`cudnnDropoutStruct`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnErrQueryMode_t`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnFilterDescriptor_t`| 1.0.0 |  |  | `hipdnnFilterDescriptor_t` |  |  |  | 
|`cudnnFilterStruct`| 1.0.0 |  |  |  |  |  |  | 
|`cudnnFoldingDirection_t`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnForwardMode_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnFusedOpsConstParamLabel_t`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnFusedOpsConstParamPack_t`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnFusedOpsConstParamStruct`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnFusedOpsPlanStruct`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnFusedOpsPlan_t`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnFusedOpsPointerPlaceHolder_t`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnFusedOpsVariantParamLabel_t`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnFusedOpsVariantParamPack_t`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnFusedOpsVariantParamStruct`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnFusedOps_t`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnGenStatsMode_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnHandle_t`| 1.0.0 |  |  | `hipdnnHandle_t` |  |  |  | 
|`cudnnIndicesType_t`| 6.0.0 |  |  | `hipdnnIndicesType_t` |  |  |  | 
|`cudnnLRNDescriptor_t`| 3.0.0 |  |  | `hipdnnLRNDescriptor_t` |  |  |  | 
|`cudnnLRNMode_t`| 3.0.0 |  |  | `hipdnnLRNMode_t` |  |  |  | 
|`cudnnLRNStruct`| 3.0.0 |  |  |  |  |  |  | 
|`cudnnLossNormalizationMode_t`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnMathType_t`| 7.0.5 |  |  | `hipdnnMathType_t` |  |  |  | 
|`cudnnMultiHeadAttnWeightKind_t`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnNanPropagation_t`| 4.0.0 |  |  | `hipdnnNanPropagation_t` |  |  |  | 
|`cudnnNormAlgo_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnNormMode_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnNormOps_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnOpTensorDescriptor_t`| 5.0.0 |  |  | `hipdnnOpTensorDescriptor_t` |  |  |  | 
|`cudnnOpTensorOp_t`| 5.0.0 |  |  | `hipdnnOpTensorOp_t` |  |  |  | 
|`cudnnOpTensorStruct`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnPersistentRNNPlan`| 6.0.0 |  |  |  |  |  |  | 
|`cudnnPersistentRNNPlan_t`| 6.0.0 |  |  | `hipdnnPersistentRNNPlan_t` |  |  |  | 
|`cudnnPointwiseMode_t`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnPoolingDescriptor_t`| 1.0.0 |  |  | `hipdnnPoolingDescriptor_t` |  |  |  | 
|`cudnnPoolingMode_t`| 1.0.0 |  |  | `hipdnnPoolingMode_t` |  |  |  | 
|`cudnnPoolingStruct`| 1.0.0 |  |  |  |  |  |  | 
|`cudnnRNNAlgo_t`| 6.0.0 |  |  | `hipdnnRNNAlgo_t` |  |  |  | 
|`cudnnRNNBiasMode_t`| 7.5.0 |  |  | `hipdnnRNNBiasMode_t` |  |  |  | 
|`cudnnRNNClipMode_t`| 7.2.1 |  |  |  |  |  |  | 
|`cudnnRNNDataDescriptor_t`| 7.2.1 |  |  |  |  |  |  | 
|`cudnnRNNDataLayout_t`| 7.2.1 |  |  |  |  |  |  | 
|`cudnnRNNDataStruct`| 7.2.1 |  |  |  |  |  |  | 
|`cudnnRNNDescriptor_t`| 5.0.0 |  |  | `hipdnnRNNDescriptor_t` |  |  |  | 
|`cudnnRNNInputMode_t`| 5.0.0 |  |  | `hipdnnRNNInputMode_t` |  |  |  | 
|`cudnnRNNMode_t`| 5.0.0 |  |  | `hipdnnRNNMode_t` |  |  |  | 
|`cudnnRNNPaddingMode_t`| 7.2.1 |  |  |  |  |  |  | 
|`cudnnRNNStruct`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnReduceTensorDescriptor_t`| 6.0.0 |  |  | `hipdnnReduceTensorDescriptor_t` |  |  |  | 
|`cudnnReduceTensorIndices_t`| 6.0.0 |  |  | `hipdnnReduceTensorIndices_t` |  |  |  | 
|`cudnnReduceTensorOp_t`| 6.0.0 |  |  | `hipdnnReduceTensorOp_t` |  |  |  | 
|`cudnnReduceTensorStruct`| 6.0.0 |  |  |  |  |  |  | 
|`cudnnReorderType_t`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnRuntimeTag_t`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnSamplerType_t`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnSeqDataAxis_t`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnSeqDataDescriptor_t`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnSeqDataStruct`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnSeverity_t`| 7.1.3 |  |  |  |  |  |  | 
|`cudnnSoftmaxAlgorithm_t`| 1.0.0 |  |  | `hipdnnSoftmaxAlgorithm_t` |  |  |  | 
|`cudnnSoftmaxMode_t`| 1.0.0 |  |  | `hipdnnSoftmaxMode_t` |  |  |  | 
|`cudnnSpatialTransformerDescriptor_t`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnSpatialTransformerStruct`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnStatus_t`| 1.0.0 |  |  | `hipdnnStatus_t` |  |  |  | 
|`cudnnTensorDescriptor_t`| 2.0.0 |  |  | `hipdnnTensorDescriptor_t` |  |  |  | 
|`cudnnTensorFormat_t`| 1.0.0 |  |  | `hipdnnTensorFormat_t` |  |  |  | 
|`cudnnTensorStruct`| 2.0.0 |  |  |  |  |  |  | 
|`cudnnTensorTransformDescriptor_t`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnTensorTransformStruct`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnWgradMode_t`| 7.5.0 |  |  |  |  |  |  | 
|`libraryPropertyType`| 6.0.0 |  |  |  |  |  |  | 
|`libraryPropertyType_t`| 6.0.0 |  |  |  |  |  |  | 

## **2. CUNN Functions**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cudnnActivationBackward`| 1.0.0 |  |  | `hipdnnActivationBackward` |  |  |  | 
|`cudnnActivationForward`| 1.0.0 |  |  | `hipdnnActivationForward` |  |  |  | 
|`cudnnAddTensor`| 2.0.0 |  |  | `hipdnnAddTensor` |  |  |  | 
|`cudnnAdvInferVersionCheck`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnAdvTrainVersionCheck`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendCreateDescriptor`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendDestroyDescriptor`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendExecute`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendFinalize`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendGetAttribute`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendInitialize`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBackendSetAttribute`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnBatchNormalizationBackward`| 4.0.0 |  |  | `hipdnnBatchNormalizationBackward` |  |  |  | 
|`cudnnBatchNormalizationBackwardEx`| 7.4.1 |  |  |  |  |  |  | 
|`cudnnBatchNormalizationForwardInference`| 4.0.0 |  |  | `hipdnnBatchNormalizationForwardInference` |  |  |  | 
|`cudnnBatchNormalizationForwardTraining`| 4.0.0 |  |  | `hipdnnBatchNormalizationForwardTraining` |  |  |  | 
|`cudnnBatchNormalizationForwardTrainingEx`| 7.4.1 |  |  |  |  |  |  | 
|`cudnnBuildRNNDynamic`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnCTCLoss`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnCTCLoss_v8`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnCnnInferVersionCheck`| 8.0.2 |  |  |  |  |  |  | 
|`cudnnCnnTrainVersionCheck`| 8.0.2 |  |  |  |  |  |  | 
|`cudnnConvolutionBackwardBias`| 1.0.0 |  |  | `hipdnnConvolutionBackwardBias` |  |  |  | 
|`cudnnConvolutionBackwardData`| 1.0.0 |  |  | `hipdnnConvolutionBackwardData` |  |  |  | 
|`cudnnConvolutionBackwardFilter`| 1.0.0 |  |  | `hipdnnConvolutionBackwardFilter` |  |  |  | 
|`cudnnConvolutionBiasActivationForward`| 6.0.0 |  |  |  |  |  |  | 
|`cudnnConvolutionForward`| 1.0.0 |  |  | `hipdnnConvolutionForward` |  |  |  | 
|`cudnnCopyAlgorithmDescriptor`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnCreate`| 1.0.0 |  |  | `hipdnnCreate` |  |  |  | 
|`cudnnCreateActivationDescriptor`| 4.0.0 |  |  | `hipdnnCreateActivationDescriptor` |  |  |  | 
|`cudnnCreateAlgorithmDescriptor`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnCreateAlgorithmPerformance`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnCreateAttnDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnCreateCTCLossDescriptor`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnCreateConvolutionDescriptor`| 1.0.0 |  |  | `hipdnnCreateConvolutionDescriptor` |  |  |  | 
|`cudnnCreateDropoutDescriptor`| 5.0.0 |  |  | `hipdnnCreateDropoutDescriptor` |  |  |  | 
|`cudnnCreateFilterDescriptor`| 1.0.0 |  |  | `hipdnnCreateFilterDescriptor` |  |  |  | 
|`cudnnCreateFusedOpsConstParamPack`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnCreateFusedOpsPlan`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnCreateFusedOpsVariantParamPack`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnCreateLRNDescriptor`| 3.0.0 |  |  | `hipdnnCreateLRNDescriptor` |  |  |  | 
|`cudnnCreateOpTensorDescriptor`| 5.0.0 |  |  | `hipdnnCreateOpTensorDescriptor` |  |  |  | 
|`cudnnCreatePersistentRNNPlan`| 6.0.0 | 8.0.1 |  | `hipdnnCreatePersistentRNNPlan` |  |  |  | 
|`cudnnCreatePoolingDescriptor`| 1.0.0 |  |  | `hipdnnCreatePoolingDescriptor` |  |  |  | 
|`cudnnCreateRNNDataDescriptor`| 7.2.1 |  |  |  |  |  |  | 
|`cudnnCreateRNNDescriptor`| 5.0.0 |  |  | `hipdnnCreateRNNDescriptor` |  |  |  | 
|`cudnnCreateReduceTensorDescriptor`| 6.0.0 |  |  | `hipdnnCreateReduceTensorDescriptor` |  |  |  | 
|`cudnnCreateSeqDataDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnCreateSpatialTransformerDescriptor`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnCreateTensorDescriptor`| 2.0.0 |  |  | `hipdnnCreateTensorDescriptor` |  |  |  | 
|`cudnnCreateTensorTransformDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnDeriveBNTensorDescriptor`| 4.0.0 |  |  | `hipdnnDeriveBNTensorDescriptor` |  |  |  | 
|`cudnnDeriveNormTensorDescriptor`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnDestroy`| 1.0.0 |  |  | `hipdnnDestroy` |  |  |  | 
|`cudnnDestroyActivationDescriptor`| 4.0.0 |  |  | `hipdnnDestroyActivationDescriptor` |  |  |  | 
|`cudnnDestroyAlgorithmDescriptor`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnDestroyAlgorithmPerformance`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnDestroyAttnDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnDestroyCTCLossDescriptor`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnDestroyConvolutionDescriptor`| 1.0.0 |  |  | `hipdnnDestroyConvolutionDescriptor` |  |  |  | 
|`cudnnDestroyDropoutDescriptor`| 5.0.0 |  |  | `hipdnnDestroyDropoutDescriptor` |  |  |  | 
|`cudnnDestroyFilterDescriptor`| 1.0.0 |  |  | `hipdnnDestroyFilterDescriptor` |  |  |  | 
|`cudnnDestroyFusedOpsConstParamPack`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnDestroyFusedOpsPlan`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnDestroyFusedOpsVariantParamPack`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnDestroyLRNDescriptor`| 3.0.0 |  |  | `hipdnnDestroyLRNDescriptor` |  |  |  | 
|`cudnnDestroyOpTensorDescriptor`| 5.0.0 |  |  | `hipdnnDestroyOpTensorDescriptor` |  |  |  | 
|`cudnnDestroyPersistentRNNPlan`| 6.0.0 | 8.0.1 |  | `hipdnnDestroyPersistentRNNPlan` |  |  |  | 
|`cudnnDestroyPoolingDescriptor`| 1.0.0 |  |  | `hipdnnDestroyPoolingDescriptor` |  |  |  | 
|`cudnnDestroyRNNDataDescriptor`| 7.2.1 |  |  |  |  |  |  | 
|`cudnnDestroyRNNDescriptor`| 5.0.0 |  |  | `hipdnnDestroyRNNDescriptor` |  |  |  | 
|`cudnnDestroyReduceTensorDescriptor`| 6.0.0 |  |  | `hipdnnDestroyReduceTensorDescriptor` |  |  |  | 
|`cudnnDestroySeqDataDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnDestroySpatialTransformerDescriptor`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnDestroyTensorDescriptor`| 2.0.0 |  |  | `hipdnnDestroyTensorDescriptor` |  |  |  | 
|`cudnnDestroyTensorTransformDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnDivisiveNormalizationBackward`| 3.0.0 |  |  |  |  |  |  | 
|`cudnnDivisiveNormalizationForward`| 3.0.0 |  |  |  |  |  |  | 
|`cudnnDropoutBackward`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnDropoutForward`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnDropoutGetReserveSpaceSize`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnDropoutGetStatesSize`| 5.0.0 |  |  | `hipdnnDropoutGetStatesSize` |  |  |  | 
|`cudnnFindConvolutionBackwardDataAlgorithm`| 3.0.0 |  |  | `hipdnnFindConvolutionBackwardDataAlgorithm` |  |  |  | 
|`cudnnFindConvolutionBackwardDataAlgorithmEx`| 5.0.0 |  |  | `hipdnnFindConvolutionBackwardDataAlgorithmEx` |  |  |  | 
|`cudnnFindConvolutionBackwardFilterAlgorithm`| 3.0.0 |  |  | `hipdnnFindConvolutionBackwardFilterAlgorithm` |  |  |  | 
|`cudnnFindConvolutionBackwardFilterAlgorithmEx`| 5.0.0 |  |  | `hipdnnFindConvolutionBackwardFilterAlgorithmEx` |  |  |  | 
|`cudnnFindConvolutionForwardAlgorithm`| 3.0.0 |  |  | `hipdnnFindConvolutionForwardAlgorithm` |  |  |  | 
|`cudnnFindConvolutionForwardAlgorithmEx`| 5.0.0 |  |  | `hipdnnFindConvolutionForwardAlgorithmEx` |  |  |  | 
|`cudnnFindRNNBackwardDataAlgorithmEx`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnFindRNNBackwardWeightsAlgorithmEx`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnFindRNNForwardInferenceAlgorithmEx`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnFindRNNForwardTrainingAlgorithmEx`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnFusedOpsExecute`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnGetActivationDescriptor`| 4.0.0 |  |  | `hipdnnGetActivationDescriptor` |  |  |  | 
|`cudnnGetAlgorithmDescriptor`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnGetAlgorithmPerformance`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnGetAlgorithmSpaceSize`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnGetAttnDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnGetBatchNormalizationBackwardExWorkspaceSize`| 7.4.1 |  |  |  |  |  |  | 
|`cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize`| 7.4.1 |  |  |  |  |  |  | 
|`cudnnGetBatchNormalizationTrainingExReserveSpaceSize`| 7.4.1 |  |  |  |  |  |  | 
|`cudnnGetCTCLossDescriptor`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnGetCTCLossDescriptorEx`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnGetCTCLossDescriptor_v8`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnGetCTCLossWorkspaceSize`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnGetCTCLossWorkspaceSize_v8`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnGetCallback`| 7.1.3 |  |  |  |  |  |  | 
|`cudnnGetConvolution2dDescriptor`| 2.0.0 |  |  | `hipdnnGetConvolution2dDescriptor` |  |  |  | 
|`cudnnGetConvolution2dForwardOutputDim`| 2.0.0 |  |  | `hipdnnGetConvolution2dForwardOutputDim` |  |  |  | 
|`cudnnGetConvolutionBackwardDataAlgorithm`| 3.0.0 | 7.6.5 | 8.0.1 | `hipdnnGetConvolutionBackwardDataAlgorithm` |  |  |  | 
|`cudnnGetConvolutionBackwardDataAlgorithmMaxCount`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnGetConvolutionBackwardDataAlgorithm_v7`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnGetConvolutionBackwardDataWorkspaceSize`| 3.0.0 |  |  | `hipdnnGetConvolutionBackwardDataWorkspaceSize` |  |  |  | 
|`cudnnGetConvolutionBackwardFilterAlgorithm`| 3.0.0 | 7.6.5 | 8.0.1 | `hipdnnGetConvolutionBackwardFilterAlgorithm` |  |  |  | 
|`cudnnGetConvolutionBackwardFilterAlgorithmMaxCount`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnGetConvolutionBackwardFilterAlgorithm_v7`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnGetConvolutionBackwardFilterWorkspaceSize`| 3.0.0 |  |  | `hipdnnGetConvolutionBackwardFilterWorkspaceSize` |  |  |  | 
|`cudnnGetConvolutionForwardAlgorithm`| 2.0.0 | 7.6.5 | 8.0.1 | `hipdnnGetConvolutionForwardAlgorithm` |  |  |  | 
|`cudnnGetConvolutionForwardAlgorithmMaxCount`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnGetConvolutionForwardAlgorithm_v7`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnGetConvolutionForwardWorkspaceSize`| 2.0.0 |  |  | `hipdnnGetConvolutionForwardWorkspaceSize` |  |  |  | 
|`cudnnGetConvolutionGroupCount`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnGetConvolutionMathType`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnGetConvolutionNdDescriptor`| 2.0.0 |  |  |  |  |  |  | 
|`cudnnGetConvolutionNdForwardOutputDim`| 2.0.0 |  |  |  |  |  |  | 
|`cudnnGetConvolutionReorderType`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnGetCudartVersion`| 6.0.0 |  |  |  |  |  |  | 
|`cudnnGetDropoutDescriptor`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnGetErrorString`| 2.0.0 |  |  | `hipdnnGetErrorString` |  |  |  | 
|`cudnnGetFilter4dDescriptor`| 2.0.0 |  |  | `hipdnnGetFilter4dDescriptor` |  |  |  | 
|`cudnnGetFilterNdDescriptor`| 2.0.0 |  |  | `hipdnnGetFilterNdDescriptor` |  |  |  | 
|`cudnnGetFilterSizeInBytes`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnGetFoldedConvBackwardDataDescriptors`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnGetFusedOpsConstParamPackAttribute`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnGetFusedOpsVariantParamPackAttribute`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnGetLRNDescriptor`| 3.0.0 |  |  | `hipdnnGetLRNDescriptor` |  |  |  | 
|`cudnnGetMultiHeadAttnBuffers`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnGetMultiHeadAttnWeights`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnGetNormalizationBackwardWorkspaceSize`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnGetNormalizationForwardTrainingWorkspaceSize`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnGetNormalizationTrainingReserveSpaceSize`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnGetOpTensorDescriptor`| 5.0.0 |  |  | `hipdnnGetOpTensorDescriptor` |  |  |  | 
|`cudnnGetPooling2dDescriptor`| 2.0.0 |  |  | `hipdnnGetPooling2dDescriptor` |  |  |  | 
|`cudnnGetPooling2dForwardOutputDim`| 2.0.0 |  |  | `hipdnnGetPooling2dForwardOutputDim` |  |  |  | 
|`cudnnGetPoolingNdDescriptor`| 2.0.0 |  |  |  |  |  |  | 
|`cudnnGetPoolingNdForwardOutputDim`| 2.0.0 |  |  |  |  |  |  | 
|`cudnnGetProperty`| 6.0.0 |  |  |  |  |  |  | 
|`cudnnGetRNNBackwardDataAlgorithmMaxCount`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnGetRNNBackwardWeightsAlgorithmMaxCount`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnGetRNNBiasMode`| 7.5.0 | 8.0.1 |  |  |  |  |  | 
|`cudnnGetRNNDataDescriptor`| 7.2.1 |  |  |  |  |  |  | 
|`cudnnGetRNNDescriptor`| 7.0.5 | 7.6.5 | 8.0.1 | `hipdnnGetRNNDescriptor` |  |  |  | 
|`cudnnGetRNNDescriptor_v6`| 8.0.1 | 8.0.1 |  |  |  |  |  | 
|`cudnnGetRNNDescriptor_v8`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnGetRNNForwardInferenceAlgorithmMaxCount`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnGetRNNForwardTrainingAlgorithmMaxCount`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnGetRNNLinLayerBiasParams`| 5.0.0 | 8.0.1 |  | `hipdnnGetRNNLinLayerBiasParams` |  |  |  | 
|`cudnnGetRNNLinLayerMatrixParams`| 5.0.0 | 8.0.1 |  | `hipdnnGetRNNLinLayerMatrixParams` |  |  |  | 
|`cudnnGetRNNMatrixMathType`| 7.1.3 | 8.0.1 |  |  |  |  |  | 
|`cudnnGetRNNPaddingMode`| 7.2.1 | 8.0.1 |  |  |  |  |  | 
|`cudnnGetRNNParamsSize`| 5.0.0 | 8.0.1 |  | `hipdnnGetRNNParamsSize` |  |  |  | 
|`cudnnGetRNNProjectionLayers`| 7.1.3 | 8.0.1 |  |  |  |  |  | 
|`cudnnGetRNNTempSpaceSizes`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnGetRNNTrainingReserveSize`| 5.0.0 | 8.0.1 |  | `hipdnnGetRNNTrainingReserveSize` |  |  |  | 
|`cudnnGetRNNWeightParams`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnGetRNNWeightSpaceSize`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnGetRNNWorkspaceSize`| 5.0.0 | 8.0.1 |  | `hipdnnGetRNNWorkspaceSize` |  |  |  | 
|`cudnnGetReduceTensorDescriptor`| 6.0.0 |  |  | `hipdnnGetReduceTensorDescriptor` |  |  |  | 
|`cudnnGetReductionIndicesSize`| 6.0.0 |  |  |  |  |  |  | 
|`cudnnGetReductionWorkspaceSize`| 6.0.0 |  |  | `hipdnnGetReductionWorkspaceSize` |  |  |  | 
|`cudnnGetSeqDataDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnGetStream`| 1.0.0 |  |  | `hipdnnGetStream` |  |  |  | 
|`cudnnGetTensor4dDescriptor`| 1.0.0 |  |  | `hipdnnGetTensor4dDescriptor` |  |  |  | 
|`cudnnGetTensorNdDescriptor`| 2.0.0 |  |  | `hipdnnGetTensorNdDescriptor` |  |  |  | 
|`cudnnGetTensorSizeInBytes`| 6.0.0 |  |  |  |  |  |  | 
|`cudnnGetTensorTransformDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnGetVersion`| 2.0.0 |  |  | `hipdnnGetVersion` |  |  |  | 
|`cudnnIm2Col`| 2.0.0 |  |  |  |  |  |  | 
|`cudnnInitTransformDest`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnLRNCrossChannelBackward`| 3.0.0 |  |  | `hipdnnLRNCrossChannelBackward` |  |  |  | 
|`cudnnLRNCrossChannelForward`| 3.0.0 |  |  | `hipdnnLRNCrossChannelForward` |  |  |  | 
|`cudnnMakeFusedOpsPlan`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnMultiHeadAttnBackwardData`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnMultiHeadAttnBackwardWeights`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnMultiHeadAttnForward`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnNormalizationBackward`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnNormalizationForwardInference`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnNormalizationForwardTraining`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnOpTensor`| 5.0.0 |  |  | `hipdnnOpTensor` |  |  |  | 
|`cudnnOpsInferVersionCheck`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnOpsTrainVersionCheck`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnPoolingBackward`| 1.0.0 |  |  | `hipdnnPoolingBackward` |  |  |  | 
|`cudnnPoolingForward`| 1.0.0 |  |  | `hipdnnPoolingForward` |  |  |  | 
|`cudnnQueryRuntimeError`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnRNNBackwardData`| 5.0.0 | 8.0.2 |  | `hipdnnRNNBackwardData` |  |  |  | 
|`cudnnRNNBackwardDataEx`| 7.2.1 | 8.0.2 |  |  |  |  |  | 
|`cudnnRNNBackwardData_v8`| 8.0.2 |  |  |  |  |  |  | 
|`cudnnRNNBackwardWeights`| 5.0.0 | 8.0.2 |  | `hipdnnRNNBackwardWeights` |  |  |  | 
|`cudnnRNNBackwardWeightsEx`| 7.2.1 | 8.0.2 |  |  |  |  |  | 
|`cudnnRNNBackwardWeights_v8`| 8.0.2 |  |  |  |  |  |  | 
|`cudnnRNNForward`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnRNNForwardInference`| 5.0.0 | 8.0.1 |  | `hipdnnRNNForwardInference` |  |  |  | 
|`cudnnRNNForwardInferenceEx`| 7.2.1 | 8.0.1 |  |  |  |  |  | 
|`cudnnRNNForwardTraining`| 5.0.0 | 8.0.1 |  | `hipdnnRNNForwardTraining` |  |  |  | 
|`cudnnRNNForwardTrainingEx`| 7.2.1 | 8.0.1 |  |  |  |  |  | 
|`cudnnRNNGetClip`| 7.2.1 | 8.0.1 |  |  |  |  |  | 
|`cudnnRNNGetClip_v8`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnRNNSetClip`| 7.2.1 | 8.0.1 |  |  |  |  |  | 
|`cudnnRNNSetClip_v8`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnReduceTensor`| 6.0.0 |  |  | `hipdnnReduceTensor` |  |  |  | 
|`cudnnReorderFilterAndBias`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnRestoreAlgorithm`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnRestoreDropoutDescriptor`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnSaveAlgorithm`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnScaleTensor`| 2.0.0 |  |  | `hipdnnScaleTensor` |  |  |  | 
|`cudnnSetActivationDescriptor`| 4.0.0 |  |  | `hipdnnSetActivationDescriptor` |  |  |  | 
|`cudnnSetAlgorithmDescriptor`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnSetAlgorithmPerformance`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnSetAttnDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnSetCTCLossDescriptor`| 7.0.5 |  |  |  |  |  |  | 
|`cudnnSetCTCLossDescriptorEx`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnSetCTCLossDescriptor_v8`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnSetCallback`| 7.1.3 |  |  |  |  |  |  | 
|`cudnnSetConvolution2dDescriptor`| 2.0.0 |  |  | `hipdnnSetConvolution2dDescriptor` |  |  |  | 
|`cudnnSetConvolutionGroupCount`| 7.0.5 |  |  | `hipdnnSetConvolutionGroupCount` |  |  |  | 
|`cudnnSetConvolutionMathType`| 7.0.5 |  |  | `hipdnnSetConvolutionMathType` |  |  |  | 
|`cudnnSetConvolutionNdDescriptor`| 2.0.0 |  |  | `hipdnnSetConvolutionNdDescriptor` |  |  |  | 
|`cudnnSetConvolutionReorderType`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnSetDropoutDescriptor`| 5.0.0 |  |  | `hipdnnSetDropoutDescriptor` |  |  |  | 
|`cudnnSetFilter4dDescriptor`| 2.0.0 |  |  | `hipdnnSetFilter4dDescriptor` |  |  |  | 
|`cudnnSetFilterNdDescriptor`| 2.0.0 |  |  | `hipdnnSetFilterNdDescriptor` |  |  |  | 
|`cudnnSetFusedOpsConstParamPackAttribute`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnSetFusedOpsVariantParamPackAttribute`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnSetLRNDescriptor`| 3.0.0 |  |  | `hipdnnSetLRNDescriptor` |  |  |  | 
|`cudnnSetOpTensorDescriptor`| 5.0.0 |  |  | `hipdnnSetOpTensorDescriptor` |  |  |  | 
|`cudnnSetPersistentRNNPlan`| 6.0.0 | 8.0.1 |  | `hipdnnSetPersistentRNNPlan` |  |  |  | 
|`cudnnSetPooling2dDescriptor`| 2.0.0 |  |  | `hipdnnSetPooling2dDescriptor` |  |  |  | 
|`cudnnSetPoolingNdDescriptor`| 2.0.0 |  |  | `hipdnnSetPoolingNdDescriptor` |  |  |  | 
|`cudnnSetRNNAlgorithmDescriptor`| 7.1.3 | 8.0.2 |  |  |  |  |  | 
|`cudnnSetRNNBiasMode`| 7.5.0 | 8.0.1 |  |  |  |  |  | 
|`cudnnSetRNNDataDescriptor`| 7.2.1 |  |  |  |  |  |  | 
|`cudnnSetRNNDescriptor`| 5.0.0 | 7.6.5 | 8.0.1 | `hipdnnSetRNNDescriptor` |  |  |  | 
|`cudnnSetRNNDescriptor_v5`| 7.0.5 | 7.6.5 | 8.0.1 | `hipdnnSetRNNDescriptor_v5` |  |  |  | 
|`cudnnSetRNNDescriptor_v6`| 6.0.0 | 8.0.1 |  | `hipdnnSetRNNDescriptor_v6` |  |  |  | 
|`cudnnSetRNNDescriptor_v8`| 8.0.1 |  |  |  |  |  |  | 
|`cudnnSetRNNMatrixMathType`| 7.0.5 | 8.0.1 |  |  |  |  |  | 
|`cudnnSetRNNPaddingMode`| 7.2.1 | 8.0.1 |  |  |  |  |  | 
|`cudnnSetRNNProjectionLayers`| 7.1.3 | 8.0.1 |  |  |  |  |  | 
|`cudnnSetReduceTensorDescriptor`| 6.0.0 |  |  | `hipdnnSetReduceTensorDescriptor` |  |  |  | 
|`cudnnSetSeqDataDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnSetSpatialTransformerNdDescriptor`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnSetStream`| 1.0.0 |  |  | `hipdnnSetStream` |  |  |  | 
|`cudnnSetTensor`| 2.0.0 |  |  | `hipdnnSetTensor` |  |  |  | 
|`cudnnSetTensor4dDescriptor`| 1.0.0 |  |  | `hipdnnSetTensor4dDescriptor` |  |  |  | 
|`cudnnSetTensor4dDescriptorEx`| 1.0.0 |  |  | `hipdnnSetTensor4dDescriptorEx` |  |  |  | 
|`cudnnSetTensorNdDescriptor`| 2.0.0 |  |  | `hipdnnSetTensorNdDescriptor` |  |  |  | 
|`cudnnSetTensorNdDescriptorEx`| 6.0.0 |  |  |  |  |  |  | 
|`cudnnSetTensorTransformDescriptor`| 7.5.0 |  |  |  |  |  |  | 
|`cudnnSoftmaxBackward`| 1.0.0 |  |  | `hipdnnSoftmaxBackward` |  |  |  | 
|`cudnnSoftmaxForward`| 1.0.0 |  |  | `hipdnnSoftmaxForward` |  |  |  | 
|`cudnnSpatialTfGridGeneratorBackward`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnSpatialTfGridGeneratorForward`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnSpatialTfSamplerBackward`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnSpatialTfSamplerForward`| 5.0.0 |  |  |  |  |  |  | 
|`cudnnTransformFilter`| 7.6.0 |  |  |  |  |  |  | 
|`cudnnTransformTensor`| 2.0.0 |  |  |  |  |  |  | 
|`cudnnTransformTensorEx`| 7.5.0 |  |  |  |  |  |  | 


\*A - Added; D - Deprecated; R - Removed