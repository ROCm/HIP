/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_HIP_FP16_H
#define HIP_INCLUDE_HIP_HIP_FP16_H

#include <hip/hip_common.h>

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/amd_detail/amd_hip_fp16.h>
#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#define HIPRT_INF_FP16 CUDART_INF_FP16
#define HIPRT_MAX_NORMAL_FP16 CUDART_MAX_NORMAL_FP16
#define HIPRT_MIN_DENORM_FP16 CUDART_MIN_DENORM_FP16
#define HIPRT_NAN_FP16 CUDART_NAN_FP16
#define HIPRT_NEG_ZERO_FP16 CUDART_NEG_ZERO_FP16
#define HIPRT_ONE_FP16 CUDART_ONE_FP16
#define HIPRT_ZERO_FP16 CUDART_ZERO_FP16

#include "cuda_fp16.h"
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif
