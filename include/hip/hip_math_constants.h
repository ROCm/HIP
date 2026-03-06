/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef HIP_MATH_CONSTANTS_H
#define HIP_MATH_CONSTANTS_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#endif

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include "hip/amd_detail/amd_hip_math_constants.h"
#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "hip/nvidia_detail/nvidia_hip_math_constants.h"
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
#endif
