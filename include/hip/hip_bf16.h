/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_HIP_BF16_H
#define HIP_INCLUDE_HIP_HIP_BF16_H

#include <hip/hip_common.h>

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/amd_detail/amd_hip_bf16.h>
#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/nvidia_detail/nvidia_hip_bf16.h>
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif  // HIP_INCLUDE_HIP_HIP_BF16_H
