/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_HIP_FP8_H
#define HIP_INCLUDE_HIP_HIP_FP8_H

#include <hip/hip_common.h>

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
// We only have fnuz defs for now, which are not supported by other platforms
#include <hip/amd_detail/amd_hip_fp8.h>
#endif

#endif  // HIP_INCLUDE_HIP_HIP_FP8_H
