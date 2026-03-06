/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_HIP_FP4_H
#define HIP_INCLUDE_HIP_HIP_FP4_H

#include <hip/hip_common.h>

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/amd_detail/amd_hip_fp4.h>
#endif

#endif  // HIP_INCLUDE_HIP_HIP_FP4_H
