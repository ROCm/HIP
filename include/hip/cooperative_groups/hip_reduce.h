/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 *  @file  hip_reduce.h
 *
 *  @brief Defines cooperative_groups reduce device API
 */
#ifndef HIP_INCLUDE_HIP_HIP_COOPERATIVE_GROUP_REDUCE_H
#define HIP_INCLUDE_HIP_HIP_COOPERATIVE_GROUP_REDUCE_H

#include <hip/hip_version.h>
#include <hip/hip_common.h>

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#if __cplusplus && defined(__clang__) && defined(__HIP__)
#include <hip/amd_detail/amd_hip_cooperative_groups_reduce.h>
#endif

#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/nvidia_detail/nvidia_hip_cooperative_groups_reduce.h>
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif  // HIP_INCLUDE_HIP_HIP_COOPERATIVE_GROUP_REDUCE_H
