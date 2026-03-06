/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_CHANNEL_DESCRIPTOR_H
#define HIP_INCLUDE_HIP_CHANNEL_DESCRIPTOR_H

// Some standard header files, these are included by hc.hpp and so want to make them avail on both
// paths to provide a consistent include env and avoid "missing symbol" errors that only appears
// on NVCC path:


#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/amd_detail/amd_channel_descriptor.h>
#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/nvidia_detail/nvidia_channel_descriptor.h>
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif
