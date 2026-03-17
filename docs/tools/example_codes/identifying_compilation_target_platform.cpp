// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstdlib>

int main()
{
    // [sphinx-amd-start]
#ifdef __HIP_PLATFORM_AMD__
    // This code path is compiled when amdclang++ is used for compilation
#endif
    // [sphinx-amd-end]

    // [sphinx-nvidia-start]
#ifdef __HIP_PLATFORM_NVIDIA__
    // This code path is compiled when nvcc is used for compilation
    // Could be compiling with CUDA language extensions enabled (for example, a ".cu file)
    // Could be in pass-through mode to an underlying host compiler (for example, a .cpp file)
#endif
    // [sphinx-nvidia-end]

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#   error "No compatible HIP platform defined!"
#endif

    return EXIT_SUCCESS;
}
