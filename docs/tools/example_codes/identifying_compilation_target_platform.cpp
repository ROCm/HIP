// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
