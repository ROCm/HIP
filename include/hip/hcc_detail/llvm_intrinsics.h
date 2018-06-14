/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 *  @file  hcc_detail/llvm_intrinsics.h
 *  @brief Contains declarations for wrapper functions for llvm intrinsics
 *         like llvm.amdgcn.s.barrier.
 */

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_LLVM_INTRINSICS_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_LLVM_INTRINSICS_H

#include "hip/hcc_detail/host_defines.h"

__device__
unsigned long __llvm_amdgcn_s_memrealtime(void) __asm("llvm.amdgcn.s.memrealtime");

__device__
unsigned __llvm_amdgcn_s_getreg(unsigned) __asm("llvm.amdgcn.s.getreg");

__device__
unsigned __llvm_amdgcn_groupstaticsize() __asm("llvm.amdgcn.groupstaticsize");

__device__
void __llvm_amdgcn_s_barrier() __asm("llvm.amdgcn.s.barrier");

#endif
