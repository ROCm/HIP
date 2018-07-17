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
__attribute__((convergent))
ulong __llvm_amdgcn_icmp_i32(uint x, uint y, uint z) __asm("llvm.amdgcn.icmp.i32");

__device__
unsigned __llvm_amdgcn_groupstaticsize() __asm("llvm.amdgcn.groupstaticsize");

__device__
unsigned int __llvm_bitrev_b32(unsigned int src0) __asm("llvm.bitreverse.i32");

__device__
uint64_t __llvm_bitrev_b64(uint64_t src0) __asm("llvm.bitreverse.i64");

extern
__device__
__attribute__((const))
unsigned int __mbcnt_lo(unsigned int x, unsigned int y) __asm("llvm.amdgcn.mbcnt.lo");

extern
__device__
__attribute__((const))
unsigned int __mbcnt_hi(unsigned int x, unsigned int y) __asm("llvm.amdgcn.mbcnt.hi");

__device__
int __llvm_amdgcn_ds_bpermute(int index, int src) __asm("llvm.amdgcn.ds.bpermute");

__device__
int __llvm_amdgcn_ds_permute(int index, int src) __asm("llvm.amdgcn.ds.permute");

__device__
int __llvm_amdgcn_ds_swizzle(int index, int pattern) __asm("llvm.amdgcn.ds.swizzle");

__device__
int __llvm_amdgcn_move_dpp(int src, int dpp_ctrl, int row_mask, int bank_mask,
                           bool bound_ctrl) __asm("llvm.amdgcn.mov.dpp.i32");

#endif
