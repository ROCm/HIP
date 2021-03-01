#!/bin/bash
# Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#set -x
HIP_BUILD_INC_DIR=$1
HIP_INC_DIR=$2
LLVM_DIR="$3/../../../"
HSA_DIR="$4"
tmp=/tmp/hip_pch.$$
mkdir -p $tmp

cat >$tmp/hip_macros.h <<EOF
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))
#define __constant__ __attribute__((constant))
#define __shared__ __attribute__((shared))

#define launch_bounds_impl0(requiredMaxThreadsPerBlock)                                            \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock)))
#define launch_bounds_impl1(requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor)                \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock),                     \
                   amdgpu_waves_per_eu(minBlocksPerMultiprocessor)))
#define select_impl_(_1, _2, impl_, ...) impl_
#define __launch_bounds__(...)                                                                     \
    select_impl_(__VA_ARGS__, launch_bounds_impl1, launch_bounds_impl0)(__VA_ARGS__)

EOF

cat >$tmp/hip_pch.h <<EOF
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
EOF

cat >$tmp/hip_pch.mcin <<EOF
 .type __hip_pch,@object
  .section .hip_pch,"aMS",@progbits,1
  .data
  .globl __hip_pch
  .globl __hip_pch_size
  .p2align 3
__hip_pch:
  .incbin "$tmp/hip.pch"
__hip_pch_size:
  .long __hip_pch_size - __hip_pch
EOF

set -x

$LLVM_DIR/bin/clang -O3 --rocm-path=$HIP_INC_DIR/.. -std=c++17 -nogpulib -isystem $HIP_INC_DIR -isystem $HIP_BUILD_INC_DIR -isystem $HSA_DIR/include --cuda-device-only -x hip $tmp/hip_pch.h -E >$tmp/pch.cui &&

cat $tmp/hip_macros.h >> $tmp/pch.cui &&

$LLVM_DIR/bin/clang -cc1 -O3 -emit-pch -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -std=c++17 -fgnuc-version=4.2.1 -o $tmp/hip.pch -x hip-cpp-output - <$tmp/pch.cui &&

$LLVM_DIR/bin/llvm-mc -o hip_pch.o $tmp/hip_pch.mcin --filetype=obj &&

rm -rf $tmp
