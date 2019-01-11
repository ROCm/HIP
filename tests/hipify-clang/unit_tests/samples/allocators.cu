// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#pragma once

#include <cuda_runtime.h>


/**
 * Allocate GPU memory for `count` elements of type `T`.
 */
template<typename T>
static T* gpuMalloc(size_t count) {
    T* ret = nullptr;
    // CHECK: hipMalloc(&ret, count * sizeof(T));
    cudaMalloc(&ret, count * sizeof(T));
    return ret;
}

