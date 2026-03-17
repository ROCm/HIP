// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>

#define HIP_CHECK(expression)                                \
{                                                            \
    const hipError_t err = expression;                       \
    if (err != hipSuccess)                                   \
    {                                                        \
        std::cout << "HIP Error: " << hipGetErrorString(err) \
              << " at line " << __LINE__ << std::endl;       \
        std::exit(EXIT_FAILURE);                             \
    }                                                        \
}

int main()
{
    // [sphinx-start]
    double * ptr;
    HIP_CHECK(hipMalloc(&ptr, sizeof(double)));
    hipPointerAttribute_t attr;
    HIP_CHECK(hipPointerGetAttributes(&attr, ptr)); /*attr.type is hipMemoryTypeDevice*/
    if(attr.type == hipMemoryTypeDevice)
        std::cout << "ptr is of type hipMemoryTypeDevice" << std::endl;

    double* ptrHost;
    HIP_CHECK(hipHostMalloc(&ptrHost, sizeof(double)));
    hipPointerAttribute_t attrHost;
    HIP_CHECK(hipPointerGetAttributes(&attrHost, ptrHost)); /*attr.type is hipMemoryTypeHost*/
    if(attrHost.type == hipMemoryTypeHost)
        std::cout << "ptrHost is of type hipMemoryTypeHost" << std::endl;
    // [sphinx-end]
    
    HIP_CHECK(hipFreeHost(ptrHost));
    HIP_CHECK(hipFree(ptr));

    return EXIT_SUCCESS;
}
