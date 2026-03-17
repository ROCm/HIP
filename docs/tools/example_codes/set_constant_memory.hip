// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>

// [sphinx-start]
__constant__ int const_array[8];

void set_constant_memory()
{
    int host_data[8] {1,2,3,4,5,6,7,8};

    if(auto err = hipMemcpyToSymbol(const_array, host_data, sizeof(int) * 8); err != hipSuccess)
        std::cerr << "HIP error " << err << ": " << hipGetErrorString(err) << std::endl;

    // call kernel that accesses const_array
}
// [sphinx-end]

int main()
{
    set_constant_memory();
    std::cout << "Success!" << std::endl;
    return EXIT_SUCCESS;
}
