/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>
#include "hip/driver_types.h"

/**
 * @brief TODO
 * 
 */
TEST_CASE("Unit_hipMalloc3D_Negative") {

    hipPitchedPtr hipPitchedPtr;
    hipExtent invalidExtent = GENERATE({0, 0, 0}, {0,0,1}, {0, 1, 0}, {1, 0, 0});
    HIP_CHECK(hipMalloc3D(&hipPitchedPtr, invalidExtent));
    HIP_CHECK(hipHostFree(hipPitchedPtr.ptr));
    /* TODO Get free memory */

    hipExtent validExtent {1,1,1};
    HIP_CHECK_ERROR(hipMalloc3D(nullptr, validExtent), hipErrorInvalidValue);
}

TEST_CASE("Unit_hipMallocPitch" ) {

    void* ptr {nullptr};
    size_t pitch;

    struct 2DExtent{
        size_t width;
        size_t height;
    };

    2DExtent extent = GENERATE({0 , 0}, {0, 1}, {1, 0});

    HIP_CHECK(hipMallocPitch(&ptr, &pitch, extent.width, extent.height));



}