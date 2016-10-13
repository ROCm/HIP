/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HCC_ACC_H
#define HCC_ACC_H
#include "hip/hip_runtime_api.h"

#if __cplusplus
#ifdef __HCC__
#include <hc.hpp>
/**
 * @brief Return hc::accelerator associated with the specified deviceId
 * @return #hipSuccess, #hipErrorInvalidDevice
 */
hipError_t hipHccGetAccelerator(int deviceId, hc::accelerator *acc);

/**
 * @brief Return hc::accelerator_view associated with the specified stream
 *
 * If stream is 0, the accelerator_view for the default stream is returned.
 * @return #hipSuccess
 */
hipError_t hipHccGetAcceleratorView(hipStream_t stream, hc::accelerator_view **av);
#endif
#endif

#endif
