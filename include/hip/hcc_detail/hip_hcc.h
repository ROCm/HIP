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

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_HCC_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_HCC_H

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


#endif // #ifdef __HCC__

hipError_t hipHccModuleLaunchKernel(hipFunction_t f,
                                    uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY,
                                    uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX,
                                    uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ,
                                    size_t sharedMemBytes,
                                    hipStream_t hStream,
                                    void **kernelParams,
                                    void **extra);

#endif // #if __cplusplus

#endif //
