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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//---
// Driver initialization and reporting:

#include <stack>

#include "hip_runtime.h"
#include "hcc_detail/hip_hcc.h"
#include "hcc_detail/trace_helper.h"

// Stack of contexts 
thread_local std::stack<ihipCtx_t *>  tls_ctxStack;


hipError_t hipInit(unsigned int flags)
{
    HIP_INIT_API(flags);
    
    hipError_t e = hipSuccess;

    // Flags must be 0
    if (flags != 0) {
        e = hipErrorInvalidValue;
    } 

    return ihipLogStatus(e);
}


hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags,  hipDevice_t device)
{
    HIP_INIT_API(ctx, flags, device); // FIXME - review if we want to init 
    hipError_t e = hipSuccess;

    *ctx = new ihipCtx_t(device, g_deviceCnt, flags);
    ihipSetTlsDefaultCtx(*ctx);
    tls_ctxStack.push(*ctx);

    return ihipLogStatus(e);
}


hipError_t hipDeviceGet(hipDevice_t *device, int deviceId)
{
    HIP_INIT_API(device, deviceId); // FIXME - review if we want to init 

    *device = ihipGetDevice(deviceId);

    hipError_t e = hipSuccess;
    if (*device == NULL) {
        e = hipErrorInvalidDevice;
    }

    return ihipLogStatus(e);
};


/**
 * @return #hipSuccess
 */
//---
hipError_t hipDriverGetVersion(int *driverVersion)
{
    HIP_INIT_API(driverVersion);

    if (driverVersion) {
        *driverVersion = 4;
    }

    return ihipLogStatus(hipSuccess);
}
