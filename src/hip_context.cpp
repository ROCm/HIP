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

#include "hip/hip_runtime.h"
#include "hip/hcc_detail/hip_hcc.h"
#include "hip/hcc_detail/trace_helper.h"

// Stack of contexts
thread_local std::stack<ihipCtx_t *>  tls_ctxStack;

void ihipCtxStackUpdate()
{
    if(tls_ctxStack.empty()) {
         tls_ctxStack.push(ihipGetTlsDefaultCtx());
    }
}

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

hipError_t hipDriverGetVersion(int *driverVersion)
{
    HIP_INIT_API(driverVersion);
    hipError_t e = hipSuccess;
    if (driverVersion) {
        *driverVersion = 4;
    }
    else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}

hipError_t hipRuntimeGetVersion(int *runtimeVersion)
{
    HIP_INIT_API(runtimeVersion);
    hipError_t e = hipSuccess;
    if (runtimeVersion) {
        *runtimeVersion = HIP_VERSION_PATCH;
    }
    else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}

hipError_t hipCtxDestroy(hipCtx_t ctx)
{
    HIP_INIT_API(ctx);
    hipError_t e = hipSuccess;
    ihipCtx_t* currentCtx= ihipGetTlsDefaultCtx();
    ihipCtx_t* primaryCtx= ((ihipDevice_t*)ctx->getDevice())->_primaryCtx;
    if(primaryCtx== ctx)
    {
        e = hipErrorInvalidValue;
    }
    else
    {
        if(currentCtx == ctx) {
            //need to destroy the ctx associated with calling thread
            tls_ctxStack.pop();
        }
        delete ctx; //As per CUDA docs , attempting to access ctx from those threads which has this ctx as current, will result in the error HIP_ERROR_CONTEXT_IS_DESTROYED.
    }

    return ihipLogStatus(e);
}

hipError_t hipCtxPopCurrent(hipCtx_t* ctx)
{
    HIP_INIT_API(ctx);
    hipError_t e = hipSuccess;
    ihipCtx_t* tempCtx;
    *ctx = ihipGetTlsDefaultCtx();
    if(!tls_ctxStack.empty()) {
        tls_ctxStack.pop();
    }
    if(!tls_ctxStack.empty()) {
        tempCtx= tls_ctxStack.top();
    }
    else {
        tempCtx = nullptr;
    }

    ihipSetTlsDefaultCtx(tempCtx); //TOD0 - Shall check for NULL?
    return ihipLogStatus(e);
}

hipError_t hipCtxPushCurrent(hipCtx_t ctx)
{
    HIP_INIT_API(ctx);
    hipError_t e = hipSuccess;
    if(ctx != NULL) {    //TODO- is this check needed?
        ihipSetTlsDefaultCtx(ctx);
        tls_ctxStack.push(ctx);
    }
    else {
        e = hipErrorInvalidContext;
    }
    return ihipLogStatus(e);
}

hipError_t hipCtxGetCurrent(hipCtx_t* ctx)
{
    HIP_INIT_API(ctx);
    hipError_t e = hipSuccess;
    if(!tls_ctxStack.empty()) {
        *ctx= tls_ctxStack.top();
    }
    else {
        *ctx = NULL;
    }
    return ihipLogStatus(e);
}

hipError_t hipCtxSetCurrent(hipCtx_t ctx)
{
    HIP_INIT_API(ctx);
    hipError_t e = hipSuccess;
    if(ctx == NULL) {
        tls_ctxStack.pop();
    }
    else {
        ihipSetTlsDefaultCtx(ctx);
        tls_ctxStack.push(ctx);
    }
    return ihipLogStatus(e);
}

hipError_t hipCtxGetDevice(hipDevice_t *device)
{
    HIP_INIT_API(device);
    hipError_t e = hipSuccess;

    ihipCtx_t *ctx = ihipGetTlsDefaultCtx();

    if(ctx == nullptr) {
        e = hipErrorInvalidContext;
    }
    else {
        *device = (ihipDevice_t*)ctx->getDevice();
    }
    return ihipLogStatus(e);
}

hipError_t hipCtxGetApiVersion (hipCtx_t ctx,int *apiVersion)
{
    HIP_INIT_API(apiVersion);

    if (apiVersion) {
        *apiVersion = 4;
    }

    return ihipLogStatus(hipSuccess);
}

hipError_t hipCtxGetCacheConfig ( hipFuncCache *cacheConfig )
{
    HIP_INIT_API(cacheConfig);

    *cacheConfig = hipFuncCachePreferNone;

    return ihipLogStatus(hipSuccess);
}

hipError_t hipCtxSetCacheConfig ( hipFuncCache cacheConfig )
{
    HIP_INIT_API(cacheConfig);

    // Nop, AMD does not support variable cache configs.

    return ihipLogStatus(hipSuccess);
}

hipError_t hipCtxSetSharedMemConfig ( hipSharedMemConfig config )
{
    HIP_INIT_API(config);

    // Nop, AMD does not support variable shared mem configs.

    return ihipLogStatus(hipSuccess);
}

hipError_t hipCtxGetSharedMemConfig ( hipSharedMemConfig * pConfig )
{
    HIP_INIT_API(pConfig);

    *pConfig = hipSharedMemBankSizeFourByte;

    return ihipLogStatus(hipSuccess);
}

hipError_t hipCtxSynchronize ( void )
{
    HIP_INIT_API(1);
    return ihipSynchronize(); //TODP Shall check validity of ctx?
}

hipError_t hipCtxGetFlags ( unsigned int* flags )
{
    HIP_INIT_API(flags);
    hipError_t e = hipSuccess;
    ihipCtx_t* tempCtx;
    tempCtx = ihipGetTlsDefaultCtx();
    *flags = tempCtx->_ctxFlags;
    return ihipLogStatus(e);
}
